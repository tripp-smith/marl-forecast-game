from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from typing import Any

import json
from urllib.parse import urlencode
from urllib.request import urlopen

from .base import NormalizedRecord

MULTI_SERIES_IDS = ["CPIAUCSL", "GDP", "UNRATE", "FEDFUNDS", "T10YIE"]


@dataclass(frozen=True)
class FredMacroAdapter:
    name: str = "fred"
    series_id: str = "CPIAUCSL"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=i)
            macro = 100.0 + (0.08 * i)
            target = 50.0 + (0.12 * i) + 0.1 * macro
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="fred_gdp_proxy",
                    target=target,
                    promo=0.0,
                    macro_index=macro,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    def _fetch_series(self, sid: str, key: str, periods: int) -> list[dict[str, Any]]:
        params = {
            "series_id": sid,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": max(5, periods),
        }
        query = urlencode(params)
        with urlopen(f"https://api.stlouisfed.org/fred/series/observations?{query}", timeout=10) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        observations = payload.get("observations", [])
        parsed = [o for o in observations if o.get("value") not in {".", None}]
        return list(reversed(parsed[-periods:]))

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        key = os.getenv("FRED_API_KEY")
        if not key:
            return self._synthetic(periods)

        try:
            parsed = self._fetch_series(self.series_id, key, periods)
            now = datetime.utcnow()
            rows: list[NormalizedRecord] = []
            for idx, obs in enumerate(parsed):
                value = float(obs["value"])
                ts = datetime.fromisoformat(obs["date"])
                rows.append(
                    NormalizedRecord(
                        timestamp=ts,
                        series_id=f"fred_{self.series_id.lower()}",
                        target=value,
                        promo=0.0,
                        macro_index=100.0 + (0.03 * idx),
                        source=self.name,
                        fetched_at=now,
                    )
                )
            return rows if rows else self._synthetic(periods)
        except Exception:
            return self._synthetic(periods)

    def fetch_multi_series(self, periods: int = 60) -> list[dict[str, Any]]:
        """Fetch multiple FRED series and merge into time-aligned rows.

        Each row has 'timestamp', 'target' (from CPIAUCSL), standard schema
        fields, and a 'macro_context' dict mapping series names to values.
        """
        key = os.getenv("FRED_API_KEY")
        if not key:
            logging.info("FRED_API_KEY not set; using synthetic multi-series proxy")
            return self._synthetic_multi(periods)

        try:
            series_data: dict[str, dict[str, float]] = {}
            for sid in MULTI_SERIES_IDS:
                observations = self._fetch_series(sid, key, periods)
                for obs in observations:
                    date_str = obs["date"]
                    value = float(obs["value"])
                    series_data.setdefault(date_str, {})[sid] = value

            all_dates = sorted(series_data.keys())
            if not all_dates:
                return self._synthetic_multi(periods)

            now = datetime.utcnow()
            rows: list[dict[str, Any]] = []
            last_known: dict[str, float] = {}
            for date_str in all_dates[-periods:]:
                values = series_data[date_str]
                for sid in MULTI_SERIES_IDS:
                    if sid in values:
                        last_known[sid] = values[sid]
                    elif sid not in last_known:
                        last_known[sid] = 0.0

                target = last_known.get("CPIAUCSL", 0.0)
                macro_ctx = {sid: last_known.get(sid, 0.0) for sid in MULTI_SERIES_IDS}
                rows.append({
                    "timestamp": datetime.fromisoformat(date_str),
                    "series_id": "fred_multi",
                    "target": target,
                    "promo": 0.0,
                    "macro_index": last_known.get("GDP", 100.0),
                    "source": self.name,
                    "fetched_at": now,
                    "macro_context": macro_ctx,
                })
            return rows if rows else self._synthetic_multi(periods)
        except Exception:
            logging.warning("FRED multi-series fetch failed; falling back to synthetic", exc_info=True)
            return self._synthetic_multi(periods)

    def _synthetic_multi(self, periods: int) -> list[dict[str, Any]]:
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[dict[str, Any]] = []
        for i in range(periods):
            ts = start + timedelta(days=i)
            macro_ctx = {
                "CPIAUCSL": 250.0 + 0.08 * i,
                "GDP": 25000.0 + 5.0 * i,
                "UNRATE": 3.5 + 0.01 * (i % 20),
                "FEDFUNDS": 5.25 - 0.005 * i,
                "T10YIE": 2.3 + 0.002 * i,
            }
            rows.append({
                "timestamp": ts,
                "series_id": "fred_multi",
                "target": macro_ctx["CPIAUCSL"],
                "promo": 0.0,
                "macro_index": macro_ctx["GDP"],
                "source": self.name,
                "fetched_at": now,
                "macro_context": macro_ctx,
            })
        return rows
