"""Eurostat adapter -- fetches EU quarterly GDP from the public Eurostat SDMX API."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen

from .base import NormalizedRecord
from .retry import RateLimiter, retry

_eurostat_rate_limiter = RateLimiter(calls_per_second=5)

_QUARTER_MONTHS = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}


@dataclass(frozen=True)
class EurostatAdapter:
    """Fetches EU quarterly GDP from the Eurostat SDMX API, with synthetic fallback."""

    name: str = "eurostat"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []
        base_year = 2015
        for i in range(periods):
            year = base_year + (i // 4)
            q = (i % 4) + 1
            month = _QUARTER_MONTHS[f"Q{q}"]
            gdp = 2800000.0 + 20000.0 * i + 5000.0 * (i % 3)
            rows.append(
                NormalizedRecord(
                    timestamp=datetime(year, month, 1),
                    series_id="eurostat_gdp",
                    target=gdp,
                    promo=0.0,
                    macro_index=gdp / 1000.0,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    @retry(max_attempts=3)
    def _fetch_api(self, periods: int) -> list[NormalizedRecord]:
        url = (
            "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/"
            "data/namq_10_gdp/Q.CLV10_MEUR.SCA.B1GQ.EA20"
            f"?format=JSON&lastNObservations={max(5, periods)}"
        )
        req = Request(url, headers={"Accept": "application/json"})
        _eurostat_rate_limiter.acquire()
        with urlopen(req, timeout=15) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

        dimension = payload.get("dimension", {})
        time_info = dimension.get("time", {}).get("category", {}).get("index", {})
        if not time_info:
            return []

        time_labels = sorted(time_info.items(), key=lambda kv: kv[1])
        values_map = payload.get("value", {})
        if not values_map:
            return []

        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []
        for label, idx in time_labels:
            val = values_map.get(str(idx))
            if val is None:
                continue
            try:
                if "Q" in label:
                    parts = label.split("-Q")
                    year = int(parts[0])
                    month = _QUARTER_MONTHS.get(f"Q{parts[1]}", 1)
                elif "-" in label:
                    year = int(label.split("-")[0])
                    month = int(label.split("-")[1])
                else:
                    year = int(label)
                    month = 1
                ts = datetime(year, month, 1)
            except (ValueError, IndexError):
                continue
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="eurostat_gdp",
                    target=float(val),
                    promo=0.0,
                    macro_index=float(val) / 1000.0,
                    source=self.name,
                    fetched_at=now,
                )
            )

        rows.sort(key=lambda r: r.timestamp)
        return rows[-periods:]

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        """Return up to *periods* EU GDP records from Eurostat or synthetic data."""
        try:
            rows = self._fetch_api(periods)
            return rows if rows else self._synthetic(periods)
        except Exception:
            logging.debug("Eurostat API fetch failed; using synthetic fallback", exc_info=True)
            return self._synthetic(periods)
