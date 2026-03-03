"""BEA adapter -- fetches NIPA GDP data (requires BEA_API_KEY env var)."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen

from .base import NormalizedRecord
from .retry import RateLimiter, retry

_bea_rate_limiter = RateLimiter(calls_per_second=5)

_QUARTER_MONTHS = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}


@dataclass(frozen=True)
class BEAAdapter:
    """Fetches NIPA GDP data from the BEA API, with synthetic fallback."""

    name: str = "bea"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []
        base_year = 2015
        for i in range(periods):
            year = base_year + (i // 4)
            q = (i % 4) + 1
            month = _QUARTER_MONTHS[f"Q{q}"]
            gdp = 18000.0 + 200.0 * i + 50.0 * (i % 3)
            rows.append(
                NormalizedRecord(
                    timestamp=datetime(year, month, 1),
                    series_id="bea_gdp",
                    target=gdp,
                    promo=0.0,
                    macro_index=gdp / 100.0,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    @retry(max_attempts=3)
    def _fetch_api(self, api_key: str, periods: int) -> list[NormalizedRecord]:
        url = (
            "https://apps.bea.gov/api/data/"
            "?method=GetData&DataSetName=NIPA&TableName=T10101"
            f"&Frequency=Q&Year=ALL&ResultFormat=JSON&UserID={api_key}"
        )
        req = Request(url, headers={"Accept": "application/json"})
        _bea_rate_limiter.acquire()
        with urlopen(req, timeout=20) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

        results = payload.get("BEAAPI", {}).get("Results", {})
        data_entries = results.get("Data", [])
        if not data_entries:
            return []

        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []
        for entry in data_entries:
            line_desc = entry.get("LineDescription", "")
            if "gross domestic product" not in line_desc.lower():
                continue
            period = entry.get("TimePeriod", "")
            val_str = entry.get("DataValue", "")
            if not period or not val_str:
                continue
            try:
                year_str, q_str = period[:4], period[4:]
                month = _QUARTER_MONTHS.get(q_str, 1)
                ts = datetime(int(year_str), month, 1)
                val = float(val_str.replace(",", ""))
            except (ValueError, KeyError):
                continue
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="bea_gdp",
                    target=val,
                    promo=0.0,
                    macro_index=val / 100.0,
                    source=self.name,
                    fetched_at=now,
                )
            )

        rows.sort(key=lambda r: r.timestamp)
        return rows[-periods:]

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        """Return up to *periods* GDP records from BEA or synthetic data."""
        api_key = os.environ.get("BEA_API_KEY", "")
        if not api_key:
            return self._synthetic(periods)

        try:
            rows = self._fetch_api(api_key, periods)
            return rows if rows else self._synthetic(periods)
        except Exception:
            logging.debug("BEA API fetch failed; using synthetic fallback", exc_info=True)
            return self._synthetic(periods)
