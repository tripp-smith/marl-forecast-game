"""BIS Policy Rate adapter -- fetches from BIS Data API v2 (public, no key)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.request import Request, urlopen

from .base import NormalizedRecord
from .retry import RateLimiter, retry

_bis_rate_limiter = RateLimiter(calls_per_second=5)


@dataclass(frozen=True)
class BISPolicyRateAdapter:
    """Fetches central-bank policy rates from the BIS SDMX API, with synthetic fallback."""

    name: str = "bis"
    country: str = "US"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=30 * i)
            rate = 5.25 - 0.05 * (i % 10)
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="bis_policy_rate",
                    target=rate,
                    promo=0.0,
                    macro_index=rate * 20.0,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    @retry(max_attempts=3)
    def _fetch_api(self, periods: int) -> list[NormalizedRecord]:
        url = (
            "https://data.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/"
            f"M.{self.country}?format=jsondata&detail=dataonly"
            f"&lastNObservations={max(5, periods)}"
        )
        req = Request(url, headers={"Accept": "application/json"})
        _bis_rate_limiter.acquire()
        with urlopen(req, timeout=15) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

        datasets = payload.get("dataSets", [{}])
        if not datasets:
            return []

        series_map = datasets[0].get("series", {})
        if not series_map:
            return []

        dims = payload.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_dim = next((d for d in dims if d.get("id") == "TIME_PERIOD"), None)
        time_values = [v["id"] for v in time_dim["values"]] if time_dim else []

        first_key = next(iter(series_map))
        observations = series_map[first_key].get("observations", {})

        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []
        for obs_idx, obs_val in sorted(observations.items(), key=lambda kv: int(kv[0])):
            idx = int(obs_idx)
            rate = float(obs_val[0])
            ts_str = time_values[idx] if idx < len(time_values) else None
            if ts_str is None:
                continue
            try:
                ts = datetime.strptime(ts_str, "%Y-%m")
            except ValueError:
                try:
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    continue

            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id=f"bis_policy_rate_{self.country.lower()}",
                    target=rate,
                    promo=0.0,
                    macro_index=rate * 20.0,
                    source=self.name,
                    fetched_at=now,
                )
            )

        return rows[-periods:]

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        """Return up to *periods* policy-rate records from BIS or synthetic data."""
        try:
            rows = self._fetch_api(periods)
            return rows if rows else self._synthetic(periods)
        except Exception:
            logging.debug("BIS API fetch failed; using synthetic fallback", exc_info=True)
            return self._synthetic(periods)
