"""Polymarket adapter — fetches prediction-market probabilities from the Gamma API."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import json
from urllib.parse import urlencode
from urllib.request import urlopen

from .base import NormalizedRecord
from .retry import RateLimiter, retry

_polymarket_rate_limiter = RateLimiter(calls_per_second=5)


@dataclass(frozen=True)
class PolymarketAdapter:
    """Fetches open-market probabilities from the Polymarket Gamma API, with synthetic fallback."""

    name: str = "polymarket"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=i)
            probability = 0.45 + (0.002 * i)
            target = 35.0 + 8.0 * probability + 0.05 * i
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="polymarket_macro_contract",
                    target=target,
                    promo=1.0 if i % 10 == 0 else 0.0,
                    macro_index=100.0 + (0.03 * i),
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    @retry(max_attempts=3)
    def _fetch_api(self, periods: int) -> list[NormalizedRecord]:
        url = "https://gamma-api.polymarket.com/markets"
        query = urlencode({"closed": "false", "limit": periods})
        _polymarket_rate_limiter.acquire()
        with urlopen(f"{url}?{query}", timeout=10) as response:
            payload: list[dict[str, Any]] = json.loads(response.read().decode("utf-8"))
        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []
        for idx, market in enumerate(payload[:periods]):
            ts = now - timedelta(days=(periods - idx))
            prob = float(market.get("probability", 0.5))
            volume = float(market.get("volume", 0.0))
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id=str(market.get("slug", "polymarket_market")),
                    target=prob,
                    promo=1.0 if volume > 100000 else 0.0,
                    macro_index=volume / 10000.0,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        """Return up to *periods* market-probability records from Polymarket or synthetic data."""
        try:
            rows = self._fetch_api(periods)
            return rows if rows else self._synthetic(periods)
        except Exception:
            return self._synthetic(periods)
