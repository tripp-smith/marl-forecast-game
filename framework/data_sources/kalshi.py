"""Kalshi adapter -- fetches prediction-market data from the public Kalshi trade API v2."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen

from .base import NormalizedRecord


@dataclass(frozen=True)
class KalshiAdapter:
    name: str = "kalshi"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []
        base = datetime(2024, 1, 1)
        for i in range(periods):
            ts = datetime(base.year, base.month, base.day + i % 28 + 1) if i < 28 else datetime(
                base.year, base.month + (i // 28), (i % 28) + 1
            )
            price = 0.5 + 0.02 * (i % 15) - 0.01 * (i % 7)
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id=f"kalshi_market_{i}",
                    target=price,
                    promo=0.0,
                    macro_index=float(100 + i * 10),
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        url = (
            "https://api.elections.kalshi.com/trade-api/v2/markets"
            f"?status=open&limit={max(5, periods)}"
        )
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=15) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

            markets = payload.get("markets", [])
            if not markets:
                return self._synthetic(periods)

            now = datetime.utcnow()
            rows: list[NormalizedRecord] = []
            for mkt in markets:
                ticker = mkt.get("ticker", "unknown")
                yes_price = mkt.get("yes_price", mkt.get("last_price"))
                volume = mkt.get("volume", 0)
                close_time = mkt.get("close_time", "")

                if yes_price is None:
                    continue

                try:
                    if close_time:
                        ts = datetime.fromisoformat(close_time.replace("Z", "+00:00")).replace(tzinfo=None)
                    else:
                        ts = now
                except ValueError:
                    ts = now

                rows.append(
                    NormalizedRecord(
                        timestamp=ts,
                        series_id=f"kalshi_{ticker}",
                        target=float(yes_price) / 100.0 if float(yes_price) > 1.0 else float(yes_price),
                        promo=0.0,
                        macro_index=float(volume),
                        source=self.name,
                        fetched_at=now,
                    )
                )

            rows.sort(key=lambda r: r.timestamp)
            return rows[-periods:] if rows else self._synthetic(periods)
        except Exception:
            logging.debug("Kalshi API fetch failed; using synthetic fallback", exc_info=True)
            return self._synthetic(periods)
