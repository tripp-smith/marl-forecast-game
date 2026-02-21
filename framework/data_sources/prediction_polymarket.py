from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import NormalizedRecord


@dataclass(frozen=True)
class PolymarketAdapter:
    name: str = "polymarket"

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
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
