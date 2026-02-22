"""BIS Policy Rate adapter."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import NormalizedRecord


@dataclass(frozen=True)
class BISPolicyRateAdapter:
    name: str = "bis"

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        """Synthetic BIS policy rate proxy; real CSV integration deferred."""
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
