from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import NormalizedRecord


@dataclass(frozen=True)
class FredMacroAdapter:
    name: str = "fred"

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
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
