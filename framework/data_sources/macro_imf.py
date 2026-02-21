from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import NormalizedRecord


@dataclass(frozen=True)
class ImfMacroAdapter:
    name: str = "imf"

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=i)
            macro = 98.5 + (0.06 * i)
            target = 42.0 + (0.11 * i) + 0.12 * macro
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="imf_weo_proxy",
                    target=target,
                    promo=0.0,
                    macro_index=macro,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows
