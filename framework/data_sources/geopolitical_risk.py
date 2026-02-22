"""Geopolitical Risk Index adapter (Caldara-Iacoviello GPR)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import NormalizedRecord


@dataclass(frozen=True)
class GeopoliticalRiskAdapter:
    name: str = "gpr"

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        """Synthetic GPR proxy; real GPR CSV integration deferred to when
        matteoiacoviello.com endpoint is verified accessible."""
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=30 * i)
            gpr_value = 100.0 + 5.0 * (i % 12) - 2.0 * (i % 7)
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="gpr_index",
                    target=gpr_value,
                    promo=0.0,
                    macro_index=gpr_value,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows
