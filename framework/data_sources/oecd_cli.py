"""OECD Composite Leading Indicators adapter."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import NormalizedRecord


@dataclass(frozen=True)
class OECDCLIAdapter:
    name: str = "oecd_cli"

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        """Synthetic OECD CLI proxy; real JSON API integration deferred."""
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=30 * i)
            cli_value = 99.5 + 0.1 * i + 0.3 * (i % 6)
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="oecd_cli",
                    target=cli_value,
                    promo=0.0,
                    macro_index=cli_value,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows
