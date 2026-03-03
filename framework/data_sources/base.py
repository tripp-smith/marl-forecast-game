"""Base types for data-source adapters: normalized records and adapter protocols."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, NamedTuple, Protocol


@dataclass(frozen=True)
class NormalizedRecord:
    """A single quantitative observation normalized across all data sources."""
    timestamp: datetime
    series_id: str
    target: float
    promo: float
    macro_index: float
    source: str
    fetched_at: datetime

    def as_row(self) -> dict[str, Any]:
        """Return the record as a plain dict suitable for DataFrame ingestion."""
        row = asdict(self)
        row["timestamp"] = self.timestamp
        row["fetched_at"] = self.fetched_at
        return row


class SourceAdapter(Protocol):
    """Protocol for quantitative data-source adapters."""

    name: str

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]: ...


class NormalizedQualRecord(NamedTuple):
    """A single qualitative observation (e.g. text release) normalized across sources."""
    timestamp: datetime
    source_id: str
    text: str
    metadata: dict[str, Any]


class QualitativeAdapter(Protocol):
    """Protocol for qualitative (text-based) data-source adapters."""

    name: str

    def fetch_releases(
        self, start_dt: datetime, end_dt: datetime
    ) -> list[NormalizedQualRecord]: ...
