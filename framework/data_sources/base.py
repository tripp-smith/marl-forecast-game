from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, NamedTuple, Protocol


@dataclass(frozen=True)
class NormalizedRecord:
    timestamp: datetime
    series_id: str
    target: float
    promo: float
    macro_index: float
    source: str
    fetched_at: datetime

    def as_row(self) -> dict:
        row = asdict(self)
        row["timestamp"] = self.timestamp
        row["fetched_at"] = self.fetched_at
        return row


class SourceAdapter(Protocol):
    name: str

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]: ...


class NormalizedQualRecord(NamedTuple):
    timestamp: datetime
    source_id: str
    text: str
    metadata: dict[str, Any]


class QualitativeAdapter(Protocol):
    name: str

    def fetch_releases(
        self, start_dt: datetime, end_dt: datetime
    ) -> list[NormalizedQualRecord]: ...
