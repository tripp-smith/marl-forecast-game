"""Earnings transcript qualitative adapter — deterministic synthetic stubs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random

from .base import NormalizedQualRecord

logger = logging.getLogger(__name__)

_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

_FISCAL_QUARTERS = [
    (1, 25),  # Q1 earnings ~late January
    (4, 25),  # Q2 earnings ~late April
    (7, 25),  # Q3 earnings ~late July
    (10, 25),  # Q4 earnings ~late October
]


def _earnings_dates(
    start_dt: datetime, end_dt: datetime
) -> list[tuple[datetime, str]]:
    """Generate quarterly earnings release dates for each ticker."""
    entries: list[tuple[datetime, str]] = []
    for year in range(start_dt.year, end_dt.year + 1):
        for month, day in _FISCAL_QUARTERS:
            dt = datetime(year, month, day, 17, 0, 0)
            if start_dt <= dt <= end_dt:
                for ticker in _TICKERS:
                    entries.append((dt, ticker))
    return sorted(entries, key=lambda e: e[0])


@dataclass(frozen=True)
class EarningsAdapter:
    """Generates deterministic synthetic earnings-call transcript stubs."""

    name: str = "earnings"
    cache_dir: str = "data/test_qualitative/earnings"
    max_records: int = 50

    def fetch_releases(
        self, start_dt: datetime, end_dt: datetime
    ) -> list[NormalizedQualRecord]:
        """Return earnings transcript records for releases within the date range."""
        entries = _earnings_dates(start_dt, end_dt)
        rng = Random(42)
        if len(entries) > self.max_records:
            entries = rng.sample(entries, self.max_records)
            entries.sort(key=lambda e: e[0])

        records: list[NormalizedQualRecord] = []
        for dt, ticker in entries:
            cached = self._try_cache(dt, ticker)
            if cached is not None:
                records.append(cached)
            else:
                records.append(self._synthetic_record(dt, ticker))
        return records

    def _try_cache(
        self, dt: datetime, ticker: str
    ) -> NormalizedQualRecord | None:
        path = (
            Path(self.cache_dir)
            / f"{dt.strftime('%Y-%m-%d')}_{ticker}.txt"
        )
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return NormalizedQualRecord(
                    timestamp=dt,
                    source_id=self.name,
                    text=text,
                    metadata={"ticker": ticker, "local_path": str(path)},
                )
        return None

    def _synthetic_record(
        self, dt: datetime, ticker: str
    ) -> NormalizedQualRecord:
        rng = Random(42 + hash(ticker) + dt.year * 100 + dt.month)
        outlooks = ["positive", "cautious", "optimistic", "uncertain"]
        drivers = [
            "strong consumer demand",
            "cloud services growth",
            "digital advertising gains",
            "margin expansion",
            "international headwinds",
        ]
        outlook = rng.choice(outlooks)
        driver = rng.choice(drivers)
        quarter = {1: "Q4", 4: "Q1", 7: "Q2", 10: "Q3"}.get(dt.month, "Q1")
        rev_growth = rng.uniform(-5, 15)
        text = (
            f"{ticker} {quarter} {dt.year} Earnings Call Transcript. "
            f"CEO remarks: We are {outlook} about the coming quarter. "
            f"Revenue grew {rev_growth:.1f}% year-over-year driven by "
            f"{driver}. CFO guidance: We expect continued momentum "
            f"with CapEx investments in AI and infrastructure."
        )

        path = (
            Path(self.cache_dir)
            / f"{dt.strftime('%Y-%m-%d')}_{ticker}.txt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

        return NormalizedQualRecord(
            timestamp=dt,
            source_id=self.name,
            text=text,
            metadata={"ticker": ticker, "synthetic": True},
        )
