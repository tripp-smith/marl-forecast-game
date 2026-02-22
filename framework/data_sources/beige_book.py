"""Beige Book qualitative adapter — Federal Reserve public archives."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any

from .base import NormalizedQualRecord

logger = logging.getLogger(__name__)

_ARCHIVE_BASE = (
    "https://www.federalreserve.gov/monetarypolicy/beigebook/"
)

_RELEASE_MONTHS = [1, 3, 4, 6, 7, 9, 10, 12]
_RELEASE_DAY = 15


def _beige_book_dates(start_dt: datetime, end_dt: datetime) -> list[datetime]:
    """Generate approximate Beige Book release dates within the range."""
    dates: list[datetime] = []
    for year in range(start_dt.year, end_dt.year + 1):
        for month in _RELEASE_MONTHS:
            dt = datetime(year, month, _RELEASE_DAY, 14, 0, 0)
            if start_dt <= dt <= end_dt:
                dates.append(dt)
    return sorted(dates)


@dataclass(frozen=True)
class BeigeBookAdapter:
    name: str = "beige_book"
    cache_dir: str = "data/test_qualitative/beige_book"

    def fetch_releases(
        self, start_dt: datetime, end_dt: datetime
    ) -> list[NormalizedQualRecord]:
        dates = _beige_book_dates(start_dt, end_dt)
        records: list[NormalizedQualRecord] = []
        for dt in dates:
            cached = self._try_cache(dt)
            if cached is not None:
                records.append(cached)
                continue
            fetched = self._try_fetch(dt)
            if fetched is not None:
                records.append(fetched)
            else:
                records.append(self._synthetic_record(dt))
        return records

    def _try_cache(self, dt: datetime) -> NormalizedQualRecord | None:
        path = Path(self.cache_dir) / f"{dt.strftime('%Y-%m-%d')}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return NormalizedQualRecord(
                    timestamp=dt,
                    source_id=self.name,
                    text=text,
                    metadata={"local_path": str(path)},
                )
        return None

    def _try_fetch(self, dt: datetime) -> NormalizedQualRecord | None:
        try:
            import requests

            slug = dt.strftime("%Y%m%d")
            url = f"{_ARCHIVE_BASE}{slug}-full.htm"
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                url = f"{_ARCHIVE_BASE}{slug}.htm"
                resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                return None

            text = self._extract_text_from_html(resp.text)
            if not text.strip():
                return None

            self._write_cache(dt, text)
            return NormalizedQualRecord(
                timestamp=dt,
                source_id=self.name,
                text=text,
                metadata={"url": url},
            )
        except Exception:
            logger.debug("BeigeBookAdapter fetch failed for %s", dt.isoformat())
            return None

    def _extract_text_from_html(self, html: str) -> str:
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _write_cache(self, dt: datetime, text: str) -> None:
        path = Path(self.cache_dir) / f"{dt.strftime('%Y-%m-%d')}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _synthetic_record(self, dt: datetime) -> NormalizedQualRecord:
        rng = Random(42 + dt.year * 1000 + dt.month)
        sentiments = ["positive", "cautious", "mixed", "stable"]
        regions = ["Northeast", "Midwest", "South", "West"]
        sentiment = rng.choice(sentiments)
        region = rng.choice(regions)
        text = (
            f"Overall economic activity was {sentiment} across districts. "
            f"The {region} region reported moderate growth in employment "
            f"and consumer spending. Labor markets remained tight with "
            f"modest wage pressures. Prices increased at a moderate pace."
        )
        return NormalizedQualRecord(
            timestamp=dt,
            source_id=self.name,
            text=text,
            metadata={"synthetic": True},
        )
