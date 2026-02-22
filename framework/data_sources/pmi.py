"""PMI Commentary qualitative adapter — ISM public reports."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random

from .base import NormalizedQualRecord

logger = logging.getLogger(__name__)


def _pmi_dates(start_dt: datetime, end_dt: datetime) -> list[datetime]:
    """Generate first-business-day dates for each month in range."""
    dates: list[datetime] = []
    year, month = start_dt.year, start_dt.month
    while True:
        day = 1
        dt = datetime(year, month, day, 10, 0, 0)
        while dt.weekday() >= 5:
            day += 1
            dt = datetime(year, month, day, 10, 0, 0)
        if dt > end_dt:
            break
        if dt >= start_dt:
            dates.append(dt)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return sorted(dates)


@dataclass(frozen=True)
class PMIAdapter:
    name: str = "pmi"
    cache_dir: str = "data/test_qualitative/pmi"

    def fetch_releases(
        self, start_dt: datetime, end_dt: datetime
    ) -> list[NormalizedQualRecord]:
        dates = _pmi_dates(start_dt, end_dt)
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

            url = (
                "https://www.ismworld.org/supply-management-news-and-reports/"
                "reports/ism-report-on-business/pmi/"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                return None
            text = self._extract_commentary(resp.text)
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
            logger.debug("PMIAdapter fetch failed for %s", dt.isoformat())
            return None

    def _extract_commentary(self, html: str) -> str:
        import re

        match = re.search(
            r'class=["\']report-commentary["\'][^>]*>(.*?)</div>',
            html,
            re.DOTALL,
        )
        if match:
            raw = match.group(1)
        else:
            raw = html
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000]

    def _write_cache(self, dt: datetime, text: str) -> None:
        path = Path(self.cache_dir) / f"{dt.strftime('%Y-%m-%d')}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _synthetic_record(self, dt: datetime) -> NormalizedQualRecord:
        rng = Random(42 + dt.year * 100 + dt.month)
        conditions = ["expanding", "contracting", "stable", "mixed"]
        pressures = ["rising", "falling", "steady", "easing"]
        condition = rng.choice(conditions)
        pressure = rng.choice(pressures)
        text = (
            f"Manufacturing PMI commentary for {dt.strftime('%B %Y')}: "
            f"The manufacturing sector was {condition}. New orders were "
            f"{rng.choice(conditions)} while supply chain pressures were "
            f"{pressure}. Prices paid by manufacturers were {pressure}. "
            f"Employment in the sector showed {rng.choice(conditions)} trends."
        )
        return NormalizedQualRecord(
            timestamp=dt,
            source_id=self.name,
            text=text,
            metadata={"synthetic": True},
        )
