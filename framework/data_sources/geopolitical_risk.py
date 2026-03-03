"""Geopolitical Risk Index adapter (Caldara-Iacoviello GPR).

Attempts to download the GPR daily CSV from matteoiacoviello.com.
Falls back to a local file path, then to synthetic data.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

from .base import NormalizedRecord
from .retry import RateLimiter, retry

GPR_CSV_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

_gpr_rate_limiter = RateLimiter(calls_per_second=5)


@dataclass(frozen=True)
class GeopoliticalRiskAdapter:
    """Fetches Caldara-Iacoviello GPR index data from a local file or remote CSV."""

    name: str = "gpr"
    local_path: str = "data/gpr_daily.csv"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
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

    def _parse_tabular(self, text: str, periods: int) -> list[NormalizedRecord]:
        """Parse tab/comma-separated GPR data. The file has columns like
        date, GPRD, GPR, ...  -- we extract the first date-like and GPR column."""
        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []

        lines = text.strip().splitlines()
        if not lines:
            return []

        sep = "\t" if "\t" in lines[0] else ","
        reader = csv.reader(lines, delimiter=sep)
        header = next(reader, None)
        if header is None:
            return []

        header_lower = [h.strip().lower() for h in header]
        date_col = None
        gpr_col = None
        for i, h in enumerate(header_lower):
            if h in {"date", "month", "day"}:
                date_col = i
            if h in {"gpr", "gprd", "gpr_index"} and gpr_col is None:
                gpr_col = i

        if date_col is None or gpr_col is None:
            return []

        for row in reader:
            if len(row) <= max(date_col, gpr_col):
                continue
            try:
                date_str = row[date_col].strip()
                for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m"):
                    try:
                        ts = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue
                value = float(row[gpr_col].strip())
                rows.append(
                    NormalizedRecord(
                        timestamp=ts,
                        series_id="gpr_index",
                        target=value,
                        promo=0.0,
                        macro_index=value,
                        source=self.name,
                        fetched_at=now,
                    )
                )
            except (ValueError, IndexError):
                continue

        rows.sort(key=lambda r: r.timestamp)
        return rows[-periods:] if rows else []

    @retry(max_attempts=3)
    def _fetch_remote(self, periods: int) -> list[NormalizedRecord]:
        _gpr_rate_limiter.acquire()
        with urlopen(GPR_CSV_URL, timeout=15) as resp:
            raw = resp.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        return self._parse_tabular(text, periods)

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        """Return up to *periods* GPR index records from local file, remote, or synthetic data."""
        p = Path(self.local_path)
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8")
                parsed = self._parse_tabular(text, periods)
                if parsed:
                    return parsed
            except Exception:
                pass

        try:
            parsed = self._fetch_remote(periods)
            if parsed:
                return parsed
        except Exception:
            logging.debug("GPR remote fetch failed; using synthetic fallback", exc_info=True)

        return self._synthetic(periods)
