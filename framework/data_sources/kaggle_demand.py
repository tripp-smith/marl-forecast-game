"""Kaggle demand-forecasting adapter — loads locally-downloaded CSVs with auto-detected columns."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .base import NormalizedRecord
from .retry import RateLimiter, retry

_kaggle_rate_limiter = RateLimiter(calls_per_second=5)

COLUMN_ALIASES = {
    "date": "timestamp",
    "ds": "timestamp",
    "sales": "target",
    "demand": "target",
    "y": "target",
    "item": "series_id",
    "store": "series_id",
    "item_id": "series_id",
}


@dataclass(frozen=True)
class KaggleDemandAdapter:
    """Loads locally-downloaded Kaggle demand datasets.

    Supports the 'Store Item Demand Forecasting Challenge' CSV format
    and similar datasets with auto-detected column mappings.
    """

    name: str = "kaggle"
    path: str = "data/kaggle_demand.csv"

    def _detect_mapping(self, headers: list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        lower_headers = [h.lower().strip() for h in headers]
        for i, h in enumerate(lower_headers):
            if h in COLUMN_ALIASES:
                canonical = COLUMN_ALIASES[h]
                if canonical not in mapping:
                    mapping[canonical] = headers[i]
            elif h in {"timestamp", "target", "series_id"}:
                mapping[h] = headers[i]
        return mapping

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.now()
        start = datetime(2022, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=i)
            target = 30.0 + 0.08 * i + 5.0 * ((i % 7) / 7.0)
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="kaggle_synthetic",
                    target=round(target, 3),
                    promo=0.0,
                    macro_index=100.0 + 0.02 * i,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        """Return up to *periods* demand records from a local Kaggle CSV or synthetic data."""
        p = Path(self.path)
        if not p.exists():
            logging.info("Kaggle CSV not found at %s; using synthetic proxy", self.path)
            return self._synthetic(periods)

        try:
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers: list[str] = list(reader.fieldnames or [])
                mapping = self._detect_mapping(headers)

                if "timestamp" not in mapping or "target" not in mapping:
                    logging.warning("Cannot auto-detect columns in %s; using synthetic", self.path)
                    return self._synthetic(periods)

                now = datetime.now()
                rows: list[NormalizedRecord] = []
                for raw in reader:
                    try:
                        ts_val = raw[mapping["timestamp"]]
                        ts = datetime.fromisoformat(ts_val) if "T" in ts_val else datetime.strptime(ts_val, "%Y-%m-%d")
                    except (ValueError, KeyError):
                        continue

                    try:
                        target = float(raw[mapping["target"]])
                    except (ValueError, KeyError):
                        continue

                    sid_col = mapping.get("series_id")
                    series_id = raw.get(sid_col, "kaggle_default") if sid_col else "kaggle_default"

                    rows.append(
                        NormalizedRecord(
                            timestamp=ts,
                            series_id=str(series_id),
                            target=target,
                            promo=0.0,
                            macro_index=100.0,
                            source=self.name,
                            fetched_at=now,
                        )
                    )

                rows.sort(key=lambda r: (r.series_id, r.timestamp))
                return rows[:periods] if rows else self._synthetic(periods)

        except Exception:
            logging.warning("Failed to load Kaggle CSV at %s; using synthetic", self.path, exc_info=True)
            return self._synthetic(periods)
