from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import csv

from .data_sources import FredMacroAdapter, PolymarketAdapter


REQUIRED_COLUMNS = ["timestamp", "series_id", "target", "promo", "macro_index"]


@dataclass(frozen=True)
class DatasetBundle:
    train: list[dict]
    valid: list[dict]
    test: list[dict]


@dataclass(frozen=True)
class DataProfile:
    source: str = "sample_csv"
    periods: int = 240
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    normalize: bool = True


def _ensure_required(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("dataset is empty")
    missing = [col for col in REQUIRED_COLUMNS if col not in rows[0].keys()]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _validate_rows(rows: list[dict]) -> None:
    _ensure_required(rows)
    per_series: dict[str, datetime] = {}
    for row in rows:
        ts = row["timestamp"]
        if row["target"] is None or row["promo"] is None or row["macro_index"] is None:
            raise ValueError("missing numeric values in row")
        last = per_series.get(row["series_id"])
        if last is not None and ts < last:
            raise ValueError("timestamps must be chronological within each series")
        per_series[row["series_id"]] = ts


def load_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    _ensure_required(rows)

    for row in rows:
        row["timestamp"] = datetime.fromisoformat(row["timestamp"])
        row["target"] = float(row["target"])
        row["promo"] = float(row["promo"])
        row["macro_index"] = float(row["macro_index"])
    ordered = sorted(rows, key=lambda x: (x["series_id"], x["timestamp"]))
    _validate_rows(ordered)
    return ordered


def load_source_rows(source: str, periods: int = 30) -> list[dict]:
    normalized = source.strip().lower()
    if normalized == "fred":
        records = FredMacroAdapter().fetch(periods)
    elif normalized == "polymarket":
        records = PolymarketAdapter().fetch(periods)
    else:
        raise ValueError(f"unknown source adapter: {source}")

    rows = [r.as_row() for r in records]
    ordered = sorted(rows, key=lambda x: (x["series_id"], x["timestamp"]))
    _validate_rows(ordered)
    return ordered


def normalize_features(rows: list[dict]) -> list[dict]:
    out = [dict(r) for r in rows]
    for col in ["promo", "macro_index"]:
        vals = [r[col] for r in out]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        std = var ** 0.5 if var > 0 else 1.0
        for r in out:
            r[col] = (r[col] - mean) / std
    return out


def chronological_split(rows: list[dict], train=0.7, valid=0.15) -> DatasetBundle:
    n = len(rows)
    train_end = int(n * train)
    valid_end = train_end + int(n * valid)
    return DatasetBundle(
        train=rows[:train_end],
        valid=rows[train_end:valid_end],
        test=rows[valid_end:],
    )


def load_dataset(profile: DataProfile, path: str | Path = "data/sample_demand.csv") -> DatasetBundle:
    if profile.source == "sample_csv":
        build_sample_dataset(path, periods=profile.periods)
        rows = load_csv(path)
    else:
        rows = load_source_rows(profile.source, periods=profile.periods)

    if profile.normalize:
        rows = normalize_features(rows)

    return chronological_split(rows, train=profile.train_ratio, valid=profile.valid_ratio)


def build_sample_dataset(path: str | Path, periods: int = 365) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2022, 1, 1)
    records = []
    for sid in ["store_1_item_1", "store_2_item_1"]:
        base = 40.0 if "store_1" in sid else 25.0
        for i in range(periods):
            ts = start + timedelta(days=i)
            weekly = 4.0 if ts.weekday() in [5, 6] else -2.0
            promo = 1 if i % 17 == 0 else 0
            macro = 100 + (i * 0.05)
            target = base + 0.15 * i + weekly + 3.0 * promo + 0.08 * macro
            records.append(
                {
                    "timestamp": ts.isoformat(),
                    "series_id": sid,
                    "target": round(target, 3),
                    "promo": promo,
                    "macro_index": round(macro, 3),
                }
            )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
