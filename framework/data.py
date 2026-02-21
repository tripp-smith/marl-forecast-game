from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import csv


REQUIRED_COLUMNS = ["timestamp", "series_id", "target", "promo", "macro_index"]


@dataclass(frozen=True)
class DatasetBundle:
    train: list[dict]
    valid: list[dict]
    test: list[dict]


def load_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    missing = [col for col in REQUIRED_COLUMNS if col not in (rows[0].keys() if rows else [])]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    for row in rows:
        row["timestamp"] = datetime.fromisoformat(row["timestamp"])
        row["target"] = float(row["target"])
        row["promo"] = float(row["promo"])
        row["macro_index"] = float(row["macro_index"])

    rows.sort(key=lambda r: (r["series_id"], r["timestamp"]))
    return rows


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
