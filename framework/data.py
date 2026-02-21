from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import csv
import json
import math

from .data_utils import ensure_source_data


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
    realtime_refresh: bool = False
    hybrid_weight: float = 0.5
    fail_on_poisoning: bool = False

    def __post_init__(self) -> None:
        if self.periods <= 0:
            raise ValueError("periods must be > 0")
        if not (0 < self.train_ratio < 1):
            raise ValueError("train_ratio must be in (0,1)")
        if not (0 <= self.valid_ratio < 1):
            raise ValueError("valid_ratio must be in [0,1)")
        if self.train_ratio + self.valid_ratio >= 1:
            raise ValueError("train_ratio + valid_ratio must be < 1")
        if not (0.0 <= self.hybrid_weight <= 1.0):
            raise ValueError("hybrid_weight must be in [0,1]")


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


def _parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def load_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    _ensure_required(rows)
    for row in rows:
        row["timestamp"] = _parse_timestamp(row["timestamp"])
        row["target"] = float(row["target"])
        row["promo"] = float(row["promo"])
        row["macro_index"] = float(row["macro_index"])
    ordered = sorted(rows, key=lambda x: (x["series_id"], x["timestamp"]))
    _validate_rows(ordered)
    return ordered


def load_json(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("json dataset must be a list of objects")
    rows = []
    for row in payload:
        parsed = dict(row)
        parsed["timestamp"] = _parse_timestamp(parsed["timestamp"])
        parsed["target"] = float(parsed["target"])
        parsed["promo"] = float(parsed["promo"])
        parsed["macro_index"] = float(parsed["macro_index"])
        rows.append(parsed)
    ordered = sorted(rows, key=lambda x: (x["series_id"], x["timestamp"]))
    _validate_rows(ordered)
    return ordered


def load_source_rows(
    source: str,
    periods: int = 30,
    *,
    realtime_refresh: bool = False,
    force_redownload: bool = False,
    cache_dir: str | Path = "data/cache",
) -> list[dict]:
    normalized = source.strip().lower()
    if normalized not in {"fred", "imf", "polymarket"}:
        raise ValueError(f"unknown source adapter: {source}")

    rows, _meta = ensure_source_data(
        normalized,
        periods=periods,
        cache_dir=cache_dir,
        force_redownload=force_redownload,
    )
    if realtime_refresh and rows:
        now = datetime.utcnow()
        rows[-1] = {**rows[-1], "fetched_at": now}
    ordered = sorted(rows, key=lambda x: (x["series_id"], x["timestamp"]))
    _validate_rows(ordered)
    return ordered


def normalize_features(rows: list[dict]) -> list[dict]:
    out = [dict(r) for r in rows]
    for col in ["promo", "macro_index"]:
        vals = [r[col] for r in out]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        std = var**0.5 if var > 0 else 1.0
        for r in out:
            r[col] = (r[col] - mean) / std
    return out


def detect_poisoning_rows(rows: list[dict], *, z_threshold: float = 6.0, mad_threshold: float = 8.0) -> list[dict]:
    targets = [r["target"] for r in rows]
    if len(targets) < 3:
        return []
    mean = sum(targets) / len(targets)
    var = sum((x - mean) ** 2 for x in targets) / max(1, len(targets) - 1)
    std = var**0.5 if var > 0 else 1.0
    sorted_vals = sorted(targets)
    median = sorted_vals[len(sorted_vals) // 2]
    abs_dev = sorted(abs(x - median) for x in targets)
    mad = abs_dev[len(abs_dev) // 2] or 1.0
    suspects = []
    for row in rows:
        z = abs((row["target"] - mean) / std)
        mz = abs(0.6745 * (row["target"] - median) / mad)
        if z >= z_threshold or mz >= mad_threshold:
            suspects.append(row)
    return suspects


def chronological_split(rows: list[dict], train=0.7, valid=0.15) -> DatasetBundle:
    if not (0 < train < 1):
        raise ValueError("train split must be in (0,1)")
    if not (0 <= valid < 1):
        raise ValueError("valid split must be in [0,1)")
    if train + valid >= 1:
        raise ValueError("train + valid split must be < 1")

    n = len(rows)
    train_end = int(n * train)
    valid_end = train_end + int(n * valid)
    return DatasetBundle(
        train=rows[:train_end],
        valid=rows[train_end:valid_end],
        test=rows[valid_end:],
    )


def build_hybrid_rows(real_rows: list[dict], synthetic_rows: list[dict], *, real_weight: float = 0.5) -> list[dict]:
    if not (0.0 <= real_weight <= 1.0):
        raise ValueError("real_weight must be in [0,1]")
    n = min(len(real_rows), len(synthetic_rows))
    take_real = int(n * real_weight)
    mixed = real_rows[:take_real] + synthetic_rows[take_real:n]
    return sorted(mixed, key=lambda x: (x["timestamp"], x["series_id"]))




def should_reject_poisoning(total_rows: int, suspect_rows: int) -> bool:
    if suspect_rows <= 0:
        return False
    # avoid hard-failing on a single suspicious point from external feeds
    # while still rejecting broader contamination patterns
    return suspect_rows >= 2 and (suspect_rows / max(1, total_rows)) >= 0.02


def load_dataset(profile: DataProfile, path: str | Path = "data/sample_demand.csv") -> DatasetBundle:
    if profile.source == "sample_csv":
        build_sample_dataset(path, periods=profile.periods)
        rows = load_csv(path)
    elif profile.source in {"fred", "imf", "polymarket"}:
        rows = load_source_rows(profile.source, periods=profile.periods, realtime_refresh=profile.realtime_refresh)
    elif profile.source == "hybrid":
        build_sample_dataset(path, periods=profile.periods)
        synthetic = load_csv(path)
        real = load_source_rows("fred", periods=profile.periods, realtime_refresh=profile.realtime_refresh)
        rows = build_hybrid_rows(real, synthetic, real_weight=profile.hybrid_weight)
    elif str(path).endswith(".json"):
        rows = load_json(path)
    else:
        rows = load_csv(path)

    suspects = detect_poisoning_rows(rows)
    if profile.fail_on_poisoning and should_reject_poisoning(len(rows), len(suspects)):
        raise ValueError("potential data poisoning detected")

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
            weekly_seasonality = 3.8 * math.sin((2 * math.pi * i) / 7.0)
            yearly = 2.1 * math.sin((2 * math.pi * i) / 365.0)
            promo = 1 if i % 17 == 0 else 0
            macro = 100 + (i * 0.05)
            target = base + 0.12 * i + weekly_seasonality + yearly + 3.0 * promo + 0.08 * macro
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
