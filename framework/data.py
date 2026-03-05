"""Dataset loading, validation, poisoning detection, and train/valid/test splitting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import csv
import json
import math

from .data_utils import build_fred_training_set, detect_poisoned_rows, ensure_source_data, fetch_qual_source_rows
from .exceptions import AdapterFetchError, DataIngestionError, PoisoningDetectedError


REQUIRED_COLUMNS = ["timestamp", "series_id", "target", "promo", "macro_index"]


@dataclass(frozen=True)
class DatasetBundle:
    """Train/valid/test split of dataset rows."""

    train: list[dict[str, Any]]
    valid: list[dict[str, Any]]
    test: list[dict[str, Any]]


@dataclass(frozen=True)
class DataProfile:
    """Configuration for dataset loading: source, split ratios, and poisoning detection."""

    source: str = "sample_csv"
    periods: int = 240
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    normalize: bool = True
    realtime_refresh: bool = False
    hybrid_weight: float = 0.5
    fail_on_poisoning: bool = False
    poisoning_threshold: float = 0.05

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
        if not (0.0 <= self.poisoning_threshold <= 1.0):
            raise ValueError("poisoning_threshold must be in [0,1]")


def _ensure_required(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise DataIngestionError("dataset is empty")
    missing = [col for col in REQUIRED_COLUMNS if col not in rows[0].keys()]
    if missing:
        raise DataIngestionError(f"missing columns: {missing}")


def _validate_rows(rows: list[dict[str, Any]]) -> None:
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


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a CSV dataset, returning chronologically-ordered rows."""
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


def load_json(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a JSON dataset, returning chronologically-ordered rows."""
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
) -> list[dict[str, Any]]:
    """Fetch rows from a named external source adapter with optional caching.

    Args:
        source: Adapter name (e.g. ``"fred"``, ``"imf"``).
        periods: Number of periods to retrieve.
        realtime_refresh: Stamp the last row with the current UTC time.
        force_redownload: Bypass cache and re-fetch from the remote source.
        cache_dir: Directory for cached adapter data.
    """
    normalized = source.strip().lower()
    _ALL_SOURCES = {
        "fred", "imf", "polymarket", "bis", "gpr", "oecd_cli",
        "kaggle", "worldbank", "bea", "kalshi", "predictit", "eurostat",
    }
    if normalized not in _ALL_SOURCES:
        raise AdapterFetchError(f"unknown source adapter: {source}")

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


def normalize_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Z-score normalize ``promo`` and ``macro_index`` columns in-place."""
    out = [dict(r) for r in rows]
    for col in ["promo", "macro_index"]:
        vals = [r[col] for r in out]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        std = var**0.5 if var > 0 else 1.0
        for r in out:
            r[col] = (r[col] - mean) / std
    return out


def detect_poisoning_rows(rows: list[dict[str, Any]], *, z_threshold: float = 6.0, mad_threshold: float = 8.0) -> list[dict[str, Any]]:
    """Detect potential data-poisoning rows via z-score and MAD thresholds."""
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


def chronological_split(rows: list[dict[str, Any]], train: float = 0.7, valid: float = 0.15) -> DatasetBundle:
    """Split *rows* chronologically into train/valid/test partitions."""
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


def build_hybrid_rows(real_rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]], *, real_weight: float = 0.5) -> list[dict[str, Any]]:
    """Blend real and synthetic rows by *real_weight*, interpolating gaps if scipy is available."""
    if not (0.0 <= real_weight <= 1.0):
        raise ValueError("real_weight must be in [0,1]")
    n = min(len(real_rows), len(synthetic_rows))
    take_real = int(n * real_weight)
    mixed = real_rows[:take_real] + synthetic_rows[take_real:n]
    mixed = sorted(mixed, key=lambda x: (x["timestamp"], x["series_id"]))

    try:
        from scipy.interpolate import interp1d
        import numpy as np

        targets = [r.get("target") for r in mixed]
        valid_idx = [i for i, t in enumerate(targets) if t is not None and not (isinstance(t, float) and math.isnan(t))]
        if len(valid_idx) >= 2 and len(valid_idx) < len(targets):
            xs = np.array(valid_idx, dtype=float)
            ys = np.array([targets[i] for i in valid_idx], dtype=float)
            f = interp1d(xs, ys, kind="linear", fill_value="extrapolate")
            for i, row in enumerate(mixed):
                t = row.get("target")
                if t is None or (isinstance(t, float) and math.isnan(t)):
                    mixed[i] = {**row, "target": float(f(i))}
    except ImportError:
        pass

    return mixed




def validate_with_schema(rows: list[dict[str, Any]]) -> list[str]:
    """Validate rows against a pydantic RowSchema. Returns list of error messages.

    If pydantic is not installed, returns an empty list (validation skipped).
    """
    try:
        from pydantic import BaseModel, ValidationError as PydValidationError

        class RowSchema(BaseModel):
            timestamp: object
            series_id: str
            target: float
            promo: float
            macro_index: float
            source: str = ""

        errors: list[str] = []
        for idx, row in enumerate(rows):
            try:
                RowSchema(**row)
            except PydValidationError as exc:
                errors.append(f"row {idx}: {exc}")
        return errors
    except ImportError:
        return []


def detect_poisoning_iqr(rows: list[dict[str, Any]], *, iqr_factor: float = 1.5) -> list[dict[str, Any]]:
    """Detect potential data poisoning using the IQR method.

    Flags rows whose ``target`` value falls outside
    ``[Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR]``.
    """
    targets = [r["target"] for r in rows]
    if len(targets) < 3:
        return []
    sorted_vals = sorted(targets)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[(3 * n) // 4]
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    return [row for row in rows if row["target"] < lower or row["target"] > upper]


def should_reject_poisoning(total_rows: int, suspect_rows: int) -> bool:
    """Return True if the fraction of suspect rows warrants dataset rejection."""
    if suspect_rows <= 0:
        return False
    # avoid hard-failing on a single suspicious point from external feeds
    # while still rejecting broader contamination patterns
    return suspect_rows >= 2 and (suspect_rows / max(1, total_rows)) >= 0.02


def load_dataset(profile: DataProfile, path: str | Path = "data/sample_demand.csv") -> DatasetBundle:
    """Load, validate, normalize, and split a dataset according to *profile*."""
    if profile.source == "sample_csv":
        build_sample_dataset(path, periods=profile.periods)
        rows = load_csv(path)
    elif profile.source == "fred_training":
        import logging
        import os
        if not os.getenv("FRED_API_KEY"):
            logging.warning("FRED_API_KEY not set; fred_training will use synthetic proxy")
        fred_rows, _meta = build_fred_training_set(periods=profile.periods)
        rows = fred_rows
    elif profile.source in {
        "fred", "imf", "polymarket", "bis", "gpr", "oecd_cli",
        "kaggle", "worldbank", "bea", "kalshi", "predictit", "eurostat",
    }:
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

    schema_errors = validate_with_schema(rows)
    if schema_errors:
        import logging as _logging
        _logging.getLogger(__name__).warning("Schema validation issues: %d rows", len(schema_errors))

    suspects_zscore = detect_poisoning_rows(rows)
    suspects_iqr = detect_poisoning_iqr(rows)
    ml_results = detect_poisoned_rows(rows, poisoning_threshold=profile.poisoning_threshold)
    seen_ids = set()
    suspects: list[dict[str, Any]] = []
    ml_suspects = [row for result in ml_results.values() for row in result.suspects]
    for s in suspects_zscore + suspects_iqr + ml_suspects:
        row_id = id(s)
        if row_id not in seen_ids:
            seen_ids.add(row_id)
            suspects.append(s)
    if profile.fail_on_poisoning and should_reject_poisoning(len(rows), len(suspects)):
        raise PoisoningDetectedError("potential data poisoning detected")

    if profile.normalize:
        rows = normalize_features(rows)

    return chronological_split(rows, train=profile.train_ratio, valid=profile.valid_ratio)


def build_qual_dataset(
    qual_adapters: tuple[str, ...],
    start_dt: datetime,
    end_dt: datetime,
    *,
    step_size_days: int = 1,
    epoch: datetime | None = None,
    z_threshold: float = 3.0,
    cache_path: str | Path = "data/cache/qual_manifest.json",
) -> dict[int, dict[str, Any]]:
    """Build a timestep-indexed qualitative dataset from multiple adapters.

    Returns a mapping from game timestep ``t`` to a dict with keys
    ``timestamp``, ``source_id``, ``text``, and ``metadata``.
    """
    from hashlib import sha256

    if epoch is None:
        epoch = start_dt

    all_records: list[dict[str, Any]] = []
    for adapter_name in qual_adapters:
        rows = fetch_qual_source_rows(adapter_name, start_dt, end_dt)
        all_records.extend(rows)

    all_records.sort(key=lambda r: r["timestamp"])

    if len(all_records) >= 3:
        lengths = [len(r.get("text", "")) for r in all_records]
        mean_len = sum(lengths) / len(lengths)
        var_len = sum((x - mean_len) ** 2 for x in lengths) / max(1, len(lengths) - 1)
        std_len = var_len ** 0.5 if var_len > 0 else 1.0
        filtered: list[dict[str, Any]] = []
        for rec, length in zip(all_records, lengths):
            z = abs((length - mean_len) / std_len)
            if z <= z_threshold:
                filtered.append(rec)
        all_records = filtered

    dataset: dict[int, dict[str, Any]] = {}
    for rec in all_records:
        ts = rec["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        delta_days = (ts - epoch).days
        step = max(0, delta_days // max(1, step_size_days))
        if step not in dataset:
            dataset[step] = rec

    text_blob = "".join(r.get("text", "") for r in all_records)
    checksum = sha256(text_blob.encode("utf-8")).hexdigest()
    manifest = {
        "checksum": checksum,
        "record_count": len(all_records),
        "adapters": list(qual_adapters),
        "start_dt": start_dt.isoformat(),
        "end_dt": end_dt.isoformat(),
    }
    manifest_path = Path(cache_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8",
    )

    return dataset


def build_sample_dataset(path: str | Path, periods: int = 365) -> None:
    """Generate a synthetic demand CSV with seasonal patterns and trends."""
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
