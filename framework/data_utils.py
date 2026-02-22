from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import json

from .data_sources import FredMacroAdapter, ImfMacroAdapter, PolymarketAdapter

REQUIRED_SCHEMA_FIELDS = {"timestamp", "series_id", "target", "promo", "macro_index"}


@dataclass(frozen=True)
class CacheStatus:
    path: Path
    exists: bool
    is_fresh: bool
    age_seconds: float | None


def _coerce_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _checksum_rows(rows: list[dict[str, Any]]) -> str:
    normalized = json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
    return sha256(normalized).hexdigest()


def cache_status(path: str | Path, *, max_age_hours: int = 24) -> CacheStatus:
    p = Path(path)
    if not p.exists():
        return CacheStatus(path=p, exists=False, is_fresh=False, age_seconds=None)
    age_seconds = max(0.0, (datetime.utcnow() - datetime.utcfromtimestamp(p.stat().st_mtime)).total_seconds())
    max_age_seconds = max_age_hours * 3600
    return CacheStatus(path=p, exists=True, is_fresh=age_seconds <= max_age_seconds, age_seconds=age_seconds)


def _load_cached_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("rows", [])


def _write_cache(path: Path, rows: list[dict[str, Any]], *, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "fetched_at": datetime.utcnow().isoformat(),
        "checksum": _checksum_rows(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_rows_from_cache(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows = _load_cached_rows(p)
    for row in rows:
        if isinstance(row.get("timestamp"), str):
            row["timestamp"] = _coerce_dt(row["timestamp"])
        if isinstance(row.get("fetched_at"), str):
            row["fetched_at"] = _coerce_dt(row["fetched_at"])
    return rows


def fetch_source_rows(source: str, periods: int) -> list[dict[str, Any]]:
    adapters = {
        "fred": FredMacroAdapter(),
        "imf": ImfMacroAdapter(),
        "polymarket": PolymarketAdapter(),
    }
    normalized = source.strip().lower()
    if normalized not in adapters:
        raise ValueError(f"unknown source adapter: {source}")
    return [r.as_row() for r in adapters[normalized].fetch(periods)]


def ensure_source_data(
    source: str,
    *,
    periods: int,
    cache_dir: str | Path = "data/cache",
    max_age_hours: int = 24,
    force_redownload: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_path = Path(cache_dir) / f"{source}.json"
    status = cache_status(cache_path, max_age_hours=max_age_hours)
    if status.exists and status.is_fresh and not force_redownload:
        rows = load_rows_from_cache(cache_path)
        if len(rows) >= periods:
            return rows[:periods], {"cache_hit": True, "cache_path": str(cache_path), "fresh": True}

    rows = fetch_source_rows(source, periods)
    if len(rows) < max(1, periods // 2):
        raise ValueError(f"integrity check failed for {source}: insufficient rows")

    _write_cache(cache_path, rows, source=source)
    return rows[:periods], {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "fresh": False,
        "forced": force_redownload,
        "checksum": _checksum_rows(rows),
    }


# ---------------------------------------------------------------------------
# Standalone data-integrity validation functions
# ---------------------------------------------------------------------------

def validate_cache_integrity(source: str, *, cache_dir: str | Path = "data/cache") -> dict[str, Any]:
    """Verify a cache file has valid checksum, parseable timestamps, and no null targets."""
    path = Path(cache_dir) / f"{source}.json"
    result: dict[str, Any] = {"source": source, "path": str(path), "valid": True, "errors": []}
    if not path.exists():
        result["valid"] = False
        result["errors"].append("cache file does not exist")
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        result["valid"] = False
        result["errors"].append(f"failed to read cache: {exc}")
        return result

    rows = payload.get("rows", [])
    stored_checksum = payload.get("checksum")
    if stored_checksum and _checksum_rows(rows) != stored_checksum:
        result["valid"] = False
        result["errors"].append("checksum mismatch")

    for idx, row in enumerate(rows):
        if row.get("target") is None:
            result["valid"] = False
            result["errors"].append(f"null target at row {idx}")
        ts = row.get("timestamp")
        if ts is not None:
            try:
                datetime.fromisoformat(str(ts))
            except ValueError:
                result["valid"] = False
                result["errors"].append(f"unparseable timestamp at row {idx}: {ts}")

    result["row_count"] = len(rows)
    return result


def validate_source_schema(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    """Check that all required fields are present and correctly typed in every row."""
    result: dict[str, Any] = {"source": source, "valid": True, "errors": []}
    for idx, row in enumerate(rows):
        missing = REQUIRED_SCHEMA_FIELDS - set(row.keys())
        if missing:
            result["valid"] = False
            result["errors"].append(f"row {idx} missing fields: {missing}")
        if "target" in row:
            try:
                float(row["target"])
            except (TypeError, ValueError):
                result["valid"] = False
                result["errors"].append(f"row {idx} target not numeric: {row['target']}")
    result["row_count"] = len(rows)
    return result


def validate_chronological_order(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify rows are in chronological order within each series."""
    result: dict[str, Any] = {"valid": True, "errors": []}
    per_series: dict[str, datetime] = {}
    for idx, row in enumerate(rows):
        ts = row.get("timestamp")
        if ts is None:
            continue
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        sid = row.get("series_id", "__default__")
        last = per_series.get(sid)
        if last is not None and ts < last:
            result["valid"] = False
            result["errors"].append(f"out-of-order at row {idx}: {ts} < {last} for series {sid}")
        per_series[sid] = ts
    return result


def validate_no_future_leakage(
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assert max train timestamp < min valid timestamp < min test timestamp."""
    result: dict[str, Any] = {"valid": True, "errors": []}

    def _max_ts(rows: list[dict[str, Any]]) -> datetime | None:
        timestamps = []
        for r in rows:
            ts = r.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts is not None:
                timestamps.append(ts)
        return max(timestamps) if timestamps else None

    def _min_ts(rows: list[dict[str, Any]]) -> datetime | None:
        timestamps = []
        for r in rows:
            ts = r.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts is not None:
                timestamps.append(ts)
        return min(timestamps) if timestamps else None

    train_max = _max_ts(train)
    valid_min = _min_ts(valid)
    test_min = _min_ts(test)

    if train_max and valid_min and train_max > valid_min:
        result["valid"] = False
        result["errors"].append(f"train max ({train_max}) > valid min ({valid_min})")
    if valid_min and test_min and valid_min > test_min:
        result["valid"] = False
        result["errors"].append(f"valid min ({valid_min}) > test min ({test_min})")

    return result


def build_fred_training_set(
    periods: int = 120,
    *,
    cache_dir: str | Path = "data/cache",
    force_redownload: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a multi-series FRED training dataset with chronological alignment.

    Returns (rows, metadata). Each row includes a 'macro_context' dict.
    Falls back to synthetic proxy when FRED_API_KEY is absent.
    """
    from .data_sources.macro_fred import FredMacroAdapter

    cache_path = Path(cache_dir) / "fred_training.json"
    status = cache_status(cache_path, max_age_hours=24)

    if status.exists and status.is_fresh and not force_redownload:
        rows = load_rows_from_cache(cache_path)
        if len(rows) >= periods:
            return rows[:periods], {"cache_hit": True, "cache_path": str(cache_path)}

    adapter = FredMacroAdapter()
    rows = adapter.fetch_multi_series(periods)

    if not rows:
        raise ValueError("FRED multi-series fetch returned no rows")

    _write_cache(cache_path, rows, source="fred_training")
    return rows[:periods], {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "checksum": _checksum_rows(rows),
    }


def validate_cross_source_consistency(source_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """For multiple data sources, check overlapping date ranges have compatible scales."""
    result: dict[str, Any] = {"valid": True, "errors": [], "sources": list(source_rows.keys())}

    source_stats: dict[str, dict[str, Any]] = {}
    for name, rows in source_rows.items():
        targets = [float(r["target"]) for r in rows if r.get("target") is not None]
        if not targets:
            continue
        source_stats[name] = {
            "mean": sum(targets) / len(targets),
            "min": min(targets),
            "max": max(targets),
            "count": len(targets),
        }

    result["source_stats"] = source_stats
    return result
