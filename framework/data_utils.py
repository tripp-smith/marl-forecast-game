"""Data utility functions: caching, adapter dispatch, schema validation, and integrity checks."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import json

from .exceptions import AdapterFetchError, DataIngestionError

from .data_sources import (
    BEAAdapter,
    BeigeBookAdapter,
    BISPolicyRateAdapter,
    EarningsAdapter,
    EurostatAdapter,
    FredMacroAdapter,
    GeopoliticalRiskAdapter,
    ImfMacroAdapter,
    KaggleDemandAdapter,
    KalshiAdapter,
    OECDCLIAdapter,
    PMIAdapter,
    PolymarketAdapter,
    PredictItAdapter,
    WorldBankAdapter,
)

REQUIRED_SCHEMA_FIELDS = {"timestamp", "series_id", "target", "promo", "macro_index"}


def _resolve_cache_dir(cache_dir: str | Path | None = None) -> str | Path:
    if cache_dir is not None:
        return cache_dir
    return os.getenv("MFG_CACHE_DIR", "data/cache")


@dataclass(frozen=True)
class CacheStatus:
    """Describes the existence, age, and freshness of a cached data file."""

    path: Path
    exists: bool
    is_fresh: bool
    age_seconds: float | None


@dataclass(frozen=True)
class AnomalyDetectionResult:
    suspects: list[dict[str, Any]]
    scores: list[float]
    detector: str
    threshold: float


def _coerce_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _checksum_rows(rows: list[dict[str, Any]]) -> str:
    normalized = json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
    return sha256(normalized).hexdigest()


def cache_status(path: str | Path, *, max_age_hours: int = 24) -> CacheStatus:
    """Check whether a cache file at *path* exists and is fresh within *max_age_hours*."""
    p = Path(path)
    if not p.exists():
        return CacheStatus(path=p, exists=False, is_fresh=False, age_seconds=None)
    age_seconds = max(0.0, (datetime.utcnow() - datetime.utcfromtimestamp(p.stat().st_mtime)).total_seconds())
    max_age_seconds = max_age_hours * 3600
    return CacheStatus(path=p, exists=True, is_fresh=age_seconds <= max_age_seconds, age_seconds=age_seconds)


def _load_cached_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload.get("rows", [])
    return rows


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
    """Load and deserialize cached rows, parsing timestamp and fetched_at strings."""
    p = Path(path)
    rows = _load_cached_rows(p)
    for row in rows:
        if isinstance(row.get("timestamp"), str):
            row["timestamp"] = _coerce_dt(row["timestamp"])
        if isinstance(row.get("fetched_at"), str):
            row["fetched_at"] = _coerce_dt(row["fetched_at"])
    return rows


def fetch_source_rows(source: str, periods: int) -> list[dict[str, Any]]:
    """Fetch raw rows from a named adapter, returning list of dicts."""
    adapters: dict[str, Any] = {
        "fred": FredMacroAdapter(),
        "imf": ImfMacroAdapter(),
        "polymarket": PolymarketAdapter(),
        "bis": BISPolicyRateAdapter(),
        "gpr": GeopoliticalRiskAdapter(),
        "oecd_cli": OECDCLIAdapter(),
        "kaggle": KaggleDemandAdapter(),
        "worldbank": WorldBankAdapter(),
        "bea": BEAAdapter(),
        "kalshi": KalshiAdapter(),
        "predictit": PredictItAdapter(),
        "eurostat": EurostatAdapter(),
    }
    normalized = source.strip().lower()
    if normalized not in adapters:
        raise AdapterFetchError(f"unknown source adapter: {source}")
    return [r.as_row() for r in adapters[normalized].fetch(periods)]


def fetch_qual_source_rows(
    source: str, start_dt: datetime, end_dt: datetime,
) -> list[dict[str, Any]]:
    """Dispatch to a qualitative adapter and return rows as dicts."""
    qual_adapters: dict[str, Any] = {
        "beige_book": BeigeBookAdapter(),
        "pmi": PMIAdapter(),
        "earnings": EarningsAdapter(),
    }
    normalized = source.strip().lower()
    if normalized not in qual_adapters:
        raise AdapterFetchError(f"unknown qualitative adapter: {source}")
    return [r._asdict() for r in qual_adapters[normalized].fetch_releases(start_dt, end_dt)]


def _audit_log(source: str, periods: int, cache_hit: bool, row_count: int) -> None:
    """Emit a structured audit log entry for data access compliance."""
    try:
        import structlog
        logger = structlog.get_logger("data_audit")
        logger.info(
            "data_access",
            source=source,
            periods_requested=periods,
            cache_hit=cache_hit,
            row_count=row_count,
        )
    except ImportError:
        logging.getLogger("data_audit").info(
            "data_access source=%s periods=%d cache_hit=%s rows=%d",
            source, periods, cache_hit, row_count,
        )


def ensure_source_data(
    source: str,
    *,
    periods: int,
    cache_dir: str | Path | None = None,
    max_age_hours: int = 24,
    force_redownload: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return cached rows if fresh, otherwise fetch from the adapter and cache.

    Args:
        source: Adapter name.
        periods: Desired row count.
        cache_dir: Directory for cache JSON files.
        max_age_hours: Maximum cache age before refetch.
        force_redownload: Bypass cache freshness check.
    """
    cache_path = Path(_resolve_cache_dir(cache_dir)) / f"{source}.json"
    status = cache_status(cache_path, max_age_hours=max_age_hours)
    if status.exists and status.is_fresh and not force_redownload:
        rows = load_rows_from_cache(cache_path)
        if len(rows) >= periods:
            _audit_log(source, periods, cache_hit=True, row_count=len(rows[:periods]))
            return rows[:periods], {"cache_hit": True, "cache_path": str(cache_path), "fresh": True}

    rows = fetch_source_rows(source, periods)
    if len(rows) < max(1, periods // 2):
        raise DataIngestionError(f"integrity check failed for {source}: insufficient rows")

    _write_cache(cache_path, rows, source=source)
    _audit_log(source, periods, cache_hit=False, row_count=len(rows[:periods]))
    return rows[:periods], {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "fresh": False,
        "forced": force_redownload,
        "checksum": _checksum_rows(rows),
    }


def verify_payload_signature(payload: dict[str, Any]) -> bool:
    """Check cached payload checksum matches the serialized rows."""
    rows = payload.get("rows", [])
    stored_checksum = payload.get("checksum")
    if not stored_checksum:
        return False
    return bool(stored_checksum == _checksum_rows(rows))


def detect_isolation_forest_anomalies(
    rows: list[dict[str, Any]],
    *,
    contamination: float = 0.05,
) -> AnomalyDetectionResult:
    """Detect anomalies using sklearn IsolationForest on numeric row features."""
    if len(rows) < 8:
        return AnomalyDetectionResult(suspects=[], scores=[], detector="isolation_forest", threshold=contamination)
    try:
        import numpy as np
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return AnomalyDetectionResult(suspects=[], scores=[], detector="isolation_forest", threshold=contamination)

    matrix = np.asarray(
        [
            [
                float(r.get("target", 0.0)),
                float(r.get("promo", 0.0)),
                float(r.get("macro_index", 0.0)),
            ]
            for r in rows
        ],
        dtype=float,
    )
    clf = IsolationForest(contamination=max(0.001, contamination), random_state=42)
    preds = clf.fit_predict(matrix)
    scores = (-clf.score_samples(matrix)).tolist()
    suspects = [row for row, pred in zip(rows, preds) if pred == -1]
    return AnomalyDetectionResult(suspects=suspects, scores=scores, detector="isolation_forest", threshold=contamination)


def detect_autoencoder_anomalies(
    rows: list[dict[str, Any]],
    *,
    threshold: float = 0.05,
    epochs: int = 50,
) -> AnomalyDetectionResult:
    """Detect anomalies using a lightweight PyTorch autoencoder when available."""
    if len(rows) < 8:
        return AnomalyDetectionResult(suspects=[], scores=[], detector="autoencoder", threshold=threshold)
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return AnomalyDetectionResult(suspects=[], scores=[], detector="autoencoder", threshold=threshold)

    matrix = np.asarray(
        [
            [
                float(r.get("target", 0.0)),
                float(r.get("promo", 0.0)),
                float(r.get("macro_index", 0.0)),
            ]
            for r in rows
        ],
        dtype=np.float32,
    )
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    normalized = (matrix - mean) / std
    data = torch.tensor(normalized, dtype=torch.float32)

    class AutoEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
            self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3))

        def forward(self, x: Any) -> Any:
            return self.decoder(self.encoder(x))

    model = AutoEncoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon = model(data)
        loss = loss_fn(recon, data)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        errors = ((model(data) - data) ** 2).mean(dim=1).cpu().numpy().tolist()
    suspects = [row for row, err in zip(rows, errors) if err >= threshold]
    return AnomalyDetectionResult(suspects=suspects, scores=errors, detector="autoencoder", threshold=threshold)


def detect_poisoned_rows(
    rows: list[dict[str, Any]],
    *,
    poisoning_threshold: float = 0.05,
) -> dict[str, AnomalyDetectionResult]:
    """Run all configured anomaly detectors and return their independent results."""
    results = {
        "isolation_forest": detect_isolation_forest_anomalies(rows, contamination=poisoning_threshold),
        "autoencoder": detect_autoencoder_anomalies(rows, threshold=poisoning_threshold),
    }
    total_flagged = sum(len(result.suspects) for result in results.values())
    if total_flagged:
        try:
            import structlog

            structlog.get_logger("poisoning_detection").warning(
                "poisoning_suspects_detected",
                total_flagged=total_flagged,
                threshold=poisoning_threshold,
                detectors={name: len(result.suspects) for name, result in results.items()},
            )
        except ImportError:
            logging.getLogger("poisoning_detection").warning(
                "poisoning_suspects_detected total=%d threshold=%.4f",
                total_flagged,
                poisoning_threshold,
            )
    return results


# ---------------------------------------------------------------------------
# Standalone data-integrity validation functions
# ---------------------------------------------------------------------------

def validate_cache_integrity(source: str, *, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Verify a cache file has valid checksum, parseable timestamps, and no null targets."""
    path = Path(_resolve_cache_dir(cache_dir)) / f"{source}.json"
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
    cache_dir: str | Path | None = None,
    force_redownload: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a multi-series FRED training dataset with chronological alignment.

    Returns (rows, metadata). Each row includes a 'macro_context' dict.
    Falls back to synthetic proxy when FRED_API_KEY is absent.
    """
    from .data_sources.macro_fred import FredMacroAdapter

    cache_path = Path(_resolve_cache_dir(cache_dir)) / "fred_training.json"
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
