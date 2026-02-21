from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import json

from .data_sources import FredMacroAdapter, ImfMacroAdapter, PolymarketAdapter


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
