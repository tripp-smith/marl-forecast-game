"""JSON trajectory export utility."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .game import GameOutputs
from .types import SimulationConfig


def _config_hash(config: SimulationConfig) -> str:
    from dataclasses import asdict
    raw = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def export_trajectories(
    game_outputs: GameOutputs,
    path: Path | str,
    *,
    config: SimulationConfig | None = None,
    seed: int | None = None,
) -> Path:
    """Write trajectory logs as a JSON file with metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "exported_at": datetime.now().isoformat(),
        "rounds_executed": game_outputs.convergence.get("rounds_executed", len(game_outputs.trajectory_logs)),
    }
    if config is not None:
        metadata["config_hash"] = _config_hash(config)
    if seed is not None:
        metadata["seed"] = seed

    payload = {
        "metadata": metadata,
        "convergence": game_outputs.convergence,
        "trajectories": game_outputs.trajectory_logs,
    }

    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p
