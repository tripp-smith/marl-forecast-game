#!/usr/bin/env python3
"""Deterministic reproducibility check for the public package API."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marl_forecast_game import GameEngine, SimulationConfig, demo_state


def _digest(values: list[float]) -> str:
    payload = ",".join(f"{value:.8f}" for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    cfg = SimulationConfig(
        horizon=24,
        max_rounds=24,
        disturbance_prob=0.2,
        disturbance_scale=1.0,
        adversarial_intensity=1.0,
        disturbance_model="gaussian",
        defense_model="ensemble",
    )
    state = demo_state(value=120.0, exogenous=0.1)

    same_seed_runs = []
    for _ in range(3):
        game = GameEngine(cfg, seed=42)
        out = game.run(state, rounds=12, disturbed=True)
        same_seed_runs.append(out.forecasts)

    shifted_seed = GameEngine(cfg, seed=43).run(state, rounds=12, disturbed=True).forecasts

    report = {
        "seed": 42,
        "rounds": 12,
        "same_seed_hashes": [_digest(run) for run in same_seed_runs],
        "same_seed_identical": same_seed_runs[0] == same_seed_runs[1] == same_seed_runs[2],
        "different_seed_hash": _digest(shifted_seed),
        "different_seed_changes_trace": shifted_seed != same_seed_runs[0],
    }

    output_dir = ROOT / "results" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reproducibility_check.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
