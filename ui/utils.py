"""Shared helpers for the Streamlit UI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("/app/results")


def discover_result_files(
    results_dir: str | Path = RESULTS_DIR,
    suffix: str = ".json",
) -> list[Path]:
    """Scan a results directory for output files, sorted newest-first."""
    d = Path(results_dir)
    if not d.is_dir():
        return []
    files = sorted(d.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def load_trajectory_logs(path: str | Path) -> list[dict[str, Any]]:
    """Load trajectory_logs from a JSON file."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "trajectory_logs" in data:
        return data["trajectory_logs"]
    return []


def load_simulation_result(path: str | Path) -> dict[str, Any]:
    """Load a full simulation result dict from JSON."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def run_simulation(
    horizon: int = 100,
    max_rounds: int = 200,
    disturbance_prob: float = 0.1,
    disturbance_scale: float = 1.0,
    adversarial_intensity: float = 1.0,
    defense_model: str = "dampening",
    disturbance_model: str = "gaussian",
    seed: int = 42,
    initial_value: float = 10.0,
    initial_exogenous: float = 0.0,
    initial_hidden_shift: float = 0.0,
    disturbed: bool = True,
) -> dict[str, Any]:
    """Run a simulation and return the output dict."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from framework.distributed import _game_outputs_to_dict
    from framework.game import ForecastGame
    from framework.metrics import mae, mape, rmse, worst_case_abs_error
    from framework.types import ForecastState, SimulationConfig

    config = SimulationConfig(
        horizon=horizon,
        max_rounds=max_rounds,
        disturbance_prob=disturbance_prob,
        disturbance_scale=disturbance_scale,
        adversarial_intensity=adversarial_intensity,
        defense_model=defense_model,
        disturbance_model=disturbance_model,
    )
    state = ForecastState(
        t=0,
        value=initial_value,
        exogenous=initial_exogenous,
        hidden_shift=initial_hidden_shift,
    )
    game = ForecastGame(config, seed=seed)
    out = game.run(state, disturbed=disturbed)
    result = _game_outputs_to_dict(out)
    result["seed"] = seed
    result["metrics"] = {
        "mae": mae(out.targets, out.forecasts),
        "rmse": rmse(out.targets, out.forecasts),
        "mape": mape(out.targets, out.forecasts),
        "worst_case": worst_case_abs_error(out.targets, out.forecasts),
    }
    return result
