from __future__ import annotations

from dataclasses import replace

from .data import build_sample_dataset, chronological_split, load_csv, load_source_rows, normalize_features
from .game import ForecastGame
from .metrics import mae, mape, rmse, robustness_delta, robustness_ratio, worst_case_abs_error
from .types import ForecastState, SimulationConfig, evolve_state


def run_verification() -> dict:
    build_sample_dataset("data/sample_demand.csv", periods=240)
    rows = normalize_features(load_csv("data/sample_demand.csv"))
    bundle = chronological_split(rows)

    cfg = SimulationConfig(
        horizon=80,
        max_rounds=200,
        disturbance_prob=0.2,
        disturbance_scale=1.2,
        runtime_backend="python",
        disturbance_model="gaussian",
        defense_model="dampening",
        enable_refactor=True,
    )
    game_clean = ForecastGame(cfg, seed=13)
    game_attack = ForecastGame(cfg, seed=13)

    init = ForecastState(t=0, value=float(bundle.train[-1]["target"]), exogenous=0.0, hidden_shift=0.0)
    clean = game_clean.run(init, disturbed=False)
    attack = game_attack.run(init, disturbed=True)

    s0 = ForecastState(t=0, value=10.0, exogenous=1.2, hidden_shift=0.0)
    s1 = evolve_state(s0, base_trend=0.4, noise=0.0, disturbance=0.1)
    deterministic_again = evolve_state(s0, base_trend=0.4, noise=0.0, disturbance=0.1)

    source_rows = load_source_rows("fred", periods=20)
    checks = {
        "split_non_empty": len(bundle.train) > 0 and len(bundle.valid) > 0 and len(bundle.test) > 0,
        "pure_transition": s1 == deterministic_again and s0 == replace(s0),
        "max_rounds_respected": len(attack.steps) <= cfg.max_rounds,
        "trajectory_present": len(attack.trajectories) == len(attack.steps),
        "source_adapter_rows": len(source_rows) == 20,
    }

    clean_metrics = {
        "mae": mae(clean.targets, clean.forecasts),
        "rmse": rmse(clean.targets, clean.forecasts),
        "mape": mape(clean.targets, clean.forecasts),
        "worst_case": worst_case_abs_error(clean.targets, clean.forecasts),
    }

    attack_metrics = {
        "mae": mae(attack.targets, attack.forecasts),
        "rmse": rmse(attack.targets, attack.forecasts),
        "mape": mape(attack.targets, attack.forecasts),
        "worst_case": worst_case_abs_error(attack.targets, attack.forecasts),
    }

    checks["attack_differs_from_clean"] = abs(attack_metrics["mae"] - clean_metrics["mae"]) > 1e-9

    robustness = {
        "mae_delta": robustness_delta(clean_metrics["mae"], attack_metrics["mae"]),
        "mae_ratio": robustness_ratio(clean_metrics["mae"], attack_metrics["mae"]),
    }

    return {
        "checks": checks,
        "clean": clean_metrics,
        "attack": attack_metrics,
        "robustness": robustness,
        "rows": {"train": len(bundle.train), "valid": len(bundle.valid), "test": len(bundle.test)},
        "convergence": attack.convergence,
    }
