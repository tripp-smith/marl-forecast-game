from __future__ import annotations

from dataclasses import replace

from .data import DataProfile, load_dataset, load_source_rows
from .game import ForecastGame
from .metrics import mae, mape, rmse, robustness_delta, robustness_ratio, worst_case_abs_error
from .observability import export_prometheus_metrics
from .types import ForecastState, SimulationConfig, evolve_state


def _scenario_metrics(cfg: SimulationConfig, init: ForecastState) -> dict:
    game_clean = ForecastGame(cfg, seed=13)
    game_attack = ForecastGame(cfg, seed=13)
    clean = game_clean.run(init, disturbed=False)
    attack = game_attack.run(init, disturbed=True)

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

    return {
        "clean": clean_metrics,
        "attack": attack_metrics,
        "robustness": {
            "mae_delta": robustness_delta(clean_metrics["mae"], attack_metrics["mae"]),
            "mae_ratio": robustness_ratio(clean_metrics["mae"], attack_metrics["mae"]),
        },
        "convergence": attack.convergence,
    }


def run_verification() -> dict:
    bundle = load_dataset(DataProfile(source="sample_csv", periods=240, normalize=True))

    cfg = SimulationConfig(
        horizon=80,
        max_rounds=200,
        disturbance_prob=0.2,
        disturbance_scale=1.2,
        adversarial_intensity=1.0,
        runtime_backend="python",
        disturbance_model="gaussian",
        defense_model="ensemble",
        enable_refactor=True,
    )

    init = ForecastState(t=0, value=float(bundle.train[-1]["target"]), exogenous=0.0, hidden_shift=0.0)
    baseline = _scenario_metrics(cfg, init)

    s0 = ForecastState(t=0, value=10.0, exogenous=1.2, hidden_shift=0.0)
    s1 = evolve_state(s0, base_trend=0.4, noise=0.0, disturbance=0.1)
    deterministic_again = evolve_state(s0, base_trend=0.4, noise=0.0, disturbance=0.1)

    source_rows = load_source_rows("fred", periods=20)
    real_bundle = load_dataset(DataProfile(source="fred", periods=60, normalize=True))
    hybrid_bundle = load_dataset(DataProfile(source="hybrid", periods=60, normalize=True, hybrid_weight=0.6))

    deterministic_runs = []
    for _ in range(100):
        run = ForecastGame(cfg, seed=99).run(init, disturbed=True)
        deterministic_runs.append(run.forecasts)
    deterministic_ok = all(deterministic_runs[0] == candidate for candidate in deterministic_runs[1:])

    stress_cfg = replace(cfg, horizon=10_000, max_rounds=10_000)
    stress_run = ForecastGame(stress_cfg, seed=1).run(init, disturbed=True)

    intensity_sweep = {}
    for intensity in [0.5, 1.0, 1.5]:
        sweep_cfg = replace(cfg, adversarial_intensity=intensity, disturbance_model="volatility", defense_model="ensemble")
        intensity_sweep[str(intensity)] = _scenario_metrics(sweep_cfg, init)["robustness"]

    checks = {
        "split_non_empty": len(bundle.train) > 0 and len(bundle.valid) > 0 and len(bundle.test) > 0,
        "pure_transition": s1 == deterministic_again,
        "max_rounds_respected": baseline["convergence"]["rounds_executed"] <= cfg.max_rounds,
        "source_adapter_rows": len(source_rows) == 20,
        "real_data_split_non_empty": len(real_bundle.train) > 0 and len(real_bundle.valid) > 0,
        "hybrid_data_non_empty": len(hybrid_bundle.train) > 0,
        "attack_differs_from_clean": abs(baseline["attack"]["mae"] - baseline["clean"]["mae"]) > 1e-9,
        "deterministic_100_runs": deterministic_ok,
        "stress_10k_rounds": stress_run.convergence["rounds_executed"] == 10_000,
    }

    return {
        "checks": checks,
        "clean": baseline["clean"],
        "attack": baseline["attack"],
        "robustness": baseline["robustness"],
        "rows": {"train": len(bundle.train), "valid": len(bundle.valid), "test": len(bundle.test)},
        "convergence": baseline["convergence"],
        "scenario_sweep": intensity_sweep,
        "prometheus_metrics_present": bool(export_prometheus_metrics()),
    }
