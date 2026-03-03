"""End-to-end verification suite testing data ingestion, determinism, and robustness."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Any

from .data import DataProfile, load_dataset, load_source_rows
from .distributed import ParallelGameRunner, RayParallelGameRunner, parallel_runner
from .game import ForecastGame
from .metrics import mae, mape, rmse, robustness_delta, robustness_ratio, worst_case_abs_error
from .observability import export_prometheus_metrics
from .types import ForecastState, SimulationConfig, evolve_state


def _scenario_metrics(cfg: SimulationConfig, init: ForecastState) -> dict[str, Any]:
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
            "rmse_delta": robustness_delta(clean_metrics["rmse"], attack_metrics["rmse"]),
            "rmse_ratio": robustness_ratio(clean_metrics["rmse"], attack_metrics["rmse"]),
            "mape_delta": robustness_delta(clean_metrics["mape"], attack_metrics["mape"]),
            "mape_ratio": robustness_ratio(clean_metrics["mape"], attack_metrics["mape"]),
            "worst_case_delta": robustness_delta(clean_metrics["worst_case"], attack_metrics["worst_case"]),
            "worst_case_ratio": robustness_ratio(clean_metrics["worst_case"], attack_metrics["worst_case"]),
        },
        "convergence": attack.convergence,
    }


def _verify_qualitative_determinism(
    base_cfg: SimulationConfig,
    init: ForecastState,
    *,
    verification_runs: int = 100,
) -> dict[str, Any]:
    """Run N identical qual-enabled simulations and assert identical outputs."""
    qual_cfg = replace(base_cfg, enable_qual=True)
    qual_dataset: dict[int, dict[str, Any]] = {
        5: {
            "timestamp": "2023-01-05",
            "source_id": "beige_book",
            "text": "Overall economic activity was moderate.",
            "metadata": {"synthetic": True},
        },
        25: {
            "timestamp": "2023-01-25",
            "source_id": "pmi",
            "text": "Manufacturing expanded with rising prices.",
            "metadata": {"synthetic": True},
        },
    }

    reference_hash: str | None = None
    all_match = True

    for _ in range(verification_runs):
        game = ForecastGame(qual_cfg, seed=42)
        game.set_qual_dataset(qual_dataset)
        output = game.run(init, rounds=qual_cfg.horizon, disturbed=False)

        h = sha256()
        for traj in output.trajectories:
            h.update(str(traj.state.qualitative_state).encode())
            h.update(str(traj.state.economic_regime).encode())
        run_hash = h.hexdigest()

        if reference_hash is None:
            reference_hash = run_hash
        elif run_hash != reference_hash:
            all_match = False
            break

    return {
        "deterministic": all_match,
        "runs": verification_runs,
        "reference_hash": reference_hash,
    }


def run_verification(backend: str = "auto", *, enable_qual: bool = False) -> dict[str, Any]:
    """Execute the full verification battery and return a results dict.

    Args:
        backend: Parallel runner backend (``"auto"``, ``"ray"``, or ``"multiprocessing"``).
        enable_qual: Include qualitative-determinism checks.
    """
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

    runner = parallel_runner(backend=backend)
    deterministic_results = runner.run_seeds(cfg, init, seeds=[99] * 100, disturbed=True)
    deterministic_ok = all(
        deterministic_results[0]["forecasts"] == r["forecasts"]
        for r in deterministic_results[1:]
    )

    stress_cfg = replace(cfg, horizon=10_000, max_rounds=10_000)
    stress_run = ForecastGame(stress_cfg, seed=1).run(init, disturbed=True)

    sweep_configs = [
        replace(cfg, adversarial_intensity=i, disturbance_model="volatility", defense_model="ensemble")
        for i in [0.5, 1.0, 1.5]
    ]
    sweep_results = runner.map_scenarios(sweep_configs, init, seeds=[13, 13, 13])
    intensity_sweep = {}
    for intensity, result in zip([0.5, 1.0, 1.5], sweep_results):
        clean_result = _scenario_metrics(
            replace(cfg, adversarial_intensity=intensity, disturbance_model="volatility", defense_model="ensemble"),
            init,
        )
        intensity_sweep[str(intensity)] = clean_result["robustness"]

    synth_bundle = load_dataset(DataProfile(source="sample_csv", periods=60, normalize=True))
    synth_init = ForecastState(t=0, value=float(synth_bundle.train[-1]["target"]), exogenous=0.0, hidden_shift=0.0)
    synth_out = ForecastGame(cfg, seed=13).run(synth_init, disturbed=False)
    hybrid_out = ForecastGame(cfg, seed=13).run(
        ForecastState(t=0, value=float(hybrid_bundle.train[-1]["target"]), exogenous=0.0, hidden_shift=0.0),
        disturbed=False,
    )
    synth_mae_val = mae(synth_out.targets, synth_out.forecasts)
    hybrid_mae_val = mae(hybrid_out.targets, hybrid_out.forecasts)
    hybrid_mae_within_10pct = (
        abs(hybrid_mae_val - synth_mae_val) / max(1e-9, synth_mae_val) < 0.10
        if synth_mae_val > 0 else True
    )

    checks = {
        "split_non_empty": len(bundle.train) > 0 and len(bundle.valid) > 0 and len(bundle.test) > 0,
        "pure_transition": s1 == deterministic_again,
        "max_rounds_respected": baseline["convergence"]["rounds_executed"] <= cfg.max_rounds,
        "source_adapter_rows": len(source_rows) >= 10,
        "real_data_split_non_empty": len(real_bundle.train) > 0 and len(real_bundle.valid) > 0,
        "hybrid_data_non_empty": len(hybrid_bundle.train) > 0,
        "attack_differs_from_clean": abs(baseline["attack"]["mae"] - baseline["clean"]["mae"]) > 1e-9,
        "deterministic_100_runs": deterministic_ok,
        "stress_10k_rounds": stress_run.convergence["rounds_executed"] == 10_000,
        "parallel_runner_available": isinstance(runner, (ParallelGameRunner, RayParallelGameRunner)),
        "hybrid_mae_within_10pct": hybrid_mae_within_10pct,
    }

    undef_cfg = replace(cfg, defense_model="identity")
    undef_metrics = _scenario_metrics(undef_cfg, init)
    undef_clean_mae = undef_metrics["clean"]["mae"]
    undef_attack_mae = undef_metrics["attack"]["mae"]
    undef_exceeds_20pct = (
        (undef_attack_mae - undef_clean_mae) / max(1e-9, undef_clean_mae) >= 0.20
        if undef_clean_mae > 0 else False
    )
    checks["undefended_attack_exceeds_20pct"] = undef_exceeds_20pct

    defended_cfg = replace(cfg, defense_model="ensemble")
    defended_metrics = _scenario_metrics(defended_cfg, init)
    defended_clean_mae = defended_metrics["clean"]["mae"]
    defended_attack_mae = defended_metrics["attack"]["mae"]
    defended_within_5pct = (
        abs(defended_attack_mae - defended_clean_mae) / max(1e-9, defended_clean_mae) < 0.05
        if defended_clean_mae > 0 else True
    )
    checks["defended_attack_within_5pct"] = defended_within_5pct

    if enable_qual:
        qual_check = _verify_qualitative_determinism(
            cfg, init, verification_runs=cfg.verification_runs,
        )
        checks["qualitative_determinism"] = qual_check["deterministic"]

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
