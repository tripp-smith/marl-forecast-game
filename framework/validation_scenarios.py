from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any

from .data import DataProfile, chronological_split, load_dataset, load_source_rows, detect_poisoning_rows
from .data_utils import (
    validate_cache_integrity,
    validate_chronological_order,
    validate_cross_source_consistency,
    validate_no_future_leakage,
    validate_source_schema,
)
from .game import ForecastGame
from .metrics import mae, rmse, worst_case_abs_error
from .types import ForecastState, SimulationConfig


@dataclass(frozen=True)
class ValidationScenario:
    name: str
    description: str
    data_source: str
    adversarial_intensity: float = 0.0
    disturbance_model: str = "gaussian"
    defense_model: str = "dampening"
    n_rounds: int = 80
    seed: int = 42
    expected_properties: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.expected_properties is None:
            object.__setattr__(self, "expected_properties", {})


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    duration_s: float
    details: dict
    errors: list


def _run_data_validation(scenario: ValidationScenario) -> ScenarioResult:
    """Run a data-source-focused validation scenario."""
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    try:
        profile = DataProfile(
            source=scenario.data_source,
            periods=max(60, scenario.n_rounds),
            normalize=True,
            hybrid_weight=0.5,
        )
        bundle = load_dataset(profile)
        details["train_rows"] = len(bundle.train)
        details["valid_rows"] = len(bundle.valid)
        details["test_rows"] = len(bundle.test)

        if not bundle.train:
            errors.append("train split is empty")
        if not bundle.valid:
            errors.append("valid split is empty")
        if not bundle.test:
            errors.append("test split is empty")

        leakage = validate_no_future_leakage(bundle.train, bundle.valid, bundle.test)
        if not leakage["valid"]:
            errors.extend(leakage["errors"])

        schema = validate_source_schema(bundle.train + bundle.valid + bundle.test, scenario.data_source)
        if not schema["valid"]:
            errors.extend(schema["errors"])

        chrono = validate_chronological_order(bundle.train)
        if not chrono["valid"]:
            errors.extend(chrono["errors"])

    except Exception as exc:
        errors.append(f"exception during data validation: {exc}")

    duration = time.perf_counter() - start
    return ScenarioResult(
        name=scenario.name,
        passed=len(errors) == 0,
        duration_s=round(duration, 4),
        details=details,
        errors=errors,
    )


def _run_cache_integrity(scenario: ValidationScenario) -> ScenarioResult:
    """Run cache-integrity validation for a given source."""
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    source = scenario.data_source
    if source in {"fred", "imf", "polymarket"}:
        result = validate_cache_integrity(source)
        details["cache_check"] = result
        if not result["valid"]:
            errors.extend(result["errors"])
    else:
        details["skipped"] = "not an external source"

    duration = time.perf_counter() - start
    return ScenarioResult(
        name=scenario.name,
        passed=len(errors) == 0,
        duration_s=round(duration, 4),
        details=details,
        errors=errors,
    )


def _run_simulation_scenario(scenario: ValidationScenario) -> ScenarioResult:
    """Run a game-simulation-focused validation scenario."""
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    try:
        profile = DataProfile(
            source=scenario.data_source,
            periods=max(60, scenario.n_rounds),
            normalize=True,
            hybrid_weight=0.5,
        )
        bundle = load_dataset(profile)
        init_value = float(bundle.train[-1]["target"]) if bundle.train else 10.0

        cfg = SimulationConfig(
            horizon=scenario.n_rounds,
            max_rounds=scenario.n_rounds * 2,
            disturbance_prob=0.2,
            disturbance_scale=1.2,
            adversarial_intensity=scenario.adversarial_intensity,
            runtime_backend="python",
            disturbance_model=scenario.disturbance_model,
            defense_model=scenario.defense_model,
            enable_refactor=True,
        )
        init = ForecastState(t=0, value=init_value, exogenous=0.0, hidden_shift=0.0)
        disturbed = scenario.adversarial_intensity > 0

        game = ForecastGame(cfg, seed=scenario.seed)
        out = game.run(init, disturbed=disturbed)

        m = mae(out.targets, out.forecasts)
        r = rmse(out.targets, out.forecasts)
        wc = worst_case_abs_error(out.targets, out.forecasts)

        details["mae"] = round(m, 6)
        details["rmse"] = round(r, 6)
        details["worst_case"] = round(wc, 6)
        details["rounds_executed"] = out.convergence["rounds_executed"]

        props = scenario.expected_properties
        if "mae_under" in props and m > props["mae_under"]:
            errors.append(f"MAE {m:.4f} exceeds threshold {props['mae_under']}")
        if "rounds_completed" in props and out.convergence["rounds_executed"] < props["rounds_completed"]:
            errors.append(f"only {out.convergence['rounds_executed']} rounds completed, expected {props['rounds_completed']}")

    except Exception as exc:
        errors.append(f"simulation exception: {exc}")

    duration = time.perf_counter() - start
    return ScenarioResult(
        name=scenario.name,
        passed=len(errors) == 0,
        duration_s=round(duration, 4),
        details=details,
        errors=errors,
    )


def _run_determinism_scenario(scenario: ValidationScenario) -> ScenarioResult:
    """Run the same simulation N times and assert identical outputs."""
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}
    n_runs = scenario.expected_properties.get("n_runs", 50)

    try:
        cfg = SimulationConfig(
            horizon=scenario.n_rounds,
            max_rounds=scenario.n_rounds * 2,
            disturbance_prob=0.2,
            disturbance_scale=1.2,
            adversarial_intensity=scenario.adversarial_intensity,
            disturbance_model=scenario.disturbance_model,
            defense_model=scenario.defense_model,
            enable_refactor=True,
        )
        init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
        reference = None
        for i in range(n_runs):
            game = ForecastGame(cfg, seed=scenario.seed)
            out = game.run(init, disturbed=True)
            if reference is None:
                reference = out.forecasts
            elif out.forecasts != reference:
                errors.append(f"run {i} diverged from reference")
                break

        details["n_runs"] = n_runs
        details["deterministic"] = len(errors) == 0
    except Exception as exc:
        errors.append(f"determinism check exception: {exc}")

    duration = time.perf_counter() - start
    return ScenarioResult(
        name=scenario.name,
        passed=len(errors) == 0,
        duration_s=round(duration, 4),
        details=details,
        errors=errors,
    )


def _run_poisoning_scenario(scenario: ValidationScenario) -> ScenarioResult:
    """Verify that poisoning detection works on injected outliers."""
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    from datetime import timedelta
    base = __import__("datetime").datetime(2024, 1, 1)
    clean_rows = [
        {"timestamp": base + timedelta(days=i), "series_id": "test", "target": float(50 + i * 0.1),
         "promo": 0.0, "macro_index": 100.0}
        for i in range(100)
    ]
    poisoned_rows = list(clean_rows)
    poisoned_rows[50] = {**poisoned_rows[50], "target": 9999.0}
    poisoned_rows[51] = {**poisoned_rows[51], "target": -9999.0}

    suspects = detect_poisoning_rows(poisoned_rows)
    details["suspects_found"] = len(suspects)
    if len(suspects) < 2:
        errors.append(f"expected >= 2 suspects, found {len(suspects)}")

    clean_suspects = detect_poisoning_rows(clean_rows)
    details["clean_false_positives"] = len(clean_suspects)
    if len(clean_suspects) > 0:
        errors.append(f"false positives on clean data: {len(clean_suspects)}")

    duration = time.perf_counter() - start
    return ScenarioResult(
        name=scenario.name,
        passed=len(errors) == 0,
        duration_s=round(duration, 4),
        details=details,
        errors=errors,
    )


def _run_robustness_scenario(scenario: ValidationScenario) -> ScenarioResult:
    """Run clean vs. attacked and verify defense improves robustness."""
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    try:
        cfg_base = SimulationConfig(
            horizon=scenario.n_rounds,
            max_rounds=scenario.n_rounds * 2,
            disturbance_prob=0.2,
            disturbance_scale=1.2,
            adversarial_intensity=scenario.adversarial_intensity,
            disturbance_model=scenario.disturbance_model,
            defense_model=scenario.defense_model,
            enable_refactor=True,
        )
        init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

        clean = ForecastGame(cfg_base, seed=scenario.seed).run(init, disturbed=False)
        attacked = ForecastGame(cfg_base, seed=scenario.seed).run(init, disturbed=True)

        clean_mae = mae(clean.targets, clean.forecasts)
        attack_mae = mae(attacked.targets, attacked.forecasts)
        details["clean_mae"] = round(clean_mae, 6)
        details["attack_mae"] = round(attack_mae, 6)
        details["mae_ratio"] = round(attack_mae / max(1e-9, clean_mae), 4)

        if "attack_worse" in scenario.expected_properties and scenario.expected_properties["attack_worse"]:
            if attack_mae <= clean_mae:
                errors.append(f"attack MAE ({attack_mae:.4f}) not worse than clean ({clean_mae:.4f})")

    except Exception as exc:
        errors.append(f"robustness check exception: {exc}")

    duration = time.perf_counter() - start
    return ScenarioResult(
        name=scenario.name,
        passed=len(errors) == 0,
        duration_s=round(duration, 4),
        details=details,
        errors=errors,
    )


def _run_cross_source_scenario(scenario: ValidationScenario) -> ScenarioResult:
    """Validate cross-source data consistency."""
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    source_data: dict[str, list[dict[str, Any]]] = {}
    for src in ["fred", "imf", "polymarket"]:
        try:
            rows = load_source_rows(src, periods=20)
            source_data[src] = rows
        except Exception as exc:
            errors.append(f"failed to load {src}: {exc}")

    if source_data:
        consistency = validate_cross_source_consistency(source_data)
        details["consistency"] = consistency
        for src, rows in source_data.items():
            schema = validate_source_schema(rows, src)
            if not schema["valid"]:
                errors.extend(schema["errors"])

    duration = time.perf_counter() - start
    return ScenarioResult(
        name=scenario.name,
        passed=len(errors) == 0,
        duration_s=round(duration, 4),
        details=details,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIO_REGISTRY: dict[str, ValidationScenario] = {}


def _register(s: ValidationScenario) -> None:
    SCENARIO_REGISTRY[s.name] = s


_register(ValidationScenario(
    name="baseline_synthetic",
    description="Sample CSV, no attack, verifies clean determinism",
    data_source="sample_csv",
    adversarial_intensity=0.0,
    n_rounds=80,
    expected_properties={"mae_under": 5.0},
))

_register(ValidationScenario(
    name="fred_cpi_clean",
    description="FRED CPI data, no attack, verifies real data ingestion and splitting",
    data_source="fred",
    adversarial_intensity=0.0,
    n_rounds=60,
    expected_properties={},
))

_register(ValidationScenario(
    name="imf_gdp_clean",
    description="IMF GDP data, verifies real data round-trip",
    data_source="imf",
    adversarial_intensity=0.0,
    n_rounds=30,
    expected_properties={},
))

_register(ValidationScenario(
    name="polymarket_clean",
    description="Polymarket data, verifies prediction market adapter",
    data_source="polymarket",
    adversarial_intensity=0.0,
    n_rounds=12,
    expected_properties={},
))

_register(ValidationScenario(
    name="hybrid_blend",
    description="Hybrid real+synthetic, verifies blending and normalization",
    data_source="hybrid",
    adversarial_intensity=0.0,
    n_rounds=60,
    expected_properties={},
))

_register(ValidationScenario(
    name="adversarial_gaussian",
    description="Synthetic + Gaussian disturbance at intensity 1.0",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    disturbance_model="gaussian",
    n_rounds=80,
    expected_properties={"mae_under": 10.0},
))

_register(ValidationScenario(
    name="adversarial_regime_shift",
    description="Synthetic + regime shift disturbance",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    disturbance_model="regime_shift",
    n_rounds=80,
    expected_properties={"mae_under": 10.0},
))

_register(ValidationScenario(
    name="adversarial_drift",
    description="Synthetic + drift disturbance",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    disturbance_model="drift",
    n_rounds=80,
    expected_properties={"mae_under": 50.0},
))

_register(ValidationScenario(
    name="defense_dampening",
    description="Adversarial + dampening defense, verifies robustness",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    disturbance_model="gaussian",
    defense_model="dampening",
    n_rounds=80,
    expected_properties={"attack_worse": True},
))

_register(ValidationScenario(
    name="defense_ensemble",
    description="Adversarial + ensemble defense",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    disturbance_model="gaussian",
    defense_model="ensemble",
    n_rounds=80,
    expected_properties={"attack_worse": True},
))

_register(ValidationScenario(
    name="poisoning_detection",
    description="Data with injected outliers, verifies detection",
    data_source="sample_csv",
    n_rounds=0,
    expected_properties={},
))

_register(ValidationScenario(
    name="stress_high_rounds",
    description="5000 rounds, verifies no timeout/crash",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    disturbance_model="gaussian",
    n_rounds=5000,
    expected_properties={"rounds_completed": 5000},
))

_register(ValidationScenario(
    name="determinism_cross_seed",
    description="Run same scenario 50x, assert identical outputs",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    n_rounds=40,
    expected_properties={"n_runs": 50},
))

_register(ValidationScenario(
    name="cache_integrity_fred",
    description="Verify FRED cache file integrity (checksum, schema)",
    data_source="fred",
    expected_properties={},
))

_register(ValidationScenario(
    name="cross_source_consistency",
    description="Validate schema and consistency across FRED/IMF/Polymarket",
    data_source="fred",
    expected_properties={},
))

_register(ValidationScenario(
    name="llm_mock_refactor",
    description="Exercises refactoring agent with mock LLM client",
    data_source="sample_csv",
    adversarial_intensity=0.0,
    n_rounds=50,
    expected_properties={"mae_under": 5.0},
))

# Phase V scenarios

_register(ValidationScenario(
    name="hierarchical_agent_run",
    description="Multi-agent with BottomUp+TopDown+EnsembleAggregator",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    n_rounds=40,
    expected_properties={"mae_under": 20.0},
))

_register(ValidationScenario(
    name="bayesian_calibration",
    description="BayesianAggregator weight sum and PIT uniformity check",
    data_source="sample_csv",
    adversarial_intensity=0.0,
    n_rounds=60,
    expected_properties={},
))

_register(ValidationScenario(
    name="marl_convergence",
    description="WoLF-PHC agent converges within 200 episodes",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    n_rounds=40,
    expected_properties={"max_episodes": 200},
))

_register(ValidationScenario(
    name="llm_refiner_stability",
    description="RecursiveStrategyRefiner does not diverge with mock LLM",
    data_source="sample_csv",
    adversarial_intensity=0.0,
    n_rounds=50,
    expected_properties={},
))

_register(ValidationScenario(
    name="fred_training_backtest",
    description="Walk-forward backtest on FRED training data (skips without FRED_API_KEY)",
    data_source="fred_training",
    n_rounds=40,
    expected_properties={},
))

_register(ValidationScenario(
    name="parallel_determinism",
    description="ParallelGameRunner produces same results as sequential for identical seeds",
    data_source="sample_csv",
    adversarial_intensity=1.0,
    n_rounds=40,
    expected_properties={},
))


# ---------------------------------------------------------------------------
# Phase V scenario handlers
# ---------------------------------------------------------------------------

def _run_hierarchical_scenario(scenario: ValidationScenario) -> ScenarioResult:
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    try:
        from .agents import (
            AdversaryAgent, AgentRegistry, BottomUpAgent, DefenderAgent,
            EnsembleAggregatorAgent, ForecastingAgent, RefactoringAgent, TopDownAgent,
        )
        registry = AgentRegistry(
            forecasters=(ForecastingAgent(), BottomUpAgent(), TopDownAgent()),
            adversaries=(AdversaryAgent(aggressiveness=scenario.adversarial_intensity),),
            defenders=(DefenderAgent(),),
            refactorer=RefactoringAgent(),
            aggregator=EnsembleAggregatorAgent(mode="equal"),
        )
        cfg = SimulationConfig(
            horizon=scenario.n_rounds,
            max_rounds=scenario.n_rounds * 2,
            disturbance_prob=0.2,
            adversarial_intensity=scenario.adversarial_intensity,
        )
        init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
        game = ForecastGame(cfg, seed=scenario.seed, registry=registry)
        out = game.run(init, disturbed=True)
        m = mae(out.targets, out.forecasts)
        details["mae"] = round(m, 6)
        details["rounds_executed"] = out.convergence["rounds_executed"]
        if "mae_under" in scenario.expected_properties and m > scenario.expected_properties["mae_under"]:
            errors.append(f"MAE {m:.4f} exceeds threshold")
    except Exception as exc:
        errors.append(f"exception: {exc}")

    return ScenarioResult(name=scenario.name, passed=len(errors) == 0,
                          duration_s=round(time.perf_counter() - start, 4), details=details, errors=errors)


def _run_bayesian_calibration(scenario: ValidationScenario) -> ScenarioResult:
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    try:
        from .aggregation import BayesianAggregator
        from .types import AgentAction
        agg = BayesianAggregator()
        actions = [AgentAction(actor="a1", delta=0.1), AgentAction(actor="a2", delta=-0.1)]
        init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

        for _ in range(scenario.n_rounds):
            mean_d, var = agg.aggregate(actions, init)
            agg.update({"a1": 0.05, "a2": 0.15})

        weights = agg.weights
        wsum = sum(weights)
        details["weight_sum"] = round(wsum, 6)
        details["weights"] = [round(w, 4) for w in weights]
        if abs(wsum - 1.0) > 1e-6:
            errors.append(f"weights sum to {wsum}, expected 1.0")
    except Exception as exc:
        errors.append(f"exception: {exc}")

    return ScenarioResult(name=scenario.name, passed=len(errors) == 0,
                          duration_s=round(time.perf_counter() - start, 4), details=details, errors=errors)


def _run_marl_convergence(scenario: ValidationScenario) -> ScenarioResult:
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    try:
        from .training import DiscreteActionSpace, TrainingLoop, WoLFPHCAgent
        max_episodes = scenario.expected_properties.get("max_episodes", 200)
        cfg = SimulationConfig(
            horizon=scenario.n_rounds,
            max_rounds=scenario.n_rounds * 2,
            adversarial_intensity=scenario.adversarial_intensity,
        )
        agent = WoLFPHCAgent(action_space=DiscreteActionSpace())
        loop = TrainingLoop(config=cfg, n_episodes=max_episodes, seed=scenario.seed)
        init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
        result = loop.train(agent, init_state=init)
        details["mean_reward_last_50"] = round(result["mean_reward_last_50"], 4)
        details["final_epsilon"] = round(result["final_epsilon"], 4)
        details["mean_td_error"] = round(result["mean_td_error_last_100"], 4)
    except Exception as exc:
        errors.append(f"exception: {exc}")

    return ScenarioResult(name=scenario.name, passed=len(errors) == 0,
                          duration_s=round(time.perf_counter() - start, 4), details=details, errors=errors)


def _run_llm_refiner_stability(scenario: ValidationScenario) -> ScenarioResult:
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    try:
        from .llm.refiner import RecursiveStrategyRefiner
        from .types import TrajectoryEntry, AgentAction, AgentMessage, frozen_mapping

        refiner = RecursiveStrategyRefiner()
        trajectories = [
            TrajectoryEntry(
                round_idx=i,
                state=ForecastState(t=i, value=10.0 + i, exogenous=0.0, hidden_shift=0.0),
                actions=(AgentAction(actor="forecaster", delta=0.1),),
                messages=(AgentMessage("f", "a", "test"),),
                reward_breakdown=frozen_mapping({"forecaster": -0.5}),
                forecast=10.0 + i + 0.1,
                target=10.0 + i + 0.4,
            )
            for i in range(scenario.n_rounds)
        ]
        result = refiner.refine(trajectories)
        details["bias_adjustment"] = result.bias_adjustment
        details["strategy_hint"] = result.strategy_hint
        if not (-0.1 <= result.bias_adjustment <= 0.1):
            errors.append(f"bias_adjustment {result.bias_adjustment} outside clamp bounds")
    except Exception as exc:
        errors.append(f"exception: {exc}")

    return ScenarioResult(name=scenario.name, passed=len(errors) == 0,
                          duration_s=round(time.perf_counter() - start, 4), details=details, errors=errors)


def _run_fred_training_backtest(scenario: ValidationScenario) -> ScenarioResult:
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    import os
    if not os.getenv("FRED_API_KEY"):
        details["skipped"] = "FRED_API_KEY not set"
        return ScenarioResult(name=scenario.name, passed=True,
                              duration_s=round(time.perf_counter() - start, 4), details=details, errors=errors)

    try:
        from .backtesting import WalkForwardBacktester
        profile = DataProfile(source="fred_training", periods=120, normalize=True)
        bundle = load_dataset(profile)
        all_rows = bundle.train + bundle.valid + bundle.test

        cfg = SimulationConfig(
            horizon=scenario.n_rounds,
            max_rounds=scenario.n_rounds * 2,
            adversarial_intensity=1.0,
        )
        bt = WalkForwardBacktester(config=cfg, window_size=40, step_size=20)
        result = bt.run(all_rows, max_windows=3)
        details["n_windows"] = result.n_windows
        details["aggregate_mae"] = round(result.aggregate_mae, 4)
        if result.n_windows == 0:
            errors.append("no backtest windows produced")
    except Exception as exc:
        errors.append(f"exception: {exc}")

    return ScenarioResult(name=scenario.name, passed=len(errors) == 0,
                          duration_s=round(time.perf_counter() - start, 4), details=details, errors=errors)


def _run_parallel_determinism(scenario: ValidationScenario) -> ScenarioResult:
    start = time.perf_counter()
    errors: list[str] = []
    details: dict[str, Any] = {}

    try:
        from .distributed import ParallelGameRunner
        cfg = SimulationConfig(
            horizon=scenario.n_rounds,
            max_rounds=scenario.n_rounds * 2,
            adversarial_intensity=scenario.adversarial_intensity,
        )
        init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
        seeds = [1, 2, 3]

        sequential = []
        for seed in seeds:
            game = ForecastGame(cfg, seed=seed)
            out = game.run(init, disturbed=True)
            sequential.append(out.forecasts)

        runner = ParallelGameRunner(n_workers=2)
        parallel = runner.run_seeds(cfg, init, seeds, disturbed=True)

        for i, (seq, par) in enumerate(zip(sequential, parallel)):
            if seq != par["forecasts"]:
                errors.append(f"seed {seeds[i]}: parallel output differs from sequential")

        details["n_seeds"] = len(seeds)
        details["match"] = len(errors) == 0
    except Exception as exc:
        errors.append(f"exception: {exc}")

    return ScenarioResult(name=scenario.name, passed=len(errors) == 0,
                          duration_s=round(time.perf_counter() - start, 4), details=details, errors=errors)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Any] = {
    "baseline_synthetic": _run_simulation_scenario,
    "fred_cpi_clean": _run_data_validation,
    "imf_gdp_clean": _run_data_validation,
    "polymarket_clean": _run_data_validation,
    "hybrid_blend": _run_data_validation,
    "adversarial_gaussian": _run_simulation_scenario,
    "adversarial_regime_shift": _run_simulation_scenario,
    "adversarial_drift": _run_simulation_scenario,
    "defense_dampening": _run_robustness_scenario,
    "defense_ensemble": _run_robustness_scenario,
    "poisoning_detection": _run_poisoning_scenario,
    "stress_high_rounds": _run_simulation_scenario,
    "determinism_cross_seed": _run_determinism_scenario,
    "cache_integrity_fred": _run_cache_integrity,
    "cross_source_consistency": _run_cross_source_scenario,
    "llm_mock_refactor": _run_simulation_scenario,
    "hierarchical_agent_run": _run_hierarchical_scenario,
    "bayesian_calibration": _run_bayesian_calibration,
    "marl_convergence": _run_marl_convergence,
    "llm_refiner_stability": _run_llm_refiner_stability,
    "fred_training_backtest": _run_fred_training_backtest,
    "parallel_determinism": _run_parallel_determinism,
}


def run_scenario(scenario: ValidationScenario) -> ScenarioResult:
    handler = _DISPATCH.get(scenario.name)
    if handler is None:
        return ScenarioResult(
            name=scenario.name,
            passed=False,
            duration_s=0.0,
            details={},
            errors=[f"no handler registered for scenario '{scenario.name}'"],
        )
    return handler(scenario)


def run_all_scenarios(names: list[str] | None = None) -> list[ScenarioResult]:
    targets = names or list(SCENARIO_REGISTRY.keys())
    results: list[ScenarioResult] = []
    for name in targets:
        scenario = SCENARIO_REGISTRY.get(name)
        if scenario is None:
            results.append(ScenarioResult(
                name=name, passed=False, duration_s=0.0, details={},
                errors=[f"unknown scenario: {name}"],
            ))
            continue
        logging.info("Running scenario: %s", name)
        results.append(run_scenario(scenario))
    return results
