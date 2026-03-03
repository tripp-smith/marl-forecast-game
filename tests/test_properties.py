"""Property-based tests using Hypothesis for MARL Forecast Game invariants."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from framework.agents import WolfpackAdversary
from framework.aggregation import BayesianAggregator
from framework.data import chronological_split, detect_poisoning_rows, normalize_features
from framework.defenses import (
    BiasGuardDefense,
    ClippingDefense,
    DampeningDefense,
    EnsembleDefense,
    defense_from_name,
)
from framework.game import ForecastGame
from framework.metrics import mae, mape, rmse, worst_case_abs_error
from framework.types import (
    AgentAction,
    AgentMessage,
    ConfidenceInterval,
    ForecastState,
    SimulationConfig,
    TrajectoryEntry,
    evolve_state,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies for core types
# ---------------------------------------------------------------------------

reasonable_floats = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
small_floats = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)

st_forecast_state = st.builds(
    ForecastState,
    t=st.integers(min_value=0, max_value=100_000),
    value=reasonable_floats,
    exogenous=small_floats,
    hidden_shift=small_floats,
)

st_agent_action = st.builds(
    AgentAction,
    actor=st.sampled_from(["forecaster", "adversary", "defender", "refactor"]),
    delta=small_floats,
)


# ---------------------------------------------------------------------------
# Property: state transition purity (deterministic)
# ---------------------------------------------------------------------------

@given(
    state=st_forecast_state,
    base_trend=small_floats,
    noise=small_floats,
    disturbance=small_floats,
)
@settings(max_examples=1000)
def test_evolve_state_is_deterministic(state, base_trend, noise, disturbance):
    s1 = evolve_state(state, base_trend=base_trend, noise=noise, disturbance=disturbance)
    s2 = evolve_state(state, base_trend=base_trend, noise=noise, disturbance=disturbance)
    assert s1 == s2
    assert s1.t == state.t + 1


# ---------------------------------------------------------------------------
# Property: frozen dataclasses are truly immutable
# ---------------------------------------------------------------------------

@given(state=st_forecast_state)
def test_forecast_state_immutability(state):
    with pytest.raises(FrozenInstanceError):
        state.value = 0.0  # type: ignore[misc]


@given(action=st_agent_action)
def test_agent_action_immutability(action):
    with pytest.raises(FrozenInstanceError):
        action.delta = 0.0  # type: ignore[misc]


def test_confidence_interval_immutability():
    ci = ConfidenceInterval(lower=0.0, upper=1.0)
    with pytest.raises(FrozenInstanceError):
        ci.lower = -1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Property: chronological_split never leaks future data into training
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=10, max_value=200),
    train_ratio=st.floats(min_value=0.2, max_value=0.7),
    valid_ratio=st.floats(min_value=0.05, max_value=0.25),
)
def test_chronological_split_no_future_leakage(n, train_ratio, valid_ratio):
    assume(train_ratio + valid_ratio < 0.98)

    base = datetime(2020, 1, 1)
    rows = [
        {"timestamp": base + timedelta(days=i), "series_id": "s1",
         "target": float(i), "promo": 0.0, "macro_index": 100.0}
        for i in range(n)
    ]
    bundle = chronological_split(rows, train=train_ratio, valid=valid_ratio)

    if bundle.train and bundle.valid:
        assert bundle.train[-1]["timestamp"] <= bundle.valid[0]["timestamp"]
    if bundle.valid and bundle.test:
        assert bundle.valid[-1]["timestamp"] <= bundle.test[0]["timestamp"]
    if bundle.train and bundle.test:
        assert bundle.train[-1]["timestamp"] <= bundle.test[0]["timestamp"]


# ---------------------------------------------------------------------------
# Property: poisoning detection monotonicity
# ---------------------------------------------------------------------------

@given(
    n_outliers=st.integers(min_value=0, max_value=5),
    base_value=st.floats(min_value=1.0, max_value=100.0),
)
def test_poisoning_detection_more_outliers_more_suspects(n_outliers, base_value):
    base_dt = datetime(2024, 1, 1)
    clean = [
        {"timestamp": base_dt + timedelta(days=i), "series_id": "s",
         "target": base_value + i * 0.01, "promo": 0.0, "macro_index": 100.0}
        for i in range(50)
    ]
    fewer = list(clean)
    more = list(clean)
    for i in range(min(n_outliers, 2)):
        fewer[i] = {**fewer[i], "target": base_value * 1000}
    for i in range(n_outliers):
        more[i] = {**more[i], "target": base_value * 1000}
    assert len(detect_poisoning_rows(more)) >= len(detect_poisoning_rows(fewer))


# ---------------------------------------------------------------------------
# Property: defense models produce bounded outputs
# ---------------------------------------------------------------------------

@given(
    forecast_delta=small_floats,
    adversary_delta=small_floats,
)
def test_clipping_defense_bounded(forecast_delta, adversary_delta):
    defense = ClippingDefense(clip=0.5)
    result = defense.defend(forecast_delta, adversary_delta)
    assert -0.5 <= result <= 0.5


@given(
    forecast_delta=small_floats,
    adversary_delta=small_floats,
)
def test_bias_guard_defense_bounded(forecast_delta, adversary_delta):
    defense = BiasGuardDefense(max_bias=0.3)
    result = defense.defend(forecast_delta, adversary_delta)
    assert abs(result) <= 0.3 + 1e-9


@given(
    forecast_delta=small_floats,
    adversary_delta=small_floats,
)
def test_dampening_defense_finite(forecast_delta, adversary_delta):
    defense = DampeningDefense()
    result = defense.defend(forecast_delta, adversary_delta)
    assert result == result  # not NaN
    assert abs(result) < float("inf")


# ---------------------------------------------------------------------------
# Property: metric non-negativity
# ---------------------------------------------------------------------------

@given(
    values=st.lists(
        st.tuples(reasonable_floats, reasonable_floats),
        min_size=2, max_size=50,
    ),
)
def test_metrics_non_negative(values):
    actual = [v[0] for v in values]
    predicted = [v[1] for v in values]
    assert mae(actual, predicted) >= 0.0
    assert rmse(actual, predicted) >= 0.0
    assert mape(actual, predicted) >= 0.0
    assert worst_case_abs_error(actual, predicted) >= 0.0


# ---------------------------------------------------------------------------
# Property: game determinism (same config + seed = identical outputs)
# ---------------------------------------------------------------------------

@given(
    seed=st.integers(min_value=1, max_value=10000),
    horizon=st.integers(min_value=5, max_value=50),
    init_value=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_game_determinism(seed, horizon, init_value):
    cfg = SimulationConfig(horizon=horizon, max_rounds=horizon * 2)
    init = ForecastState(t=0, value=init_value, exogenous=0.0, hidden_shift=0.0)
    out1 = ForecastGame(cfg, seed=seed).run(init, disturbed=True)
    out2 = ForecastGame(cfg, seed=seed).run(init, disturbed=True)
    assert out1.forecasts == out2.forecasts
    assert out1.targets == out2.targets


# ---------------------------------------------------------------------------
# Property: evolve_state preserves timestep monotonicity
# ---------------------------------------------------------------------------

@given(
    state=st_forecast_state,
    n_steps=st.integers(min_value=1, max_value=20),
)
def test_evolve_state_timestep_monotonic(state, n_steps):
    s = state
    for _ in range(n_steps):
        prev_t = s.t
        s = evolve_state(s, base_trend=0.4, noise=0.0, disturbance=0.0)
        assert s.t == prev_t + 1


# ---------------------------------------------------------------------------
# Property: normalization produces zero-mean features
# ---------------------------------------------------------------------------

@given(n=st.integers(min_value=5, max_value=100))
def test_normalization_zero_mean(n):
    base_dt = datetime(2024, 1, 1)
    rows = [
        {"timestamp": base_dt + timedelta(days=i), "series_id": "s",
         "target": float(i), "promo": float(i % 3), "macro_index": 100.0 + i * 0.5}
        for i in range(n)
    ]
    normed = normalize_features(rows)
    promo_mean = sum(r["promo"] for r in normed) / len(normed)
    macro_mean = sum(r["macro_index"] for r in normed) / len(normed)
    assert abs(promo_mean) < 1e-9
    assert abs(macro_mean) < 1e-9


# ---------------------------------------------------------------------------
# Property: evolve_state outputs are always finite
# ---------------------------------------------------------------------------

@given(
    state=st_forecast_state,
    base_trend=small_floats,
    noise=small_floats,
    disturbance=small_floats,
)
@settings(max_examples=500)
def test_evolve_state_finite_outputs(state, base_trend, noise, disturbance):
    import math
    s1 = evolve_state(state, base_trend=base_trend, noise=noise, disturbance=disturbance)
    assert math.isfinite(s1.value)
    assert math.isfinite(s1.exogenous)
    assert math.isfinite(s1.hidden_shift)


# ---------------------------------------------------------------------------
# Property: reward breakdown values are consistent across agents
# ---------------------------------------------------------------------------

@given(
    seed=st.integers(min_value=1, max_value=5000),
    init_value=reasonable_floats,
)
@settings(max_examples=50)
def test_reward_breakdown_consistency(seed, init_value):
    cfg = SimulationConfig(horizon=5, max_rounds=10)
    init = ForecastState(t=0, value=init_value, exogenous=0.0, hidden_shift=0.0)
    out = ForecastGame(cfg, seed=seed).run(init, disturbed=True)
    for step in out.steps:
        rb = step.reward_breakdown
        assert "forecaster" in rb
        assert "adversary" in rb
        assert "defender" in rb
        assert abs(rb["forecaster"] + rb["adversary"]) < 1e-9
        assert abs(rb["forecaster"] - rb["defender"]) < 1e-9


# ---------------------------------------------------------------------------
# Property: confidence intervals always span a finite positive range
# ---------------------------------------------------------------------------

@given(
    seed=st.integers(min_value=1, max_value=5000),
    init_value=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_confidence_interval_finite_range(seed, init_value):
    import math
    cfg = SimulationConfig(horizon=5, max_rounds=10)
    init = ForecastState(t=0, value=init_value, exogenous=0.0, hidden_shift=0.0)
    out = ForecastGame(cfg, seed=seed).run(init, disturbed=True)
    for step in out.steps:
        assert math.isfinite(step.confidence.lower)
        assert math.isfinite(step.confidence.upper)
        assert step.confidence.upper >= step.confidence.lower


# ---------------------------------------------------------------------------
# Property: trajectory log length equals steps executed
# ---------------------------------------------------------------------------

@given(
    horizon=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=1, max_value=5000),
)
@settings(max_examples=50)
def test_trajectory_log_length_matches_steps(horizon, seed):
    cfg = SimulationConfig(horizon=horizon, max_rounds=horizon + 10)
    init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
    out = ForecastGame(cfg, seed=seed).run(init, disturbed=True)
    assert len(out.trajectory_logs) == len(out.steps)
    assert len(out.forecasts) == len(out.steps)
    assert len(out.targets) == len(out.steps)


# ---------------------------------------------------------------------------
# Property: validate_immutability catches non-frozen dataclasses
# ---------------------------------------------------------------------------

def test_validate_immutability_rejects_mutable():
    from dataclasses import dataclass as dc
    from framework.types import validate_immutability

    @dc
    class MutableState:
        x: int = 0

    with pytest.raises(TypeError, match="must be frozen"):
        validate_immutability(MutableState)


# ---------------------------------------------------------------------------
# Property: convergence_threshold triggers early stop
# ---------------------------------------------------------------------------

@given(
    init_value=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20)
def test_convergence_threshold_early_stop(init_value):
    cfg = SimulationConfig(
        horizon=200, max_rounds=200, convergence_threshold=0.001,
        disturbance_prob=1.0, disturbance_scale=5.0,
    )
    init = ForecastState(t=0, value=init_value, exogenous=0.0, hidden_shift=0.0)
    out = ForecastGame(cfg, seed=42).run(init, disturbed=True)
    if out.convergence.get("reason") == "divergence_threshold_exceeded":
        assert out.convergence["rounds_executed"] < 200


# ---------------------------------------------------------------------------
# Property: Kelly-Criterion BMA weights always sum to 1
# ---------------------------------------------------------------------------

@given(
    errors=st.lists(small_floats, min_size=2, max_size=6),
)
@settings(max_examples=50)
def test_kelly_bma_weights_sum_invariant(errors):
    agg = BayesianAggregator()
    names = [f"agent_{i}" for i in range(len(errors))]
    agg._ensure_init(names)
    agg.update({n: e for n, e in zip(names, errors)})
    assert sum(agg.weights) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Property: Wolfpack coalition never exceeds total number of forecasters
# ---------------------------------------------------------------------------

@given(
    n_agents=st.integers(min_value=2, max_value=10),
    threshold=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_wolfpack_target_set_bounded(n_agents, threshold):
    import random as stdlib_random
    wolf = WolfpackAdversary(correlation_threshold=threshold)
    rng = stdlib_random.Random(42)
    names = [f"agent_{i}" for i in range(n_agents)]
    for _ in range(30):
        for name in names:
            wolf.record_residual(name, rng.gauss(0, 1))
    primary = names[0]
    _, coalition = wolf.select_targets(primary)
    assert len(coalition) + 1 <= n_agents


# ---------------------------------------------------------------------------
# T22: Data ingestion hypothesis tests (max_examples=200)
# ---------------------------------------------------------------------------

@given(
    periods=st.integers(min_value=10, max_value=500),
    train_ratio=st.floats(min_value=0.3, max_value=0.7),
    valid_ratio=st.floats(min_value=0.05, max_value=0.2),
    normalize=st.booleans(),
)
@settings(max_examples=200)
def test_data_ingestion_hypothesis(periods, train_ratio, valid_ratio, normalize):
    import tempfile
    from pathlib import Path
    from framework.data import DataProfile, load_dataset, build_sample_dataset
    assume(train_ratio + valid_ratio < 0.95)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "sample.csv"
        build_sample_dataset(p, periods=periods)
        profile = DataProfile(
            source="sample_csv",
            periods=periods,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            normalize=normalize,
        )
        bundle = load_dataset(profile, path=p)
        total = len(bundle.train) + len(bundle.valid) + len(bundle.test)
        assert total > 0
        assert len(bundle.train) > 0


# ---------------------------------------------------------------------------
# T24: KS distribution tests for disturbances
# ---------------------------------------------------------------------------

def test_gaussian_disturbance_ks():
    """KS test: GaussianDisturbance samples follow a normal distribution."""
    from random import Random
    from framework.disturbances import GaussianDisturbance
    from framework.types import ForecastState, SimulationConfig
    try:
        from scipy.stats import kstest, norm
    except ImportError:
        pytest.skip("scipy not installed")

    model = GaussianDisturbance()
    rng = Random(42)
    cfg = SimulationConfig(disturbance_prob=1.0, disturbance_scale=1.0, adversarial_intensity=1.0)
    s = ForecastState(t=0, value=0.0, exogenous=0.0, hidden_shift=0.0)
    samples = [model.sample(s, rng, cfg) for _ in range(1000)]
    nonzero = [x for x in samples if x != 0.0]
    if len(nonzero) >= 100:
        stat, pval = kstest(nonzero, "norm", args=(0, cfg.disturbance_scale * cfg.adversarial_intensity))
        assert pval > 0.01, f"KS test failed: stat={stat}, p={pval}"


def test_hmm_disturbance_finite_samples():
    """HMM disturbance produces finite, non-NaN samples."""
    import math
    from random import Random
    from framework.disturbances import HMMRegimeShiftDisturbance
    from framework.types import ForecastState, SimulationConfig

    model = HMMRegimeShiftDisturbance()
    rng = Random(42)
    cfg = SimulationConfig(disturbance_prob=1.0)
    s = ForecastState(t=0, value=0.0, exogenous=0.0, hidden_shift=0.0)
    samples = [model.sample(s, rng, cfg) for _ in range(1000)]
    assert all(math.isfinite(x) for x in samples)
    mean_val = sum(samples) / len(samples)
    std_val = (sum((x - mean_val) ** 2 for x in samples) / len(samples)) ** 0.5
    assert abs(mean_val) < 2 * std_val + 1.0


def test_garch_disturbance_finite_samples():
    """GARCH disturbance produces finite, non-NaN samples."""
    import math
    from random import Random
    from framework.disturbances import GARCHDisturbance
    from framework.types import ForecastState, SimulationConfig

    model = GARCHDisturbance()
    rng = Random(42)
    cfg = SimulationConfig(disturbance_prob=1.0)
    s = ForecastState(t=0, value=0.0, exogenous=0.0, hidden_shift=0.0)
    samples = [model.sample(s, rng, cfg) for _ in range(1000)]
    assert all(math.isfinite(x) for x in samples)
    mean_val = sum(samples) / len(samples)
    std_val = (sum((x - mean_val) ** 2 for x in samples) / len(samples)) ** 0.5
    assert abs(mean_val) < 2 * std_val + 1.0


# ---------------------------------------------------------------------------
# T26: CI invariance -- epsilon_convergence is finite
# ---------------------------------------------------------------------------

@given(
    convergence_threshold=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=1, max_value=5000),
)
@settings(max_examples=100)
def test_epsilon_convergence_finite(convergence_threshold, seed):
    import math
    cfg = SimulationConfig(
        horizon=30, max_rounds=30, convergence_threshold=convergence_threshold,
    )
    init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
    out = ForecastGame(cfg, seed=seed).run(init, disturbed=True)
    ec = out.convergence.get("epsilon_convergence")
    assert ec is not None
    assert math.isfinite(ec)
    assert ec >= 0


# ---------------------------------------------------------------------------
# T42: Poisoning rejection under varied attack
# ---------------------------------------------------------------------------

@given(
    n_outliers=st.integers(min_value=0, max_value=10),
    base_value=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    z_threshold=st.floats(min_value=2.0, max_value=6.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_poisoning_rejection_varied_attack(n_outliers, base_value, z_threshold):
    base_dt = datetime(2024, 1, 1)
    rows = [
        {"timestamp": base_dt + timedelta(days=i), "series_id": "s",
         "target": base_value + i * 0.01, "promo": 0.0, "macro_index": 100.0}
        for i in range(50)
    ]
    for i in range(n_outliers):
        rows[i] = {**rows[i], "target": base_value * 1000}
    detected = detect_poisoning_rows(rows, z_threshold=z_threshold)
    if n_outliers > 0 and base_value * 1000 > base_value + 50 * 0.01:
        assert len(detected) >= min(n_outliers, 1)
