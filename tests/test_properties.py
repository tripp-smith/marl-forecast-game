"""Property-based tests using Hypothesis for MARL Forecast Game invariants."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

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
