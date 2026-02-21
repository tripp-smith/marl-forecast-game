import pytest
import random

from framework.data import (
    DataProfile,
    build_sample_dataset,
    chronological_split,
    load_csv,
    load_dataset,
    load_source_rows,
    normalize_features,
)
from framework.game import ForecastGame
from framework.strategy_runtime import HaskellRLMRuntime, PythonStrategyRuntime, runtime_from_name
from framework.types import ForecastState, SimulationConfig, evolve_state


def test_data_split_and_order(tmp_path):
    p = tmp_path / "sample.csv"
    build_sample_dataset(p, periods=60)
    rows = normalize_features(load_csv(p))
    bundle = chronological_split(rows)
    assert len(bundle.train) > len(bundle.valid) > 0
    assert bundle.train[-1]["timestamp"] <= bundle.valid[0]["timestamp"]


def test_transition_is_pure_property():
    rng = random.Random(11)
    for _ in range(200):
        value = rng.uniform(-100, 100)
        exogenous = rng.uniform(-5, 5)
        noise = rng.uniform(-2, 2)
        disturbance = rng.uniform(-2, 2)
        s0 = ForecastState(t=0, value=value, exogenous=exogenous, hidden_shift=0.0)
        s1 = evolve_state(s0, base_trend=0.4, noise=noise, disturbance=disturbance)
        s2 = evolve_state(s0, base_trend=0.4, noise=noise, disturbance=disturbance)
        assert s1 == s2
        assert s0.t == 0


def test_game_round_cap():
    cfg = SimulationConfig(horizon=400, max_rounds=200)
    game = ForecastGame(cfg, seed=9)
    out = game.run(ForecastState(t=0, value=50.0, exogenous=0.0, hidden_shift=0.0), disturbed=True)
    assert len(out.steps) == 200
    assert out.convergence["round_cap_hit"] is True


def test_zero_rounds_and_negative_rounds_do_not_crash():
    cfg = SimulationConfig(horizon=10, max_rounds=20)
    game = ForecastGame(cfg, seed=2)
    s0 = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
    assert len(game.run(s0, rounds=0).steps) == 0
    assert len(game.run(s0, rounds=-5).steps) == 0


def test_runtime_backend_selection_and_fallback():
    assert isinstance(runtime_from_name("python"), PythonStrategyRuntime)
    assert isinstance(runtime_from_name("haskell"), HaskellRLMRuntime)
    assert isinstance(runtime_from_name("unknown"), PythonStrategyRuntime)


def test_confidence_interval_invariant_property():
    rng = random.Random(21)
    cfg = SimulationConfig(horizon=5, max_rounds=10, disturbance_model="volatility")
    game = ForecastGame(cfg, seed=3)
    for _ in range(100):
        state_val = rng.uniform(-20, 20)
        exogenous = rng.uniform(-3, 3)
        out = game.run(ForecastState(t=0, value=state_val, exogenous=exogenous, hidden_shift=0.0), disturbed=True)
        for step in out.steps:
            assert step.confidence.lower <= step.forecast <= step.confidence.upper


def test_source_adapter_schema():
    rows = load_source_rows("polymarket", periods=12)
    assert len(rows) == 12
    first = rows[0]
    assert {"timestamp", "series_id", "target", "promo", "macro_index", "source", "fetched_at"}.issubset(first.keys())


def test_load_dataset_real_profile_works():
    bundle = load_dataset(DataProfile(source="fred", periods=45, normalize=True))
    assert len(bundle.train) > 0
    assert len(bundle.valid) > 0


def test_immutability_of_reward_breakdown_mapping():
    cfg = SimulationConfig(horizon=3, max_rounds=5)
    out = ForecastGame(cfg, seed=5).run(ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0))
    with pytest.raises(TypeError):
        out.steps[0].reward_breakdown["forecaster"] = 0.0


def test_validation_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(
        "timestamp,series_id,target,promo\n"
        "2024-01-03T00:00:00,s1,1.0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_csv(p)


def test_llm_refactor_stub_path_executes():
    cfg = SimulationConfig(horizon=6, max_rounds=10, enable_refactor=True, enable_llm_refactor=True)
    out = ForecastGame(cfg, seed=7).run(ForecastState(t=0, value=8.0, exogenous=0.1, hidden_shift=0.0))
    assert len(out.steps) == 6
