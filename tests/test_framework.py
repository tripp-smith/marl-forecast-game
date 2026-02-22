from __future__ import annotations

import json
import random

import pytest

from framework.agents import AdversaryAgent
from framework.data import (
    DataProfile,
    build_hybrid_rows,
    build_sample_dataset,
    chronological_split,
    detect_poisoning_rows,
    load_csv,
    load_dataset,
    load_json,
    load_source_rows,
    normalize_features,
    should_reject_poisoning,
)
from framework.defenses import StackedDefense, defense_from_name
from framework.data_utils import cache_status, ensure_source_data
from framework.llm.ollama_interface import OllamaInterface

from framework.disturbances import disturbance_from_name
from framework.game import ForecastGame
from framework.llm import DSPyLikeRepl
from framework.llm.ollama import OllamaClient
from framework.strategy_runtime import (
    DeterministicPromptClient,
    HaskellRLMRuntime,
    PromptStrategyRuntime,
    PythonStrategyRuntime,
    runtime_from_name,
)
from framework.types import ForecastState, SimulationConfig, evolve_state


def test_data_split_and_order(tmp_path):
    p = tmp_path / "sample.csv"
    build_sample_dataset(p, periods=60)
    rows = normalize_features(load_csv(p))
    bundle = chronological_split(rows)
    assert len(bundle.train) > len(bundle.valid) > 0
    assert bundle.train[-1]["timestamp"] <= bundle.valid[0]["timestamp"]


def test_json_loader(tmp_path):
    p = tmp_path / "sample.json"
    p.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "series_id": "s1",
                    "target": 1.0,
                    "promo": 0,
                    "macro_index": 100.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    rows = load_json(p)
    assert len(rows) == 1


def test_transition_is_pure_property():
    rng = random.Random(11)
    for _ in range(500):
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
    for _ in range(200):
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


def test_load_dataset_real_profile_works_for_fred_and_imf_and_hybrid():
    fred_bundle = load_dataset(DataProfile(source="fred", periods=45, normalize=True))
    imf_bundle = load_dataset(DataProfile(source="imf", periods=45, normalize=True))
    hybrid_bundle = load_dataset(DataProfile(source="hybrid", periods=45, normalize=True, hybrid_weight=0.4))
    assert len(fred_bundle.train) > 0
    assert len(imf_bundle.valid) > 0
    assert len(hybrid_bundle.test) > 0


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


def test_simulation_config_validation_raises_for_invalid_prob():
    with pytest.raises(ValueError):
        SimulationConfig(disturbance_prob=1.5)


def test_data_profile_validation_raises_for_invalid_ratios():
    with pytest.raises(ValueError):
        DataProfile(train_ratio=0.8, valid_ratio=0.25)


def test_chronological_split_validation_raises_for_invalid_splits():
    with pytest.raises(ValueError):
        chronological_split([{"x": 1}], train=0.0, valid=0.2)


def test_new_disturbance_models_are_reachable():
    s = ForecastState(t=12, value=10.0, exogenous=0.5, hidden_shift=0.1)
    cfg = SimulationConfig(disturbance_prob=1.0, disturbance_scale=1.0)
    rng = random.Random(7)
    for name in ["regime_shift", "volatility_burst", "drift"]:
        model = disturbance_from_name(name)
        val = model.sample(s, rng, cfg)
        assert isinstance(val, float)


def test_stacked_defense_parser_works():
    model = defense_from_name("stack:clipping,dampening")
    assert isinstance(model, StackedDefense)
    correction = model.defend(0.2, -0.5)
    assert isinstance(correction, float)


def test_prompt_runtime_supports_mocked_llm_response():
    runtime = PromptStrategyRuntime(client=DeterministicPromptClient("0.123"))
    state = ForecastState(t=0, value=1.0, exogenous=0.0, hidden_shift=0.0)
    assert runtime.forecast_delta(state) == pytest.approx(0.123)


def test_adversary_cost_reduces_attack_magnitude():
    state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
    aggressive = AdversaryAgent(aggressiveness=1.0, attack_cost=0.0).act(state)
    costly = AdversaryAgent(aggressiveness=1.0, attack_cost=2.0).act(state)
    assert abs(costly.delta) < abs(aggressive.delta)


def test_game_outputs_include_structured_trajectory_logs():
    cfg = SimulationConfig(horizon=4, max_rounds=10)
    out = ForecastGame(cfg, seed=1).run(ForecastState(t=0, value=2.0, exogenous=0.0, hidden_shift=0.0))
    assert len(out.trajectory_logs) == 4
    assert {"round_idx", "state", "actions", "forecast", "target", "reward", "disturbance", "messages"}.issubset(
        out.trajectory_logs[0].keys()
    )


def test_poisoning_detector_flags_outlier():
    from datetime import datetime, timedelta

    base = datetime(2024, 1, 1)
    rows = [
        {"timestamp": base + timedelta(days=i), "series_id": "s", "target": float(x), "promo": 0.0, "macro_index": 1.0}
        for i, x in enumerate([1, 1, 1, 1, 1000])
    ]
    assert detect_poisoning_rows(rows)


def test_hybrid_row_builder_preserves_chronological_order(tmp_path):
    p = tmp_path / "sample.csv"
    build_sample_dataset(p, periods=20)
    synthetic = load_csv(p)
    real = load_source_rows("fred", periods=20)
    mixed = build_hybrid_rows(real, synthetic, real_weight=0.5)
    assert mixed[0]["timestamp"] <= mixed[-1]["timestamp"]


def test_llm_repl_stub_can_be_constructed_without_calling_network():
    repl = DSPyLikeRepl(client=OllamaClient())
    assert repl.client.base_url.startswith("http")


def test_should_reject_poisoning_requires_multiple_suspects():
    assert should_reject_poisoning(100, 1) is False
    assert should_reject_poisoning(100, 2) is True


def test_load_dataset_can_enforce_poisoning_on_strict_mode(tmp_path):
    p = tmp_path / "poison.csv"
    p.write_text(
        "timestamp,series_id,target,promo,macro_index\n"
        "2024-01-01T00:00:00,s1,1,0,1\n"
        "2024-01-02T00:00:00,s1,1,0,1\n"
        "2024-01-03T00:00:00,s1,1,0,1\n"
        "2024-01-04T00:00:00,s1,1000,0,1\n"
        "2024-01-05T00:00:00,s1,2000,0,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_dataset(DataProfile(source="csv", periods=5, normalize=False, fail_on_poisoning=True), path=p)


def test_data_utils_cache_and_force_redownload(tmp_path):
    rows_a, meta_a = ensure_source_data("fred", periods=5, cache_dir=tmp_path, force_redownload=True)
    n = len(rows_a)
    assert n >= 3
    assert meta_a["cache_hit"] is False
    rows_b, meta_b = ensure_source_data("fred", periods=n, cache_dir=tmp_path, force_redownload=False)
    assert len(rows_b) == n
    assert meta_b["cache_hit"] is True


def test_cache_status_detects_missing_file(tmp_path):
    status = cache_status(tmp_path / "missing.json", max_age_hours=1)
    assert status.exists is False
    assert status.is_fresh is False


def test_cache_freshness_property_window():
    for hours in [1, 2, 6, 12, 24, 48, 96]:
        status = cache_status("data/sample_demand.csv", max_age_hours=hours + 1_000)
        assert status.exists is True
        assert status.is_fresh is True


def test_ollama_interface_availability_handles_unreachable_host():
    client = OllamaInterface(base_url="http://127.0.0.1:9")
    assert client.is_available() is False
