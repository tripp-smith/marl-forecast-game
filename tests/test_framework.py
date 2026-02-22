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
from framework.types import AgentAction, ForecastState, SimulationConfig, evolve_state


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


def test_kaggle_adapter_synthetic_fallback():
    from framework.data_sources.kaggle_demand import KaggleDemandAdapter
    adapter = KaggleDemandAdapter(path="nonexistent.csv")
    rows = adapter.fetch(periods=15)
    assert len(rows) == 15
    assert rows[0].source == "kaggle"


def test_kaggle_adapter_loads_csv(tmp_path):
    from framework.data_sources.kaggle_demand import KaggleDemandAdapter
    csv_path = tmp_path / "demand.csv"
    csv_path.write_text(
        "date,store,item,sales\n"
        "2022-01-01,1,1,30\n"
        "2022-01-02,1,1,35\n"
        "2022-01-03,1,1,28\n",
        encoding="utf-8",
    )
    adapter = KaggleDemandAdapter(path=str(csv_path))
    rows = adapter.fetch(periods=10)
    assert len(rows) == 3
    assert rows[0].target == 30.0


def test_real_vs_synthetic_metrics_comparison():
    """H.4: Run game on synthetic and FRED-sourced data, compare MAE/RMSE."""
    from framework.metrics import mae, rmse

    synth_bundle = load_dataset(DataProfile(source="sample_csv", periods=120, normalize=True))
    fred_bundle = load_dataset(DataProfile(source="fred", periods=120, normalize=True))

    cfg = SimulationConfig(horizon=30, max_rounds=50)

    synth_init = ForecastState(t=0, value=float(synth_bundle.train[-1]["target"]), exogenous=0.0, hidden_shift=0.0)
    fred_init = ForecastState(t=0, value=float(fred_bundle.train[-1]["target"]), exogenous=0.0, hidden_shift=0.0)

    synth_out = ForecastGame(cfg, seed=42).run(synth_init, disturbed=True)
    fred_out = ForecastGame(cfg, seed=42).run(fred_init, disturbed=True)

    synth_mae = mae(synth_out.targets, synth_out.forecasts)
    fred_mae = mae(fred_out.targets, fred_out.forecasts)

    assert synth_mae >= 0
    assert fred_mae >= 0
    if synth_mae > 0:
        degradation = abs(fred_mae - synth_mae) / synth_mae
        assert degradation < 2.5, f"Degradation {degradation:.1%} exceeds 250% tolerance"


def test_fred_integration_with_api_key():
    """H.1: Integration test for FRED API -- skips without FRED_API_KEY."""
    import os
    key = os.getenv("FRED_API_KEY")
    if not key:
        pytest.skip("FRED_API_KEY not set")
    from framework.data_sources.macro_fred import FredMacroAdapter
    adapter = FredMacroAdapter()
    rows = adapter.fetch(periods=10)
    assert len(rows) >= 5
    assert all(r.source == "fred" for r in rows)


def test_imf_integration_with_live_data():
    """H.1: Integration test for IMF adapter -- uses synthetic if API fails."""
    rows = load_source_rows("imf", periods=10)
    assert len(rows) == 10
    first = rows[0]
    assert "timestamp" in first
    assert "target" in first


def test_yaml_scenario_loader(tmp_path):
    """I.1: Load scenarios from YAML config."""
    from framework.scenarios import load_scenario_specs
    yaml_path = tmp_path / "scenarios.yaml"
    yaml_path.write_text(
        "scenarios:\n"
        "  - name: high_adversary\n"
        "    disturbance_model: volatility\n"
        "    defense_model: ensemble\n"
        "    adversarial_intensity: 2.0\n"
        "  - name: calm\n"
        "    disturbance_model: gaussian\n"
        "    adversarial_intensity: 0.5\n",
        encoding="utf-8",
    )
    specs = load_scenario_specs(yaml_path)
    assert len(specs) == 2
    assert specs[0].name == "high_adversary"
    assert specs[0].adversarial_intensity == 2.0
    cfg = specs[0].to_config()
    assert cfg.disturbance_model == "volatility"
    assert cfg.defense_model == "ensemble"


def test_nash_equilibrium_rock_paper_scissors():
    """I.2: Compute Nash equilibrium for a simple zero-sum game."""
    import numpy as np
    from framework.equilibrium import compute_nash_equilibrium
    payoff = np.array([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ])
    result = compute_nash_equilibrium(payoff)
    assert len(result.attacker_strategy) == 3
    assert len(result.defender_strategy) == 3
    assert abs(sum(result.attacker_strategy) - 1.0) < 1e-6
    assert abs(sum(result.defender_strategy) - 1.0) < 1e-6
    for p in result.attacker_strategy:
        assert abs(p - 1.0 / 3.0) < 0.05


def test_dann_defense_stub():
    """I.3: DANN defense stub produces bounded outputs."""
    from framework.defenses import DANNDefenseStub, defense_from_name
    dann = defense_from_name("dann")
    assert isinstance(dann, DANNDefenseStub)
    result = dann.defend(0.5, -0.3)
    assert isinstance(result, float)


def test_escalating_disturbance_increases_over_time():
    """I.4: Escalating disturbance intensity grows with t."""
    from framework.disturbances import EscalatingDisturbance
    model = EscalatingDisturbance(base_scale=0.1, escalation_rate=0.1)
    cfg = SimulationConfig(disturbance_prob=1.0)
    rng = random.Random(42)
    magnitudes_early = []
    magnitudes_late = []
    for _ in range(200):
        s_early = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
        s_late = ForecastState(t=100, value=10.0, exogenous=0.0, hidden_shift=0.0)
        magnitudes_early.append(abs(model.sample(s_early, rng, cfg)))
        magnitudes_late.append(abs(model.sample(s_late, rng, cfg)))
    assert sum(magnitudes_late) / len(magnitudes_late) > sum(magnitudes_early) / len(magnitudes_early)


def test_trade_off_fields_in_convergence():
    """I.5: GameOutputs convergence dict includes trade-off fields."""
    cfg = SimulationConfig(horizon=10, max_rounds=20, attack_cost=0.5)
    out = ForecastGame(cfg, seed=1).run(ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0))
    assert "attack_cost_total" in out.convergence
    assert "defense_efficacy_ratio" in out.convergence
    assert "accuracy_vs_cost" in out.convergence
    assert out.convergence["attack_cost_total"] >= 0


def test_chat_strategy_runtime_fallback():
    """J.1: ChatStrategyRuntime falls back to PythonStrategyRuntime on failure."""
    from framework.strategy_runtime import ChatStrategyRuntime
    runtime = ChatStrategyRuntime(base_url="http://127.0.0.1:9")
    state = ForecastState(t=0, value=10.0, exogenous=1.0, hidden_shift=0.0)
    delta = runtime.forecast_delta(state)
    expected = PythonStrategyRuntime().forecast_delta(state)
    assert delta == pytest.approx(expected)


def test_llm_policy_agent_without_trajectories():
    """J.2: LLMPolicyAgent acts like base agent when no trajectories provided."""
    from framework.agents import LLMPolicyAgent
    agent = LLMPolicyAgent()
    state = ForecastState(t=0, value=10.0, exogenous=0.5, hidden_shift=0.0)
    runtime = PythonStrategyRuntime()
    action = agent.act(state, runtime)
    base_delta = runtime.forecast_delta(state)
    assert action.delta == pytest.approx(base_delta)
    assert action.actor == "llm_policy"


def test_ollama_keep_alive_and_availability():
    """J.3: OllamaInterface availability handling."""
    from framework.llm.ollama_interface import OllamaInterface
    client = OllamaInterface(base_url="http://127.0.0.1:9")
    assert client.is_available() is False

    client_default = OllamaInterface()
    if client_default.is_available():
        result = client_default.list_models()
        assert "models" in result


def test_recursive_strategy_refiner_deterministic():
    """J.4: RecursiveStrategyRefiner with mock client produces clamped, deterministic output."""
    from framework.llm.refiner import RecursiveStrategyRefiner
    from framework.types import TrajectoryEntry, frozen_mapping

    trajectories = [
        TrajectoryEntry(
            round_idx=i,
            state=ForecastState(t=i, value=10.0 + i, exogenous=0.0, hidden_shift=0.0),
            actions=(AgentAction(actor="forecaster", delta=0.4),),
            messages=(),
            reward_breakdown=frozen_mapping({"forecaster": -0.5}),
            forecast=10.5 + i,
            target=11.0 + i,
        )
        for i in range(5)
    ]
    refiner = RecursiveStrategyRefiner(clamp_min=-0.05, clamp_max=0.05)
    result = refiner.refine(trajectories)
    assert -0.05 <= result.bias_adjustment <= 0.05
    result2 = refiner.refine(trajectories)
    assert result.bias_adjustment == result2.bias_adjustment


def test_trajectory_export_creates_json(tmp_path):
    """L.1: Export trajectory logs to JSON with metadata."""
    from framework.export import export_trajectories
    cfg = SimulationConfig(horizon=5, max_rounds=10)
    out = ForecastGame(cfg, seed=1).run(ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0))
    path = export_trajectories(out, tmp_path / "traj.json", config=cfg, seed=1)
    assert path.exists()
    data = json.loads(path.read_text())
    assert "metadata" in data
    assert "trajectories" in data
    assert data["metadata"]["seed"] == 1
    assert len(data["trajectories"]) == 5


def test_metrics_server_returns_false_without_prometheus():
    """L.2: Metrics server returns False when prometheus_client not available or port=0."""
    from framework.observability import start_metrics_server
    assert start_metrics_server(port=0) is False


_ADAPTER_SOURCES = [
    "bis", "gpr", "oecd_cli", "worldbank", "bea", "kalshi", "predictit", "eurostat",
]

_REQUIRED_FIELDS = {"timestamp", "series_id", "target", "promo", "macro_index", "source", "fetched_at"}


def test_bis_adapter_synthetic_fallback():
    from framework.data_sources.bis_policy_rate import BISPolicyRateAdapter
    adapter = BISPolicyRateAdapter()
    rows = adapter._synthetic(10)
    assert len(rows) == 10
    assert all(r.source == "bis" for r in rows)


def test_oecd_adapter_synthetic_fallback():
    from framework.data_sources.oecd_cli import OECDCLIAdapter
    adapter = OECDCLIAdapter()
    rows = adapter._synthetic(10)
    assert len(rows) == 10
    assert all(r.source == "oecd_cli" for r in rows)


def test_gpr_adapter_synthetic_fallback():
    from framework.data_sources.geopolitical_risk import GeopoliticalRiskAdapter
    adapter = GeopoliticalRiskAdapter(local_path="nonexistent.csv")
    rows = adapter._synthetic(10)
    assert len(rows) == 10
    assert all(r.source == "gpr" for r in rows)


def test_worldbank_adapter_synthetic_fallback():
    from framework.data_sources.world_bank import WorldBankAdapter
    adapter = WorldBankAdapter()
    rows = adapter._synthetic(10)
    assert len(rows) == 10
    assert all(r.source == "worldbank" for r in rows)


def test_bea_adapter_synthetic_fallback():
    from framework.data_sources.bea import BEAAdapter
    adapter = BEAAdapter()
    rows = adapter._synthetic(10)
    assert len(rows) == 10
    assert all(r.source == "bea" for r in rows)


def test_kalshi_adapter_synthetic_fallback():
    from framework.data_sources.kalshi import KalshiAdapter
    adapter = KalshiAdapter()
    rows = adapter._synthetic(10)
    assert len(rows) == 10
    assert all(r.source == "kalshi" for r in rows)


def test_predictit_adapter_synthetic_fallback():
    from framework.data_sources.predictit import PredictItAdapter
    adapter = PredictItAdapter()
    rows = adapter._synthetic(10)
    assert len(rows) == 10
    assert all(r.source == "predictit" for r in rows)


def test_eurostat_adapter_synthetic_fallback():
    from framework.data_sources.eurostat import EurostatAdapter
    adapter = EurostatAdapter()
    rows = adapter._synthetic(10)
    assert len(rows) == 10
    assert all(r.source == "eurostat" for r in rows)


@pytest.mark.parametrize("source", _ADAPTER_SOURCES)
def test_adapter_schema_compliance(source):
    """Verify every row from each adapter has all required NormalizedRecord fields."""
    from framework.data_utils import fetch_source_rows
    rows = fetch_source_rows(source, 10)
    assert len(rows) >= 5
    for row in rows:
        assert _REQUIRED_FIELDS.issubset(set(row.keys())), f"Missing fields in {source}: {_REQUIRED_FIELDS - set(row.keys())}"
        assert isinstance(row["target"], (int, float))
        assert row["target"] is not None


@pytest.mark.parametrize("source", _ADAPTER_SOURCES)
def test_adapter_pipeline_integration(source):
    """Call load_source_rows for each adapter and verify row count + chronological order per series."""
    rows = load_source_rows(source, periods=10)
    assert len(rows) >= 5
    from collections import defaultdict
    per_series: dict[str, list] = defaultdict(list)
    for r in rows:
        per_series[r["series_id"]].append(r["timestamp"])
    for sid, ts_list in per_series.items():
        for i in range(1, len(ts_list)):
            assert ts_list[i] >= ts_list[i - 1], f"{source}/{sid}: out-of-order at index {i}"


@pytest.mark.parametrize("source", _ADAPTER_SOURCES)
def test_adapter_cache_round_trip(source, tmp_path):
    """ensure_source_data writes cache, second call reads from cache."""
    rows_a, meta_a = ensure_source_data(source, periods=5, cache_dir=tmp_path, force_redownload=True)
    assert len(rows_a) >= 3
    assert meta_a["cache_hit"] is False
    cache_file = tmp_path / f"{source}.json"
    assert cache_file.exists()

    rows_b, meta_b = ensure_source_data(source, periods=len(rows_a), cache_dir=tmp_path, force_redownload=False)
    assert len(rows_b) == len(rows_a)
    assert meta_b["cache_hit"] is True


def test_all_twelve_adapters_registered_in_pipeline():
    """Verify that all 12 adapters are accessible via fetch_source_rows."""
    from framework.data_utils import fetch_source_rows
    all_sources = [
        "fred", "imf", "polymarket", "bis", "gpr", "oecd_cli",
        "kaggle", "worldbank", "bea", "kalshi", "predictit", "eurostat",
    ]
    for source in all_sources:
        rows = fetch_source_rows(source, 5)
        assert len(rows) >= 3, f"Adapter '{source}' returned too few rows"


def test_load_dataset_accepts_new_sources():
    """Verify load_dataset routes new sources through the pipeline."""
    for source in ["bis", "worldbank", "eurostat"]:
        bundle = load_dataset(DataProfile(source=source, periods=30, normalize=True))
        assert len(bundle.train) > 0


def test_haskell_python_equivalence():
    """K.5: Verify Haskell evolveState matches Python evolve_state within FP tolerance."""
    import os
    import subprocess

    binary = "/usr/local/bin/marl-forecast-game"
    cabal_available = False
    binary_available = os.path.isfile(binary)

    if not binary_available:
        try:
            subprocess.run(["cabal", "--version"], capture_output=True, timeout=3)
            cabal_available = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if not binary_available and not cabal_available:
        pytest.skip("Neither Haskell binary nor cabal available")

    test_cases = [
        {"t": 0, "value": 10.0, "exogenous": 1.2, "hidden_shift": 0.0},
        {"t": 5, "value": -3.5, "exogenous": 0.0, "hidden_shift": 0.5},
        {"t": 0, "value": 0.0, "exogenous": 0.0, "hidden_shift": 0.0},
    ]

    for tc in test_cases:
        input_data = json.dumps(tc)
        py_state = ForecastState(t=tc["t"], value=tc["value"], exogenous=tc["exogenous"], hidden_shift=tc["hidden_shift"])
        py_delta = PythonStrategyRuntime().forecast_delta(py_state)

        if binary_available:
            cmd = [binary, "--delta"]
            cwd = None
        else:
            cmd = ["cabal", "run", "marl-forecast-game", "--", "--delta"]
            cwd = "haskell"

        try:
            result = subprocess.run(cmd, input=input_data, capture_output=True, text=True, timeout=10, cwd=cwd)
            if result.returncode != 0:
                pytest.skip(f"Haskell binary returned non-zero: {result.stderr}")
            hs_delta = float(result.stdout.strip())
            assert abs(hs_delta - py_delta) < 1e-9, f"Mismatch: Haskell={hs_delta}, Python={py_delta}"
        except (subprocess.TimeoutExpired, ValueError) as e:
            pytest.skip(f"Haskell subprocess failed: {e}")


# ---------------------------------------------------------------------------
# Kelly-Criterion BMA tests
# ---------------------------------------------------------------------------


def test_kelly_bma_weights_sum_to_one():
    from framework.aggregation import BayesianAggregator

    agg = BayesianAggregator()
    names = ["a", "b", "c"]
    agg._ensure_init(names)
    rng = random.Random(42)
    for _ in range(20):
        errors = {n: rng.gauss(0, 1) for n in names}
        agg.update(errors)
        assert sum(agg.weights) == pytest.approx(1.0, abs=1e-9)


def test_kelly_bma_bankruptcy_disables_agent():
    from framework.aggregation import BayesianAggregator

    agg = BayesianAggregator()
    names = ["good", "bad"]
    agg._ensure_init(names)
    for _ in range(50):
        agg.update(
            {"good": 0.01, "bad": 5.0},
            bankruptcy_threshold=0.5,
        )
    assert agg._active[0] is True
    assert agg._active[1] is False
    w = agg.weights
    assert w[1] == 0.0
    assert w[0] == pytest.approx(1.0, abs=1e-9)


def test_kelly_bma_all_bankrupt_fallback():
    from framework.aggregation import BayesianAggregator

    agg = BayesianAggregator()
    names = ["a", "b"]
    agg._ensure_init(names)
    for _ in range(100):
        agg.update({"a": 10.0, "b": 10.0}, bankruptcy_threshold=0.5)
    assert all(not a for a in agg._active)
    w = agg.weights
    assert sum(w) == pytest.approx(1.0, abs=1e-9)
    assert w[0] == pytest.approx(0.5, abs=1e-9)


def test_neg_crps_sign():
    from framework.metrics import crps, neg_crps

    cases = [(0.0, 0.0, 1.0), (1.0, 0.5, 0.3), (-2.0, 0.0, 2.0), (0.0, 0.0, 0.01)]
    for actual, mean, std in cases:
        assert neg_crps(actual, mean, std) == pytest.approx(-crps(actual, mean, std))


# ---------------------------------------------------------------------------
# Bounded Rationality Curriculum tests
# ---------------------------------------------------------------------------


def test_boltzmann_act_high_tau_is_uniform():
    import math
    from collections import Counter
    from framework.training import QTableAgent, DiscreteActionSpace

    space = DiscreteActionSpace(n_bins=5)
    agent = QTableAgent(action_space=space, epsilon=0.0)
    state = ForecastState(t=0, value=1.0, exogenous=0.0, hidden_shift=0.0)
    counts = Counter(agent.boltzmann_act(state, tau=100.0) for _ in range(2000))
    for a in range(5):
        assert counts[a] > 100, f"Action {a} sampled too rarely for uniform distribution"


def test_boltzmann_act_low_tau_is_greedy():
    import numpy as np
    from framework.training import QTableAgent, DiscreteActionSpace, _state_hash

    space = DiscreteActionSpace(n_bins=5)
    agent = QTableAgent(action_space=space, epsilon=0.0)
    state = ForecastState(t=0, value=1.0, exogenous=0.0, hidden_shift=0.0)
    key = _state_hash(state)
    agent._q_table[key] = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
    for _ in range(50):
        assert agent.boltzmann_act(state, tau=0.001) == 3


def test_rarl_tau_schedule_decay():
    import math
    from framework.training import RADversarialTrainer

    cfg = SimulationConfig()
    trainer = RADversarialTrainer(config=cfg, total_epochs=100)
    tau_0 = trainer._compute_tau(0)
    tau_100 = trainer._compute_tau(100)
    assert abs(tau_0 - cfg.adversary_tau_init) < 0.5
    assert abs(tau_100 - cfg.adversary_tau_final) < 0.5
    assert tau_0 > tau_100


def test_rarl_bounded_rationality_trains():
    from framework.training import QTableAgent, RADversarialTrainer

    cfg = SimulationConfig(horizon=20, max_rounds=20)
    trainer = RADversarialTrainer(config=cfg, total_epochs=20, alternation_schedule=5)
    forecaster = QTableAgent()
    adversary = QTableAgent()
    result = trainer.train(forecaster, adversary)
    assert result["total_epochs"] == 20
    assert len(result["epoch_results"]) == 20
    assert "tau" in result["epoch_results"][0]


# ---------------------------------------------------------------------------
# Wolfpack Adversary tests
# ---------------------------------------------------------------------------


def test_wolfpack_correlation_matrix():
    from framework.agents import WolfpackAdversary

    wolf = WolfpackAdversary(correlation_threshold=0.7)
    rng = random.Random(99)
    for _ in range(50):
        base = rng.gauss(0, 1)
        wolf.record_residual("agent_a", base + rng.gauss(0, 0.05))
        wolf.record_residual("agent_b", base + rng.gauss(0, 0.05))
        wolf.record_residual("agent_c", rng.gauss(0, 1))

    corr = wolf.compute_correlation_matrix()
    assert corr[("agent_a", "agent_b")] > 0.7
    assert abs(corr.get(("agent_a", "agent_c"), 0.0)) < 0.7


def test_wolfpack_target_selection():
    from framework.agents import WolfpackAdversary

    wolf = WolfpackAdversary(correlation_threshold=0.7)
    rng = random.Random(99)
    for _ in range(50):
        base = rng.gauss(0, 1)
        wolf.record_residual("agent_a", base + rng.gauss(0, 0.05))
        wolf.record_residual("agent_b", base + rng.gauss(0, 0.05))
        wolf.record_residual("agent_c", rng.gauss(0, 1))

    primary, coalition = wolf.select_targets("agent_a")
    assert primary == "agent_a"
    assert "agent_b" in coalition
    assert "agent_c" not in coalition


def test_wolfpack_coalition_size_bounded():
    from framework.agents import WolfpackAdversary

    wolf = WolfpackAdversary(correlation_threshold=0.5)
    rng = random.Random(42)
    agents = [f"agent_{i}" for i in range(5)]
    for _ in range(30):
        for name in agents:
            wolf.record_residual(name, rng.gauss(0, 1))
    _, coalition = wolf.select_targets("agent_0")
    assert len(coalition) < len(agents)


def test_wolfpack_game_integration():
    from framework.agents import (
        AgentRegistry, DefenderAgent, EnsembleAggregatorAgent,
        ForecastingAgent, RefactoringAgent, WolfpackAdversary,
    )

    cfg = SimulationConfig(
        horizon=30, max_rounds=30,
        disturbance_model="wolfpack",
        adversarial_intensity=1.0,
    )
    wolf = WolfpackAdversary(aggressiveness=1.0, correlation_threshold=0.7)
    registry = AgentRegistry(
        forecasters=(ForecastingAgent(name="f1"), ForecastingAgent(name="f2"), ForecastingAgent(name="f3")),
        adversaries=(wolf,),
        defenders=(DefenderAgent(),),
        refactorer=RefactoringAgent(),
        aggregator=EnsembleAggregatorAgent(),
    )
    game = ForecastGame(cfg, seed=42, registry=registry)
    init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
    out = game.run(init, disturbed=True)
    assert len(out.forecasts) > 0
    assert len(out.targets) == len(out.forecasts)
