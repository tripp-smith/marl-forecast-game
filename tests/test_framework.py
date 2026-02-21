from framework.data import build_sample_dataset, chronological_split, load_csv, load_source_rows, normalize_features
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


def test_transition_is_pure():
    s0 = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
    s1 = evolve_state(s0, base_trend=0.4, noise=0.1, disturbance=0.0)
    s2 = evolve_state(s0, base_trend=0.4, noise=0.1, disturbance=0.0)
    assert s1 == s2
    assert s0.t == 0


def test_game_round_cap():
    cfg = SimulationConfig(horizon=400, max_rounds=200)
    game = ForecastGame(cfg, seed=9)
    out = game.run(ForecastState(t=0, value=50.0, exogenous=0.0, hidden_shift=0.0), disturbed=True)
    assert len(out.steps) == 200
    assert out.convergence["round_cap_hit"] is True


def test_runtime_backend_selection_and_fallback():
    assert isinstance(runtime_from_name("python"), PythonStrategyRuntime)
    assert isinstance(runtime_from_name("haskell"), HaskellRLMRuntime)
    assert isinstance(runtime_from_name("unknown"), PythonStrategyRuntime)


def test_confidence_and_messages_present():
    cfg = SimulationConfig(horizon=10, max_rounds=20, disturbance_model="shift")
    game = ForecastGame(cfg, seed=3)
    out = game.run(ForecastState(t=0, value=12.0, exogenous=0.1, hidden_shift=0.0), disturbed=True)
    assert len(out.confidence) == len(out.steps)
    assert out.steps[0].confidence.lower <= out.steps[0].forecast <= out.steps[0].confidence.upper
    assert len(out.steps[0].messages) == 3


def test_source_adapter_schema():
    rows = load_source_rows("polymarket", periods=12)
    assert len(rows) == 12
    first = rows[0]
    assert {"timestamp", "series_id", "target", "promo", "macro_index", "source", "fetched_at"}.issubset(first.keys())
