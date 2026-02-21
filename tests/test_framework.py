from framework.data import build_sample_dataset, chronological_split, load_csv, normalize_features
from framework.game import ForecastGame
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
