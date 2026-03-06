from marl_forecast_game import GameEngine, ForecastState, SimulationConfig, demo_state


def test_public_api_imports_and_runs() -> None:
    cfg = SimulationConfig(horizon=5, max_rounds=5, defense_model="ensemble")
    state = demo_state(value=100.0, exogenous=0.0)
    assert isinstance(state, ForecastState)

    game = GameEngine(cfg, seed=42)
    out = game.run(state, rounds=5, disturbed=True)

    assert len(out.forecasts) == 5
    assert len(out.targets) == 5
