from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from framework.agents import EvolutionaryAgentPopulation, ForecastingAgent, StrategyVariant, WolfpackAdversary
from framework.strategy_runtime import PythonStrategyRuntime
from framework.types import ForecastState


def test_evolutionary_population_shifts_toward_high_fitness_strategy():
    population = EvolutionaryAgentPopulation(
        variants=(
            StrategyVariant("forecaster_variant_0", "forecaster", 0.2, 1.0, 0.0),
            StrategyVariant("forecaster_variant_1", "forecaster", 0.2, 1.0, 0.0),
            StrategyVariant("forecaster_variant_2", "forecaster", 0.2, 1.0, 0.0),
            StrategyVariant("forecaster_variant_3", "forecaster", 0.2, 1.0, 0.0),
            StrategyVariant("forecaster_variant_4", "forecaster", 0.2, 1.0, 0.0),
        ),
        evolution_rate=0.8,
        prune_threshold=0.0,
    )
    reward_history = {
        "forecaster_variant_0": [1.0] * 10,
        "forecaster_variant_1": [0.1] * 10,
        "forecaster_variant_2": [0.05] * 10,
        "forecaster_variant_3": [0.02] * 10,
        "forecaster_variant_4": [0.01] * 10,
    }

    evolved = population
    for _ in range(5):
        evolved = evolved.evolve(reward_history)

    proportions = {variant.name: variant.proportion for variant in evolved.variants_for("forecaster")}
    assert proportions["forecaster_variant_0"] > 0.8
    assert population.variants_for("forecaster")[0].proportion == pytest.approx(0.2)


def test_evolutionary_population_prunes_bankrupt_variants():
    population = EvolutionaryAgentPopulation(
        variants=(
            StrategyVariant("adversary_variant_0", "adversary", 0.5, 1.0, 0.0, bankroll=1.0),
            StrategyVariant("adversary_variant_1", "adversary", 0.5, 1.0, 0.0, bankroll=1e-6),
        ),
        evolution_rate=1.0,
        prune_threshold=0.0,
        bankruptcy_floor=1e-5,
    )

    evolved = population.evolve(
        {
            "adversary_variant_0": [1.0],
            "adversary_variant_1": [0.0],
        }
    )
    names = [variant.name for variant in evolved.variants_for("adversary")]
    assert names == ["adversary_variant_0"]


def test_forecasting_agent_retries_llm_parse_failures(monkeypatch, caplog):
    class StubRepl:
        def __init__(self) -> None:
            self.responses = iter(
                (
                    {"completion": "invalid"},
                    {"completion": "still_invalid"},
                    {"completion": "0.25"},
                )
            )

        def run_turn(self, prompt: str, **_: object) -> dict[str, str]:
            return next(self.responses)

    sleeps: list[float] = []
    monkeypatch.setattr("framework.agents.base.time.sleep", sleeps.append)

    agent = ForecastingAgent(llm_repl=StubRepl())  # type: ignore[arg-type]
    state = ForecastState(t=0, value=10.0, exogenous=0.5, hidden_shift=0.0)
    action = agent.act(state, PythonStrategyRuntime(), round_idx=1)

    assert action.delta == pytest.approx((0.8 * 0.6) + (0.2 * 0.25))
    assert sleeps == [0.5, 1.0]
    assert "LLM parse failed" in caplog.text


def test_forecasting_agent_falls_back_after_exhausting_parse_retries(monkeypatch):
    class StubRepl:
        def run_turn(self, prompt: str, **_: object) -> dict[str, str]:
            return {"completion": "not_a_float"}

    sleeps: list[float] = []
    monkeypatch.setattr("framework.agents.base.time.sleep", sleeps.append)

    state = ForecastState(t=0, value=10.0, exogenous=0.5, hidden_shift=0.0)
    action = ForecastingAgent(llm_repl=StubRepl()).act(  # type: ignore[arg-type]
        state,
        PythonStrategyRuntime(),
        round_idx=1,
    )

    assert action.delta == pytest.approx(0.6)
    assert sleeps == [0.5, 1.0, 2.0]


def test_wolfpack_adversary_is_frozen():
    wolf = WolfpackAdversary()
    with pytest.raises(FrozenInstanceError):
        wolf.aggressiveness = 2.0
