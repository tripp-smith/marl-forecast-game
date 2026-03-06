from __future__ import annotations

import pytest

from framework.agents import EvolutionaryAgentPopulation, StrategyVariant


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
