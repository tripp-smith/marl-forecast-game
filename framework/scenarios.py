"""Monte Carlo scenario generation producing trajectory fans."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List, Tuple

from .game import ForecastGame
from .types import ForecastState, SimulationConfig


@dataclass(frozen=True)
class ScenarioFan:
    """Percentile trajectories across Monte Carlo replications."""

    n_replications: int
    n_rounds: int
    p10: Tuple[float, ...]
    p25: Tuple[float, ...]
    p50: Tuple[float, ...]
    p75: Tuple[float, ...]
    p90: Tuple[float, ...]
    mean_mae: float


@dataclass(frozen=True)
class ScenarioGenerator:
    config: SimulationConfig
    n_replications: int = 1000

    def generate(self, initial: ForecastState, base_seed: int = 0) -> ScenarioFan:
        all_forecasts: list[list[float]] = []
        all_targets: list[list[float]] = []

        for rep in range(self.n_replications):
            game = ForecastGame(self.config, seed=base_seed + rep)
            out = game.run(initial, disturbed=True)
            all_forecasts.append(out.forecasts)
            all_targets.append(out.targets)

        if not all_forecasts or not all_forecasts[0]:
            return ScenarioFan(
                n_replications=self.n_replications,
                n_rounds=0,
                p10=(), p25=(), p50=(), p75=(), p90=(),
                mean_mae=0.0,
            )

        n_rounds = len(all_forecasts[0])
        percentiles: dict[str, list[float]] = {"p10": [], "p25": [], "p50": [], "p75": [], "p90": []}

        for t in range(n_rounds):
            values = sorted(f[t] for f in all_forecasts if t < len(f))
            n = len(values)
            percentiles["p10"].append(values[max(0, int(n * 0.10))])
            percentiles["p25"].append(values[max(0, int(n * 0.25))])
            percentiles["p50"].append(values[max(0, int(n * 0.50))])
            percentiles["p75"].append(values[max(0, min(n - 1, int(n * 0.75)))])
            percentiles["p90"].append(values[max(0, min(n - 1, int(n * 0.90)))])

        total_mae = 0.0
        for forecasts, targets in zip(all_forecasts, all_targets):
            if forecasts and targets:
                total_mae += sum(abs(f - t) for f, t in zip(forecasts, targets)) / len(forecasts)
        mean_mae = total_mae / max(1, len(all_forecasts))

        return ScenarioFan(
            n_replications=self.n_replications,
            n_rounds=n_rounds,
            p10=tuple(percentiles["p10"]),
            p25=tuple(percentiles["p25"]),
            p50=tuple(percentiles["p50"]),
            p75=tuple(percentiles["p75"]),
            p90=tuple(percentiles["p90"]),
            mean_mae=mean_mae,
        )
