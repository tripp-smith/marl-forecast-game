"""Monte Carlo scenario generation producing trajectory fans and YAML config."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from typing import Any, List, Tuple

import yaml  # type: ignore[import-untyped]

from .game import ForecastGame
from .types import ForecastState, SimulationConfig


# ---------------------------------------------------------------------------
# YAML scenario configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioSpec:
    """A single scenario parsed from YAML."""
    name: str
    disturbance_model: str = "gaussian"
    defense_model: str = "dampening"
    adversarial_intensity: float = 1.0
    disturbance_prob: float = 0.1
    disturbance_scale: float = 1.0
    horizon: int = 100
    max_rounds: int = 200
    attack_cost: float = 0.0

    def to_config(self, base: SimulationConfig | None = None) -> SimulationConfig:
        """Convert this spec to a SimulationConfig, optionally overlaying on *base*."""
        if base is not None:
            return replace(
                base,
                disturbance_model=self.disturbance_model,
                defense_model=self.defense_model,
                adversarial_intensity=self.adversarial_intensity,
                disturbance_prob=self.disturbance_prob,
                disturbance_scale=self.disturbance_scale,
                horizon=self.horizon,
                max_rounds=self.max_rounds,
                attack_cost=self.attack_cost,
            )
        return SimulationConfig(
            disturbance_model=self.disturbance_model,
            defense_model=self.defense_model,
            adversarial_intensity=self.adversarial_intensity,
            disturbance_prob=self.disturbance_prob,
            disturbance_scale=self.disturbance_scale,
            horizon=self.horizon,
            max_rounds=self.max_rounds,
            attack_cost=self.attack_cost,
        )


def load_scenario_specs(path: str | Path) -> list[ScenarioSpec]:
    """Load scenario definitions from a YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenarios_raw = data if isinstance(data, list) else data.get("scenarios", [])
    specs: list[ScenarioSpec] = []
    for entry in scenarios_raw:
        specs.append(ScenarioSpec(
            name=entry.get("name", "unnamed"),
            disturbance_model=entry.get("disturbance_model", "gaussian"),
            defense_model=entry.get("defense_model", "dampening"),
            adversarial_intensity=float(entry.get("adversarial_intensity", 1.0)),
            disturbance_prob=float(entry.get("disturbance_prob", 0.1)),
            disturbance_scale=float(entry.get("disturbance_scale", 1.0)),
            horizon=int(entry.get("horizon", 100)),
            max_rounds=int(entry.get("max_rounds", 200)),
            attack_cost=float(entry.get("attack_cost", 0.0)),
        ))
    return specs


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
    """Runs Monte Carlo replications and produces percentile forecast fans."""

    config: SimulationConfig
    n_replications: int = 1000

    def generate(self, initial: ForecastState, base_seed: int = 0) -> ScenarioFan:
        """Run *n_replications* games and compute percentile trajectories."""
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
