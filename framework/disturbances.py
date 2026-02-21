from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from .types import ForecastState, SimulationConfig


class DisturbanceModel(Protocol):
    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float: ...


@dataclass(frozen=True)
class GaussianDisturbance:
    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() <= config.disturbance_prob:
            return rng.gauss(0, config.disturbance_scale)
        return 0.0


@dataclass(frozen=True)
class ShiftDisturbance:
    shift: float = 0.35

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() <= config.disturbance_prob:
            return self.shift
        return 0.0


@dataclass(frozen=True)
class EvasionDisturbance:
    factor: float = 0.2

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() <= config.disturbance_prob:
            sign = 1.0 if state.value >= 0 else -1.0
            return sign * config.disturbance_scale * self.factor
        return 0.0


def disturbance_from_name(name: str) -> DisturbanceModel:
    normalized = name.strip().lower()
    if normalized in {"gaussian", "default"}:
        return GaussianDisturbance()
    if normalized in {"shift", "regime_shift"}:
        return ShiftDisturbance()
    if normalized in {"evasion", "evasion_like"}:
        return EvasionDisturbance()
    return GaussianDisturbance()
