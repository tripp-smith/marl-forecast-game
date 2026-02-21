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
            return rng.gauss(0, config.disturbance_scale * config.adversarial_intensity)
        return 0.0


@dataclass(frozen=True)
class ShiftDisturbance:
    shift: float = 0.35

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() <= config.disturbance_prob:
            return self.shift * config.adversarial_intensity
        return 0.0


@dataclass(frozen=True)
class EvasionDisturbance:
    factor: float = 0.2

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() <= config.disturbance_prob:
            sign = 1.0 if state.value >= 0 else -1.0
            return sign * config.disturbance_scale * self.factor * config.adversarial_intensity
        return 0.0


@dataclass(frozen=True)
class VolatilityScaledDisturbance:
    min_scale: float = 0.05

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() > config.disturbance_prob:
            return 0.0
        dynamic_scale = max(self.min_scale, abs(state.hidden_shift) + abs(state.exogenous))
        return rng.gauss(0, config.disturbance_scale * dynamic_scale * config.adversarial_intensity)


@dataclass(frozen=True)
class RegimeShiftDisturbance:
    level_shift: float = 0.8
    every_n_steps: int = 12

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if self.every_n_steps <= 0:
            return 0.0
        if state.t > 0 and state.t % self.every_n_steps == 0 and rng.random() <= config.disturbance_prob:
            sign = -1.0 if rng.random() < 0.5 else 1.0
            return sign * self.level_shift * config.adversarial_intensity
        return 0.0


@dataclass(frozen=True)
class VolatilityBurstDisturbance:
    burst_scale: float = 2.5

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() > config.disturbance_prob:
            return 0.0
        scale = config.disturbance_scale * config.adversarial_intensity * self.burst_scale
        return rng.gauss(0, scale)


@dataclass(frozen=True)
class DriftDisturbance:
    step_drift: float = 0.03

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() > config.disturbance_prob:
            return 0.0
        direction = 1.0 if state.exogenous >= 0 else -1.0
        return direction * self.step_drift * (state.t + 1) * config.adversarial_intensity


def disturbance_from_name(name: str) -> DisturbanceModel:
    normalized = name.strip().lower()
    if normalized in {"gaussian", "default"}:
        return GaussianDisturbance()
    if normalized in {"shift", "regime_shift_basic"}:
        return ShiftDisturbance()
    if normalized in {"evasion", "evasion_like"}:
        return EvasionDisturbance()
    if normalized in {"volatility", "volatility_scaled", "vol_scaled"}:
        return VolatilityScaledDisturbance()
    if normalized in {"regime_shift", "regime"}:
        return RegimeShiftDisturbance()
    if normalized in {"volatility_burst", "burst"}:
        return VolatilityBurstDisturbance()
    if normalized in {"drift", "systematic_drift"}:
        return DriftDisturbance()
    return GaussianDisturbance()
