from __future__ import annotations

import logging
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


@dataclass(frozen=True)
class HistoricalDisturbance:
    """Samples disturbances from a fitted distribution on cached historical residuals."""

    cache_path: str = "data/cache/fred.json"
    fallback_std: float = 0.3

    def _load_residuals(self) -> list[float]:
        import json
        from pathlib import Path
        p = Path(self.cache_path)
        if not p.exists():
            return []
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            rows = payload.get("rows", [])
            targets = [float(r["target"]) for r in rows if r.get("target") is not None]
            if len(targets) < 3:
                return []
            mean = sum(targets) / len(targets)
            return [t - mean for t in targets]
        except Exception:
            return []

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() > config.disturbance_prob:
            return 0.0
        residuals = self._load_residuals()
        if residuals:
            sampled = rng.choice(residuals)
            scale = max(0.01, sum(abs(r) for r in residuals) / len(residuals))
            return (sampled / scale) * config.disturbance_scale * config.adversarial_intensity
        return rng.gauss(0, self.fallback_std * config.adversarial_intensity)


@dataclass(frozen=True)
class EscalatingDisturbance:
    """Disturbance that intensifies linearly with round index."""

    base_scale: float = 0.1
    escalation_rate: float = 0.01

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() > config.disturbance_prob:
            return 0.0
        effective_scale = self.base_scale + self.escalation_rate * state.t
        return rng.gauss(0, effective_scale * config.adversarial_intensity)


@dataclass(frozen=True)
class WolfpackDisturbance:
    """Multi-target disturbance: primary (full scale) for the target agent, secondary (reduced) for coalition."""

    primary_scale: float = 1.0
    secondary_scale: float = 0.5

    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        if rng.random() > config.disturbance_prob:
            return 0.0
        return rng.gauss(0, config.disturbance_scale * config.adversarial_intensity * self.primary_scale)

    def sample_secondary(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float:
        """Correlated disturbance applied to coalition members."""
        if rng.random() > config.disturbance_prob:
            return 0.0
        return rng.gauss(0, config.disturbance_scale * config.adversarial_intensity * self.secondary_scale)


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
    if normalized in {"historical", "historical_residual"}:
        return HistoricalDisturbance()
    if normalized in {"escalating", "escalate"}:
        return EscalatingDisturbance()
    if normalized in {"wolfpack", "wolf_pack"}:
        return WolfpackDisturbance()
    logging.warning("Unknown disturbance model '%s', defaulting to GaussianDisturbance", name)
    return GaussianDisturbance()
