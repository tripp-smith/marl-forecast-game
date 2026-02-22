from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol


class DefenseModel(Protocol):
    def defend(self, forecast_delta: float, adversary_delta: float) -> float: ...


@dataclass(frozen=True)
class DampeningDefense:
    dampening: float = 0.6

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        correction = -(adversary_delta * self.dampening)
        correction -= 0.1 * max(-1.0, min(1.0, forecast_delta))
        return correction


@dataclass(frozen=True)
class ClippingDefense:
    clip: float = 0.2

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        correction = -adversary_delta
        return max(-self.clip, min(self.clip, correction))


@dataclass(frozen=True)
class BiasGuardDefense:
    max_bias: float = 0.12

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        if abs(adversary_delta) < self.max_bias:
            return -adversary_delta
        return -self.max_bias if adversary_delta > 0 else self.max_bias


@dataclass(frozen=True)
class EnsembleDefense:
    dampening: DampeningDefense = DampeningDefense()
    clipping: ClippingDefense = ClippingDefense(clip=0.25)
    bias_guard: BiasGuardDefense = BiasGuardDefense()

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        damp = self.dampening.defend(forecast_delta, adversary_delta)
        clip = self.clipping.defend(forecast_delta, adversary_delta)
        bias = self.bias_guard.defend(forecast_delta, adversary_delta)
        return (damp + clip + bias) / 3.0


@dataclass(frozen=True)
class StackedDefense:
    first: DefenseModel
    second: DefenseModel

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        first_out = self.first.defend(forecast_delta, adversary_delta)
        return self.second.defend(forecast_delta + first_out, adversary_delta + first_out)


def defense_from_name(name: str) -> DefenseModel:
    normalized = name.strip().lower()
    if normalized.startswith("stack:"):
        _, models = normalized.split(":", 1)
        parts = [p.strip() for p in models.split(",") if p.strip()]
        if len(parts) >= 2:
            return StackedDefense(defense_from_name(parts[0]), defense_from_name(parts[1]))
    if normalized in {"dampening", "default"}:
        return DampeningDefense()
    if normalized in {"clipping", "clip"}:
        return ClippingDefense()
    if normalized in {"bias_guard", "bias"}:
        return BiasGuardDefense()
    if normalized in {"ensemble", "filter_ensemble"}:
        return EnsembleDefense()
    logging.warning("Unknown defense model '%s', defaulting to DampeningDefense", name)
    return DampeningDefense()
