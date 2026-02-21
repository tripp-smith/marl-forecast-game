from __future__ import annotations

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
class EnsembleDefense:
    dampening: DampeningDefense = DampeningDefense()
    clipping: ClippingDefense = ClippingDefense(clip=0.25)

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        damp = self.dampening.defend(forecast_delta, adversary_delta)
        clip = self.clipping.defend(forecast_delta, adversary_delta)
        return 0.5 * (damp + clip)


def defense_from_name(name: str) -> DefenseModel:
    normalized = name.strip().lower()
    if normalized in {"dampening", "default"}:
        return DampeningDefense()
    if normalized in {"clipping", "clip"}:
        return ClippingDefense()
    if normalized in {"ensemble", "filter_ensemble"}:
        return EnsembleDefense()
    return DampeningDefense()
