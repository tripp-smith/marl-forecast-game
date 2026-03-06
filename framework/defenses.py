"""Defense models that correct forecasts against adversarial perturbations."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol


class DefenseModel(Protocol):
    """Protocol for defense strategies that produce correction deltas."""

    def defend(self, forecast_delta: float, adversary_delta: float) -> float: ...


@dataclass(frozen=True)
class IdentityDefense:
    """No-op defense used for ablations and undefended baselines."""

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        return 0.0


@dataclass(frozen=True)
class DampeningDefense:
    """Proportional dampening of the adversary delta with forecast bias correction."""

    dampening: float = 0.6

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        correction = -(adversary_delta * self.dampening)
        correction -= 0.1 * max(-1.0, min(1.0, forecast_delta))
        return correction


@dataclass(frozen=True)
class ClippingDefense:
    """Hard-clips the adversary correction to a bounded range."""

    clip: float = 0.2

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        correction = -adversary_delta
        return max(-self.clip, min(self.clip, correction))


@dataclass(frozen=True)
class BiasGuardDefense:
    """Caps the correction magnitude at a configurable maximum bias threshold."""

    max_bias: float = 0.12

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        if abs(adversary_delta) < self.max_bias:
            return -adversary_delta
        return -self.max_bias if adversary_delta > 0 else self.max_bias


@dataclass(frozen=True)
class EnsembleDefense:
    """Averages dampening, clipping, and bias-guard corrections."""

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
    """Chains two defense models sequentially."""

    first: DefenseModel
    second: DefenseModel

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        first_out = self.first.defend(forecast_delta, adversary_delta)
        return self.second.defend(forecast_delta + first_out, adversary_delta + first_out)


@dataclass(frozen=True)
class DANNDefenseStub:
    """Domain Adversarial Neural Network defense stub.

    Simulates domain-adaptation-style correction with a configurable
    domain weight blending between source and target corrections.
    """

    dampening: float = 0.6
    domain_weight: float = 0.3

    def defend(self, forecast_delta: float, adversary_delta: float) -> float:
        source_correction = -(adversary_delta * self.dampening)
        target_correction = -0.1 * max(-1.0, min(1.0, forecast_delta))
        return (1.0 - self.domain_weight) * source_correction + self.domain_weight * target_correction


def defense_from_name(name: str) -> DefenseModel:
    """Instantiate a defense model by name, defaulting to dampening."""
    normalized = name.strip().lower()
    if normalized.startswith("stack:"):
        _, models = normalized.split(":", 1)
        parts = [p.strip() for p in models.split(",") if p.strip()]
        if len(parts) >= 2:
            return StackedDefense(defense_from_name(parts[0]), defense_from_name(parts[1]))
    if normalized in {"identity", "none", "noop", "no_op"}:
        return IdentityDefense()
    if normalized in {"dampening", "default"}:
        return DampeningDefense()
    if normalized in {"clipping", "clip"}:
        return ClippingDefense()
    if normalized in {"bias_guard", "bias"}:
        return BiasGuardDefense()
    if normalized in {"ensemble", "filter_ensemble"}:
        return EnsembleDefense()
    if normalized in {"dann", "domain_adversarial"}:
        return DANNDefenseStub()
    logging.warning("Unknown defense model '%s', defaulting to DampeningDefense", name)
    return DampeningDefense()
