from __future__ import annotations

from dataclasses import dataclass

from .types import AgentAction, ForecastState


@dataclass(frozen=True)
class ForecastingAgent:
    name: str = "forecaster"

    def act(self, state: ForecastState) -> AgentAction:
        # momentum + exogenous signal
        return AgentAction(actor=self.name, delta=0.55 + 0.35 * state.exogenous)


@dataclass(frozen=True)
class AdversaryAgent:
    name: str = "adversary"
    aggressiveness: float = 1.0

    def act(self, state: ForecastState) -> AgentAction:
        # attack in opposite direction of latent trend
        sign = -1.0 if state.value >= 0 else 1.0
        return AgentAction(actor=self.name, delta=sign * 0.4 * self.aggressiveness)


@dataclass(frozen=True)
class DefenderAgent:
    name: str = "defender"
    dampening: float = 0.6

    def act(self, forecast_action: AgentAction, adversary_action: AgentAction) -> AgentAction:
        correction = -(adversary_action.delta * self.dampening)
        # lightly regularize forecaster overreaction
        correction -= 0.1 * max(-1.0, min(1.0, forecast_action.delta))
        return AgentAction(actor=self.name, delta=correction)
