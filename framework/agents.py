from __future__ import annotations

from dataclasses import dataclass

from .defenses import defense_from_name
from .llm import LLMRefactorClient, MockLLMRefactorClient, RefactorRequest
from .strategy_runtime import StrategyRuntime
from .types import AgentAction, ForecastState


@dataclass(frozen=True)
class ForecastingAgent:
    name: str = "forecaster"

    def act(self, state: ForecastState, runtime: StrategyRuntime) -> AgentAction:
        return AgentAction(actor=self.name, delta=runtime.forecast_delta(state))


@dataclass(frozen=True)
class AdversaryAgent:
    name: str = "adversary"
    aggressiveness: float = 1.0

    def act(self, state: ForecastState) -> AgentAction:
        sign = -1.0 if state.value >= 0 else 1.0
        return AgentAction(actor=self.name, delta=sign * 0.4 * self.aggressiveness)


@dataclass(frozen=True)
class DefenderAgent:
    name: str = "defender"

    def act(self, forecast_action: AgentAction, adversary_action: AgentAction, defense_model: str) -> AgentAction:
        defense = defense_from_name(defense_model)
        correction = defense.defend(forecast_action.delta, adversary_action.delta)
        return AgentAction(actor=self.name, delta=correction)


@dataclass(frozen=True)
class RefactoringAgent:
    name: str = "refactor"
    step_size: float = 0.02
    llm_client: LLMRefactorClient = MockLLMRefactorClient(step_size=0.02)

    def revise(self, latest_error: float, *, use_llm: bool = False) -> float:
        if use_llm:
            suggestion = self.llm_client.suggest(
                RefactorRequest(latest_error=latest_error, strategy_name=self.name)
            )
            return suggestion.bias_adjustment
        return -self.step_size if latest_error > 0 else self.step_size
