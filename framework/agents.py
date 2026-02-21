from __future__ import annotations

from dataclasses import dataclass, field

from .defenses import defense_from_name
from .llm import DSPyLikeRepl, LLMRefactorClient, MockLLMRefactorClient, OllamaClient, RefactorRequest
from .strategy_runtime import StrategyRuntime
from .types import AgentAction, ForecastState


@dataclass(frozen=True)
class ForecastingAgent:
    name: str = "forecaster"
    llm_repl: DSPyLikeRepl | None = None

    def act(self, state: ForecastState, runtime: StrategyRuntime) -> AgentAction:
        base_delta = runtime.forecast_delta(state)
        if self.llm_repl is None:
            return AgentAction(actor=self.name, delta=base_delta)

        prompt = f"Given value={state.value:.4f}, exogenous={state.exogenous:.4f}, suggest scalar delta"
        try:
            turn = self.llm_repl.run_turn(prompt)
            llm_delta = float(turn["completion"].strip().split()[0])
            return AgentAction(actor=self.name, delta=(0.8 * base_delta) + (0.2 * llm_delta))
        except Exception:
            return AgentAction(actor=self.name, delta=base_delta)


@dataclass(frozen=True)
class AdversaryAgent:
    name: str = "adversary"
    aggressiveness: float = 1.0
    attack_cost: float = 0.0

    def act(self, state: ForecastState) -> AgentAction:
        sign = -1.0 if state.value >= 0 else 1.0
        base = sign * 0.4 * self.aggressiveness
        penalty = min(abs(base), self.attack_cost * 0.2)
        delta = base - (penalty if base > 0 else -penalty)
        return AgentAction(actor=self.name, delta=delta)


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
    llm_client: LLMRefactorClient = field(default_factory=lambda: MockLLMRefactorClient(step_size=0.02))

    def revise(self, latest_error: float, *, use_llm: bool = False) -> float:
        if use_llm:
            suggestion = self.llm_client.suggest(
                RefactorRequest(latest_error=latest_error, strategy_name=self.name)
            )
            return suggestion.bias_adjustment
        return -self.step_size if latest_error > 0 else self.step_size


@dataclass(frozen=True)
class SafeAgentExecutor:
    fallback_delta: float = 0.0

    def execute(self, fn, *args, **kwargs) -> AgentAction:
        try:
            return fn(*args, **kwargs)
        except Exception:
            return AgentAction(actor="fallback", delta=self.fallback_delta)


def default_ollama_repl() -> DSPyLikeRepl:
    return DSPyLikeRepl(client=OllamaClient())
