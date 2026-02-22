from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

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
        expected_trend = 0.4 + 0.4 * state.exogenous
        direction = -1.0 if expected_trend >= 0 else 1.0
        base = direction * 0.4 * self.aggressiveness
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


# ---------------------------------------------------------------------------
# Hierarchical agents (Phase R)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BottomUpAgent:
    """Segment-level forecaster operating on granular state slices."""

    name: str = "bottom_up"
    segment_weight: float = 0.3

    def act(self, state: ForecastState, runtime: StrategyRuntime) -> AgentAction:
        seg_contribution = 0.0
        if state.segment_values:
            seg_contribution = sum(state.segment_values.values()) / max(1, len(state.segment_values))
        base_delta = runtime.forecast_delta(state)
        delta = (1.0 - self.segment_weight) * base_delta + self.segment_weight * seg_contribution
        return AgentAction(actor=self.name, delta=delta)


@dataclass(frozen=True)
class TopDownAgent:
    """Macro-level adjuster using macro_context for GDP/PMI-based corrections."""

    name: str = "top_down"
    macro_sensitivity: float = 0.2

    def act(self, state: ForecastState) -> AgentAction:
        macro_signal = 0.0
        if state.macro_context:
            macro_signal = sum(state.macro_context.values()) / max(1, len(state.macro_context))
        delta = self.macro_sensitivity * macro_signal
        return AgentAction(actor=self.name, delta=delta)


@dataclass(frozen=True)
class EnsembleAggregatorAgent:
    """Combines multiple agent actions into a single weighted delta."""

    name: str = "aggregator"
    mode: str = "equal"

    def aggregate(self, actions: List[AgentAction], reward_history: dict[str, float] | None = None) -> AgentAction:
        if not actions:
            return AgentAction(actor=self.name, delta=0.0)

        if self.mode == "reward_proportional" and reward_history:
            weights = []
            for a in actions:
                r = reward_history.get(a.actor, 0.0)
                weights.append(max(0.0, r + 1.0))
            total = sum(weights) or 1.0
            weighted = sum(w * a.delta for w, a in zip(weights, actions)) / total
            return AgentAction(actor=self.name, delta=weighted)

        total = sum(a.delta for a in actions)
        return AgentAction(actor=self.name, delta=total / len(actions))


# ---------------------------------------------------------------------------
# Agent registry and executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRegistry:
    """Flexible container for variable numbers of agents."""

    forecasters: tuple[ForecastingAgent | BottomUpAgent, ...] = ()
    adversaries: tuple[AdversaryAgent, ...] = ()
    defenders: tuple[DefenderAgent, ...] = ()
    refactorer: RefactoringAgent | None = None
    aggregator: EnsembleAggregatorAgent = field(default_factory=EnsembleAggregatorAgent)


@dataclass(frozen=True)
class SafeAgentExecutor:
    fallback_delta: float = 0.0

    def execute(self, fn, *args, **kwargs) -> AgentAction:
        try:
            return fn(*args, **kwargs)
        except Exception:
            logging.warning("SafeAgentExecutor caught exception in %s, using fallback", getattr(fn, "__name__", fn), exc_info=True)
            return AgentAction(actor="fallback", delta=self.fallback_delta)


def default_ollama_repl() -> DSPyLikeRepl:
    return DSPyLikeRepl(client=OllamaClient())
