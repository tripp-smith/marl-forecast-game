from __future__ import annotations

import logging
import math
from collections import defaultdict
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
# Wolfpack adversary (mutable: tracks residual history)
# ---------------------------------------------------------------------------


@dataclass
class WolfpackAdversary:
    """Coordinated adversary targeting correlated forecaster clusters."""

    name: str = "wolfpack"
    aggressiveness: float = 1.0
    attack_cost: float = 0.0
    correlation_threshold: float = 0.7
    _residuals: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list), repr=False)

    def record_residual(self, agent_name: str, residual: float) -> None:
        self._residuals[agent_name].append(residual)

    @staticmethod
    def _pearson(xs: list[float], ys: list[float]) -> float:
        n = min(len(xs), len(ys))
        if n < 3:
            return 0.0
        mx = sum(xs[:n]) / n
        my = sum(ys[:n]) / n
        cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
        sx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)) / n)
        sy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)) / n)
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return cov / (sx * sy)

    def compute_correlation_matrix(self) -> dict[tuple[str, str], float]:
        names = list(self._residuals.keys())
        matrix: dict[tuple[str, str], float] = {}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i < j:
                    rho = self._pearson(self._residuals[a], self._residuals[b])
                    matrix[(a, b)] = rho
                    matrix[(b, a)] = rho
        return matrix

    def select_targets(self, primary_target: str) -> tuple[str, list[str]]:
        corr = self.compute_correlation_matrix()
        coalition = [
            name for name in self._residuals
            if name != primary_target and abs(corr.get((primary_target, name), 0.0)) >= self.correlation_threshold
        ]
        return primary_target, coalition

    def act(self, state: ForecastState, *, is_primary: bool = True) -> AgentAction:
        expected_trend = 0.4 + 0.4 * state.exogenous
        direction = -1.0 if expected_trend >= 0 else 1.0
        scale = self.aggressiveness if is_primary else self.aggressiveness * 0.5
        base = direction * 0.4 * scale
        penalty = min(abs(base), self.attack_cost * 0.2)
        delta = base - (penalty if base > 0 else -penalty)
        return AgentAction(actor=self.name, delta=delta)


# ---------------------------------------------------------------------------
# Agent registry and executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRegistry:
    """Flexible container for variable numbers of agents."""

    forecasters: tuple[ForecastingAgent | BottomUpAgent, ...] = ()
    adversaries: tuple[AdversaryAgent | WolfpackAdversary, ...] = ()
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


@dataclass(frozen=True)
class LLMPolicyAgent:
    """Wraps a ForecastingAgent and periodically refines its bias via LLM trajectory analysis."""

    name: str = "llm_policy"
    base_agent: ForecastingAgent = field(default_factory=ForecastingAgent)
    refine_every_n: int = 10
    clamp_min: float = -0.1
    clamp_max: float = 0.1

    def act(self, state: ForecastState, runtime: "StrategyRuntime", trajectories: list | None = None, round_idx: int = 0) -> AgentAction:
        from .llm.refiner import RecursiveStrategyRefiner
        from .types import TrajectoryEntry

        base_action = self.base_agent.act(state, runtime)

        if trajectories and round_idx > 0 and round_idx % self.refine_every_n == 0:
            refiner = RecursiveStrategyRefiner()
            result = refiner.refine(trajectories)
            adjusted_delta = base_action.delta + result.bias_adjustment
            return AgentAction(actor=self.name, delta=adjusted_delta)

        return AgentAction(actor=self.name, delta=base_action.delta)


def default_ollama_repl() -> DSPyLikeRepl:
    return DSPyLikeRepl(client=OllamaClient())
