"""Hierarchical and composite forecasting agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..strategy_runtime import StrategyRuntime
from ..types import AgentAction, ForecastState
from .base import ForecastingAgent


@dataclass(frozen=True)
class BottomUpAgent:
    """Segment-level forecaster operating on granular state slices."""

    name: str = "bottom_up"
    segment_weight: float = 0.3

    def act(self, state: ForecastState, runtime: StrategyRuntime) -> AgentAction:
        """Blend segment-level signals with the base runtime forecast."""
        seg_contribution = 0.0
        if state.segment_values:
            seg_contribution = sum(state.segment_values.values()) / max(1, len(state.segment_values))
        base_delta = runtime.forecast_delta(state)
        delta = (1.0 - self.segment_weight) * base_delta + self.segment_weight * seg_contribution
        return AgentAction(actor=self.name, delta=delta)


@dataclass(frozen=True)
class TopDownAgent:
    """Macro-level adjuster using macro context for economy-wide corrections."""

    name: str = "top_down"
    macro_sensitivity: float = 0.2

    def act(self, state: ForecastState) -> AgentAction:
        """Convert macro context into a top-down forecast adjustment."""
        macro_signal = 0.0
        if state.macro_context:
            macro_signal = sum(state.macro_context.values()) / max(1, len(state.macro_context))
        regime_modifier = 1.0 + 0.1 * (state.economic_regime - 1)
        delta = self.macro_sensitivity * macro_signal * regime_modifier
        return AgentAction(actor=self.name, delta=delta)


@dataclass(frozen=True)
class EnsembleAggregatorAgent:
    """Combine multiple forecast actions into a single weighted delta."""

    name: str = "aggregator"
    mode: str = "equal"

    def aggregate(self, actions: list[AgentAction], reward_history: dict[str, float] | None = None) -> AgentAction:
        """Aggregate a list of forecast actions into a single action."""
        if not actions:
            return AgentAction(actor=self.name, delta=0.0)

        if self.mode == "reward_proportional" and reward_history:
            weights = []
            for action in actions:
                reward = reward_history.get(action.actor, 0.0)
                weights.append(max(0.0, reward + 1.0))
            total = sum(weights) or 1.0
            weighted = sum(weight * action.delta for weight, action in zip(weights, actions)) / total
            return AgentAction(actor=self.name, delta=weighted)

        total = sum(action.delta for action in actions)
        return AgentAction(actor=self.name, delta=total / len(actions))


@dataclass(frozen=True)
class LLMPolicyAgent:
    """Wrap a forecasting agent and periodically refine its output using trajectories."""

    name: str = "llm_policy"
    base_agent: ForecastingAgent = field(default_factory=ForecastingAgent)
    refine_every_n: int = 10
    clamp_min: float = -0.1
    clamp_max: float = 0.1

    def act(
        self,
        state: ForecastState,
        runtime: StrategyRuntime,
        trajectories: list[Any] | None = None,
        round_idx: int = 0,
    ) -> AgentAction:
        """Return the current forecast action, optionally refined from trajectories."""
        from ..llm.refiner import RecursiveStrategyRefiner

        base_action = self.base_agent.act(state, runtime, round_idx=round_idx)
        if trajectories and round_idx > 0 and round_idx % self.refine_every_n == 0:
            refiner = RecursiveStrategyRefiner(clamp_min=self.clamp_min, clamp_max=self.clamp_max)
            result = refiner.refine(trajectories)
            adjusted_delta = base_action.delta + result.bias_adjustment
            return AgentAction(actor=self.name, delta=adjusted_delta)

        return AgentAction(actor=self.name, delta=base_action.delta)
