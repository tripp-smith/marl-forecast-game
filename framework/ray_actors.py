"""Ray actor wrappers for stateful agent representations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import ray
except ImportError:
    ray = None  # type: ignore[assignment,unused-ignore]

from .agents import AdversaryAgent, DefenderAgent, ForecastingAgent
from .strategy_runtime import runtime_from_name
from .types import AgentAction, ForecastState, frozen_mapping


def _state_from_dict(d: dict[str, Any]) -> ForecastState:
    return ForecastState(
        t=d["t"],
        value=d["value"],
        exogenous=d["exogenous"],
        hidden_shift=d["hidden_shift"],
        segment_id=d.get("segment_id", ""),
        segment_values=frozen_mapping(d.get("segment_values", {})),
        macro_context=frozen_mapping(d.get("macro_context", {})),
    )


def _action_to_dict(a: AgentAction) -> dict[str, Any]:
    return {"actor": a.actor, "delta": a.delta}


class _RayForecasterActorImpl:
    """Stateful forecaster maintaining cumulative rewards and Q-table weights."""

    def __init__(self, name: str = "forecaster", runtime_backend: str = "python") -> None:
        self.agent = ForecastingAgent(name=name)
        self.runtime = runtime_from_name(runtime_backend)
        self.cumulative_reward: float = 0.0

    def act(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = _state_from_dict(state_dict)
        action = self.agent.act(state, self.runtime)
        return _action_to_dict(action)

    def update_reward(self, reward: float) -> None:
        self.cumulative_reward += reward

    def get_reward(self) -> float:
        return self.cumulative_reward


class _RayAdversaryActorImpl:
    """Stateful adversary with internal memory for strategic adaptation."""

    def __init__(self, name: str = "adversary", aggressiveness: float = 1.0, attack_cost: float = 0.0) -> None:
        self.agent = AdversaryAgent(name=name, aggressiveness=aggressiveness, attack_cost=attack_cost)
        self.cumulative_reward: float = 0.0
        self.attack_history: list[float] = []

    def act(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = _state_from_dict(state_dict)
        action = self.agent.act(state)
        self.attack_history.append(action.delta)
        return _action_to_dict(action)

    def update_reward(self, reward: float) -> None:
        self.cumulative_reward += reward

    def get_reward(self) -> float:
        return self.cumulative_reward


class _RayDefenderActorImpl:
    """Stateful defender actor."""

    def __init__(self, name: str = "defender") -> None:
        self.agent = DefenderAgent(name=name)
        self.cumulative_reward: float = 0.0

    def act(self, forecast_dict: dict[str, Any], adversary_dict: dict[str, Any], defense_model: str) -> dict[str, Any]:
        f_action = AgentAction(**forecast_dict)
        a_action = AgentAction(**adversary_dict)
        action = self.agent.act(f_action, a_action, defense_model)
        return _action_to_dict(action)

    def update_reward(self, reward: float) -> None:
        self.cumulative_reward += reward

    def get_reward(self) -> float:
        return self.cumulative_reward


if ray is not None:
    RayForecasterActor = ray.remote(_RayForecasterActorImpl)
    RayAdversaryActor = ray.remote(_RayAdversaryActorImpl)
    RayDefenderActor = ray.remote(_RayDefenderActorImpl)
else:
    RayForecasterActor = _RayForecasterActorImpl  # type: ignore[assignment,misc,unused-ignore]
    RayAdversaryActor = _RayAdversaryActorImpl  # type: ignore[assignment,misc,unused-ignore]
    RayDefenderActor = _RayDefenderActorImpl  # type: ignore[assignment,misc,unused-ignore]


@dataclass
class ActorRegistry:
    """Holds Ray actor handles instead of direct agent instances."""

    forecasters: list[Any] = field(default_factory=list)
    adversaries: list[Any] = field(default_factory=list)
    defenders: list[Any] = field(default_factory=list)
