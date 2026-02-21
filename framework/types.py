from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class ForecastState:
    """Immutable state for the forecasting Markov game."""

    t: int
    value: float
    exogenous: float
    hidden_shift: float


@dataclass(frozen=True)
class AgentAction:
    actor: str
    delta: float


@dataclass(frozen=True)
class AgentMessage:
    sender: str
    receiver: str
    payload: str


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float


@dataclass(frozen=True)
class TrajectoryEntry:
    round_idx: int
    state: ForecastState
    actions: Tuple[AgentAction, ...]
    messages: Tuple[AgentMessage, ...]
    reward_breakdown: Dict[str, float]
    forecast: float
    target: float


@dataclass(frozen=True)
class StepResult:
    next_state: ForecastState
    actions: Tuple[AgentAction, ...]
    reward_breakdown: Dict[str, float]
    forecast: float
    target: float
    confidence: ConfidenceInterval
    messages: Tuple[AgentMessage, ...]


@dataclass(frozen=True)
class SimulationConfig:
    horizon: int = 100
    max_rounds: int = 200
    base_noise_std: float = 0.15
    disturbance_prob: float = 0.1
    disturbance_scale: float = 1.0
    runtime_backend: str = "python"
    disturbance_model: str = "gaussian"
    defense_model: str = "dampening"
    enable_refactor: bool = True


def evolve_state(
    state: ForecastState,
    *,
    base_trend: float,
    noise: float,
    disturbance: float,
) -> ForecastState:
    """Pure transition function for the environment."""

    new_value = state.value + base_trend + 0.4 * state.exogenous + noise + disturbance
    new_exogenous = 0.6 * state.exogenous + 0.2 * disturbance
    return replace(
        state,
        t=state.t + 1,
        value=new_value,
        exogenous=new_exogenous,
        hidden_shift=disturbance,
    )
