from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Tuple


_EMPTY_MAPPING: Mapping[str, float] = MappingProxyType({})


def frozen_mapping(values: dict[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class ForecastState:
    """Immutable state for the forecasting Markov game."""

    t: int
    value: float
    exogenous: float
    hidden_shift: float
    segment_id: str = ""
    segment_values: Mapping[str, float] = field(default_factory=lambda: _EMPTY_MAPPING)
    macro_context: Mapping[str, float] = field(default_factory=lambda: _EMPTY_MAPPING)


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
class ProbabilisticForecast:
    mean: float
    variance: float
    quantiles: Tuple[float, ...] = ()


@dataclass(frozen=True)
class TrajectoryEntry:
    round_idx: int
    state: ForecastState
    actions: Tuple[AgentAction, ...]
    messages: Tuple[AgentMessage, ...]
    reward_breakdown: Mapping[str, float]
    forecast: float
    target: float


@dataclass(frozen=True)
class StepResult:
    next_state: ForecastState
    actions: Tuple[AgentAction, ...]
    reward_breakdown: Mapping[str, float]
    forecast: float
    target: float
    confidence: ConfidenceInterval
    messages: Tuple[AgentMessage, ...]
    probabilistic: ProbabilisticForecast | None = None


@dataclass(frozen=True)
class SimulationConfig:
    horizon: int = 100
    max_rounds: int = 200
    max_round_timeout_s: float = 1.0
    base_noise_std: float = 0.15
    disturbance_prob: float = 0.1
    disturbance_scale: float = 1.0
    adversarial_intensity: float = 1.0
    runtime_backend: str = "python"
    disturbance_model: str = "gaussian"
    defense_model: str = "dampening"
    enable_refactor: bool = True
    enable_llm_refactor: bool = False
    attack_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ValueError("horizon must be >= 0")
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be >= 0")
        if self.max_round_timeout_s <= 0:
            raise ValueError("max_round_timeout_s must be > 0")
        if self.base_noise_std < 0:
            raise ValueError("base_noise_std must be >= 0")
        if not (0.0 <= self.disturbance_prob <= 1.0):
            raise ValueError("disturbance_prob must be in [0, 1]")
        if self.disturbance_scale < 0:
            raise ValueError("disturbance_scale must be >= 0")
        if self.adversarial_intensity < 0:
            raise ValueError("adversarial_intensity must be >= 0")
        if self.attack_cost < 0:
            raise ValueError("attack_cost must be >= 0")


def evolve_state(
    state: ForecastState,
    *,
    base_trend: float,
    noise: float,
    disturbance: float,
    coeff_map: dict[str, float] | None = None,
) -> ForecastState:
    macro_contribution = 0.0
    if coeff_map and state.macro_context:
        for k, coeff in coeff_map.items():
            if k in state.macro_context:
                macro_contribution += coeff * state.macro_context[k]

    new_value = state.value + base_trend + 0.4 * state.exogenous + noise + disturbance + macro_contribution
    new_exogenous = 0.6 * state.exogenous + 0.2 * disturbance
    return replace(
        state,
        t=state.t + 1,
        value=new_value,
        exogenous=new_exogenous,
        hidden_shift=disturbance,
    )
