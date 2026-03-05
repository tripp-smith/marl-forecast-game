"""Core immutable data types for the forecasting game simulation."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from .exceptions import ConfigValidationError


_EMPTY_MAPPING: Mapping[str, float] = MappingProxyType({})


def frozen_mapping(values: dict[str, float]) -> Mapping[str, float]:
    """Wrap a dict as an immutable MappingProxyType."""
    return MappingProxyType(dict(values))


def validate_immutability(*classes: type) -> None:
    """Assert that all given dataclasses are frozen. Called at import time."""
    for cls in classes:
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass")
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        if not params.frozen:
            raise TypeError(f"{cls.__name__} must be frozen")


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
    qualitative_state: tuple[int, ...] = ()
    raw_qual_state: tuple[float, ...] = ()
    raw_qual_text: str | None = None
    economic_regime: int = 0
    last_qual_update_step: int = -1


@dataclass(frozen=True)
class AgentAction:
    """A single agent's proposed delta adjustment."""

    actor: str
    delta: float


@dataclass(frozen=True)
class AgentMessage:
    """Inter-agent communication payload."""

    sender: str
    receiver: str
    payload: str


@dataclass(frozen=True)
class ConfidenceInterval:
    """Lower and upper bounds of a forecast confidence interval."""

    lower: float
    upper: float


@dataclass(frozen=True)
class ProbabilisticForecast:
    """Gaussian predictive distribution with optional quantiles."""

    mean: float
    variance: float
    quantiles: Tuple[float, ...] = ()


@dataclass(frozen=True)
class TrajectoryEntry:
    """Single-round record of state, actions, messages, rewards, and forecast vs. target."""

    round_idx: int
    state: ForecastState
    actions: Tuple[AgentAction, ...]
    messages: Tuple[AgentMessage, ...]
    reward_breakdown: Mapping[str, float]
    forecast: float
    target: float


@dataclass(frozen=True)
class StepResult:
    """Result of a single simulation step including next state and metrics."""

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
    """Complete configuration for a forecast game simulation run."""

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
    convergence_threshold: float = 0.0
    adversary_tau_init: float = 5.0
    adversary_tau_final: float = 0.1
    tau_decay_rate: float = 0.05
    bankruptcy_threshold: float = 0.01
    wolfpack_correlation_threshold: float = 0.7
    enable_qual: bool = False
    decay_rate: float = 0.01
    feature_dim: int = 3
    regime_classes: int = 3
    max_context_tokens: int = 4096
    qualitative_extractor_prompt: str = (
        "Extract discrete [-1,0,1] for sentiment, uncertainty, guidance "
        "from the following text as JSON: {text}"
    )
    regime_prompt: str = (
        "Classify economic regime [recession=0, expansion=1, stagflation=2] "
        "from quantitative={quant} and qualitative={qual}. Reply with a single integer."
    )
    summary_prompt: str = (
        "Summarize the following to 200 tokens retaining sentiment and uncertainty: {text}"
    )
    qual_adapters: tuple[str, ...] = ("beige_book", "pmi", "earnings")
    verification_runs: int = 100
    mnpo_eta: float = 1.0
    mnpo_beta: float = 0.1
    mnpo_population_size: int = 10
    start_date: str = ""
    agent_type: str = "forecaster"
    marl_qtable_path: str = ""
    marl_algorithm: str = "q"
    rl_backend: str = "tabular"
    rl_algorithm: str = "dqn"
    gpu_enabled: bool = False
    replay_buffer_size: int = 2048
    rl_batch_size: int = 64
    target_update_interval: int = 25
    epsilon_final: float = 0.01
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_decay: float = 0.995
    llm_provider: str = "ollama"
    llm_model: str = "default"
    llm_fallback_to_local: bool = True
    poisoning_threshold: float = 0.05
    agent_specs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ConfigValidationError("horizon must be >= 0")
        if self.max_rounds < 0:
            raise ConfigValidationError("max_rounds must be >= 0")
        if self.max_round_timeout_s <= 0:
            raise ConfigValidationError("max_round_timeout_s must be > 0")
        if self.base_noise_std < 0:
            raise ConfigValidationError("base_noise_std must be >= 0")
        if not (0.0 <= self.disturbance_prob <= 1.0):
            raise ConfigValidationError("disturbance_prob must be in [0, 1]")
        if self.disturbance_scale < 0:
            raise ConfigValidationError("disturbance_scale must be >= 0")
        if self.adversarial_intensity < 0:
            raise ConfigValidationError("adversarial_intensity must be >= 0")
        if self.attack_cost < 0:
            raise ConfigValidationError("attack_cost must be >= 0")
        if self.convergence_threshold < 0:
            raise ConfigValidationError("convergence_threshold must be >= 0")
        if self.adversary_tau_init <= 0:
            raise ConfigValidationError("adversary_tau_init must be > 0")
        if self.adversary_tau_final <= 0:
            raise ConfigValidationError("adversary_tau_final must be > 0")
        if self.tau_decay_rate <= 0:
            raise ConfigValidationError("tau_decay_rate must be > 0")
        if not (0.0 < self.bankruptcy_threshold < 1.0):
            raise ConfigValidationError("bankruptcy_threshold must be in (0, 1)")
        if not (0.0 < self.wolfpack_correlation_threshold <= 1.0):
            raise ConfigValidationError("wolfpack_correlation_threshold must be in (0, 1]")
        if self.decay_rate < 0:
            raise ConfigValidationError("decay_rate must be >= 0")
        if self.feature_dim < 1:
            raise ConfigValidationError("feature_dim must be >= 1")
        if self.regime_classes < 1:
            raise ConfigValidationError("regime_classes must be >= 1")
        if self.max_context_tokens < 1:
            raise ConfigValidationError("max_context_tokens must be >= 1")
        if self.mnpo_eta <= 0:
            raise ConfigValidationError("mnpo_eta must be > 0")
        if self.mnpo_beta <= 0:
            raise ConfigValidationError("mnpo_beta must be > 0")
        if self.mnpo_population_size < 1:
            raise ConfigValidationError("mnpo_population_size must be >= 1")
        if self.marl_algorithm not in {"q", "wolf", "rarl"}:
            raise ConfigValidationError("marl_algorithm must be one of: q, wolf, rarl")
        if self.rl_backend not in {"tabular", "deep"}:
            raise ConfigValidationError("rl_backend must be one of: tabular, deep")
        if self.rl_algorithm not in {"dqn", "ppo"}:
            raise ConfigValidationError("rl_algorithm must be one of: dqn, ppo")
        if self.replay_buffer_size < 1:
            raise ConfigValidationError("replay_buffer_size must be >= 1")
        if self.rl_batch_size < 1:
            raise ConfigValidationError("rl_batch_size must be >= 1")
        if self.target_update_interval < 1:
            raise ConfigValidationError("target_update_interval must be >= 1")
        if not (0.0 <= self.epsilon_final <= 1.0):
            raise ConfigValidationError("epsilon_final must be in [0, 1]")
        if self.temperature_init <= 0:
            raise ConfigValidationError("temperature_init must be > 0")
        if self.temperature_final <= 0:
            raise ConfigValidationError("temperature_final must be > 0")
        if self.temperature_decay <= 0:
            raise ConfigValidationError("temperature_decay must be > 0")
        if not (0.0 <= self.poisoning_threshold <= 1.0):
            raise ConfigValidationError("poisoning_threshold must be in [0, 1]")


def decay_qualitative_state(
    raw: tuple[float, ...],
    current_step: int,
    release_step: int,
    decay_rate: float,
) -> tuple[int, ...]:
    """Apply exponential decay to a raw qualitative tensor and round to integers."""
    if release_step < 0 or not raw:
        return ()
    dt = max(0, current_step - release_step)
    factor = math.exp(-decay_rate * dt)
    return tuple(round(v * factor) for v in raw)


def evolve_state(
    state: ForecastState,
    *,
    base_trend: float,
    noise: float,
    disturbance: float,
    coeff_map: dict[str, float] | None = None,
    qual_override: dict[str, Any] | None = None,
    decay_rate: float = 0.01,
) -> ForecastState:
    """Advance *state* by one time step, applying trend, noise, and disturbance.

    Args:
        state: Current forecast state.
        base_trend: Deterministic trend component.
        noise: Stochastic noise component.
        disturbance: Adversarial or environmental disturbance.
        coeff_map: Optional macro-context coefficient mapping.
        qual_override: Optional qualitative field overrides.
        decay_rate: Exponential decay rate for qualitative state.
    """
    macro_contribution = 0.0
    if coeff_map and state.macro_context:
        for k, coeff in coeff_map.items():
            if k in state.macro_context:
                macro_contribution += coeff * state.macro_context[k]

    new_value = state.value + base_trend + 0.4 * state.exogenous + noise + disturbance + macro_contribution
    new_exogenous = 0.6 * state.exogenous + 0.2 * disturbance
    new_t = state.t + 1

    qual_fields: dict[str, Any] = {}
    if qual_override:
        qual_fields = dict(qual_override)
    elif state.raw_qual_state:
        qual_fields["qualitative_state"] = decay_qualitative_state(
            state.raw_qual_state, new_t, state.last_qual_update_step, decay_rate,
        )

    return replace(
        state,
        t=new_t,
        value=new_value,
        exogenous=new_exogenous,
        hidden_shift=disturbance,
        **qual_fields,
    )


validate_immutability(
    ForecastState, AgentAction, AgentMessage, ConfidenceInterval,
    ProbabilisticForecast, TrajectoryEntry, StepResult, SimulationConfig,
)
