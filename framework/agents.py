"""Agent definitions for the forecasting game: forecasters, adversaries, defenders, and utilities."""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, List, Literal, cast

import numpy as np

from .defenses import defense_from_name
from .llm import DSPyLikeRepl, LLMRefactorClient, MockLLMRefactorClient, OllamaClient, RefactorRequest
from .observability import record_marl_policy_loaded
from .strategy_runtime import StrategyRuntime
from .types import AgentAction, ForecastState


@dataclass(frozen=True)
class ForecastingAgent:
    """Primary forecasting agent that produces delta adjustments, optionally via LLM."""

    name: str = "forecaster"
    llm_repl: DSPyLikeRepl | None = None

    def act(self, state: ForecastState, runtime: StrategyRuntime, *, round_idx: int | None = None) -> AgentAction:
        """Produce a forecast delta for *state* using *runtime* and optional LLM refinement."""
        base_delta = runtime.forecast_delta(state)
        if self.llm_repl is None:
            return AgentAction(actor=self.name, delta=base_delta)

        prompt = f"Given value={state.value:.4f}, exogenous={state.exogenous:.4f}, suggest scalar delta"
        if state.raw_qual_text:
            prompt += f" | Qualitative: {state.raw_qual_text[:512]} | Regime: {state.economic_regime}"
        try:
            turn = self.llm_repl.run_turn(prompt, round_idx=round_idx, agent=self.name)
            llm_delta = float(turn["completion"].strip().split()[0])
            return AgentAction(actor=self.name, delta=(0.8 * base_delta) + (0.2 * llm_delta))
        except Exception:
            return AgentAction(actor=self.name, delta=base_delta)


@dataclass(frozen=True)
class QLearnedAgent(ForecastingAgent):
    """Forecasting or adversarial agent backed by a saved tabular MARL policy."""

    q_table_path: str | None = None
    algorithm: Literal["q", "wolf", "rarl"] = "q"
    _q_agent: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.q_table_path:
            return
        from .training import QTableAgent, RADversarialTrainer, WoLFPHCAgent

        loader: dict[str, Any] = {
            "q": QTableAgent,
            "wolf": WoLFPHCAgent,
            "rarl": RADversarialTrainer,
        }
        loaded = loader[self.algorithm].load(self.q_table_path)
        if hasattr(loaded, "epsilon"):
            loaded.epsilon = 0.0
        object.__setattr__(self, "_q_agent", loaded)
        record_marl_policy_loaded(self.algorithm)

    def act(
        self,
        state: ForecastState,
        runtime: StrategyRuntime | None = None,
        *,
        round_idx: int | None = None,
    ) -> AgentAction:
        if self._q_agent is None:
            if runtime is None:
                return AgentAction(actor=self.name, delta=0.0)
            return super().act(state, runtime, round_idx=round_idx)
        idx = self._q_agent.act(state)
        delta = self._q_agent.action_space.action_to_delta(idx)
        logging.getLogger(__name__).info("QLearnedAgent delta=%s actor=%s", f"{delta:.4f}", self.name)
        return AgentAction(actor=self.name, delta=delta)


@dataclass(frozen=True)
class AdversaryAgent:
    """Adversarial agent that injects directional noise to degrade forecasts."""

    name: str = "adversary"
    aggressiveness: float = 1.0
    attack_cost: float = 0.0

    def act(self, state: ForecastState) -> AgentAction:
        """Compute an adversarial delta opposing the expected trend."""
        expected_trend = 0.4 + 0.4 * state.exogenous
        direction = -1.0 if expected_trend >= 0 else 1.0
        base = direction * 0.4 * self.aggressiveness
        penalty = min(abs(base), self.attack_cost * 0.2)
        delta = base - (penalty if base > 0 else -penalty)
        return AgentAction(actor=self.name, delta=delta)


@dataclass(frozen=True)
class DefenderAgent:
    """Defensive agent that corrects forecasts against adversarial perturbations."""

    name: str = "defender"
    llm_client: OllamaClient | None = None

    def act(self, forecast_action: AgentAction, adversary_action: AgentAction, defense_model: str) -> AgentAction:
        """Compute a defensive correction using the named defense model or optional LLM."""
        if self.llm_client is not None and abs(adversary_action.delta) > 0.3:
            try:
                import json as _json
                state_json = _json.dumps({
                    "forecast_delta": forecast_action.delta,
                    "adversary_delta": adversary_action.delta,
                    "defense_model": defense_model,
                })
                prompt = (
                    f"Given adversarial state: {state_json}\n"
                    "Suggest a single float correction delta to defend the forecast. "
                    "Reply with ONLY a number."
                )
                raw = self.llm_client.generate(prompt, agent=self.name)
                from .llm.audit import get_llm_log
                llm_delta = float(raw.strip().split()[0])
                return AgentAction(actor=self.name, delta=llm_delta)
            except Exception:
                pass
        defense = defense_from_name(defense_model)
        correction = defense.defend(forecast_action.delta, adversary_action.delta)
        return AgentAction(actor=self.name, delta=correction)


@dataclass(frozen=True)
class RefactoringAgent:
    """Bias-correction agent that revises the forecast via error feedback or LLM suggestions."""

    name: str = "refactor"
    step_size: float = 0.02
    llm_client: LLMRefactorClient = field(default_factory=lambda: MockLLMRefactorClient(step_size=0.02))

    def revise(self, latest_error: float, *, use_llm: bool = False, round_idx: int | None = None) -> float:
        """Return a bias adjustment based on *latest_error*, optionally using LLM."""
        if use_llm:
            suggestion = self.llm_client.suggest(
                RefactorRequest(latest_error=latest_error, strategy_name=self.name),
                round_idx=round_idx,
                agent=self.name,
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
        regime_modifier = 1.0 + 0.1 * (state.economic_regime - 1)
        delta = self.macro_sensitivity * macro_signal * regime_modifier
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


@dataclass(frozen=True)
class StrategyVariant:
    """Parameterized strategy candidate used by evolutionary population updates."""

    name: str
    role: str
    proportion: float
    delta_scale: float
    delta_bias: float
    bankroll: float = 1.0
    subgroup: str = "default"


@dataclass(frozen=True)
class EvolutionaryAgentPopulation:
    """Immutable pool of strategy variants updated by replicator dynamics."""

    variants: tuple[StrategyVariant, ...]
    evolution_rate: float = 0.05
    prune_threshold: float = 0.1
    bankruptcy_floor: float = 0.0

    @classmethod
    def bootstrap(
        cls,
        *,
        population_size: int = 20,
        evolution_rate: float = 0.05,
        seed: int = 42,
        roles: tuple[str, ...] = ("forecaster", "adversary", "defender"),
        subgroup_count: int = 2,
    ) -> "EvolutionaryAgentPopulation":
        rng = Random(seed)
        variants: list[StrategyVariant] = []
        per_role = max(1, population_size // max(1, len(roles)))
        for role in roles:
            for idx in range(per_role):
                subgroup = f"group_{idx % max(1, subgroup_count)}"
                variants.append(
                    StrategyVariant(
                        name=f"{role}_variant_{idx}",
                        role=role,
                        proportion=1.0 / per_role,
                        delta_scale=max(0.25, rng.uniform(0.6, 1.4)),
                        delta_bias=rng.uniform(-0.15, 0.15),
                        subgroup=subgroup,
                    )
                )
        return cls(variants=tuple(variants), evolution_rate=evolution_rate)

    def variants_for(self, role: str, *, subgroup: str | None = None) -> tuple[StrategyVariant, ...]:
        items = tuple(v for v in self.variants if v.role == role and (subgroup is None or v.subgroup == subgroup))
        if items:
            return items
        return tuple(v for v in self.variants if v.role == role)

    def sample(self, role: str, rng: Random, *, subgroup: str | None = None) -> StrategyVariant | None:
        items = self.variants_for(role, subgroup=subgroup)
        if not items:
            return None
        weights = [max(1e-9, v.proportion) for v in items]
        return rng.choices(items, weights=weights, k=1)[0]

    @staticmethod
    def apply_variant(action: AgentAction, variant: StrategyVariant | None) -> AgentAction:
        if variant is None:
            return action
        return AgentAction(
            actor=action.actor,
            delta=(action.delta * variant.delta_scale) + variant.delta_bias,
        )

    def evolve(
        self,
        reward_history: dict[str, list[float]],
    ) -> "EvolutionaryAgentPopulation":
        updated: list[StrategyVariant] = []
        for role in sorted({variant.role for variant in self.variants}):
            role_variants = [variant for variant in self.variants if variant.role == role]
            if not role_variants:
                continue
            raw_fitness = np.asarray(
                [float(np.mean(reward_history.get(variant.name, [0.0]))) for variant in role_variants],
                dtype=float,
            )
            if raw_fitness.size == 0:
                continue
            min_fit = float(raw_fitness.min())
            max_fit = float(raw_fitness.max())
            if max_fit - min_fit < 1e-12:
                normalized = np.ones_like(raw_fitness)
            else:
                normalized = (raw_fitness - min_fit) / (max_fit - min_fit)
            base_props = np.asarray([variant.proportion for variant in role_variants], dtype=float)
            centered = normalized - float(normalized.mean())
            next_props = base_props * (1.0 + (self.evolution_rate * centered))
            next_props = np.clip(next_props, 1e-9, None)
            next_props = next_props / max(next_props.sum(), 1e-12)
            survivors: list[StrategyVariant] = []
            for variant, fitness, proportion in zip(role_variants, normalized, next_props):
                edge = (2.0 * fitness) - 1.0
                kelly_fraction = max(-0.95, min(edge, 1.0))
                bankroll = variant.bankroll * max(0.0, 1.0 + kelly_fraction * self.evolution_rate)
                if fitness < self.prune_threshold or bankroll <= self.bankruptcy_floor:
                    continue
                survivors.append(
                    StrategyVariant(
                        name=variant.name,
                        role=variant.role,
                        proportion=float(proportion),
                        delta_scale=variant.delta_scale,
                        delta_bias=variant.delta_bias,
                        bankroll=float(bankroll),
                        subgroup=variant.subgroup,
                    )
                )
            if not survivors:
                best_idx = int(np.argmax(normalized))
                best = role_variants[best_idx]
                survivors = [
                    StrategyVariant(
                        name=best.name,
                        role=best.role,
                        proportion=1.0,
                        delta_scale=best.delta_scale,
                        delta_bias=best.delta_bias,
                        bankroll=max(best.bankroll, 1e-6),
                        subgroup=best.subgroup,
                    )
                ]
            else:
                total = sum(item.proportion for item in survivors) or 1.0
                survivors = [
                    StrategyVariant(
                        name=item.name,
                        role=item.role,
                        proportion=item.proportion / total,
                        delta_scale=item.delta_scale,
                        delta_bias=item.delta_bias,
                        bankroll=item.bankroll,
                        subgroup=item.subgroup,
                    )
                    for item in survivors
                ]
            updated.extend(survivors)
        return EvolutionaryAgentPopulation(
            variants=tuple(updated),
            evolution_rate=self.evolution_rate,
            prune_threshold=self.prune_threshold,
            bankruptcy_floor=self.bankruptcy_floor,
        )


# ---------------------------------------------------------------------------
# Agent registry and executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRegistry:
    """Flexible container for variable numbers of agents."""

    forecasters: tuple[ForecastingAgent | BottomUpAgent | TopDownAgent | QLearnedAgent, ...] = ()
    adversaries: tuple[AdversaryAgent | WolfpackAdversary | QLearnedAgent, ...] = ()
    defenders: tuple[DefenderAgent, ...] = ()
    refactorer: RefactoringAgent | None = None
    aggregator: EnsembleAggregatorAgent = field(default_factory=EnsembleAggregatorAgent)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AgentRegistry:
        """Build a registry from an agent list config.

        Example:
            {"agents": ["forecaster", "qlearned-adversary", "defender"]}
        """
        specs = config.get("agents", ("forecaster", "adversary", "defender"))

        forecasters: list[ForecastingAgent | BottomUpAgent | TopDownAgent | QLearnedAgent] = []
        adversaries: list[AdversaryAgent | WolfpackAdversary | QLearnedAgent] = []
        defenders: list[DefenderAgent] = []
        refactorer: RefactoringAgent | None = None

        for item in specs:
            if isinstance(item, str):
                agent_type = item
                name = item
                kwargs: dict[str, Any] = {}
            else:
                agent_type = item.get("type", "forecaster")
                name = item.get("name", agent_type)
                kwargs = dict(item.get("kwargs", {}))

            if agent_type == "qlearned-adversary":
                agent_type = "qlearned"
                kwargs["role"] = "adversary"
            elif agent_type == "qlearned-forecaster":
                agent_type = "qlearned"
                kwargs["role"] = "forecaster"

            role = str(kwargs.pop("role", ""))
            agent = create_agent(agent_type=agent_type, name=name, **kwargs)

            if role == "adversary":
                adversaries.append(agent)
                continue
            if role == "forecaster":
                forecasters.append(agent)
                continue

            if isinstance(agent, DefenderAgent):
                defenders.append(agent)
            elif isinstance(agent, RefactoringAgent):
                refactorer = agent
            elif isinstance(agent, (AdversaryAgent, WolfpackAdversary)):
                adversaries.append(agent)
            elif isinstance(agent, QLearnedAgent) and "adversary" in name:
                adversaries.append(agent)
            else:
                forecasters.append(agent)

        return cls(
            forecasters=tuple(forecasters),
            adversaries=tuple(adversaries),
            defenders=tuple(defenders),
            refactorer=refactorer,
            aggregator=EnsembleAggregatorAgent(mode=str(config.get("aggregator_mode", "equal"))),
        )


def create_agent(agent_type: str, name: str, **kwargs: Any) -> Any:
    """Create an agent instance from a short type string."""
    if agent_type == "forecaster":
        return ForecastingAgent(name=name)
    if agent_type == "adversary":
        return AdversaryAgent(
            name=name,
            aggressiveness=float(kwargs.get("aggressiveness", 1.0)),
            attack_cost=float(kwargs.get("attack_cost", 0.0)),
        )
    if agent_type == "defender":
        return DefenderAgent(name=name)
    if agent_type == "refactor":
        return RefactoringAgent(name=name)
    if agent_type == "bottom_up":
        return BottomUpAgent(name=name, segment_weight=float(kwargs.get("segment_weight", 0.3)))
    if agent_type == "top_down":
        return TopDownAgent(name=name, macro_sensitivity=float(kwargs.get("macro_sensitivity", 0.2)))
    if agent_type == "wolfpack":
        return WolfpackAdversary(
            name=name,
            aggressiveness=float(kwargs.get("aggressiveness", 1.0)),
            attack_cost=float(kwargs.get("attack_cost", 0.0)),
            correlation_threshold=float(kwargs.get("correlation_threshold", 0.7)),
        )
    if agent_type == "qlearned":
        algorithm = cast(Literal["q", "wolf", "rarl"], str(kwargs.get("algorithm", "q")))
        return QLearnedAgent(
            name=name,
            q_table_path=kwargs.get("q_table_path") or kwargs.get("q_table"),
            algorithm=algorithm,
        )
    raise ValueError(f"Unknown agent_type: {agent_type}")


@dataclass(frozen=True)
class SafeAgentExecutor:
    """Exception-safe wrapper that falls back to a zero-delta action on failure."""

    fallback_delta: float = 0.0

    def execute(self, fn: Callable[..., AgentAction], *args: Any, **kwargs: Any) -> AgentAction:
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

    def act(self, state: ForecastState, runtime: "StrategyRuntime", trajectories: list[Any] | None = None, round_idx: int = 0) -> AgentAction:
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
    """Create a DSPyLikeRepl backed by the default OllamaClient."""
    return DSPyLikeRepl(client=OllamaClient())
