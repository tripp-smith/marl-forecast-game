"""Multi-agent forecasting game engine with adversarial disturbances and defenses."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import logging
from random import Random
import time
from typing import Any, Callable, List

import numpy as np

from .agents import (
    AdversaryAgent,
    AgentRegistry,
    BottomUpAgent,
    DefenderAgent,
    EvolutionaryAgentPopulation,
    EnsembleAggregatorAgent,
    ForecastingAgent,
    RefactoringAgent,
    SafeAgentExecutor,
    TopDownAgent,
    WolfpackAdversary,
)
from .aggregation import BayesianAggregator
from .disturbances import disturbance_from_name
from .equilibria import BayesianBeliefState, bayesian_likelihood_from_observation, compute_correlated_equilibrium
from .topology import CoalitionTopologyManager
from .observability import (
    ROUND_COUNTER,
    ROUND_LATENCY,
    GameObserver,
    create_span,
    record_agent_metrics,
    record_alert,
    record_disturbance,
)
from .llm.audit import get_llm_log
from .strategy_runtime import runtime_from_name
from .types import (
    AgentAction,
    AgentMessage,
    ConfidenceInterval,
    ForecastState,
    ProbabilisticForecast,
    SimulationConfig,
    StepResult,
    TrajectoryEntry,
    evolve_state,
    frozen_mapping,
)


AgentFactory = Callable[[SimulationConfig], tuple[ForecastingAgent, AdversaryAgent, DefenderAgent, RefactoringAgent]]


def _state_to_dict(state: ForecastState) -> dict[str, Any]:
    return {
        "t": state.t,
        "value": state.value,
        "exogenous": state.exogenous,
        "hidden_shift": state.hidden_shift,
        "segment_id": state.segment_id,
        "segment_values": dict(state.segment_values),
        "macro_context": dict(state.macro_context),
        "qualitative_state": list(state.qualitative_state),
        "raw_qual_text": state.raw_qual_text,
        "economic_regime": state.economic_regime,
        "last_qual_update_step": state.last_qual_update_step,
    }


@dataclass(frozen=True)
class RoundActions:
    """Per-round action bundle produced before state evolution."""

    all_forecast_actions: tuple[AgentAction, ...]
    forecast_action: AgentAction
    adversary_action: AgentAction
    defender_action: AgentAction
    forecast_variants: tuple[tuple[str, str], ...] = ()
    adversary_variant_name: str = ""
    defender_variant_name: str = ""
    wolfpack_adversaries: tuple[WolfpackAdversary, ...] = ()
    quarantined: bool = False


@dataclass(frozen=True)
class RoundOutcome:
    """Forecast and state-transition artifacts for a single round."""

    disturbance: float
    sabotage_penalty: float
    forecast: float
    next_state: ForecastState
    target: float


@dataclass(frozen=True)
class RewardUpdate:
    """Reward, error, and bias updates derived from a round transition."""

    error: float
    reward: float
    reward_breakdown: dict[str, float]
    refactor_bias: float


@dataclass(frozen=True)
class GameOutputs:
    """Immutable container for all simulation outputs: steps, trajectories, and convergence info."""

    steps: List[StepResult]
    trajectories: List[TrajectoryEntry]
    trajectory_logs: List[dict[str, Any]]
    forecasts: List[float]
    targets: List[float]
    confidence: List[ConfidenceInterval]
    convergence: dict[str, Any]
    llm_calls: List[dict[str, Any]] = ()  # type: ignore[assignment]
    wall_clock_s: float = 0.0
    evolutionary_population: EvolutionaryAgentPopulation | None = None
    coalition_graph: dict[str, Any] | None = None


def default_agent_factory(config: SimulationConfig) -> tuple[ForecastingAgent, AdversaryAgent, DefenderAgent, RefactoringAgent]:
    """Create the default set of agents from a simulation config."""
    return (
        ForecastingAgent(),
        AdversaryAgent(aggressiveness=config.adversarial_intensity, attack_cost=config.attack_cost),
        DefenderAgent(),
        RefactoringAgent(),
    )


def _build_registry(config: SimulationConfig, factory: AgentFactory) -> AgentRegistry:
    """Convert a legacy 4-tuple AgentFactory result into an AgentRegistry."""
    forecaster, adversary, defender, refactorer = factory(config)
    return AgentRegistry(
        forecasters=(forecaster,),
        adversaries=(adversary,),
        defenders=(defender,),
        refactorer=refactorer,
        aggregator=EnsembleAggregatorAgent(),
    )


class ForecastGame:
    """Orchestrates multi-agent forecast rounds with adversarial disturbances and defenses."""

    def __init__(
        self,
        config: SimulationConfig,
        seed: int = 7,
        *,
        agent_factory: AgentFactory = default_agent_factory,
        registry: AgentRegistry | None = None,
        evolutionary_population: EvolutionaryAgentPopulation | None = None,
    ):
        self.config = config
        self._rng = Random(seed)
        self.runtime = runtime_from_name(config.runtime_backend)
        self.disturbance = disturbance_from_name(config.disturbance_model)

        if registry is not None:
            self._registry = registry
        else:
            self._registry = _build_registry(config, agent_factory)

        self.forecaster = self._registry.forecasters[0] if self._registry.forecasters else ForecastingAgent()
        self.adversary = self._registry.adversaries[0] if self._registry.adversaries else AdversaryAgent()
        self.defender = self._registry.defenders[0] if self._registry.defenders else DefenderAgent()
        self.refactor = self._registry.refactorer or RefactoringAgent()

        self._seed = seed
        self.safe_exec = SafeAgentExecutor()
        _logger = GameObserver().logger()
        try:
            self.logger = _logger.bind(simulation_seed=seed)
        except (AttributeError, TypeError):
            self.logger = _logger
        self.bayesian_agg = BayesianAggregator()
        self.evolutionary_population = evolutionary_population
        if self.evolutionary_population is None and config.dynamics == "evolutionary":
            self.evolutionary_population = EvolutionaryAgentPopulation.bootstrap(
                population_size=config.population_size,
                evolution_rate=config.evolution_rate,
                seed=seed,
            )
        labels = tuple(f"type_{idx}" for idx, _ in enumerate(config.prior_alpha[:-1])) + ("adversarial",)
        self._belief_state = BayesianBeliefState(labels=labels, alpha=config.prior_alpha)
        self._topology = CoalitionTopologyManager(reform_interval=config.coalition_reform_interval)

        self._start_date_offset: int = 0
        self._start_date_parsed: datetime | None = None
        if config.start_date:
            try:
                self._start_date_parsed = datetime.fromisoformat(config.start_date)
            except (ValueError, TypeError):
                self._start_date_parsed = None

        self._qual_extractor = None
        self._regime_classifier = None
        self._qual_dataset: dict[int, dict[str, Any]] = {}
        if config.enable_qual:
            try:
                from .qualitative import QualitativeExtractor, RegimeClassifier

                self._qual_extractor = QualitativeExtractor(
                    feature_dim=config.feature_dim,
                )
                self._regime_classifier = RegimeClassifier(
                    n_regimes=config.regime_classes,
                )
            except Exception:
                logging.debug("Qualitative components unavailable; running without")

    def set_qual_dataset(self, dataset: dict[int, dict[str, Any]]) -> None:
        """Inject a pre-built qualitative dataset (timestep -> record mapping)."""
        self._qual_dataset = dict(dataset)

    def set_dataset_bundle(self, rows: list[dict[str, Any]]) -> None:
        """Set dataset rows and compute start_date offset if configured."""
        if self._start_date_parsed is not None and rows:
            for idx, row in enumerate(rows):
                ts = row.get("timestamp")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                if ts is not None and ts >= self._start_date_parsed:
                    self._start_date_offset = idx
                    break

    def run(self, initial: ForecastState, rounds: int | None = None, *, disturbed: bool = True) -> GameOutputs:
        """Execute the simulation from *initial* for up to *rounds* steps.

        Args:
            initial: Starting forecast state.
            rounds: Number of rounds (defaults to ``config.horizon``).
            disturbed: Whether adversarial disturbances are active.
        """
        requested_rounds = rounds if rounds is not None else self.config.horizon
        n_rounds = max(0, min(requested_rounds, self.config.max_rounds))

        with create_span("simulation.run", {"seed": self._seed, "disturbed": disturbed, "n_rounds": n_rounds}):
            return self._run_inner(initial, n_rounds, disturbed=disturbed)

    def _compute_actions(
        self,
        state: ForecastState,
        *,
        round_idx: int,
        disturbed: bool,
        cumulative_rewards: dict[str, float],
        coalition_payload: tuple[tuple[str, ...], ...],
        wolfpack_residuals: dict[str, list[float]],
    ) -> RoundActions:
        """Compute forecast, adversary, and defender actions for a round.

        Args:
            state: Current immutable simulation state.
            round_idx: Current round index.
            disturbed: Whether disturbances are enabled for the run.
            cumulative_rewards: Running reward totals keyed by agent id.
            coalition_payload: Dynamic coalition membership for the round.
            wolfpack_residuals: Residual history keyed by forecaster id.

        Returns:
            Fully materialized round actions including metadata needed downstream.
        """
        all_forecast_actions: list[AgentAction] = []
        forecast_variants: list[tuple[str, str]] = []
        for forecast_agent in self._registry.forecasters:
            if isinstance(forecast_agent, TopDownAgent):
                action = self.safe_exec.execute(forecast_agent.act, state)
            elif isinstance(forecast_agent, BottomUpAgent):
                action = self.safe_exec.execute(forecast_agent.act, state, self.runtime)
            else:
                action = self.safe_exec.execute(forecast_agent.act, state, self.runtime, round_idx=round_idx)
            if self.evolutionary_population is not None:
                subgroup = coalition_payload[0][0] if coalition_payload else None
                variant = self.evolutionary_population.sample("forecaster", self._rng, subgroup=subgroup)
                action = self.evolutionary_population.apply_variant(action, variant)
                if variant is not None:
                    forecast_variants.append((action.actor, variant.name))
            all_forecast_actions.append(action)

        adversary_variant_name = ""
        wolfpack_adversaries = tuple(
            adversary for adversary in self._registry.adversaries if isinstance(adversary, WolfpackAdversary)
        )
        if disturbed and wolfpack_adversaries and len(all_forecast_actions) > 1:
            wolf = wolfpack_adversaries[0]
            if self.bayesian_agg.weights:
                primary_idx = max(range(len(self.bayesian_agg.weights)), key=lambda i: self.bayesian_agg.weights[i])
                primary_name = self.bayesian_agg.agent_names[primary_idx]
            else:
                primary_name = all_forecast_actions[0].actor
            _, coalition = wolf.select_targets(primary_name, wolfpack_residuals)
            targeted = {primary_name, *coalition}
            perturbed_actions: list[AgentAction] = []
            for forecast_action in all_forecast_actions:
                if forecast_action.actor in targeted:
                    is_primary = forecast_action.actor == primary_name
                    wolf_action = wolf.act(state, is_primary=is_primary)
                    perturbed_actions.append(
                        AgentAction(actor=forecast_action.actor, delta=forecast_action.delta + wolf_action.delta)
                    )
                else:
                    perturbed_actions.append(forecast_action)
            all_forecast_actions = perturbed_actions
            adversary_action = AgentAction(actor=wolf.name, delta=0.0)
        elif disturbed:
            adversary_actions = [self.safe_exec.execute(adversary.act, state) for adversary in self._registry.adversaries]
            if self.evolutionary_population is not None and adversary_actions:
                adjusted_adversary_actions: list[AgentAction] = []
                for adversary_action in adversary_actions:
                    subgroup = coalition_payload[-1][0] if coalition_payload else None
                    variant = self.evolutionary_population.sample("adversary", self._rng, subgroup=subgroup)
                    adjusted_adversary_actions.append(self.evolutionary_population.apply_variant(adversary_action, variant))
                    if variant is not None:
                        adversary_variant_name = variant.name
                adversary_actions = adjusted_adversary_actions
            adversary_action = (
                adversary_actions[0]
                if len(adversary_actions) == 1
                else AgentAction(
                    actor="adversary",
                    delta=sum(action.delta for action in adversary_actions) / max(1, len(adversary_actions)),
                )
            )
        else:
            adversary_action = AgentAction(
                actor=self._registry.adversaries[0].name if self._registry.adversaries else "adversary",
                delta=0.0,
            )

        forecast_action = (
            all_forecast_actions[0]
            if len(all_forecast_actions) == 1
            else self._registry.aggregator.aggregate(all_forecast_actions, cumulative_rewards)
        )
        defender_actions = [
            self.safe_exec.execute(defender.act, forecast_action, adversary_action, self.config.defense_model)
            for defender in self._registry.defenders
        ]
        defender_variant_name = ""
        if self.evolutionary_population is not None and defender_actions:
            adjusted_defender_actions: list[AgentAction] = []
            for defender_action in defender_actions:
                variant = self.evolutionary_population.sample("defender", self._rng)
                adjusted_defender_actions.append(self.evolutionary_population.apply_variant(defender_action, variant))
                if variant is not None:
                    defender_variant_name = variant.name
            defender_actions = adjusted_defender_actions
        defender_action = (
            defender_actions[0]
            if len(defender_actions) == 1
            else AgentAction(
                actor="defender",
                delta=sum(action.delta for action in defender_actions) / max(1, len(defender_actions)),
            )
        )

        if self.config.equilibrium_type == "correlated":
            target_proxy = state.value + 0.4 + (0.4 * state.exogenous)
            f_candidates = np.array(
                [0.8 * forecast_action.delta, forecast_action.delta, 1.2 * forecast_action.delta],
                dtype=float,
            )
            a_candidates = np.array(
                [0.8 * adversary_action.delta, adversary_action.delta, 1.2 * adversary_action.delta],
                dtype=float,
            )
            forecaster_payoff = np.zeros((len(f_candidates), len(a_candidates)))
            adversary_payoff = np.zeros_like(forecaster_payoff)
            for left_idx, forecast_delta in enumerate(f_candidates):
                for right_idx, adversary_delta in enumerate(a_candidates):
                    proxy_error = abs(
                        target_proxy - (state.value + forecast_delta + adversary_delta + defender_action.delta)
                    )
                    forecaster_payoff[left_idx, right_idx] = -proxy_error
                    adversary_payoff[left_idx, right_idx] = proxy_error
            ce = compute_correlated_equilibrium((forecaster_payoff, adversary_payoff))
            forecast_idx, adversary_idx = ce.sample_actions(self._rng)
            forecast_action = AgentAction(actor=forecast_action.actor, delta=float(f_candidates[forecast_idx]))
            adversary_action = AgentAction(actor=adversary_action.actor, delta=float(a_candidates[adversary_idx]))

        quarantined = False
        if self.config.equilibrium_type == "bayesian":
            posterior_adv = self._belief_state.posterior[-1]
            if posterior_adv >= self.config.quarantine_threshold:
                quarantined = True
                adversary_action = AgentAction(actor=adversary_action.actor, delta=0.0)

        return RoundActions(
            all_forecast_actions=tuple(all_forecast_actions),
            forecast_action=forecast_action,
            adversary_action=adversary_action,
            defender_action=defender_action,
            forecast_variants=tuple(forecast_variants),
            adversary_variant_name=adversary_variant_name,
            defender_variant_name=defender_variant_name,
            wolfpack_adversaries=wolfpack_adversaries,
            quarantined=quarantined,
        )

    def _apply_disturbances(
        self,
        state: ForecastState,
        actions: RoundActions,
        *,
        refactor_bias: float,
        disturbed: bool,
    ) -> RoundOutcome:
        """Apply disturbances and evolve the state for a round.

        Args:
            state: Current immutable simulation state.
            actions: Pre-computed actions for the round.
            refactor_bias: Running bias correction carried into the round.
            disturbed: Whether disturbances are enabled.

        Returns:
            Forecast values and the resulting next state.
        """
        disturbance_val = self.disturbance.sample(state, self._rng, self.config) if disturbed else 0.0
        forecast = (
            state.value
            + actions.forecast_action.delta
            + actions.adversary_action.delta
            + actions.defender_action.delta
            + refactor_bias
        )
        sabotage_penalty = 0.0
        if disturbed and self.config.sabotage_prob > 0.0 and self._rng.random() <= self.config.sabotage_prob:
            sabotage_penalty = self._rng.uniform(-0.2, 0.2) * max(
                1.0,
                abs(actions.forecast_action.delta) + abs(actions.adversary_action.delta),
            )
            forecast += sabotage_penalty
        noise = self._rng.gauss(0, self.config.base_noise_std)

        qual_override: dict[str, Any] | None = None
        if self.config.enable_qual and self._qual_dataset and state.t in self._qual_dataset:
            qual_record = self._qual_dataset[state.t]
            qual_text = qual_record.get("text", "")
            if self._qual_extractor is not None:
                raw_tensor = self._qual_extractor.extract(
                    qual_text,
                    self.config.qualitative_extractor_prompt,
                )
            else:
                raw_tensor = (0,) * self.config.feature_dim
            regime = 0
            if self._regime_classifier is not None:
                regime = self._regime_classifier.classify(
                    dict(state.macro_context),
                    raw_tensor,
                    self.config.regime_prompt,
                )
            qual_override = {
                "qualitative_state": raw_tensor,
                "raw_qual_state": tuple(float(value) for value in raw_tensor),
                "raw_qual_text": qual_text[: self.config.max_context_tokens] if qual_text else None,
                "economic_regime": regime,
                "last_qual_update_step": state.t,
            }

        next_state = evolve_state(
            state,
            base_trend=0.4,
            noise=noise,
            disturbance=disturbance_val,
            qual_override=qual_override,
            decay_rate=self.config.decay_rate,
        )
        return RoundOutcome(
            disturbance=disturbance_val,
            sabotage_penalty=sabotage_penalty,
            forecast=forecast,
            next_state=next_state,
            target=next_state.value,
        )

    def _update_rewards_and_bias(
        self,
        state: ForecastState,
        actions: RoundActions,
        outcome: RoundOutcome,
        *,
        round_idx: int,
        refactor_bias: float,
        previous_abs_error: float,
        wolfpack_residuals: dict[str, list[float]],
    ) -> RewardUpdate:
        """Compute rewards, update bias, and refresh Bayesian aggregation state.

        Args:
            state: Round-start simulation state.
            actions: Action bundle used for the round.
            outcome: Forecast and transition results for the round.
            round_idx: Current round index.
            refactor_bias: Bias entering the round.
            previous_abs_error: Absolute error from the prior round.
            wolfpack_residuals: Residual history keyed by forecasting agent id.

        Returns:
            Reward update payload containing the new error and bias.
        """
        error = outcome.target - outcome.forecast
        reward = -abs(error)
        benign_mass, suspicious_mass = bayesian_likelihood_from_observation(
            actions.adversary_action.delta,
            outcome.disturbance + outcome.sabotage_penalty,
        )
        if len(self._belief_state.alpha) > 2:
            base = benign_mass / max(1, len(self._belief_state.alpha) - 1)
            likelihood = tuple(base for _ in range(len(self._belief_state.alpha) - 1)) + (suspicious_mass,)
        else:
            likelihood = (benign_mass, suspicious_mass)
        self._belief_state = self._belief_state.update(likelihood)

        record_agent_metrics(actions.forecast_action.actor, "forecaster", actions.forecast_action.delta, max(0.0, reward))
        record_agent_metrics(
            actions.adversary_action.actor,
            "adversary",
            actions.adversary_action.delta,
            max(0.0, -reward),
        )
        record_agent_metrics(actions.defender_action.actor, "defender", actions.defender_action.delta, max(0.0, reward))
        record_disturbance(outcome.disturbance != 0.0, abs(error) > previous_abs_error)

        if len(actions.all_forecast_actions) > 1:
            agent_errors: dict[str, float] = {}
            agent_means: dict[str, float] = {}
            agent_stds: dict[str, float] = {}
            for forecast_action in actions.all_forecast_actions:
                forecast_value = state.value + forecast_action.delta
                agent_means[forecast_action.actor] = forecast_value
                agent_stds[forecast_action.actor] = self.config.base_noise_std
                agent_errors[forecast_action.actor] = outcome.target - forecast_value
            self.bayesian_agg.update(
                agent_errors,
                means=agent_means,
                stds=agent_stds,
                bankruptcy_threshold=self.config.bankruptcy_threshold,
            )
            for forecast_action in actions.all_forecast_actions:
                wolfpack_residuals.setdefault(forecast_action.actor, []).append(agent_errors[forecast_action.actor])

        next_refactor_bias = refactor_bias
        if self.config.enable_refactor and self.refactor is not None:
            next_refactor_bias += self.refactor.revise(
                error,
                use_llm=self.config.enable_llm_refactor,
                round_idx=round_idx,
            )

        return RewardUpdate(
            error=error,
            reward=reward,
            reward_breakdown={
                "forecaster": reward,
                "adversary": -reward,
                "defender": reward,
            },
            refactor_bias=next_refactor_bias,
        )

    def _run_inner(self, initial: ForecastState, n_rounds: int, *, disturbed: bool) -> GameOutputs:
        wall_start = time.perf_counter()
        llm_log = get_llm_log()
        llm_log.clear()

        state = initial
        steps: List[StepResult] = []
        trajectories: List[TrajectoryEntry] = []
        trajectory_logs: List[dict[str, Any]] = []
        forecasts: List[float] = []
        targets: List[float] = []
        confidence: List[ConfidenceInterval] = []
        refactor_bias = 0.0
        cumulative_rewards: dict[str, float] = {}
        variant_rewards: dict[str, list[float]] = {}
        rolling_errors: List[float] = []
        convergence_reason: str = "completed"
        prev_abs_error: float = 0.0
        cap_hits: int = 0
        coalition_graph: dict[str, Any] | None = None
        wolfpack_residuals: dict[str, list[float]] = {}

        for idx in range(n_rounds):
            start = time.perf_counter()

            coalition_payload: tuple[tuple[str, ...], ...] = ()
            if self.config.coalitions == "dynamic":
                known_agents = [agent.name for agent in self._registry.forecasters] + [agent.name for agent in self._registry.adversaries]
                utilities = {name: cumulative_rewards.get(name, 0.0) for name in known_agents}
                affinities: dict[tuple[str, str], float] = {}
                for left in known_agents:
                    for right in known_agents:
                        if left >= right:
                            continue
                        affinities[(left, right)] = 1.0 / (1.0 + abs(utilities[left] - utilities[right]))
                coalition_payload = self._topology.reform(
                    utilities,
                    affinities,
                    round_idx=idx,
                    dynamic=True,
                )
                coalition_graph = self._topology.graph_payload()

            actions = self._compute_actions(
                state,
                round_idx=idx,
                disturbed=disturbed,
                cumulative_rewards=cumulative_rewards,
                coalition_payload=coalition_payload,
                wolfpack_residuals=wolfpack_residuals,
            )
            outcome = self._apply_disturbances(state, actions, refactor_bias=refactor_bias, disturbed=disturbed)
            update = self._update_rewards_and_bias(
                state,
                actions,
                outcome,
                round_idx=idx,
                refactor_bias=refactor_bias,
                previous_abs_error=prev_abs_error,
                wolfpack_residuals=wolfpack_residuals,
            )
            reward_breakdown = frozen_mapping(update.reward_breakdown)
            previous_abs_error = prev_abs_error
            prev_abs_error = abs(update.error)

            rolling_errors.append(abs(update.error))
            if self.config.convergence_threshold > 0 and len(rolling_errors) >= 20:
                window = rolling_errors[-20:]
                rolling_mae = sum(window) / len(window)
                if rolling_mae > self.config.convergence_threshold:
                    convergence_reason = "divergence_threshold_exceeded"
                    record_alert("mae_exceeds_threshold")

            for forecast_action in actions.all_forecast_actions:
                cumulative_rewards[forecast_action.actor] = cumulative_rewards.get(forecast_action.actor, 0.0) + update.reward
            for _, variant_name in actions.forecast_variants:
                variant_rewards.setdefault(variant_name, []).append(update.reward)
            if actions.adversary_variant_name:
                variant_rewards.setdefault(actions.adversary_variant_name, []).append(-update.reward)
            if actions.defender_variant_name:
                variant_rewards.setdefault(actions.defender_variant_name, []).append(update.reward)

            band = abs(outcome.disturbance) + self.config.base_noise_std + 0.05
            ci = ConfidenceInterval(lower=outcome.forecast - band, upper=outcome.forecast + band)
            messages = (
                AgentMessage("forecaster", "adversary", f"proposal={actions.forecast_action.delta:.4f}"),
                AgentMessage("adversary", "defender", f"attack={actions.adversary_action.delta:.4f}"),
                AgentMessage("defender", "refactor", f"defense={actions.defender_action.delta:.4f}"),
            )

            step = StepResult(
                next_state=outcome.next_state,
                actions=(actions.forecast_action, actions.adversary_action, actions.defender_action),
                reward_breakdown=reward_breakdown,
                forecast=outcome.forecast,
                target=outcome.target,
                confidence=ci,
                messages=messages,
            )
            traj = TrajectoryEntry(
                round_idx=idx,
                state=state,
                actions=(actions.forecast_action, actions.adversary_action, actions.defender_action),
                messages=messages,
                reward_breakdown=reward_breakdown,
                forecast=outcome.forecast,
                target=outcome.target,
            )
            elapsed = time.perf_counter() - start
            if elapsed > self.config.max_round_timeout_s:
                cap_hits += 1
                try:
                    self.logger.warning("round_timeout", round_idx=idx, elapsed=elapsed)
                except TypeError:
                    self.logger.warning(f"round_timeout round_idx={idx} elapsed={elapsed:.6f}")
                break

            if ROUND_COUNTER is not None:
                ROUND_COUNTER.inc()
            if ROUND_LATENCY is not None:
                ROUND_LATENCY.observe(elapsed)
            try:
                self.logger.info("round_complete", round_idx=idx, reward=update.reward, disturbance=outcome.disturbance)
            except TypeError:
                self.logger.info(
                    f"round_complete round_idx={idx} reward={update.reward:.6f} disturbance={outcome.disturbance:.6f}"
                )

            cost_this_round = abs(actions.adversary_action.delta) * self.config.attack_cost
            dist_magnitude = abs(outcome.disturbance)
            error_delta_val = abs(abs(update.error) - previous_abs_error) if idx > 0 else abs(update.error)
            cost_benefit = {
                "round": idx,
                "attack_cost": cost_this_round,
                "disturbance_magnitude": dist_magnitude,
                "error_delta": error_delta_val,
                "cost_benefit_ratio": error_delta_val / max(1e-9, cost_this_round) if cost_this_round > 0 else 0.0,
            }

            bma_snapshot = (
                list(self.bayesian_agg.weights)
                if len(actions.all_forecast_actions) > 1 and self.bayesian_agg.weights
                else None
            )
            trajectory_logs.append(
                {
                    "round_idx": idx,
                    "state": _state_to_dict(state),
                    "actions": [
                        asdict(action)
                        for action in (actions.forecast_action, actions.adversary_action, actions.defender_action)
                    ],
                    "forecast": outcome.forecast,
                    "target": outcome.target,
                    "reward": update.reward,
                    "disturbance": outcome.disturbance,
                    "messages": [asdict(m) for m in messages],
                    "elapsed_s": elapsed,
                    "bma_weights": bma_snapshot,
                    "confidence": {"lower": ci.lower, "upper": ci.upper},
                    "cost_benefit": cost_benefit,
                    "coalitions": [list(group) for group in coalition_payload],
                    "quarantined": actions.quarantined,
                    "sabotage_penalty": outcome.sabotage_penalty,
                    "posterior": list(self._belief_state.posterior),
                }
            )
            steps.append(step)
            trajectories.append(traj)
            forecasts.append(outcome.forecast)
            targets.append(outcome.target)
            confidence.append(ci)
            state = outcome.next_state
            refactor_bias = update.refactor_bias

            if convergence_reason != "completed":
                break

        total_attack_cost = sum(
            abs(log["actions"][1]["delta"]) * self.config.attack_cost
            for log in trajectory_logs
        )
        clean_errors = [abs(log["target"] - log["forecast"]) for log in trajectory_logs]
        mean_error = sum(clean_errors) / max(1, len(clean_errors))
        defense_efficacy = 1.0 - (mean_error / max(1e-9, mean_error + total_attack_cost))

        if rolling_errors and len(rolling_errors) >= 20:
            window = rolling_errors[-20:]
            epsilon_convergence = sum(window) / len(window)
        elif rolling_errors:
            epsilon_convergence = sum(rolling_errors) / len(rolling_errors)
        else:
            epsilon_convergence = 0.0

        convergence = {
            "rounds_executed": len(steps),
            "max_rounds": self.config.max_rounds,
            "round_cap_hit": len(steps) == self.config.max_rounds,
            "reason": convergence_reason,
            "attack_cost_total": total_attack_cost,
            "defense_efficacy_ratio": defense_efficacy,
            "accuracy_vs_cost": mean_error / max(1e-9, total_attack_cost) if total_attack_cost > 0 else 0.0,
            "epsilon_convergence": epsilon_convergence,
            "cap_hits": cap_hits,
            "coalition_modularity": self._topology.modularity(),
            "posterior": list(self._belief_state.posterior),
        }
        if self.evolutionary_population is not None and variant_rewards:
            self.evolutionary_population = self.evolutionary_population.evolve(variant_rewards)
        wall_clock_s = time.perf_counter() - wall_start
        return GameOutputs(
            steps=steps,
            trajectories=trajectories,
            trajectory_logs=trajectory_logs,
            forecasts=forecasts,
            targets=targets,
            confidence=confidence,
            convergence=convergence,
            llm_calls=llm_log.to_dicts(),
            wall_clock_s=wall_clock_s,
            evolutionary_population=self.evolutionary_population,
            coalition_graph=coalition_graph,
        )
