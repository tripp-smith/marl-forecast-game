"""Multi-agent forecasting game engine with adversarial disturbances and defenses."""
from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import asdict, dataclass, replace
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
from .disturbances import WolfpackDisturbance, disturbance_from_name
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

            all_forecast_actions: list[AgentAction] = []
            forecast_variants: list[tuple[str, str]] = []
            for f_agent in self._registry.forecasters:
                if isinstance(f_agent, TopDownAgent):
                    action = self.safe_exec.execute(f_agent.act, state)
                elif isinstance(f_agent, BottomUpAgent):
                    action = self.safe_exec.execute(f_agent.act, state, self.runtime)
                else:
                    action = self.safe_exec.execute(f_agent.act, state, self.runtime, round_idx=idx)
                if self.evolutionary_population is not None:
                    subgroup = coalition_payload[0][0] if coalition_payload else None
                    variant = self.evolutionary_population.sample("forecaster", self._rng, subgroup=subgroup)
                    action = self.evolutionary_population.apply_variant(action, variant)
                    if variant is not None:
                        forecast_variants.append((action.actor, variant.name))
                all_forecast_actions.append(action)

            adversary_variant_name = ""
            wolfpack_adversaries = [
                a for a in self._registry.adversaries if isinstance(a, WolfpackAdversary)
            ]
            if disturbed and wolfpack_adversaries and len(all_forecast_actions) > 1:
                wolf = wolfpack_adversaries[0]
                bma_w = self.bayesian_agg.weights
                if bma_w:
                    primary_idx = max(range(len(bma_w)), key=lambda i: bma_w[i])
                    primary_name = self.bayesian_agg.agent_names[primary_idx]
                else:
                    primary_name = all_forecast_actions[0].actor
                _, coalition = wolf.select_targets(primary_name)
                targeted = {primary_name} | set(coalition)
                perturbed_actions: list[AgentAction] = []
                for fa in all_forecast_actions:
                    if fa.actor in targeted:
                        is_primary = fa.actor == primary_name
                        wp_action = wolf.act(state, is_primary=is_primary)
                        perturbed_actions.append(AgentAction(actor=fa.actor, delta=fa.delta + wp_action.delta))
                    else:
                        perturbed_actions.append(fa)
                all_forecast_actions = perturbed_actions
                a_action = AgentAction(actor=wolf.name, delta=0.0)
            elif disturbed:
                adv_actions = [self.safe_exec.execute(a.act, state) for a in self._registry.adversaries]
                adversary_variant_name = ""
                if self.evolutionary_population is not None and adv_actions:
                    adjusted_adv_actions: list[AgentAction] = []
                    for adv_action in adv_actions:
                        subgroup = coalition_payload[-1][0] if coalition_payload else None
                        variant = self.evolutionary_population.sample("adversary", self._rng, subgroup=subgroup)
                        adjusted_adv_actions.append(self.evolutionary_population.apply_variant(adv_action, variant))
                        if variant is not None:
                            adversary_variant_name = variant.name
                    adv_actions = adjusted_adv_actions
                a_action = adv_actions[0] if len(adv_actions) == 1 else \
                    AgentAction(actor="adversary", delta=sum(a.delta for a in adv_actions) / max(1, len(adv_actions)))
            else:
                adversary_variant_name = ""
                a_action = AgentAction(actor=self._registry.adversaries[0].name if self._registry.adversaries else "adversary", delta=0.0)

            f_action = all_forecast_actions[0] if len(all_forecast_actions) == 1 else \
                self._registry.aggregator.aggregate(all_forecast_actions, cumulative_rewards)

            def_actions = [self.safe_exec.execute(d.act, f_action, a_action, self.config.defense_model) for d in self._registry.defenders]
            defender_variant_name = ""
            if self.evolutionary_population is not None and def_actions:
                adjusted_def_actions: list[AgentAction] = []
                for def_action in def_actions:
                    variant = self.evolutionary_population.sample("defender", self._rng)
                    adjusted_def_actions.append(self.evolutionary_population.apply_variant(def_action, variant))
                    if variant is not None:
                        defender_variant_name = variant.name
                def_actions = adjusted_def_actions
            d_action = def_actions[0] if len(def_actions) == 1 else \
                AgentAction(actor="defender", delta=sum(d.delta for d in def_actions) / max(1, len(def_actions)))

            if self.config.equilibrium_type == "correlated":
                target_proxy = state.value + 0.4 + (0.4 * state.exogenous)
                f_candidates = np.array([0.8 * f_action.delta, f_action.delta, 1.2 * f_action.delta], dtype=float)
                a_candidates = np.array([0.8 * a_action.delta, a_action.delta, 1.2 * a_action.delta], dtype=float)
                forecaster_payoff = np.zeros((len(f_candidates), len(a_candidates)))
                adversary_payoff = np.zeros_like(forecaster_payoff)
                for i, f_delta in enumerate(f_candidates):
                    for j, a_delta in enumerate(a_candidates):
                        proxy_error = abs(target_proxy - (state.value + f_delta + a_delta + d_action.delta))
                        forecaster_payoff[i, j] = -proxy_error
                        adversary_payoff[i, j] = proxy_error
                ce = compute_correlated_equilibrium((forecaster_payoff, adversary_payoff))
                f_idx, a_idx = ce.sample_actions(self._rng)
                f_action = AgentAction(actor=f_action.actor, delta=float(f_candidates[f_idx]))
                a_action = AgentAction(actor=a_action.actor, delta=float(a_candidates[a_idx]))

            quarantined = False
            if self.config.equilibrium_type == "bayesian":
                posterior_adv = self._belief_state.posterior[-1]
                if posterior_adv >= self.config.quarantine_threshold:
                    quarantined = True
                    a_action = AgentAction(actor=a_action.actor, delta=0.0)

            disturbance_val = self.disturbance.sample(state, self._rng, self.config) if disturbed else 0.0
            forecast = state.value + f_action.delta + a_action.delta + d_action.delta + refactor_bias
            sabotage_penalty = 0.0
            if disturbed and self.config.sabotage_prob > 0.0 and self._rng.random() <= self.config.sabotage_prob:
                sabotage_penalty = self._rng.uniform(-0.2, 0.2) * max(1.0, abs(f_action.delta) + abs(a_action.delta))
                forecast += sabotage_penalty
            noise = self._rng.gauss(0, self.config.base_noise_std)

            qual_override: dict[str, Any] | None = None
            if self.config.enable_qual and self._qual_dataset and state.t in self._qual_dataset:
                qual_rec = self._qual_dataset[state.t]
                qual_text = qual_rec.get("text", "")
                if self._qual_extractor is not None:
                    raw_tensor = self._qual_extractor.extract(
                        qual_text, self.config.qualitative_extractor_prompt,
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
                    "raw_qual_state": tuple(float(v) for v in raw_tensor),
                    "raw_qual_text": qual_text[:self.config.max_context_tokens] if qual_text else None,
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
            target = next_state.value
            error = target - forecast
            reward = -abs(error)
            benign_mass, suspicious_mass = bayesian_likelihood_from_observation(a_action.delta, disturbance_val + sabotage_penalty)
            if len(self._belief_state.alpha) > 2:
                base = benign_mass / max(1, len(self._belief_state.alpha) - 1)
                likelihood = tuple(base for _ in range(len(self._belief_state.alpha) - 1)) + (suspicious_mass,)
            else:
                likelihood = (benign_mass, suspicious_mass)
            self._belief_state = self._belief_state.update(likelihood)

            record_agent_metrics(f_action.actor, "forecaster", f_action.delta, max(0.0, reward))
            record_agent_metrics(a_action.actor, "adversary", a_action.delta, max(0.0, -reward))
            record_agent_metrics(d_action.actor, "defender", d_action.delta, max(0.0, reward))
            record_disturbance(disturbance_val != 0.0, abs(error) > prev_abs_error)
            prev_abs_error = abs(error)

            if len(all_forecast_actions) > 1:
                agent_errors: dict[str, float] = {}
                agent_means: dict[str, float] = {}
                agent_stds: dict[str, float] = {}
                for fa in all_forecast_actions:
                    fa_forecast = state.value + fa.delta
                    agent_means[fa.actor] = fa_forecast
                    agent_stds[fa.actor] = self.config.base_noise_std
                    agent_errors[fa.actor] = target - fa_forecast
                self.bayesian_agg.update(
                    agent_errors,
                    means=agent_means,
                    stds=agent_stds,
                    bankruptcy_threshold=self.config.bankruptcy_threshold,
                )
                for wolf in wolfpack_adversaries:
                    for fa in all_forecast_actions:
                        wolf.record_residual(fa.actor, agent_errors[fa.actor])

            if self.config.enable_refactor and self.refactor is not None:
                refactor_bias += self.refactor.revise(error, use_llm=self.config.enable_llm_refactor, round_idx=idx)

            rolling_errors.append(abs(error))
            if self.config.convergence_threshold > 0 and len(rolling_errors) >= 20:
                window = rolling_errors[-20:]
                rolling_mae = sum(window) / len(window)
                if rolling_mae > self.config.convergence_threshold:
                    convergence_reason = "divergence_threshold_exceeded"
                    record_alert("mae_exceeds_threshold")

            for a in all_forecast_actions:
                cumulative_rewards[a.actor] = cumulative_rewards.get(a.actor, 0.0) + reward
            for _, variant_name in forecast_variants:
                variant_rewards.setdefault(variant_name, []).append(reward)
            if adversary_variant_name:
                variant_rewards.setdefault(adversary_variant_name, []).append(-reward)
            if defender_variant_name:
                variant_rewards.setdefault(defender_variant_name, []).append(reward)

            band = abs(disturbance_val) + self.config.base_noise_std + 0.05
            ci = ConfidenceInterval(lower=forecast - band, upper=forecast + band)
            messages = (
                AgentMessage("forecaster", "adversary", f"proposal={f_action.delta:.4f}"),
                AgentMessage("adversary", "defender", f"attack={a_action.delta:.4f}"),
                AgentMessage("defender", "refactor", f"defense={d_action.delta:.4f}"),
            )

            reward_breakdown = frozen_mapping({"forecaster": reward, "adversary": -reward, "defender": reward})

            step = StepResult(
                next_state=next_state,
                actions=(f_action, a_action, d_action),
                reward_breakdown=reward_breakdown,
                forecast=forecast,
                target=target,
                confidence=ci,
                messages=messages,
            )
            traj = TrajectoryEntry(
                round_idx=idx,
                state=state,
                actions=(f_action, a_action, d_action),
                messages=messages,
                reward_breakdown=reward_breakdown,
                forecast=forecast,
                target=target,
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
                self.logger.info("round_complete", round_idx=idx, reward=reward, disturbance=disturbance_val)
            except TypeError:
                self.logger.info(f"round_complete round_idx={idx} reward={reward:.6f} disturbance={disturbance_val:.6f}")

            cost_this_round = abs(a_action.delta) * self.config.attack_cost
            dist_magnitude = abs(disturbance_val)
            error_delta_val = abs(abs(error) - prev_abs_error) if idx > 0 else abs(error)
            cost_benefit = {
                "round": idx,
                "attack_cost": cost_this_round,
                "disturbance_magnitude": dist_magnitude,
                "error_delta": error_delta_val,
                "cost_benefit_ratio": error_delta_val / max(1e-9, cost_this_round) if cost_this_round > 0 else 0.0,
            }

            bma_snapshot = list(self.bayesian_agg.weights) if len(all_forecast_actions) > 1 and self.bayesian_agg.weights else None
            trajectory_logs.append(
                {
                    "round_idx": idx,
                    "state": _state_to_dict(state),
                    "actions": [asdict(a) for a in (f_action, a_action, d_action)],
                    "forecast": forecast,
                    "target": target,
                    "reward": reward,
                    "disturbance": disturbance_val,
                    "messages": [asdict(m) for m in messages],
                    "elapsed_s": elapsed,
                    "bma_weights": bma_snapshot,
                    "confidence": {"lower": ci.lower, "upper": ci.upper},
                    "cost_benefit": cost_benefit,
                    "coalitions": [list(group) for group in coalition_payload],
                    "quarantined": quarantined,
                    "sabotage_penalty": sabotage_penalty,
                    "posterior": list(self._belief_state.posterior),
                }
            )
            steps.append(step)
            trajectories.append(traj)
            forecasts.append(forecast)
            targets.append(target)
            confidence.append(ci)
            state = next_state

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
