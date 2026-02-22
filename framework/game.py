from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from random import Random
import time
from typing import Any, Callable, List

from .agents import (
    AdversaryAgent,
    AgentRegistry,
    BottomUpAgent,
    DefenderAgent,
    EnsembleAggregatorAgent,
    ForecastingAgent,
    RefactoringAgent,
    SafeAgentExecutor,
    TopDownAgent,
    WolfpackAdversary,
)
from .aggregation import BayesianAggregator
from .disturbances import WolfpackDisturbance, disturbance_from_name
from .observability import (
    ROUND_COUNTER,
    ROUND_LATENCY,
    GameObserver,
    create_span,
    record_agent_metrics,
    record_alert,
    record_disturbance,
)
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


def _state_to_dict(state: ForecastState) -> dict:
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
    steps: List[StepResult]
    trajectories: List[TrajectoryEntry]
    trajectory_logs: List[dict]
    forecasts: List[float]
    targets: List[float]
    confidence: List[ConfidenceInterval]
    convergence: dict


def default_agent_factory(config: SimulationConfig) -> tuple[ForecastingAgent, AdversaryAgent, DefenderAgent, RefactoringAgent]:
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
    def __init__(
        self,
        config: SimulationConfig,
        seed: int = 7,
        *,
        agent_factory: AgentFactory = default_agent_factory,
        registry: AgentRegistry | None = None,
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

        self._qual_extractor = None
        self._regime_classifier = None
        self._qual_dataset: dict[int, dict] = {}
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

    def set_qual_dataset(self, dataset: dict[int, dict]) -> None:
        """Inject a pre-built qualitative dataset (timestep -> record mapping)."""
        self._qual_dataset = dict(dataset)

    def run(self, initial: ForecastState, rounds: int | None = None, *, disturbed: bool = True) -> GameOutputs:
        requested_rounds = rounds if rounds is not None else self.config.horizon
        n_rounds = max(0, min(requested_rounds, self.config.max_rounds))

        with create_span("simulation.run", {"seed": self._seed, "disturbed": disturbed, "n_rounds": n_rounds}):
            return self._run_inner(initial, n_rounds, disturbed=disturbed)

    def _run_inner(self, initial: ForecastState, n_rounds: int, *, disturbed: bool) -> GameOutputs:
        state = initial
        steps: List[StepResult] = []
        trajectories: List[TrajectoryEntry] = []
        trajectory_logs: List[dict] = []
        forecasts: List[float] = []
        targets: List[float] = []
        confidence: List[ConfidenceInterval] = []
        refactor_bias = 0.0
        cumulative_rewards: dict[str, float] = {}
        rolling_errors: List[float] = []
        convergence_reason: str = "completed"
        prev_abs_error: float = 0.0

        for idx in range(n_rounds):
            start = time.perf_counter()

            all_forecast_actions: list[AgentAction] = []
            for f_agent in self._registry.forecasters:
                if isinstance(f_agent, TopDownAgent):
                    action = self.safe_exec.execute(f_agent.act, state)
                else:
                    action = self.safe_exec.execute(f_agent.act, state, self.runtime)
                all_forecast_actions.append(action)

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
                a_action = adv_actions[0] if len(adv_actions) == 1 else \
                    AgentAction(actor="adversary", delta=sum(a.delta for a in adv_actions) / max(1, len(adv_actions)))
            else:
                a_action = AgentAction(actor=self._registry.adversaries[0].name if self._registry.adversaries else "adversary", delta=0.0)

            f_action = all_forecast_actions[0] if len(all_forecast_actions) == 1 else \
                self._registry.aggregator.aggregate(all_forecast_actions, cumulative_rewards)

            def_actions = [self.safe_exec.execute(d.act, f_action, a_action, self.config.defense_model) for d in self._registry.defenders]
            d_action = def_actions[0] if len(def_actions) == 1 else \
                AgentAction(actor="defender", delta=sum(d.delta for d in def_actions) / max(1, len(def_actions)))

            disturbance_val = self.disturbance.sample(state, self._rng, self.config) if disturbed else 0.0
            forecast = state.value + f_action.delta + a_action.delta + d_action.delta + refactor_bias
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
                refactor_bias += self.refactor.revise(error, use_llm=self.config.enable_llm_refactor)

            rolling_errors.append(abs(error))
            if self.config.convergence_threshold > 0 and len(rolling_errors) >= 20:
                window = rolling_errors[-20:]
                rolling_mae = sum(window) / len(window)
                if rolling_mae > self.config.convergence_threshold:
                    convergence_reason = "divergence_threshold_exceeded"
                    record_alert("mae_exceeds_threshold")

            for a in all_forecast_actions:
                cumulative_rewards[a.actor] = cumulative_rewards.get(a.actor, 0.0) + reward

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

        convergence = {
            "rounds_executed": len(steps),
            "max_rounds": self.config.max_rounds,
            "round_cap_hit": len(steps) == self.config.max_rounds,
            "reason": convergence_reason,
            "attack_cost_total": total_attack_cost,
            "defense_efficacy_ratio": defense_efficacy,
            "accuracy_vs_cost": mean_error / max(1e-9, total_attack_cost) if total_attack_cost > 0 else 0.0,
        }
        return GameOutputs(
            steps=steps,
            trajectories=trajectories,
            trajectory_logs=trajectory_logs,
            forecasts=forecasts,
            targets=targets,
            confidence=confidence,
            convergence=convergence,
        )
