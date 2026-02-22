from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random
import time
from typing import Callable, List

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
)
from .disturbances import disturbance_from_name
from .observability import ROUND_COUNTER, ROUND_LATENCY, GameObserver
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

        self.safe_exec = SafeAgentExecutor()
        self.logger = GameObserver().logger()

    def run(self, initial: ForecastState, rounds: int | None = None, *, disturbed: bool = True) -> GameOutputs:
        requested_rounds = rounds if rounds is not None else self.config.horizon
        n_rounds = max(0, min(requested_rounds, self.config.max_rounds))
        state = initial
        steps: List[StepResult] = []
        trajectories: List[TrajectoryEntry] = []
        trajectory_logs: List[dict] = []
        forecasts: List[float] = []
        targets: List[float] = []
        confidence: List[ConfidenceInterval] = []
        refactor_bias = 0.0
        cumulative_rewards: dict[str, float] = {}

        for idx in range(n_rounds):
            start = time.perf_counter()

            all_forecast_actions: list[AgentAction] = []
            for f_agent in self._registry.forecasters:
                if isinstance(f_agent, TopDownAgent):
                    action = self.safe_exec.execute(f_agent.act, state)
                else:
                    action = self.safe_exec.execute(f_agent.act, state, self.runtime)
                all_forecast_actions.append(action)

            f_action = all_forecast_actions[0] if len(all_forecast_actions) == 1 else \
                self._registry.aggregator.aggregate(all_forecast_actions, cumulative_rewards)

            if disturbed:
                adv_actions = [self.safe_exec.execute(a.act, state) for a in self._registry.adversaries]
                a_action = adv_actions[0] if len(adv_actions) == 1 else \
                    AgentAction(actor="adversary", delta=sum(a.delta for a in adv_actions) / max(1, len(adv_actions)))
            else:
                a_action = AgentAction(actor=self._registry.adversaries[0].name if self._registry.adversaries else "adversary", delta=0.0)

            def_actions = [self.safe_exec.execute(d.act, f_action, a_action, self.config.defense_model) for d in self._registry.defenders]
            d_action = def_actions[0] if len(def_actions) == 1 else \
                AgentAction(actor="defender", delta=sum(d.delta for d in def_actions) / max(1, len(def_actions)))

            disturbance_val = self.disturbance.sample(state, self._rng, self.config) if disturbed else 0.0
            forecast = state.value + f_action.delta + a_action.delta + d_action.delta + refactor_bias
            noise = self._rng.gauss(0, self.config.base_noise_std)
            next_state = evolve_state(state, base_trend=0.4, noise=noise, disturbance=disturbance_val)
            target = next_state.value
            error = target - forecast
            reward = -abs(error)

            if self.config.enable_refactor and self.refactor is not None:
                refactor_bias += self.refactor.revise(error, use_llm=self.config.enable_llm_refactor)

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

        convergence = {
            "rounds_executed": len(steps),
            "max_rounds": self.config.max_rounds,
            "round_cap_hit": len(steps) == self.config.max_rounds,
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
