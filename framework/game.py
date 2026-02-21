from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List

from .agents import AdversaryAgent, DefenderAgent, ForecastingAgent, RefactoringAgent
from .disturbances import disturbance_from_name
from .strategy_runtime import runtime_from_name
from .types import AgentMessage, ConfidenceInterval, ForecastState, SimulationConfig, StepResult, TrajectoryEntry, evolve_state


@dataclass(frozen=True)
class GameOutputs:
    steps: List[StepResult]
    trajectories: List[TrajectoryEntry]
    forecasts: List[float]
    targets: List[float]
    confidence: List[ConfidenceInterval]
    convergence: dict


class ForecastGame:
    def __init__(self, config: SimulationConfig, seed: int = 7):
        self.config = config
        self._rng = Random(seed)
        self.runtime = runtime_from_name(config.runtime_backend)
        self.disturbance = disturbance_from_name(config.disturbance_model)
        self.forecaster = ForecastingAgent()
        self.adversary = AdversaryAgent()
        self.defender = DefenderAgent()
        self.refactor = RefactoringAgent()

    def run(self, initial: ForecastState, rounds: int | None = None, *, disturbed: bool = True) -> GameOutputs:
        n_rounds = min(rounds or self.config.horizon, self.config.max_rounds)
        state = initial
        steps: List[StepResult] = []
        trajectories: List[TrajectoryEntry] = []
        forecasts: List[float] = []
        targets: List[float] = []
        confidence: List[ConfidenceInterval] = []
        refactor_bias = 0.0

        for idx in range(n_rounds):
            f_action = self.forecaster.act(state, self.runtime)
            a_action = self.adversary.act(state)
            d_action = self.defender.act(f_action, a_action, self.config.defense_model)

            disturbance = self.disturbance.sample(state, self._rng, self.config) if disturbed else 0.0

            forecast = state.value + f_action.delta + a_action.delta + d_action.delta + refactor_bias
            noise = self._rng.gauss(0, self.config.base_noise_std)
            next_state = evolve_state(state, base_trend=0.4, noise=noise, disturbance=disturbance)
            target = next_state.value
            error = target - forecast
            reward = -abs(error)

            if self.config.enable_refactor:
                refactor_bias += self.refactor.revise(error)

            band = abs(disturbance) + self.config.base_noise_std + 0.05
            ci = ConfidenceInterval(lower=forecast - band, upper=forecast + band)
            messages = (
                AgentMessage("forecaster", "adversary", f"proposal={f_action.delta:.4f}"),
                AgentMessage("adversary", "defender", f"attack={a_action.delta:.4f}"),
                AgentMessage("defender", "refactor", f"defense={d_action.delta:.4f}"),
            )

            step = StepResult(
                next_state=next_state,
                actions=(f_action, a_action, d_action),
                reward_breakdown={"forecaster": reward, "adversary": -reward, "defender": reward},
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
                reward_breakdown=step.reward_breakdown,
                forecast=forecast,
                target=target,
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
            forecasts=forecasts,
            targets=targets,
            confidence=confidence,
            convergence=convergence,
        )
