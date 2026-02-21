from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List

from .agents import AdversaryAgent, DefenderAgent, ForecastingAgent
from .types import ForecastState, SimulationConfig, StepResult, evolve_state


@dataclass(frozen=True)
class GameOutputs:
    steps: List[StepResult]
    forecasts: List[float]
    targets: List[float]


class ForecastGame:
    def __init__(self, config: SimulationConfig, seed: int = 7):
        self.config = config
        self._rng = Random(seed)
        self.forecaster = ForecastingAgent()
        self.adversary = AdversaryAgent()
        self.defender = DefenderAgent()

    def run(self, initial: ForecastState, rounds: int | None = None, *, disturbed: bool = True) -> GameOutputs:
        n_rounds = min(rounds or self.config.horizon, self.config.max_rounds)
        state = initial
        steps: List[StepResult] = []
        forecasts: List[float] = []
        targets: List[float] = []

        for _ in range(n_rounds):
            f_action = self.forecaster.act(state)
            a_action = self.adversary.act(state)
            d_action = self.defender.act(f_action, a_action)

            disturbance = 0.0
            if disturbed and self._rng.random() <= self.config.disturbance_prob:
                disturbance = self._rng.gauss(0, self.config.disturbance_scale)

            forecast = state.value + f_action.delta + a_action.delta + d_action.delta
            noise = self._rng.gauss(0, self.config.base_noise_std)
            next_state = evolve_state(
                state,
                base_trend=0.4,
                noise=noise,
                disturbance=disturbance,
            )
            target = next_state.value
            reward = -abs(target - forecast)
            step = StepResult(
                next_state=next_state,
                actions=(f_action, a_action, d_action),
                reward_breakdown={
                    "forecaster": reward,
                    "adversary": -reward,
                    "defender": reward,
                },
                forecast=forecast,
                target=target,
            )
            steps.append(step)
            forecasts.append(forecast)
            targets.append(target)
            state = next_state

        return GameOutputs(steps=steps, forecasts=forecasts, targets=targets)
