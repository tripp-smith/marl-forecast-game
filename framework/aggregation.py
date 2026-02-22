"""Bayesian Model Averaging aggregation with Kelly-Criterion bankroll weights."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .metrics import neg_crps
from .types import AgentAction, ForecastState, ProbabilisticForecast


@dataclass
class BayesianAggregator:
    """Maintains per-agent bankroll weights updated via Kelly-Criterion exponential growth."""

    agent_names: list[str] = field(default_factory=list)
    _bankrolls: list[float] = field(default_factory=list, repr=False)
    _active: list[bool] = field(default_factory=list, repr=False)
    _observation_variance: float = 1.0
    _initialized: bool = field(default=False, repr=False)

    def _ensure_init(self, names: list[str]) -> None:
        if self._initialized:
            return
        self.agent_names = list(names)
        self._bankrolls = [1.0] * len(names)
        self._active = [True] * len(names)
        self._initialized = True

    @property
    def weights(self) -> list[float]:
        if not self._bankrolls:
            return []
        active_total = sum(b for b, a in zip(self._bankrolls, self._active) if a)
        if active_total <= 0:
            n = len(self._bankrolls)
            return [1.0 / n] * n
        return [
            (b / active_total if a else 0.0)
            for b, a in zip(self._bankrolls, self._active)
        ]

    def update(
        self,
        agent_errors: dict[str, float],
        means: dict[str, float] | None = None,
        stds: dict[str, float] | None = None,
        bankruptcy_threshold: float = 0.01,
    ) -> None:
        """Update bankrolls via exponential growth scored by negative CRPS."""
        for i, name in enumerate(self.agent_names):
            if name not in agent_errors or not self._active[i]:
                continue
            err = agent_errors[name]
            if means is not None and stds is not None and name in means and name in stds:
                actual = means[name] + err
                score = neg_crps(actual, means[name], stds[name])
            else:
                score = -abs(err)
            self._bankrolls[i] *= math.exp(score)
            if self._bankrolls[i] < bankruptcy_threshold:
                self._active[i] = False

    def aggregate(
        self,
        actions: list[AgentAction],
        state: ForecastState,
    ) -> tuple[float, float]:
        """Return (weighted_mean_delta, variance) across agent deltas."""
        names = [a.actor for a in actions]
        self._ensure_init(names)

        w = self.weights
        name_to_weight = dict(zip(self.agent_names, w))

        weighted_deltas = []
        weight_sum = 0.0
        for action in actions:
            wi = name_to_weight.get(action.actor, 1.0 / max(1, len(actions)))
            weighted_deltas.append(wi * action.delta)
            weight_sum += wi

        mean_delta = sum(weighted_deltas) / max(1e-12, weight_sum)

        variance = 0.0
        for action in actions:
            wi = name_to_weight.get(action.actor, 1.0 / max(1, len(actions)))
            variance += wi * (action.delta - mean_delta) ** 2
        variance /= max(1e-12, weight_sum)

        return mean_delta, variance

    def make_probabilistic(
        self,
        forecast_mean: float,
        variance: float,
        base_noise_std: float,
    ) -> ProbabilisticForecast:
        """Build a ProbabilisticForecast with quantiles from aggregation results."""
        total_var = variance + base_noise_std ** 2
        std = math.sqrt(max(0.0, total_var))
        z_scores = (-1.2816, -0.6745, 0.0, 0.6745, 1.2816)
        quantiles = tuple(forecast_mean + z * std for z in z_scores)
        return ProbabilisticForecast(
            mean=forecast_mean,
            variance=total_var,
            quantiles=quantiles,
        )
