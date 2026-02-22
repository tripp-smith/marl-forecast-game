"""Bayesian Model Averaging aggregation for multi-agent forecasts."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .types import AgentAction, ForecastState, ProbabilisticForecast


@dataclass
class BayesianAggregator:
    """Maintains per-agent posterior weights updated via Gaussian log-likelihood."""

    agent_names: list[str] = field(default_factory=list)
    _log_weights: list[float] = field(default_factory=list, repr=False)
    _observation_variance: float = 1.0
    _initialized: bool = field(default=False, repr=False)

    def _ensure_init(self, names: list[str]) -> None:
        if self._initialized:
            return
        self.agent_names = list(names)
        self._log_weights = [0.0] * len(names)
        self._initialized = True

    @property
    def weights(self) -> list[float]:
        if not self._log_weights:
            return []
        max_lw = max(self._log_weights)
        exp_weights = [math.exp(lw - max_lw) for lw in self._log_weights]
        total = sum(exp_weights) or 1.0
        return [w / total for w in exp_weights]

    def update(self, agent_errors: dict[str, float]) -> None:
        """Update posterior weights given per-agent forecast errors."""
        for i, name in enumerate(self.agent_names):
            if name in agent_errors:
                err = agent_errors[name]
                ll = -0.5 * (err ** 2) / self._observation_variance
                self._log_weights[i] += ll

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
