"""Bayesian hyperparameter optimization for SimulationConfig tuning."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from .game import ForecastGame, GameOutputs
from .metrics import mae
from .types import ForecastState, SimulationConfig


@dataclass
class BayesianOptimizer:
    """Tunes SimulationConfig fields by minimizing MAE on a validation split."""

    base_config: SimulationConfig
    init_state: ForecastState
    seed: int = 42
    n_iterations: int = 25

    def _objective(self, disturbance_prob: float, adversarial_intensity: float, base_noise_std: float) -> float:
        cfg = replace(
            self.base_config,
            disturbance_prob=max(0.0, min(1.0, disturbance_prob)),
            adversarial_intensity=max(0.0, adversarial_intensity),
            base_noise_std=max(0.0, base_noise_std),
        )
        game = ForecastGame(cfg, seed=self.seed)
        out = game.run(self.init_state, disturbed=True)
        return -mae(out.targets, out.forecasts)

    def optimize(self) -> dict[str, Any]:
        """Run Bayesian optimization and return best parameters and MAE."""
        try:
            from bayes_opt import BayesianOptimization
        except ImportError:
            return {"error": "bayesian-optimization not installed", "best_params": {}, "best_mae": float("inf")}

        pbounds = {
            "disturbance_prob": (0.01, 0.5),
            "adversarial_intensity": (0.1, 3.0),
            "base_noise_std": (0.01, 0.5),
        }
        optimizer = BayesianOptimization(f=self._objective, pbounds=pbounds, random_state=self.seed, verbose=0)
        optimizer.maximize(init_points=5, n_iter=self.n_iterations)
        best = optimizer.max
        if best is None:
            return {"error": "optimization failed", "best_params": {}, "best_mae": float("inf")}
        return {
            "best_params": best["params"],
            "best_mae": -best["target"],
        }
