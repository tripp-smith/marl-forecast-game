"""Walk-forward backtesting and sensitivity analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from .game import ForecastGame
from .metrics import mae, rmse, crps, mean_crps
from .types import ForecastState, SimulationConfig


@dataclass(frozen=True)
class WindowResult:
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    mae: float
    rmse: float
    n_forecasts: int


@dataclass(frozen=True)
class BacktestResult:
    n_windows: int
    window_results: tuple[WindowResult, ...]
    aggregate_mae: float
    aggregate_rmse: float


@dataclass
class WalkForwardBacktester:
    """Sliding-window train/evaluate backtester."""

    config: SimulationConfig
    window_size: int = 60
    step_size: int = 20
    seed: int = 42

    def run(self, rows: list[dict], *, max_windows: int = 50) -> BacktestResult:
        n = len(rows)
        if n < self.window_size + self.step_size:
            return BacktestResult(n_windows=0, window_results=(), aggregate_mae=0.0, aggregate_rmse=0.0)

        window_results: list[WindowResult] = []
        all_forecasts: list[float] = []
        all_targets: list[float] = []

        start = 0
        window_idx = 0

        while start + self.window_size + self.step_size <= n and window_idx < max_windows:
            train_end = start + self.window_size
            test_end = min(train_end + self.step_size, n)
            test_rows = rows[train_end:test_end]

            train_last = rows[train_end - 1]
            init_value = float(train_last["target"])
            init = ForecastState(t=0, value=init_value, exogenous=0.0, hidden_shift=0.0)

            game = ForecastGame(self.config, seed=self.seed + window_idx)
            out = game.run(init, rounds=len(test_rows), disturbed=True)

            if out.forecasts and out.targets:
                w_mae = mae(out.targets, out.forecasts)
                w_rmse = rmse(out.targets, out.forecasts)
                all_forecasts.extend(out.forecasts)
                all_targets.extend(out.targets)
            else:
                w_mae = 0.0
                w_rmse = 0.0

            window_results.append(WindowResult(
                window_idx=window_idx,
                train_start=start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
                mae=w_mae,
                rmse=w_rmse,
                n_forecasts=len(out.forecasts),
            ))

            start += self.step_size
            window_idx += 1

        agg_mae = mae(all_targets, all_forecasts) if all_forecasts else 0.0
        agg_rmse = rmse(all_targets, all_forecasts) if all_forecasts else 0.0

        return BacktestResult(
            n_windows=len(window_results),
            window_results=tuple(window_results),
            aggregate_mae=agg_mae,
            aggregate_rmse=agg_rmse,
        )


@dataclass
class SensitivityAnalyzer:
    """Per-factor perturbation analysis for macro_context fields."""

    config: SimulationConfig
    perturbation_std: float = 1.0
    seed: int = 42

    def analyze(
        self,
        init_state: ForecastState,
        factors: list[str] | None = None,
    ) -> dict[str, float]:
        if factors is None:
            factors = list(init_state.macro_context.keys()) if init_state.macro_context else []

        if not factors:
            return {}

        baseline_game = ForecastGame(self.config, seed=self.seed)
        baseline_out = baseline_game.run(init_state, disturbed=True)
        baseline_mae = mae(baseline_out.targets, baseline_out.forecasts) if baseline_out.forecasts else 0.0

        importance: dict[str, float] = {}

        for factor in factors:
            perturbed_ctx = dict(init_state.macro_context)
            current_val = perturbed_ctx.get(factor, 0.0)
            perturbed_ctx[factor] = current_val + self.perturbation_std

            from .types import frozen_mapping
            perturbed_state = replace(
                init_state,
                macro_context=frozen_mapping(perturbed_ctx),
            )

            game = ForecastGame(self.config, seed=self.seed)
            out = game.run(perturbed_state, disturbed=True)
            perturbed_mae = mae(out.targets, out.forecasts) if out.forecasts else 0.0

            importance[factor] = abs(perturbed_mae - baseline_mae)

        total = sum(importance.values()) or 1.0
        return {k: v / total for k, v in importance.items()}
