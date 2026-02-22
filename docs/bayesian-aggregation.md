# Bayesian Aggregation, Backtesting, and Scenarios

## Overview

The framework provides Bayesian Model Averaging (BMA) for combining multi-agent forecasts into calibrated probabilistic predictions, along with walk-forward backtesting, sensitivity analysis, Monte Carlo scenario generation, and surrogate-based hyperparameter optimization.

## BayesianAggregator

The `BayesianAggregator` class maintains per-agent posterior weights that are updated after each round based on forecast accuracy.

### Weight Update

Posterior weights are maintained in log-space and updated via Gaussian log-likelihood:

```
log_weight[i] += -0.5 * error[i]^2 / observation_variance
```

Normalized weights are computed using the log-sum-exp trick for numerical stability:

```python
max_lw = max(log_weights)
exp_weights = [exp(lw - max_lw) for lw in log_weights]
weights = [w / sum(exp_weights) for w in exp_weights]
```

Weights always sum to 1.0. Agents with smaller errors accumulate larger weights over time.

### Aggregation

The `aggregate(actions, state)` method returns `(weighted_mean_delta, variance)`:

```
mean_delta = sum(w_i * delta_i) / sum(w_i)
variance = sum(w_i * (delta_i - mean_delta)^2) / sum(w_i)
```

### Usage

```python
from framework.aggregation import BayesianAggregator
from framework.types import AgentAction, ForecastState

agg = BayesianAggregator()
actions = [AgentAction("agent_a", 0.3), AgentAction("agent_b", -0.1)]
state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

mean_delta, variance = agg.aggregate(actions, state)
agg.update({"agent_a": 0.05, "agent_b": 0.15})  # per-agent errors
```

## ProbabilisticForecast

The `make_probabilistic()` method generates a full distributional forecast:

```python
@dataclass(frozen=True)
class ProbabilisticForecast:
    mean: float
    variance: float
    quantiles: Tuple[float, ...]  # (p10, p25, p50, p75, p90)
```

Total variance combines agent disagreement with base noise:

```
total_variance = agent_variance + base_noise_std^2
```

Quantiles are computed from the Gaussian inverse CDF at z-scores:

| Quantile | z-score |
|---|---|
| p10 | -1.2816 |
| p25 | -0.6745 |
| p50 | 0.0 |
| p75 | 0.6745 |
| p90 | 1.2816 |

## Calibration Metrics

### Probability Integral Transform (PIT)

Evaluates whether predicted distributions are well-calibrated:

```python
pit_score(actual, forecast_mean, forecast_std) -> float
```

Computes `CDF(actual)` under `N(mean, std)` using the error function. A perfectly calibrated model produces uniformly distributed PIT scores.

### Continuous Ranked Probability Score (CRPS)

A strictly proper scoring rule for distributional forecasts. Implemented as the closed-form Gaussian CRPS:

```python
crps(actual, forecast_mean, forecast_std) -> float
```

```
CRPS = std * (z * (2 * CDF(z) - 1) + 2 * PDF(z) - 1/sqrt(pi))
```

where `z = (actual - mean) / std`. Lower CRPS indicates better probabilistic forecasting.

### Interval Coverage

Fraction of actual values falling within predicted confidence intervals:

```python
interval_coverage(actuals, lowers, uppers) -> float
```

Target coverage for a 90% interval should be approximately 0.90.

## Walk-Forward Backtesting

The `WalkForwardBacktester` evaluates model performance using a sliding-window approach:

```python
@dataclass
class WalkForwardBacktester:
    config: SimulationConfig
    window_size: int = 60    # training window length
    step_size: int = 20      # step between windows
    seed: int = 42
```

### Process

1. For each window position:
   - Train split: `rows[start : start + window_size]`
   - Test split: `rows[start + window_size : start + window_size + step_size]`
2. Initialize `ForecastState` from the last training row's target value.
3. Run `ForecastGame` for `len(test_rows)` rounds.
4. Compute MAE and RMSE for the window.
5. Advance start by `step_size` and repeat.

### Result Structure

```python
@dataclass(frozen=True)
class BacktestResult:
    n_windows: int
    window_results: tuple[WindowResult, ...]
    aggregate_mae: float
    aggregate_rmse: float
```

Each `WindowResult` contains per-window `mae`, `rmse`, train/test indices, and `n_forecasts`.

## Sensitivity Analysis

The `SensitivityAnalyzer` measures the importance of each macroeconomic factor by perturbation:

```python
@dataclass
class SensitivityAnalyzer:
    config: SimulationConfig
    perturbation_std: float = 1.0
```

### Process

1. Run a baseline simulation from the given initial state.
2. For each factor in `macro_context`:
   - Create a perturbed state with the factor increased by `perturbation_std`.
   - Run the simulation and compute MAE.
   - Record `importance[factor] = |perturbed_MAE - baseline_MAE|`.
3. Normalize importance values to sum to 1.0.

### Output

```python
{"GDP": 0.35, "UNRATE": 0.25, "CPIAUCSL": 0.20, "FEDFUNDS": 0.15, "T10YIE": 0.05}
```

## Monte Carlo Scenario Generation

The `ScenarioGenerator` produces percentile trajectory fans across many replications:

```python
@dataclass(frozen=True)
class ScenarioGenerator:
    config: SimulationConfig
    n_replications: int = 1000
```

### Process

1. Run `n_replications` games with different seeds (`base_seed + i`).
2. At each time step, collect all forecast values across replications.
3. Sort and extract percentiles (p10, p25, p50, p75, p90).
4. Compute mean MAE across all replications.

### Result Structure

```python
@dataclass(frozen=True)
class ScenarioFan:
    n_replications: int
    n_rounds: int
    p10: Tuple[float, ...]   # pessimistic trajectory
    p25: Tuple[float, ...]
    p50: Tuple[float, ...]   # median trajectory
    p75: Tuple[float, ...]
    p90: Tuple[float, ...]   # optimistic trajectory
    mean_mae: float
```

The fan chart provides a visual representation of forecast uncertainty across time.

## Bayesian Hyperparameter Optimization

The `BayesianOptimizer` tunes `SimulationConfig` parameters using surrogate-based optimization:

```python
@dataclass
class BayesianOptimizer:
    base_config: SimulationConfig
    init_state: ForecastState
    seed: int = 42
    n_iterations: int = 25
```

### Tuned Parameters

| Parameter | Bounds | Description |
|---|---|---|
| `disturbance_prob` | [0.01, 0.5] | Probability of disturbance per round |
| `adversarial_intensity` | [0.1, 3.0] | Adversary aggressiveness |
| `base_noise_std` | [0.01, 0.5] | Baseline noise standard deviation |

### Objective

The optimizer minimizes MAE by running `ForecastGame` with each parameter configuration and computing the negative MAE as the objective value. It uses the `bayesian-optimization` library's Gaussian Process surrogate model with 5 initial random points and `n_iterations` exploration steps.

### Result

```python
{
    "best_params": {
        "disturbance_prob": 0.08,
        "adversarial_intensity": 0.5,
        "base_noise_std": 0.12
    },
    "best_mae": 0.35
}
```

If `bayesian-optimization` is not installed, the optimizer returns an error dict with `best_mae=inf`.
