"""Accuracy, robustness, and probabilistic calibration metrics."""
from __future__ import annotations

import math
from typing import Iterable


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Absolute Error between actual and predicted sequences."""
    pairs = list(zip(actual, predicted))
    return sum(abs(a - p) for a, p in pairs) / max(1, len(pairs))


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Root Mean Squared Error between actual and predicted sequences."""
    pairs = list(zip(actual, predicted))
    mse = sum((a - p) ** 2 for a, p in pairs) / max(1, len(pairs))
    return float(mse**0.5)


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Absolute Percentage Error (skips zero-actual entries)."""
    pairs = [(a, p) for a, p in zip(actual, predicted) if a != 0]
    return 100.0 * sum(abs((a - p) / a) for a, p in pairs) / max(1, len(pairs))


def worst_case_abs_error(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Maximum absolute error across all actual/predicted pairs."""
    pairs = list(zip(actual, predicted))
    return max((abs(a - p) for a, p in pairs), default=0.0)


def robustness_delta(clean_value: float, attack_value: float) -> float:
    """Absolute difference between attacked and clean metric values."""
    return attack_value - clean_value


def robustness_ratio(clean_value: float, attack_value: float) -> float:
    """Ratio of attacked to clean metric values (0.0 if clean is zero)."""
    if clean_value == 0:
        return 0.0
    return attack_value / clean_value


# ---------------------------------------------------------------------------
# Probabilistic / calibration metrics (Phase S)
# ---------------------------------------------------------------------------

def pit_score(actual: float, forecast_mean: float, forecast_std: float) -> float:
    """Probability Integral Transform: CDF(actual) under N(mean, std)."""
    if forecast_std <= 0:
        return 0.5
    z = (actual - forecast_mean) / forecast_std
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def pit_scores(actuals: Iterable[float], means: Iterable[float], stds: Iterable[float]) -> list[float]:
    """Compute PIT scores for sequences of actuals, means, and standard deviations."""
    return [pit_score(a, m, s) for a, m, s in zip(actuals, means, stds)]


def crps(actual: float, forecast_mean: float, forecast_std: float) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution."""
    if forecast_std <= 0:
        return abs(actual - forecast_mean)
    z = (actual - forecast_mean) / forecast_std
    pdf_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return forecast_std * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / math.sqrt(math.pi))


def neg_crps(actual: float, forecast_mean: float, forecast_std: float) -> float:
    """Negative CRPS: higher is better. Suitable for exponential bankroll scoring."""
    return -crps(actual, forecast_mean, forecast_std)


def mean_crps(actuals: Iterable[float], means: Iterable[float], stds: Iterable[float]) -> float:
    """Average CRPS across sequences of actuals, means, and standard deviations."""
    scores = [crps(a, m, s) for a, m, s in zip(actuals, means, stds)]
    return sum(scores) / max(1, len(scores))


def interval_coverage(
    actuals: Iterable[float],
    lowers: Iterable[float],
    uppers: Iterable[float],
) -> float:
    """Fraction of actuals falling within [lower, upper]."""
    hits = 0
    total = 0
    for a, lo, hi in zip(actuals, lowers, uppers):
        total += 1
        if lo <= a <= hi:
            hits += 1
    return hits / max(1, total)
