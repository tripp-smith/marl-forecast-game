from __future__ import annotations

from typing import Iterable


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = list(zip(actual, predicted))
    return sum(abs(a - p) for a, p in pairs) / max(1, len(pairs))


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = list(zip(actual, predicted))
    mse = sum((a - p) ** 2 for a, p in pairs) / max(1, len(pairs))
    return mse**0.5


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = [(a, p) for a, p in zip(actual, predicted) if a != 0]
    return 100.0 * sum(abs((a - p) / a) for a, p in pairs) / max(1, len(pairs))


def worst_case_abs_error(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = list(zip(actual, predicted))
    return max((abs(a - p) for a, p in pairs), default=0.0)


def robustness_delta(clean_value: float, attack_value: float) -> float:
    return attack_value - clean_value


def robustness_ratio(clean_value: float, attack_value: float) -> float:
    if clean_value == 0:
        return 0.0
    return attack_value / clean_value
