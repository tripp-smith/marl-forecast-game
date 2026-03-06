"""Correlated-equilibrium and Bayesian-uncertainty helpers."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class CorrelatedEquilibriumResult:
    joint_probabilities: tuple[tuple[float, ...], ...]
    expected_payoffs: tuple[float, ...]
    success: bool

    def sample_actions(self, rng: Random) -> tuple[int, int]:
        flat = [value for row in self.joint_probabilities for value in row]
        if not flat:
            return 0, 0
        idx = rng.choices(range(len(flat)), weights=flat, k=1)[0]
        cols = len(self.joint_probabilities[0]) if self.joint_probabilities else 1
        return idx // cols, idx % cols


@dataclass(frozen=True)
class BayesianBeliefState:
    """Dirichlet-backed posterior over hidden agent types."""

    labels: tuple[str, ...]
    alpha: tuple[float, ...]

    @property
    def posterior(self) -> tuple[float, ...]:
        total = sum(self.alpha) or 1.0
        return tuple(value / total for value in self.alpha)

    def update(self, likelihood: Sequence[float]) -> "BayesianBeliefState":
        if len(likelihood) != len(self.alpha):
            raise ValueError("likelihood dimensionality must match alpha")
        posterior_alpha = tuple(float(a) + max(0.0, float(l)) for a, l in zip(self.alpha, likelihood))
        return BayesianBeliefState(labels=self.labels, alpha=posterior_alpha)

    def probability(self, label: str) -> float:
        for current, value in zip(self.labels, self.posterior):
            if current == label:
                return value
        raise KeyError(label)


def compute_correlated_equilibrium(payoff_matrices: Sequence[np.ndarray]) -> CorrelatedEquilibriumResult:
    """Solve a two-player correlated equilibrium maximizing total welfare."""
    if len(payoff_matrices) != 2:
        raise ValueError("expected exactly two payoff matrices")
    u0 = np.asarray(payoff_matrices[0], dtype=float)
    u1 = np.asarray(payoff_matrices[1], dtype=float)
    if u0.shape != u1.shape or u0.ndim != 2:
        raise ValueError("payoff matrices must be same-shaped 2-D arrays")
    m, n = u0.shape
    num_vars = m * n
    c = -(u0 + u1).reshape(-1)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    for action in range(m):
        for deviation in range(m):
            if action == deviation:
                continue
            row = np.zeros(num_vars)
            for opp in range(n):
                idx = action * n + opp
                row[idx] = -(u0[action, opp] - u0[deviation, opp])
            a_ub.append(row)
            b_ub.append(0.0)

    for action in range(n):
        for deviation in range(n):
            if action == deviation:
                continue
            row = np.zeros(num_vars)
            for opp in range(m):
                idx = opp * n + action
                row[idx] = -(u1[opp, action] - u1[opp, deviation])
            a_ub.append(row)
            b_ub.append(0.0)

    a_eq = np.ones((1, num_vars))
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * num_vars

    result = linprog(
        c,
        A_ub=np.stack(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        uniform = tuple(tuple(1.0 / num_vars for _ in range(n)) for _ in range(m))
        return CorrelatedEquilibriumResult(joint_probabilities=uniform, expected_payoffs=(0.0, 0.0), success=False)

    joint = result.x.reshape(m, n)
    expected = (
        float(np.sum(joint * u0)),
        float(np.sum(joint * u1)),
    )
    return CorrelatedEquilibriumResult(
        joint_probabilities=tuple(tuple(float(x) for x in row) for row in joint),
        expected_payoffs=expected,
        success=True,
    )


def bayesian_likelihood_from_observation(
    adversary_delta: float,
    disturbance: float,
) -> tuple[float, float]:
    """Map observed attack-like behavior into benign/adversarial likelihood mass."""
    attack_signal = abs(adversary_delta) + abs(disturbance)
    suspicious = min(1.0, attack_signal / 1.5)
    benign = max(0.0, 1.0 - suspicious)
    return benign, suspicious
