"""Nash equilibrium computation for discretized attacker-defender games."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class NashResult:
    """Mixed-strategy Nash equilibrium: attacker/defender strategies and game value."""

    attacker_strategy: Tuple[float, ...]
    defender_strategy: Tuple[float, ...]
    game_value: float


def compute_nash_equilibrium(payoff_matrix: np.ndarray) -> NashResult:
    """Compute minimax Nash equilibrium for a two-player zero-sum game.

    Args:
        payoff_matrix: (m x n) matrix where entry (i,j) is the attacker's
            payoff when attacker plays action i and defender plays action j.

    Returns:
        NashResult with mixed strategies and game value.
    """
    m, n = payoff_matrix.shape

    c_att = np.zeros(m + 1)
    c_att[-1] = -1.0

    A_ub_att = np.zeros((n, m + 1))
    A_ub_att[:, :m] = -payoff_matrix.T
    A_ub_att[:, -1] = 1.0
    b_ub_att = np.zeros(n)

    A_eq_att = np.zeros((1, m + 1))
    A_eq_att[0, :m] = 1.0
    b_eq_att = np.array([1.0])

    bounds_att = [(0.0, None)] * m + [(None, None)]
    result_att = linprog(c_att, A_ub=A_ub_att, b_ub=b_ub_att,
                         A_eq=A_eq_att, b_eq=b_eq_att, bounds=bounds_att,
                         method="highs")

    c_def = np.zeros(n + 1)
    c_def[-1] = 1.0

    A_ub_def = np.zeros((m, n + 1))
    A_ub_def[:, :n] = payoff_matrix
    A_ub_def[:, -1] = -1.0
    b_ub_def = np.zeros(m)

    A_eq_def = np.zeros((1, n + 1))
    A_eq_def[0, :n] = 1.0
    b_eq_def = np.array([1.0])

    bounds_def = [(0.0, None)] * n + [(None, None)]
    result_def = linprog(c_def, A_ub=A_ub_def, b_ub=b_ub_def,
                         A_eq=A_eq_def, b_eq=b_eq_def, bounds=bounds_def,
                         method="highs")

    if result_att.success and result_def.success:
        attacker_strat = tuple(float(x) for x in result_att.x[:m])
        defender_strat = tuple(float(x) for x in result_def.x[:n])
        game_value = float(result_att.x[-1])
    else:
        attacker_strat = tuple(1.0 / m for _ in range(m))
        defender_strat = tuple(1.0 / n for _ in range(n))
        game_value = 0.0

    return NashResult(
        attacker_strategy=attacker_strat,
        defender_strategy=defender_strat,
        game_value=game_value,
    )
