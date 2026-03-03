"""MNPO objective utilities.

Core Definitions (copy these verbatim into docstrings)
Preference Oracle P(x, y_i, {y_j}_{j≠i}) ∈ [0,1]
• Returns the probability that forecast y_i is preferred over the set of opponent forecasts {y_j}.
• In your framework: use negative CRPS (or PIT calibration score + Kelly BMA) as the underlying reward, then convert to probability via sigmoid or Bradley-Terry:\nP(y_i ≻ y_j) = 1 / (1 + exp(-(r_i - r_j)/β)), where r = -CRPS (lower CRPS = better).\nFor multiple opponents: average the win probabilities or use Plackett-Luce ranking.
Multiplayer Objective (for any policy π_i)\nJ(π_i, {π_j}{j≠i}) = E_x [ E{y_iπ_i, y_jπ_j} P(y_i ≻ {y_j}) ] - τ KL(π_i || π_ref)
Nash Equilibrium (target): A policy π* where no player can improve by unilaterally changing strategy.
"""
from __future__ import annotations

import numpy as np


def q_values_to_log_probs(q_values: np.ndarray) -> np.ndarray:
    """Convert Q-values to normalized log-probabilities via softmax."""
    shifted = q_values - np.max(q_values)
    probs = np.exp(shifted)
    probs /= probs.sum()
    return np.log(np.clip(probs, 1e-12, 1.0))


def mnpo_loss(
    policy_logprobs: np.ndarray,
    opponent_logprobs: list[np.ndarray],
    winners: np.ndarray,
    losers: np.ndarray,
    lambdas: list[float],
    eta: float = 1.0,
    beta: float = 0.1,
) -> float:
    h = (policy_logprobs[winners] - policy_logprobs[losers])
    for lmb, opp_log in zip(lambdas, opponent_logprobs):
        h = h - lmb * (opp_log[winners] - opp_log[losers])
    target = eta / (2 * beta)
    return float(np.mean((h - target) ** 2))


def tabular_closed_form_update(
    candidate_prob: np.ndarray,
    opponent_probs: list[np.ndarray],
    preferences: list[float],
    eta: float,
) -> np.ndarray:
    """Closed-form multiplicative policy update for tabular policies."""
    if not opponent_probs:
        return candidate_prob
    n = len(opponent_probs) + 1
    geo = np.ones_like(candidate_prob)
    for opp in opponent_probs:
        geo *= np.clip(opp, 1e-12, 1.0)
    geo = geo ** (1.0 / max(1, n - 1))
    pref_bonus = np.exp((eta / max(1, n - 1)) * np.array(preferences))
    updated = np.clip(geo * pref_bonus, 1e-12, None)
    updated = updated / updated.sum()
    return updated
