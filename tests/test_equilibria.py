from __future__ import annotations

import numpy as np

from framework.equilibria import BayesianBeliefState, compute_correlated_equilibrium


def test_correlated_equilibrium_beats_independent_mixed_baseline():
    payoff = np.array([[-10.0, 1.0], [1.0, -10.0]])
    result = compute_correlated_equilibrium((payoff, payoff))

    correlated_total = sum(result.expected_payoffs)
    mixed_nash_total = float(np.sum(np.full((2, 2), 0.25) * (payoff + payoff)))
    assert result.success is True
    assert correlated_total > mixed_nash_total * 1.1


def test_bayesian_belief_state_updates_toward_suspicious_type():
    belief = BayesianBeliefState(labels=("benign", "adversarial"), alpha=(1.0, 1.0))
    updated = belief.update((0.1, 0.9))
    assert updated.probability("adversarial") > belief.probability("adversarial")
