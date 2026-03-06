"""Multi-agent adversarial forecasting framework."""

from .equilibria import BayesianBeliefState, CorrelatedEquilibriumResult, compute_correlated_equilibrium

__all__ = [
    "BayesianBeliefState",
    "CorrelatedEquilibriumResult",
    "compute_correlated_equilibrium",
]
