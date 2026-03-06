"""Stable public API for academic and notebook-facing use."""
from __future__ import annotations

from framework.game import ForecastGame, GameOutputs
from framework.types import ForecastState, SimulationConfig


GameEngine = ForecastGame


def demo_state(*, value: float = 100.0, exogenous: float = 0.0, hidden_shift: float = 0.0) -> ForecastState:
    """Return a small default state for quickstarts and notebooks."""
    return ForecastState(t=0, value=value, exogenous=exogenous, hidden_shift=hidden_shift)


__all__ = ["GameEngine", "GameOutputs", "ForecastState", "SimulationConfig", "demo_state"]
