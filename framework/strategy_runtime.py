from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import ForecastState


class StrategyRuntime(Protocol):
    """Runtime abstraction for strategy/policy evaluation backends."""

    def forecast_delta(self, state: ForecastState) -> float: ...


@dataclass(frozen=True)
class PythonStrategyRuntime:
    """Default deterministic local runtime."""

    def forecast_delta(self, state: ForecastState) -> float:
        return 0.55 + 0.35 * state.exogenous


@dataclass(frozen=True)
class HaskellRLMRuntime:
    """Compatibility shim for future dspy-repl HaskellRLM integration.

    The fallback implementation intentionally mirrors Python runtime behavior
    so tests remain deterministic while allowing backend selection wiring.
    """

    fallback: PythonStrategyRuntime = PythonStrategyRuntime()

    def forecast_delta(self, state: ForecastState) -> float:
        return self.fallback.forecast_delta(state)


def runtime_from_name(name: str) -> StrategyRuntime:
    normalized = name.strip().lower()
    if normalized in {"python", "default"}:
        return PythonStrategyRuntime()
    if normalized in {"haskell", "haskellrlm"}:
        return HaskellRLMRuntime()
    return PythonStrategyRuntime()
