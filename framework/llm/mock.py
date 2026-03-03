"""Deterministic mock LLM refactor client for testing without a live model."""
from __future__ import annotations

from dataclasses import dataclass

from .base import RefactorRequest, RefactorSuggestion


@dataclass(frozen=True)
class MockLLMRefactorClient:
    """Deterministic refactor client that applies a fixed step in the error-reducing direction."""

    step_size: float = 0.02

    def suggest(self, request: RefactorRequest, **_kwargs: object) -> RefactorSuggestion:
        """Return a bias adjustment of +/- *step_size* based on the error sign."""
        delta = -self.step_size if request.latest_error > 0 else self.step_size
        return RefactorSuggestion(
            bias_adjustment=delta,
            rationale=f"deterministic mock adjustment for {request.strategy_name}",
        )
