from __future__ import annotations

from dataclasses import dataclass

from .base import RefactorRequest, RefactorSuggestion


@dataclass(frozen=True)
class MockLLMRefactorClient:
    step_size: float = 0.02

    def suggest(self, request: RefactorRequest) -> RefactorSuggestion:
        delta = -self.step_size if request.latest_error > 0 else self.step_size
        return RefactorSuggestion(
            bias_adjustment=delta,
            rationale=f"deterministic mock adjustment for {request.strategy_name}",
        )
