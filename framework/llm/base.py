"""Base types and protocol for LLM-driven strategy refactoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RefactorRequest:
    """Input to an LLM refactor client describing the current strategy error."""

    latest_error: float
    strategy_name: str


@dataclass(frozen=True)
class RefactorSuggestion:
    """LLM-produced suggestion containing a bias adjustment and rationale."""

    bias_adjustment: float
    rationale: str


class LLMRefactorClient(Protocol):
    """Protocol for any client that can suggest strategy refactoring adjustments."""

    def suggest(self, request: RefactorRequest, **kwargs: object) -> RefactorSuggestion: ...
