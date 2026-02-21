from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RefactorRequest:
    latest_error: float
    strategy_name: str


@dataclass(frozen=True)
class RefactorSuggestion:
    bias_adjustment: float
    rationale: str


class LLMRefactorClient(Protocol):
    def suggest(self, request: RefactorRequest) -> RefactorSuggestion: ...
