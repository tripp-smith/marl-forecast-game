"""LLM call audit trail -- captures every prompt, response, and latency."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMCallRecord:
    """Single recorded LLM interaction."""

    timestamp: float
    round_idx: int | None
    agent: str
    call_type: str  # "generate", "embed", "refactor"
    model: str
    prompt: str
    response: str
    latency_ms: float
    error: str | None = None


class LLMCallLog:
    """Thread-safe accumulator for LLM call records."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[LLMCallRecord] = []

    def record(
        self,
        *,
        round_idx: int | None = None,
        agent: str = "",
        call_type: str = "generate",
        model: str = "",
        prompt: str = "",
        response: str = "",
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> LLMCallRecord:
        rec = LLMCallRecord(
            timestamp=time.time(),
            round_idx=round_idx,
            agent=agent,
            call_type=call_type,
            model=model,
            prompt=prompt,
            response=response,
            latency_ms=latency_ms,
            error=error,
        )
        with self._lock:
            self._entries.append(rec)
        return rec

    def entries(self) -> list[LLMCallRecord]:
        with self._lock:
            return list(self._entries)

    def to_dicts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(e) for e in self._entries]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_GLOBAL_LOG = LLMCallLog()


def get_llm_log() -> LLMCallLog:
    """Return the process-wide LLM audit log singleton."""
    return _GLOBAL_LOG
