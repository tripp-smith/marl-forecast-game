"""LLM-driven policy suggestion module.

Wraps :class:`OllamaRefactorClient` to simulate LLM-driven code
suggestions for agent policy adjustments.

Sub-phase: I (LLM Integration Completion).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .llm.ollama import OllamaRefactorClient

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMRefactorStub:
    """Simulates LLM-driven code suggestions for agent policy adjustments.

    Uses :class:`OllamaRefactorClient` under the hood. On any error the
    original ``policy_code`` is returned unchanged.
    """

    client: OllamaRefactorClient = field(default_factory=OllamaRefactorClient)

    def suggest_policy_update(
        self,
        policy_code: str,
        performance_history: list[float],
    ) -> str:
        """Construct a prompt from *policy_code* and *performance_history*, call
        the LLM, and return the suggested updated code string."""
        prompt = (
            "You are an RL policy optimizer. Given the following policy implementation:\n\n"
            f"```\n{policy_code}\n```\n\n"
            f"Performance history (recent MAE values): {performance_history[-20:]}\n\n"
            "Suggest an improved version of this policy code. "
            "Return ONLY the updated code, no explanations."
        )
        try:
            response = self.client.client.generate(prompt, agent="llm_refactor_stub")
            return response.strip() if response.strip() else policy_code
        except Exception:
            _logger.debug("LLMRefactorStub LLM call failed; returning original policy")
            return policy_code
