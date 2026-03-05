"""LLM-driven recursive strategy refinement via trajectory analysis."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from .base import RefactorRequest
from .client import provider_client_from_config
from .mock import MockLLMRefactorClient
from .ollama import OllamaRefactorClient
from ..mnpo_loss import mnpo_loss
from ..types import TrajectoryEntry


@dataclass(frozen=True)
class RefinementResult:
    """Output of a refinement pass: clamped bias, hint, and raw diagnostics."""

    bias_adjustment: float
    strategy_hint: str
    raw_response: str
    strategy_chain: tuple[str, ...] = ()


class QueryCapableClient(Protocol):
    def query(self, prompt: str, model: str | None = None, **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class RecursiveStrategyRefiner:
    """Feeds trajectory summaries to LLM for strategy improvement suggestions."""

    client: OllamaRefactorClient | MockLLMRefactorClient | QueryCapableClient = MockLLMRefactorClient()
    clamp_min: float = -0.1
    clamp_max: float = 0.1
    provider: str = "ollama"
    model: str = "default"

    def __post_init__(self) -> None:
        if self.client is None:
            object.__setattr__(
                self,
                "client",
                provider_client_from_config(self.provider, model=self.model),
            )

    def refine(self, trajectories: list[TrajectoryEntry]) -> RefinementResult:
        """Analyse *trajectories* and return a clamped bias adjustment suggestion."""
        if not trajectories:
            return RefinementResult(bias_adjustment=0.0, strategy_hint="no data", raw_response="")

        rewards = []
        errors = []
        for traj in trajectories:
            r = traj.reward_breakdown.get("forecaster", 0.0)
            rewards.append(r)
            errors.append(abs(traj.target - traj.forecast))

        mean_reward = sum(rewards) / len(rewards)
        worst_error = max(errors)
        mean_error = sum(errors) / len(errors)

        base_prompt = (
            "Generate a resilient forecasting strategy update as JSON with keys "
            "'bias_adjustment' and 'rationale'. "
            f"Mean reward={mean_reward:.6f}; mean error={mean_error:.6f}; worst error={worst_error:.6f}."
        )
        refinement_prompt = (
            "Few-shot example: state drift high -> output {'bias_adjustment': -0.02, "
            "'rationale': 'trim overprediction during drift'}.\n"
            f"Now refine this forecaster strategy using: {base_prompt}"
        )
        strategy_chain: tuple[str, ...] = ()
        suggestion = None
        if hasattr(self.client, "query"):
            try:
                base_response = self.client.query(base_prompt, model=self.model)
                final_response = self.client.query(refinement_prompt + f"\nBase draft: {base_response}", model=self.model)
                strategy_chain = (base_response, final_response)
                parsed = json.loads(final_response.strip())
                suggestion = type("Suggestion", (), {
                    "bias_adjustment": float(parsed["bias_adjustment"]),
                    "rationale": str(parsed.get("rationale", "LLM-generated")),
                })()
            except Exception:
                logging.debug("RecursiveStrategyRefiner query path failed; falling back to suggest()", exc_info=True)
        if suggestion is None:
            suggestion = self.client.suggest(  # type: ignore[union-attr]
                RefactorRequest(latest_error=mean_error, strategy_name="forecaster")
            )
            strategy_chain = (suggestion.rationale,)

        clamped = max(self.clamp_min, min(self.clamp_max, suggestion.bias_adjustment))

        return RefinementResult(
            bias_adjustment=clamped,
            strategy_hint=suggestion.rationale,
            raw_response=f"mean_reward={mean_reward:.4f} worst_error={worst_error:.4f}",
            strategy_chain=strategy_chain,
        )


    def mnpo_refine(self, pair_logprobs: list[tuple[float, float]], eta: float = 1.0, beta: float = 0.1) -> float:
        """Compute MNPO-style preference loss for strategy refinement loops."""
        if not pair_logprobs:
            return 0.0
        import numpy as np

        policy = np.array([p[0] for p in pair_logprobs], dtype=float)
        baseline = np.array([p[1] for p in pair_logprobs], dtype=float)
        winners = np.arange(len(pair_logprobs), dtype=int)
        losers = np.zeros(len(pair_logprobs), dtype=int)
        stacked = np.empty(len(pair_logprobs) * 2, dtype=float)
        stacked[0::2] = policy
        stacked[1::2] = baseline
        return float(mnpo_loss(stacked, [], winners, losers, [], eta=eta, beta=beta))
