"""Adversarial, defensive, and bias-correction agents."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..defenses import defense_from_name
from ..llm import LLMRefactorClient, MockLLMRefactorClient, OllamaClient, RefactorRequest
from ..types import AgentAction, ForecastState


@dataclass(frozen=True)
class AdversaryAgent:
    """Adversarial agent that injects directional noise to degrade forecasts."""

    name: str = "adversary"
    aggressiveness: float = 1.0
    attack_cost: float = 0.0

    def act(self, state: ForecastState) -> AgentAction:
        """Compute an adversarial delta opposing the expected trend.

        Args:
            state: Current immutable game state.

        Returns:
            Adversarial action opposing the base trend direction.
        """
        expected_trend = 0.4 + 0.4 * state.exogenous
        direction = -1.0 if expected_trend >= 0 else 1.0
        base = direction * 0.4 * self.aggressiveness
        penalty = min(abs(base), self.attack_cost * 0.2)
        delta = base - (penalty if base > 0 else -penalty)
        return AgentAction(actor=self.name, delta=delta)


@dataclass(frozen=True)
class DefenderAgent:
    """Defensive agent that corrects forecasts against adversarial perturbations."""

    name: str = "defender"
    llm_client: OllamaClient | None = None

    def act(self, forecast_action: AgentAction, adversary_action: AgentAction, defense_model: str) -> AgentAction:
        """Compute a defensive correction for a forecast.

        Args:
            forecast_action: Baseline forecasting action.
            adversary_action: Adversarial perturbation.
            defense_model: Named defense strategy.

        Returns:
            Defensive correction action.
        """
        if self.llm_client is not None and abs(adversary_action.delta) > 0.3:
            try:
                import json as _json

                state_json = _json.dumps(
                    {
                        "forecast_delta": forecast_action.delta,
                        "adversary_delta": adversary_action.delta,
                        "defense_model": defense_model,
                    }
                )
                prompt = (
                    f"Given adversarial state: {state_json}\n"
                    "Suggest a single float correction delta to defend the forecast. "
                    "Reply with ONLY a number."
                )
                raw = self.llm_client.generate(prompt, agent=self.name)
                llm_delta = float(raw.strip().split()[0])
                return AgentAction(actor=self.name, delta=llm_delta)
            except Exception:
                pass
        defense = defense_from_name(defense_model)
        correction = defense.defend(forecast_action.delta, adversary_action.delta)
        return AgentAction(actor=self.name, delta=correction)


@dataclass(frozen=True)
class RefactoringAgent:
    """Bias-correction agent that revises the forecast via error feedback or LLM suggestions."""

    name: str = "refactor"
    step_size: float = 0.02
    llm_client: LLMRefactorClient = field(default_factory=lambda: MockLLMRefactorClient(step_size=0.02))

    def revise(self, latest_error: float, *, use_llm: bool = False, round_idx: int | None = None) -> float:
        """Return a bias adjustment based on the latest error.

        Args:
            latest_error: Signed forecast error for the completed round.
            use_llm: Whether to request the adjustment from the LLM client.
            round_idx: Optional round index for observability.

        Returns:
            Bias adjustment to apply to subsequent forecasts.
        """
        if use_llm:
            suggestion = self.llm_client.suggest(
                RefactorRequest(latest_error=latest_error, strategy_name=self.name),
                round_idx=round_idx,
                agent=self.name,
            )
            return suggestion.bias_adjustment
        return -self.step_size if latest_error > 0 else self.step_size


@dataclass(frozen=True)
class WolfpackAdversary:
    """Coordinated adversary targeting correlated forecaster clusters."""

    name: str = "wolfpack"
    aggressiveness: float = 1.0
    attack_cost: float = 0.0
    correlation_threshold: float = 0.7

    @staticmethod
    def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
        """Compute a bounded Pearson correlation between two residual sequences."""
        n = min(len(xs), len(ys))
        if n < 3:
            return 0.0
        mx = sum(xs[:n]) / n
        my = sum(ys[:n]) / n
        cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
        sx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)) / n)
        sy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)) / n)
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return cov / (sx * sy)

    def compute_correlation_matrix(
        self,
        residual_history: Mapping[str, Sequence[float]],
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise residual correlations from external history.

        Args:
            residual_history: Residual time series keyed by forecasting agent id.

        Returns:
            Symmetric correlation lookup keyed by ordered agent pairs.
        """
        names = list(residual_history.keys())
        matrix: dict[tuple[str, str], float] = {}
        for i, left in enumerate(names):
            for j, right in enumerate(names):
                if i < j:
                    rho = self._pearson(residual_history[left], residual_history[right])
                    matrix[(left, right)] = rho
                    matrix[(right, left)] = rho
        return matrix

    def select_targets(
        self,
        primary_target: str,
        residual_history: Mapping[str, Sequence[float]],
    ) -> tuple[str, list[str]]:
        """Select the primary target and its correlated coalition.

        Args:
            primary_target: Highest-value forecasting target for the current round.
            residual_history: Residual history keyed by forecasting agent id.

        Returns:
            Primary target and correlated coalition members above the threshold.
        """
        corr = self.compute_correlation_matrix(residual_history)
        coalition = [
            name
            for name in residual_history
            if name != primary_target and abs(corr.get((primary_target, name), 0.0)) >= self.correlation_threshold
        ]
        return primary_target, coalition

    def act(self, state: ForecastState, *, is_primary: bool = True) -> AgentAction:
        """Return the wolfpack disturbance for a target.

        Args:
            state: Current immutable game state.
            is_primary: Whether the target is the pack's primary focus.

        Returns:
            Disturbance action for the selected target.
        """
        expected_trend = 0.4 + 0.4 * state.exogenous
        direction = -1.0 if expected_trend >= 0 else 1.0
        scale = self.aggressiveness if is_primary else self.aggressiveness * 0.5
        base = direction * 0.4 * scale
        penalty = min(abs(base), self.attack_cost * 0.2)
        delta = base - (penalty if base > 0 else -penalty)
        return AgentAction(actor=self.name, delta=delta)
