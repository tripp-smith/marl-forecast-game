"""Base forecasting agents and shared execution helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..llm import DSPyLikeRepl, OllamaClient
from ..observability import record_marl_policy_loaded
from ..strategy_runtime import StrategyRuntime
from ..types import AgentAction, ForecastState


LOGGER = logging.getLogger(__name__)
LLM_PARSE_BACKOFF_S: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class ForecastingAgent:
    """Primary forecasting agent with optional LLM refinement."""

    name: str = "forecaster"
    llm_repl: DSPyLikeRepl | None = None

    def _build_prompt(self, state: ForecastState) -> str:
        """Build the LLM prompt for a forecasting step.

        Args:
            state: Current immutable game state.

        Returns:
            Prompt text describing the current quantitative and qualitative state.
        """
        prompt = f"Given value={state.value:.4f}, exogenous={state.exogenous:.4f}, suggest scalar delta"
        if state.raw_qual_text:
            prompt += f" | Qualitative: {state.raw_qual_text[:512]} | Regime: {state.economic_regime}"
        return prompt

    @staticmethod
    def _parse_completion_delta(turn: dict[str, Any]) -> float:
        """Parse the first numeric token from an LLM turn payload.

        Args:
            turn: Response payload returned by ``DSPyLikeRepl.run_turn``.

        Returns:
            Parsed scalar delta.

        Raises:
            ValueError: If the completion is missing or does not begin with a float.
        """
        completion = str(turn.get("completion", "")).strip()
        if not completion:
            raise ValueError("LLM completion was empty")
        return float(completion.split()[0])

    def act(self, state: ForecastState, runtime: StrategyRuntime, *, round_idx: int | None = None) -> AgentAction:
        """Produce a forecast delta for the given state.

        Args:
            state: Current immutable game state.
            runtime: Runtime used for the baseline forecast policy.
            round_idx: Optional round index for observability.

        Returns:
            Forecast delta proposed by the agent.
        """
        base_delta = runtime.forecast_delta(state)
        if self.llm_repl is None:
            return AgentAction(actor=self.name, delta=base_delta)

        prompt = self._build_prompt(state)
        for attempt, backoff_s in enumerate((0.0, *LLM_PARSE_BACKOFF_S), start=1):
            if backoff_s > 0:
                time.sleep(backoff_s)
            try:
                turn = self.llm_repl.run_turn(prompt, round_idx=round_idx, agent=self.name)
                llm_delta = self._parse_completion_delta(turn)
                blended_delta = (0.8 * base_delta) + (0.2 * llm_delta)
                return AgentAction(actor=self.name, delta=blended_delta)
            except ValueError as exc:
                LOGGER.warning("LLM parse failed for %s on attempt %s: %s", self.name, attempt, exc)
            except Exception:
                LOGGER.warning("LLM refinement failed for %s", self.name, exc_info=True)
                break
        return AgentAction(actor=self.name, delta=base_delta)


@dataclass(frozen=True)
class QLearnedAgent(ForecastingAgent):
    """Forecasting or adversarial agent backed by a saved tabular MARL policy."""

    q_table_path: str | None = None
    algorithm: Literal["q", "wolf", "rarl"] = "q"
    _q_agent: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the backing tabular policy, if configured."""
        if not self.q_table_path:
            return
        from ..training import QTableAgent, RADversarialTrainer, WoLFPHCAgent

        loader: dict[str, Any] = {
            "q": QTableAgent,
            "wolf": WoLFPHCAgent,
            "rarl": RADversarialTrainer,
        }
        loaded = loader[self.algorithm].load(self.q_table_path)
        if hasattr(loaded, "epsilon"):
            loaded.epsilon = 0.0
        object.__setattr__(self, "_q_agent", loaded)
        record_marl_policy_loaded(self.algorithm)

    def act(
        self,
        state: ForecastState,
        runtime: StrategyRuntime | None = None,
        *,
        round_idx: int | None = None,
    ) -> AgentAction:
        """Return the policy-backed action for a state.

        Example:
            ``agent.act(state) -> AgentAction(actor="forecaster", delta=0.05)``

        Args:
            state: Current immutable game state.
            runtime: Optional fallback runtime when no Q-table is loaded.
            round_idx: Optional round index for compatibility with the base agent.

        Returns:
            Agent action containing the chosen forecast delta.
        """
        if self._q_agent is None:
            if runtime is None:
                return AgentAction(actor=self.name, delta=0.0)
            return super().act(state, runtime, round_idx=round_idx)
        idx = self._q_agent.act(state)
        delta = self._q_agent.action_space.action_to_delta(idx)
        LOGGER.info("QLearnedAgent delta=%s actor=%s", f"{delta:.4f}", self.name)
        return AgentAction(actor=self.name, delta=delta)


@dataclass(frozen=True)
class SafeAgentExecutor:
    """Exception-safe wrapper that falls back to a zero-delta action on failure."""

    fallback_delta: float = 0.0

    def execute(self, fn: Callable[..., AgentAction], *args: Any, **kwargs: Any) -> AgentAction:
        """Run an agent function with fallback handling.

        Args:
            fn: Agent callable to execute.
            *args: Positional arguments passed to ``fn``.
            **kwargs: Keyword arguments passed to ``fn``.

        Returns:
            The agent action or a fallback zero-delta action on failure.
        """
        try:
            return fn(*args, **kwargs)
        except Exception:
            LOGGER.warning(
                "SafeAgentExecutor caught exception in %s, using fallback",
                getattr(fn, "__name__", fn),
                exc_info=True,
            )
            return AgentAction(actor="fallback", delta=self.fallback_delta)


def default_ollama_repl() -> DSPyLikeRepl:
    """Create a ``DSPyLikeRepl`` backed by the default Ollama client."""
    return DSPyLikeRepl(client=OllamaClient())
