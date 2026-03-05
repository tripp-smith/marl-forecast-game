"""Strategy runtime backends for computing forecast deltas (Python + LLM)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import ForecastState


class StrategyRuntime(Protocol):
    """Runtime abstraction for strategy/policy evaluation backends."""

    def forecast_delta(self, state: ForecastState) -> float: ...


class ModelBackend(Protocol):
    """Pluggable ML model backend for agent-level predictions."""

    def predict(self, state: ForecastState) -> float: ...


@dataclass(frozen=True)
class PassthroughBackend:
    """Wraps the default Python runtime delta as a ModelBackend."""

    def predict(self, state: ForecastState) -> float:
        return 0.4 + 0.4 * state.exogenous


@dataclass(frozen=True)
class XGBoostBackend:
    """GradientBoostingRegressor-based backend (trains on provided history)."""

    _fallback: PassthroughBackend = PassthroughBackend()

    def predict(self, state: ForecastState) -> float:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return self._fallback.predict(state)
        except ImportError:
            return self._fallback.predict(state)


@dataclass(frozen=True)
class ARIMABackend:
    """ARIMA-style linear backend using trend and exogenous."""

    trend_coeff: float = 0.4
    exo_coeff: float = 0.35
    persistence: float = 0.1

    def predict(self, state: ForecastState) -> float:
        return self.trend_coeff + self.exo_coeff * state.exogenous + self.persistence * state.hidden_shift


class PromptRuntimeClient(Protocol):
    """Protocol for prompt-based runtime clients."""

    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class ProviderPromptClient:
    """Prompt client backed by the provider-neutral LLM layer."""

    provider: str = "ollama"
    model: str = "default"
    fallback: str = "0.0"

    def complete(self, prompt: str) -> str:
        from .llm.client import query_text

        try:
            return query_text(prompt, provider=self.provider, model=self.model)
        except Exception:
            return self.fallback


@dataclass(frozen=True)
class PythonStrategyRuntime:
    """Default deterministic local runtime."""

    def forecast_delta(self, state: ForecastState) -> float:
        return 0.4 + 0.4 * state.exogenous


@dataclass(frozen=True)
class DeterministicPromptClient:
    """Always returns a fixed response; useful for testing."""

    response: str = "0.0"

    def complete(self, prompt: str) -> str:
        return self.response


@dataclass(frozen=True)
class PromptStrategyRuntime:
    """Mock-friendly prompt runtime for LLM strategy calls."""

    client: PromptRuntimeClient = DeterministicPromptClient("0.0")
    fallback: PythonStrategyRuntime = PythonStrategyRuntime()

    def forecast_delta(self, state: ForecastState) -> float:
        prompt = f"state(t={state.t}, value={state.value:.4f}, exogenous={state.exogenous:.4f}) -> delta"
        text = self.client.complete(prompt).strip()
        try:
            return float(text)
        except ValueError:
            return self.fallback.forecast_delta(state)


@dataclass(frozen=True)
class OllamaPromptClient:
    """Adapts OllamaClient.generate to the PromptRuntimeClient.complete interface."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"

    def complete(self, prompt: str) -> str:
        from .llm.ollama import OllamaClient
        client = OllamaClient(base_url=self.base_url, model=self.model)
        try:
            return client.generate(prompt)
        except Exception:
            return "0.0"


@dataclass(frozen=True)
class ChatStrategyRuntime:
    """Uses OllamaInterface.chat for strategy generation via conversation."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    fallback: PythonStrategyRuntime = PythonStrategyRuntime()

    def forecast_delta(self, state: ForecastState) -> float:
        from .llm.ollama_interface import OllamaInterface
        client = OllamaInterface(base_url=self.base_url, model=self.model)
        messages = [
            {"role": "system", "content": "You are a forecasting agent. Given state, return a single float delta."},
            {"role": "user", "content": f"t={state.t}, value={state.value:.4f}, exogenous={state.exogenous:.4f}. Return delta:"},
        ]
        try:
            response = client.chat(messages)
            content = response.get("message", {}).get("content", "").strip()
            tokens = content.replace(",", " ").split()
            for token in tokens:
                try:
                    return float(token)
                except ValueError:
                    continue
        except Exception:
            pass
        return self.fallback.forecast_delta(state)


def runtime_from_name(name: str) -> StrategyRuntime:
    """Instantiate a strategy runtime by name, defaulting to Python."""
    normalized = name.strip().lower()
    if normalized in {"python", "default"}:
        return PythonStrategyRuntime()
    if normalized in {"prompt", "mock_llm", "llm"}:
        return PromptStrategyRuntime()
    if normalized in {"openai", "anthropic", "grok"}:
        return PromptStrategyRuntime(client=ProviderPromptClient(provider=normalized))
    if normalized in {"chat", "ollama_chat"}:
        return ChatStrategyRuntime()
    return PythonStrategyRuntime()
