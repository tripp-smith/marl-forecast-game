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
    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class PythonStrategyRuntime:
    """Default deterministic local runtime."""

    def forecast_delta(self, state: ForecastState) -> float:
        return 0.4 + 0.4 * state.exogenous


@dataclass(frozen=True)
class HaskellRLMRuntime:
    """Haskell subprocess bridge with Python fallback.

    Tries the pre-built binary first (/usr/local/bin/marl-forecast-game),
    then falls back to cabal run, then to PythonStrategyRuntime.
    """

    fallback: PythonStrategyRuntime = PythonStrategyRuntime()
    haskell_dir: str = "haskell"
    binary_path: str = "/usr/local/bin/marl-forecast-game"
    timeout_s: float = 2.0

    def forecast_delta(self, state: ForecastState) -> float:
        import json
        import os
        import subprocess

        input_data = json.dumps({
            "t": state.t,
            "value": state.value,
            "exogenous": state.exogenous,
            "hidden_shift": state.hidden_shift,
        })

        if os.path.isfile(self.binary_path):
            try:
                result = subprocess.run(
                    [self.binary_path, "--delta"],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return float(result.stdout.strip())
            except Exception:
                pass

        try:
            result = subprocess.run(
                ["cabal", "run", "marl-forecast-game", "--", "--delta"],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=self.haskell_dir,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass

        return self.fallback.forecast_delta(state)


@dataclass(frozen=True)
class DeterministicPromptClient:
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
    normalized = name.strip().lower()
    if normalized in {"python", "default"}:
        return PythonStrategyRuntime()
    if normalized in {"haskell", "haskellrlm"}:
        return HaskellRLMRuntime()
    if normalized in {"prompt", "mock_llm", "llm"}:
        return PromptStrategyRuntime()
    if normalized in {"chat", "ollama_chat"}:
        return ChatStrategyRuntime()
    return PythonStrategyRuntime()
