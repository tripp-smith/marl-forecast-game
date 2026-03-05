"""Provider-neutral LLM clients with retries, rate limiting, and cost tracking."""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import requests  # type: ignore[import-untyped]

from .audit import get_llm_log


DEFAULT_COSTS_PER_1K: dict[str, float] = {
    "ollama": 0.0,
    "openai": 0.002,
    "anthropic": 0.003,
    "grok": 0.005,
}


@dataclass(frozen=True)
class LLMQueryResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


@dataclass
class CostTracker:
    entries: list[LLMQueryResult] = field(default_factory=list)

    def record(self, result: LLMQueryResult) -> None:
        self.entries.append(result)

    @property
    def total_cost_usd(self) -> float:
        return sum(item.estimated_cost_usd for item in self.entries)


class BaseProviderClient(ABC):
    provider_name: str = "unknown"

    def __init__(
        self,
        *,
        model: str = "default",
        rate_limit_seconds: float = 0.0,
        max_retries: int = 3,
        timeout_seconds: float = 20.0,
        cost_per_1k_tokens: float | None = None,
    ) -> None:
        self.model = model
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.cost_per_1k_tokens = DEFAULT_COSTS_PER_1K[self.provider_name] if cost_per_1k_tokens is None else cost_per_1k_tokens
        self.cost_tracker = CostTracker()
        self._last_call_ts = 0.0

    def query(self, prompt: str, model: str | None = None, **kwargs: Any) -> str:
        target_model = model or self.model
        self._respect_rate_limit()
        error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                result = self._perform_query(prompt, model=target_model, **kwargs)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.cost_tracker.record(result)
                get_llm_log().record(
                    agent=str(kwargs.get("agent", "")),
                    round_idx=kwargs.get("round_idx"),
                    call_type="query",
                    model=target_model,
                    prompt=prompt,
                    response=result.text,
                    latency_ms=latency_ms,
                )
                return result.text
            except Exception as exc:
                error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        assert error is not None
        raise error

    def _respect_rate_limit(self) -> None:
        if self.rate_limit_seconds <= 0:
            return
        now = time.time()
        wait = self.rate_limit_seconds - (now - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    def _estimate_result(self, text: str, model: str, prompt: str) -> LLMQueryResult:
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(text.split()))
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = (total_tokens / 1000.0) * self.cost_per_1k_tokens
        return LLMQueryResult(
            text=text,
            provider=self.provider_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost,
        )

    @abstractmethod
    def _perform_query(self, prompt: str, *, model: str, **kwargs: Any) -> LLMQueryResult:
        raise NotImplementedError


class OllamaProviderClient(BaseProviderClient):
    provider_name = "ollama"

    def __init__(self, *, base_url: str | None = None, keep_alive: str = "5m", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.keep_alive = keep_alive

    def _perform_query(self, prompt: str, *, model: str, **kwargs: Any) -> LLMQueryResult:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False, "keep_alive": self.keep_alive}
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        text = str(response.json().get("response", ""))
        return self._estimate_result(text, model=model, prompt=prompt)


class OpenAIProviderClient(BaseProviderClient):
    provider_name = "openai"

    def __init__(self, *, api_key: str | None = None, base_url: str = "https://api.openai.com/v1", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url

    def _perform_query(self, prompt: str, *, model: str, **kwargs: Any) -> LLMQueryResult:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.0),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        body = response.json()
        text = str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))
        return self._estimate_result(text, model=model, prompt=prompt)


class AnthropicProviderClient(BaseProviderClient):
    provider_name = "anthropic"

    def __init__(self, *, api_key: str | None = None, base_url: str = "https://api.anthropic.com/v1/messages", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = base_url

    def _perform_query(self, prompt: str, *, model: str, **kwargs: Any) -> LLMQueryResult:
        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 256),
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        response = requests.post(self.base_url, json=payload, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        body = response.json()
        blocks = body.get("content", [])
        text = ""
        if blocks:
            text = str(blocks[0].get("text", ""))
        return self._estimate_result(text, model=model, prompt=prompt)


class GrokProviderClient(BaseProviderClient):
    provider_name = "grok"

    def __init__(self, *, api_key: str | None = None, base_url: str = "https://api.x.ai/v1/chat/completions", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("GROK_API_KEY", "")
        self.base_url = base_url

    def _perform_query(self, prompt: str, *, model: str, **kwargs: Any) -> LLMQueryResult:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.0),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.base_url, json=payload, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        body = response.json()
        text = str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))
        return self._estimate_result(text, model=model, prompt=prompt)


def provider_client_from_config(provider: str, *, model: str = "default", fallback_to_local: bool = True) -> BaseProviderClient:
    normalized = provider.strip().lower()
    if normalized == "ollama":
        return OllamaProviderClient(model=model)
    if normalized == "openai":
        try:
            return OpenAIProviderClient(model=model)
        except Exception:
            if fallback_to_local:
                return OllamaProviderClient(model=model)
            raise
    if normalized == "anthropic":
        try:
            return AnthropicProviderClient(model=model)
        except Exception:
            if fallback_to_local:
                return OllamaProviderClient(model=model)
            raise
    if normalized == "grok":
        try:
            return GrokProviderClient(model=model)
        except Exception:
            if fallback_to_local:
                return OllamaProviderClient(model=model)
            raise
    raise ValueError(f"Unknown LLM provider: {provider}")


def query_text(prompt: str, *, provider: str = "ollama", model: str = "default", fallback_to_local: bool = True, **kwargs: Any) -> str:
    client = provider_client_from_config(provider, model=model, fallback_to_local=fallback_to_local)
    return client.query(prompt, model=model, **kwargs)
