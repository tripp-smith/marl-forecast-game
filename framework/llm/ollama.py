"""Ollama-backed LLM clients for generation, embedding, and strategy refactoring."""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
from typing import Any

import json
import os
from urllib.request import urlopen

from .audit import get_llm_log
from .base import RefactorRequest, RefactorSuggestion
from .mock import MockLLMRefactorClient


@dataclass(frozen=True)
class OllamaClient:
    """Low-level client for the Ollama generate and embeddings APIs."""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = "llama3.2"
    keep_alive: str = "5m"

    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        temperature: float | None = None,
        format_schema: str | None = None,
        round_idx: int | None = None,
        agent: str = "",
    ) -> str:
        """Send a prompt to Ollama and return the response text.

        Args:
            prompt: The input text.
            seed: Optional RNG seed for reproducibility.
            temperature: Sampling temperature override.
            format_schema: Optional JSON-schema string for structured output.
            round_idx: Game round index for audit logging.
            agent: Agent identifier for audit logging.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        if seed is not None or temperature is not None:
            options: dict[str, Any] = {}
            if seed is not None:
                options["seed"] = seed
            if temperature is not None:
                options["temperature"] = temperature
            payload["options"] = options
        if format_schema is not None:
            payload["format"] = format_schema
        data = json.dumps(payload).encode("utf-8")
        req_url = f"{self.base_url}/api/generate"
        t0 = _time.perf_counter()
        error_msg: str | None = None
        response_text = ""
        try:
            with urlopen(req_url, data=data, timeout=10) as resp:
                body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            response_text = str(body.get("response", ""))
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            latency_ms = (_time.perf_counter() - t0) * 1000
            get_llm_log().record(
                round_idx=round_idx,
                agent=agent,
                call_type="generate",
                model=self.model,
                prompt=prompt,
                response=response_text,
                latency_ms=latency_ms,
                error=error_msg,
            )
        return response_text

    def query(self, prompt: str, model: str | None = None, **kwargs: Any) -> str:
        """Provider-neutral alias used by the multi-provider abstraction."""
        target_model = self.model if model is None else model
        client = self if target_model == self.model else OllamaClient(base_url=self.base_url, model=target_model, keep_alive=self.keep_alive)
        return client.generate(prompt, **kwargs)

    def embeddings(
        self,
        text: str,
        *,
        round_idx: int | None = None,
        agent: str = "",
    ) -> list[float]:
        """Return an embedding vector for *text* via the Ollama embeddings API."""
        payload = {"model": self.model, "prompt": text, "keep_alive": self.keep_alive}
        data = json.dumps(payload).encode("utf-8")
        req_url = f"{self.base_url}/api/embeddings"
        t0 = _time.perf_counter()
        error_msg: str | None = None
        response_text = ""
        try:
            with urlopen(req_url, data=data, timeout=10) as resp:
                body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            embedding = body.get("embedding", [])
            response_text = f"[{len(embedding)} dims]"
        except Exception as exc:
            error_msg = str(exc)
            embedding = []
            raise
        finally:
            latency_ms = (_time.perf_counter() - t0) * 1000
            get_llm_log().record(
                round_idx=round_idx,
                agent=agent,
                call_type="embed",
                model=self.model,
                prompt=text[:256],
                response=response_text,
                latency_ms=latency_ms,
                error=error_msg,
            )
        return [float(x) for x in embedding]


@dataclass(frozen=True)
class DSPyLikeRepl:
    """Convenience wrapper that pairs a generate call with an embedding call."""

    client: OllamaClient

    def run_turn(
        self,
        prompt: str,
        *,
        round_idx: int | None = None,
        agent: str = "",
    ) -> dict[str, Any]:
        """Generate a completion and its embedding, returning both in a dict."""
        completion = self.client.generate(
            prompt, round_idx=round_idx, agent=agent,
        )
        vectors = self.client.embeddings(
            completion[:256] if completion else "",
            round_idx=round_idx,
            agent=agent,
        )
        return {
            "prompt": prompt,
            "completion": completion,
            "embedding_size": len(vectors),
            "model": self.client.model,
            "keep_alive": self.client.keep_alive,
        }


@dataclass(frozen=True)
class OllamaRefactorClient:
    """LLMRefactorClient implementation backed by Ollama generate API.

    Falls back to MockLLMRefactorClient behaviour on network or parse errors.
    """

    client: OllamaClient = OllamaClient()
    _fallback: MockLLMRefactorClient = MockLLMRefactorClient()

    def suggest(
        self,
        request: RefactorRequest,
        **kwargs: object,
    ) -> RefactorSuggestion:
        """Query Ollama for a bias adjustment; falls back to mock on failure."""
        prompt = (
            f"Given a forecasting strategy '{request.strategy_name}' with latest error "
            f"{request.latest_error:.6f}, suggest a numeric bias adjustment (single float) "
            f"and a one-sentence rationale. Reply as JSON: "
            f'{{"bias_adjustment": <float>, "rationale": "<string>"}}'
        )
        r_idx = kwargs.get("round_idx")
        r_agent = str(kwargs.get("agent", "refactor"))
        round_idx: int | None = None
        if isinstance(r_idx, (int, float, str)):
            try:
                round_idx = int(r_idx)
            except ValueError:
                round_idx = None
        try:
            raw = self.client.generate(
                prompt,
                round_idx=round_idx,
                agent=r_agent,
            )
            parsed = json.loads(raw.strip())
            return RefactorSuggestion(
                bias_adjustment=float(parsed["bias_adjustment"]),
                rationale=str(parsed.get("rationale", "LLM-generated")),
            )
        except Exception:
            logging.debug("OllamaRefactorClient falling back to mock for '%s'", request.strategy_name)
            return self._fallback.suggest(request)
