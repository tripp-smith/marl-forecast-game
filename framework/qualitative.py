"""Deterministic qualitative extraction and regime classification via LLM."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .llm.ollama import OllamaClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualitativeExtractor:
    """Maps raw narrative text to a bounded discrete tensor in {-1, 0, 1}^k."""

    client: OllamaClient = field(default_factory=OllamaClient)
    seed: int = 42
    temperature: float = 0.0
    feature_dim: int = 3

    def extract(self, text: str, prompt_template: str) -> tuple[int, ...]:
        prompt = prompt_template.format(text=text)
        try:
            raw = self.client.generate(
                prompt, seed=self.seed, temperature=self.temperature,
            )
            parsed = json.loads(raw.strip())
            values = (
                int(max(-1, min(1, parsed.get("sentiment", 0)))),
                int(max(-1, min(1, parsed.get("uncertainty", 0)))),
                int(max(-1, min(1, parsed.get("guidance", 0)))),
            )
            return values[: self.feature_dim]
        except Exception:
            logger.debug("QualitativeExtractor falling back to zero tensor")
            return (0,) * self.feature_dim


@dataclass(frozen=True)
class RegimeClassifier:
    """Deterministic regime classifier fusing quantitative and qualitative signals."""

    client: OllamaClient = field(default_factory=OllamaClient)
    seed: int = 42
    temperature: float = 0.0
    n_regimes: int = 3

    def classify(
        self,
        quant_dict: dict[str, float],
        qual_state: tuple[int, ...],
        prompt_template: str,
    ) -> int:
        prompt = prompt_template.format(quant=quant_dict, qual=qual_state)
        try:
            raw = self.client.generate(
                prompt, seed=self.seed, temperature=self.temperature,
            )
            regime = int(raw.strip())
            return max(0, min(self.n_regimes - 1, regime))
        except Exception:
            logger.debug("RegimeClassifier falling back to regime 0")
            return 0
