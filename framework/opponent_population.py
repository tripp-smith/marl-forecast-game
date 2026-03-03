"""Opponent population management for MNPO training."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OpponentPopulation:
    """Stores historical/heterogeneous opponents used by MNPO."""

    mode: str = "TD"
    max_size: int = 10
    _population: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def add_opponent(self, policy_snapshot: dict[str, Any], weight: float = 1.0) -> None:
        entry = {"snapshot": copy.deepcopy(policy_snapshot), "weight": float(weight)}
        self._population.append(entry)
        if len(self._population) > self.max_size:
            self._population = self._population[-self.max_size :]

    def get_mixture(self, n: int = 5) -> list[tuple[dict[str, Any], float]]:
        if not self._population:
            return []
        sample = self._population[-max(1, n) :]
        if self.mode.upper() == "TD":
            raw_weights = [0.5 ** (len(sample) - 1 - idx) for idx in range(len(sample))]
        else:
            raw_weights = [1.0 for _ in sample]

        total = sum(raw_weights)
        lambdas = [w / total for w in raw_weights]
        return [(copy.deepcopy(entry["snapshot"]), lam) for entry, lam in zip(sample, lambdas)]

    def save_population(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._population, indent=2), encoding="utf-8")

    def load_population(self, path: str | Path) -> None:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        self._population = list(data)[-self.max_size :]

    @property
    def size(self) -> int:
        return len(self._population)
