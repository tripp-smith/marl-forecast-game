"""Preference-pair generation for MNPO training.

Core Definitions (copy these verbatim into docstrings)
Preference Oracle P(x, y_i, {y_j}_{j≠i}) ∈ [0,1]
• Returns the probability that forecast y_i is preferred over the set of opponent forecasts {y_j}.
• In your framework: use negative CRPS (or PIT calibration score + Kelly BMA) as the underlying reward, then convert to probability via sigmoid or Bradley-Terry:\nP(y_i ≻ y_j) = 1 / (1 + exp(-(r_i - r_j)/β)), where r = -CRPS (lower CRPS = better).\nFor multiple opponents: average the win probabilities or use Plackett-Luce ranking.
Multiplayer Objective (for any policy π_i)\nJ(π_i, {π_j}{j≠i}) = E_x [ E{y_iπ_i, y_jπ_j} P(y_i ≻ {y_j}) ] - τ KL(π_i || π_ref)
Nash Equilibrium (target): A policy π* where no player can improve by unilaterally changing strategy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from .llm.ollama import OllamaRefactorClient
from .metrics import crps


@dataclass
class MNPOOracle:
    mode: str = "crps_based"
    beta: float = 0.1
    seed: int = 42
    llm_client: OllamaRefactorClient | None = None

    def _score(self, actual: float, forecast: float) -> float:
        return -crps(actual=actual, forecast_mean=forecast, forecast_std=1.0)

    def _winner_from_row(self, row: dict[str, Any]) -> tuple[float, float, str]:
        actual = float(row["target"])
        cand = float(row["candidate_forecast"])
        opp = float(row["opponent_forecast"])
        if self.mode == "llm_judge" and self.llm_client is not None:
            prompt = (
                "Choose the better forecast for the given actual value. "
                f"actual={actual:.4f}, candidate={cand:.4f}, opponent={opp:.4f}. "
                "Reply only with candidate or opponent."
            )
            out = self.llm_client._post_json("/api/generate", {"model": self.llm_client.model, "prompt": prompt})
            text = str(out.get("response", "candidate")).lower()
            return ((cand, opp, "candidate") if "candidate" in text else (opp, cand, "opponent"))

        cand_s = self._score(actual, cand)
        opp_s = self._score(actual, opp)
        if abs(cand_s - opp_s) < 1e-12:
            idx = int(row.get("state", {}).get("t", 0))
            return ((cand, opp, "candidate") if idx % 2 == 0 else (opp, cand, "opponent"))
        return ((cand, opp, "candidate") if cand_s >= opp_s else (opp, cand, "opponent"))

    def generate_pairs(
        self,
        trajectories: list[dict[str, Any]],
        num_pairs: int = 512,
        oracle_id: str | None = None,
        save_dir: str | Path = "data/preferences",
    ) -> list[tuple[dict[str, Any], float, float]]:
        rng = Random(self.seed)
        if not trajectories:
            return []

        pairs: list[tuple[dict[str, Any], float, float]] = []
        unique: set[tuple[int, float, float]] = set()
        max_attempts = max(num_pairs * 50, 1000)
        attempts = 0

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1
            i = rng.randrange(0, len(trajectories))
            row = trajectories[i]
            winner, loser, winner_label = self._winner_from_row(row)
            key = (i, round(winner, 6), round(loser, 6))
            if key in unique:
                continue
            unique.add(key)
            x = {
                "state": row.get("state", {}),
                "target": row.get("target"),
                "oracle_id": oracle_id,
                "idx": i,
                "winner_label": winner_label,
            }
            pairs.append((x, winner, loser))

        if pairs:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = [
                {"x": x, "winner": winner, "loser": loser}
                for x, winner, loser in pairs
            ]
            (out_dir / f"mnpo_pairs_{oracle_id or 'default'}.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

        return pairs
