"""Distributed/parallel game execution via multiprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

from .game import ForecastGame, GameOutputs
from .types import ForecastState, SimulationConfig, frozen_mapping


def _state_to_primitives(state: ForecastState) -> dict[str, Any]:
    return {
        "t": state.t,
        "value": state.value,
        "exogenous": state.exogenous,
        "hidden_shift": state.hidden_shift,
        "segment_id": state.segment_id,
        "segment_values": dict(state.segment_values),
        "macro_context": dict(state.macro_context),
    }


def _state_from_primitives(d: dict[str, Any]) -> ForecastState:
    return ForecastState(
        t=d["t"],
        value=d["value"],
        exogenous=d["exogenous"],
        hidden_shift=d["hidden_shift"],
        segment_id=d.get("segment_id", ""),
        segment_values=frozen_mapping(d.get("segment_values", {})),
        macro_context=frozen_mapping(d.get("macro_context", {})),
    )


def _run_single(args: tuple[SimulationConfig, dict[str, Any], int, bool]) -> dict[str, Any]:
    config, state_dict, seed, disturbed = args
    init_state = _state_from_primitives(state_dict)
    game = ForecastGame(config, seed=seed)
    out = game.run(init_state, disturbed=disturbed)
    return {
        "seed": seed,
        "forecasts": out.forecasts,
        "targets": out.targets,
        "convergence": out.convergence,
    }


@dataclass
class ParallelGameRunner:
    """Runs multiple game instances concurrently using multiprocessing."""

    n_workers: int = 4

    def map_scenarios(
        self,
        configs: list[SimulationConfig],
        init_state: ForecastState,
        seeds: list[int] | None = None,
        disturbed: bool = True,
    ) -> list[dict[str, Any]]:
        if seeds is None:
            seeds = list(range(len(configs)))

        state_dict = _state_to_primitives(init_state)
        args_list = [(cfg, state_dict, seed, disturbed) for cfg, seed in zip(configs, seeds)]

        with Pool(processes=min(self.n_workers, len(args_list))) as pool:
            results = pool.map(_run_single, args_list)

        return results

    def run_seeds(
        self,
        config: SimulationConfig,
        init_state: ForecastState,
        seeds: list[int],
        disturbed: bool = True,
    ) -> list[dict[str, Any]]:
        configs = [config] * len(seeds)
        return self.map_scenarios(configs, init_state, seeds, disturbed)
