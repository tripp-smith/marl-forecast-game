"""Distributed/parallel game execution via multiprocessing or Ray."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

from .game import ForecastGame, GameOutputs
from .types import ForecastState, SimulationConfig, frozen_mapping

try:
    import ray
except ImportError:
    ray = None  # type: ignore[assignment,unused-ignore]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

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


def _config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def _config_from_dict(d: dict[str, Any]) -> SimulationConfig:
    return SimulationConfig(**d)


def _game_outputs_to_dict(out: GameOutputs) -> dict[str, Any]:
    return {
        "trajectory_logs": out.trajectory_logs,
        "forecasts": out.forecasts,
        "targets": out.targets,
        "convergence": out.convergence,
        "confidence": [{"lower": ci.lower, "upper": ci.upper} for ci in out.confidence],
        "llm_calls": list(out.llm_calls) if out.llm_calls else [],
        "wall_clock_s": out.wall_clock_s,
    }


# ---------------------------------------------------------------------------
# Multiprocessing runner (original)
# ---------------------------------------------------------------------------

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
        """Run each config/seed pair in parallel and return result dicts."""
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
        """Run the same config across multiple seeds in parallel."""
        configs = [config] * len(seeds)
        return self.map_scenarios(configs, init_state, seeds, disturbed)


# ---------------------------------------------------------------------------
# Ray runner
# ---------------------------------------------------------------------------

def _ray_run_single_fn(
    config_dict: dict[str, Any],
    state_dict: dict[str, Any],
    seed: int,
    disturbed: bool,
    trace_ctx: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a single simulation. Designed to run as a Ray remote task."""
    if trace_ctx:
        try:
            from opentelemetry import context as otel_context
            from opentelemetry.propagate import extract
            otel_context.attach(extract(trace_ctx))
        except ImportError:
            pass

    config = _config_from_dict(config_dict)
    init_state = _state_from_primitives(state_dict)
    game = ForecastGame(config, seed=seed)
    out = game.run(init_state, disturbed=disturbed)
    result = _game_outputs_to_dict(out)
    result["seed"] = seed
    return result


if ray is not None:
    _ray_run_single = ray.remote(num_cpus=1)(_ray_run_single_fn)
else:
    _ray_run_single = None  # type: ignore[assignment,unused-ignore]


@dataclass(frozen=True)
class FaultToleranceConfig:
    """Retry and restart limits for Ray fault tolerance."""

    max_task_retries: int = 3
    retry_delay_s: float = 1.0
    actor_max_restarts: int = 3


@dataclass
class RayParallelGameRunner:
    """Runs multiple game instances concurrently using Ray."""

    address: str | None = None
    fault_tolerance: FaultToleranceConfig = dataclasses.field(default_factory=FaultToleranceConfig)

    def _ensure_init(self) -> None:
        if ray is None:
            from .exceptions import SimulationError
            raise SimulationError("ray is not installed")
        if not ray.is_initialized():
            ray.init(address=self.address, ignore_reinit_error=True)

    def map_scenarios(
        self,
        configs: list[SimulationConfig],
        init_state: ForecastState,
        seeds: list[int] | None = None,
        disturbed: bool = True,
    ) -> list[dict[str, Any]]:
        self._ensure_init()
        if seeds is None:
            seeds = list(range(len(configs)))

        state_dict = _state_to_primitives(init_state)

        trace_ctx: dict[str, str] | None = None
        try:
            from opentelemetry.propagate import inject
            trace_ctx = {}
            inject(trace_ctx)
        except ImportError:
            pass

        task = _ray_run_single.options(  # type: ignore[union-attr,unused-ignore]
            max_retries=self.fault_tolerance.max_task_retries,
            retry_exceptions=True,
        )
        futures = [
            task.remote(_config_to_dict(cfg), state_dict, seed, disturbed, trace_ctx)
            for cfg, seed in zip(configs, seeds)
        ]
        results: list[dict[str, Any]] = ray.get(futures)
        return results

    def run_seeds(
        self,
        config: SimulationConfig,
        init_state: ForecastState,
        seeds: list[int],
        disturbed: bool = True,
    ) -> list[dict[str, Any]]:
        return self.map_scenarios([config] * len(seeds), init_state, seeds, disturbed)

    def shutdown(self) -> None:
        """Shut down the Ray runtime if initialized."""
        if ray is not None and ray.is_initialized():
            ray.shutdown()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def parallel_runner(
    backend: str = "auto",
    n_workers: int = 4,
    ray_address: str | None = None,
) -> ParallelGameRunner | RayParallelGameRunner:
    """Return the appropriate parallel runner based on *backend*.

    ``"auto"`` selects Ray when available, falling back to multiprocessing.
    ``"ray"`` forces Ray (raises if unavailable).
    ``"multiprocessing"`` forces stdlib multiprocessing.
    """
    if backend == "ray" or (backend == "auto" and ray is not None):
        return RayParallelGameRunner(address=ray_address)
    if backend == "ray":
        raise RuntimeError("ray is not installed but backend='ray' was requested")
    return ParallelGameRunner(n_workers=n_workers)
