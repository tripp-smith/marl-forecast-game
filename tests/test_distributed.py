"""Tests for distributed/parallel game execution (T21)."""
from __future__ import annotations

import dataclasses

import pytest

from framework.distributed import (
    FaultToleranceConfig,
    ParallelGameRunner,
    RayParallelGameRunner,
    _config_from_dict,
    _config_to_dict,
    _game_outputs_to_dict,
    _state_from_primitives,
    _state_to_primitives,
    parallel_runner,
)
from framework.game import ForecastGame, GameOutputs
from framework.types import ForecastState, SimulationConfig, frozen_mapping


@pytest.fixture
def simple_config() -> SimulationConfig:
    return SimulationConfig(horizon=10, max_rounds=20)


@pytest.fixture
def simple_state() -> ForecastState:
    return ForecastState(
        t=0,
        value=10.0,
        exogenous=0.5,
        hidden_shift=0.0,
        segment_id="seg_a",
        segment_values=frozen_mapping({"a": 1.0, "b": 2.0}),
        macro_context=frozen_mapping({"gdp": 0.3}),
    )


class TestConfigSerialization:
    def test_round_trip(self, simple_config: SimulationConfig) -> None:
        d = _config_to_dict(simple_config)
        restored = _config_from_dict(d)
        assert restored == simple_config

    def test_all_fields_preserved(self, simple_config: SimulationConfig) -> None:
        d = _config_to_dict(simple_config)
        for field in dataclasses.fields(simple_config):
            assert field.name in d


class TestStateSerialization:
    def test_round_trip(self, simple_state: ForecastState) -> None:
        d = _state_to_primitives(simple_state)
        restored = _state_from_primitives(d)
        assert restored.t == simple_state.t
        assert restored.value == simple_state.value
        assert restored.exogenous == simple_state.exogenous
        assert dict(restored.segment_values) == dict(simple_state.segment_values)
        assert dict(restored.macro_context) == dict(simple_state.macro_context)


class TestGameOutputsSerialization:
    def test_preserves_forecasts_targets(self, simple_config: SimulationConfig, simple_state: ForecastState) -> None:
        game = ForecastGame(simple_config, seed=42)
        out = game.run(simple_state, disturbed=False)
        d = _game_outputs_to_dict(out)
        assert d["forecasts"] == out.forecasts
        assert d["targets"] == out.targets
        assert len(d["trajectory_logs"]) == len(out.trajectory_logs)
        assert len(d["confidence"]) == len(out.confidence)
        assert d["convergence"] == out.convergence


class TestFaultToleranceConfig:
    def test_defaults(self) -> None:
        ftc = FaultToleranceConfig()
        assert ftc.max_task_retries == 3
        assert ftc.retry_delay_s == 1.0
        assert ftc.actor_max_restarts == 3

    def test_frozen(self) -> None:
        ftc = FaultToleranceConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ftc.max_task_retries = 5  # type: ignore[misc]


class TestParallelRunner:
    def test_multiprocessing_backend(self) -> None:
        runner = parallel_runner(backend="multiprocessing")
        assert isinstance(runner, ParallelGameRunner)

    def test_auto_backend_returns_runner(self) -> None:
        runner = parallel_runner(backend="auto")
        assert isinstance(runner, (ParallelGameRunner, RayParallelGameRunner))

    def test_multiprocessing_run_seeds(self, simple_config: SimulationConfig, simple_state: ForecastState) -> None:
        runner = ParallelGameRunner(n_workers=2)
        results = runner.run_seeds(simple_config, simple_state, seeds=[42, 42], disturbed=False)
        assert len(results) == 2
        assert results[0]["forecasts"] == results[1]["forecasts"]

    def test_multiprocessing_map_scenarios(self, simple_config: SimulationConfig, simple_state: ForecastState) -> None:
        configs = [simple_config, simple_config]
        runner = ParallelGameRunner(n_workers=2)
        results = runner.map_scenarios(configs, simple_state, seeds=[7, 7])
        assert len(results) == 2
        assert results[0]["forecasts"] == results[1]["forecasts"]
