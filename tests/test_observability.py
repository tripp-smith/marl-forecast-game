"""Tests for observability metrics and logging (T21)."""
from __future__ import annotations

import pytest

from framework.observability import (
    AGENT_DELTA,
    AGENT_REWARD,
    DISTURBANCE_COUNT,
    DISTURBANCE_SUCCESS,
    ROUND_COUNTER,
    SIM_MAE,
    SIM_RMSE,
    GameObserver,
    export_prometheus_metrics,
    record_agent_metrics,
    record_disturbance,
    record_simulation_metrics,
)


class TestMetricDefinitions:
    """Verify metric objects are created when prometheus_client is installed."""

    @pytest.mark.skipif(SIM_MAE is None, reason="prometheus_client not installed")
    def test_sim_metrics_exist(self) -> None:
        assert SIM_MAE is not None
        assert SIM_RMSE is not None

    @pytest.mark.skipif(AGENT_DELTA is None, reason="prometheus_client not installed")
    def test_agent_metrics_exist(self) -> None:
        assert AGENT_DELTA is not None
        assert AGENT_REWARD is not None

    @pytest.mark.skipif(DISTURBANCE_COUNT is None, reason="prometheus_client not installed")
    def test_disturbance_metrics_exist(self) -> None:
        assert DISTURBANCE_COUNT is not None
        assert DISTURBANCE_SUCCESS is not None


class TestRecordSimulationMetrics:
    @pytest.mark.skipif(SIM_MAE is None, reason="prometheus_client not installed")
    def test_sets_gauge_values(self) -> None:
        record_simulation_metrics(
            seed=42, disturbed=True,
            mae_val=1.5, rmse_val=2.0, mape_val=10.0,
            worst=5.0, duration=0.5, rounds=100,
        )
        output = export_prometheus_metrics()
        assert "marl_sim_mae" in output

    def test_no_error_without_prometheus(self) -> None:
        record_simulation_metrics(
            seed=1, disturbed=False,
            mae_val=0.0, rmse_val=0.0, mape_val=0.0,
            worst=0.0, duration=0.0, rounds=0,
        )


class TestRecordAgentMetrics:
    def test_no_error_without_prometheus(self) -> None:
        record_agent_metrics("forecaster_0", "forecaster", 0.5, 1.0)

    @pytest.mark.skipif(AGENT_DELTA is None, reason="prometheus_client not installed")
    def test_agent_delta_in_output(self) -> None:
        record_agent_metrics("test_agent", "forecaster", 0.3, 0.0)
        output = export_prometheus_metrics()
        assert "marl_agent_delta" in output


class TestRecordDisturbance:
    def test_no_error_without_prometheus(self) -> None:
        record_disturbance(True, True)
        record_disturbance(False, False)

    @pytest.mark.skipif(DISTURBANCE_COUNT is None, reason="prometheus_client not installed")
    def test_disturbance_in_output(self) -> None:
        record_disturbance(True, False)
        output = export_prometheus_metrics()
        assert "marl_disturbance_injections_total" in output


class TestExportMetrics:
    def test_returns_string(self) -> None:
        result = export_prometheus_metrics()
        assert isinstance(result, str)

    @pytest.mark.skipif(ROUND_COUNTER is None, reason="prometheus_client not installed")
    def test_contains_round_counter(self) -> None:
        output = export_prometheus_metrics()
        assert "marl_game_rounds_total" in output


class TestGameObserver:
    def test_logger_returns_logger(self) -> None:
        obs = GameObserver()
        logger = obs.logger()
        assert logger is not None

    def test_custom_name(self) -> None:
        obs = GameObserver(logger_name="test_game")
        logger = obs.logger()
        assert logger is not None


class TestStructuredLogging:
    def test_structlog_json_output(self) -> None:
        try:
            import structlog
        except ImportError:
            pytest.skip("structlog not installed")

        logger = structlog.get_logger("test")
        bound = logger.bind(simulation_seed=42, trace_id="abc123")
        assert bound is not None


# ---------------------------------------------------------------------------
# T29: Prometheus scrape verification tests
# ---------------------------------------------------------------------------


class TestPrometheusScrapeVerification:
    @pytest.mark.skipif(SIM_MAE is None, reason="prometheus_client not installed")
    def test_scenario_label_in_output(self) -> None:
        record_simulation_metrics(
            seed=1, disturbed=False,
            mae_val=1.23, rmse_val=1.5, mape_val=5.0,
            worst=3.0, duration=0.1, rounds=50,
            scenario="test",
        )
        output = export_prometheus_metrics()
        assert "marl_sim_mae" in output
        assert 'scenario="test"' in output
        assert 'seed="1"' in output
        assert 'disturbed="false"' in output

    @pytest.mark.skipif(SIM_MAE is None, reason="prometheus_client not installed")
    def test_recorded_value_matches(self) -> None:
        record_simulation_metrics(
            seed=99, disturbed=True,
            mae_val=2.75, rmse_val=3.0, mape_val=8.0,
            worst=6.0, duration=0.2, rounds=100,
            scenario="verify",
        )
        output = export_prometheus_metrics()
        assert "2.75" in output
