"""Observability: structured logging, Prometheus metrics, OpenTelemetry tracing, and alerting."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from typing import Any, Callable, Generator

# ---------------------------------------------------------------------------
# Structured logging (structlog)
# ---------------------------------------------------------------------------

try:
    import structlog as _structlog

    _structlog.configure(
        processors=[
            _structlog.stdlib.add_log_level,
            _structlog.processors.TimeStamper(fmt="iso"),
            _structlog.processors.JSONRenderer(),
        ],
        wrapper_class=_structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=_structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
except Exception:  # pragma: no cover
    _structlog = None  # type: ignore[assignment,unused-ignore]

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_Counter: Any = None
_Histogram: Any = None
_Gauge: Any = None
_generate_latest: Callable[..., bytes] | None = None

try:
    from prometheus_client import Counter as _Counter
    from prometheus_client import Gauge as _Gauge
    from prometheus_client import Histogram as _Histogram
    from prometheus_client import generate_latest as _generate_latest
except Exception:  # pragma: no cover
    pass

ROUND_COUNTER = _Counter("marl_game_rounds_total", "Total game rounds") if _Counter is not None else None
ROUND_LATENCY = (
    _Histogram("marl_game_round_latency_seconds", "Round execution latency in seconds") if _Histogram is not None else None
)

SIM_MAE = _Gauge("marl_sim_mae", "Simulation MAE", ["seed", "disturbed", "scenario"]) if _Gauge is not None else None
SIM_RMSE = _Gauge("marl_sim_rmse", "Simulation RMSE", ["seed", "disturbed", "scenario"]) if _Gauge is not None else None
SIM_MAPE = _Gauge("marl_sim_mape", "Simulation MAPE", ["seed", "disturbed", "scenario"]) if _Gauge is not None else None
SIM_WORST = _Gauge("marl_sim_worst_case", "Worst-case error", ["seed", "disturbed", "scenario"]) if _Gauge is not None else None
SIM_DURATION = _Histogram("marl_sim_duration_seconds", "Simulation wall-clock time") if _Histogram is not None else None
SIM_ROUNDS = _Gauge("marl_sim_rounds", "Rounds executed", ["seed"]) if _Gauge is not None else None

AGENT_DELTA = _Histogram("marl_agent_delta", "Agent delta per round", ["actor", "role"]) if _Histogram is not None else None
AGENT_REWARD = _Counter("marl_agent_reward_total", "Cumulative agent reward", ["actor"]) if _Counter is not None else None

DISTURBANCE_COUNT = _Counter("marl_disturbance_injections_total", "Disturbance injection count") if _Counter is not None else None
DISTURBANCE_SUCCESS = _Counter("marl_disturbance_success_total", "Disturbances that increased error") if _Counter is not None else None
ALERT_ANOMALY = _Counter("marl_alert_anomaly_total", "Alert threshold breaches", ["alert_type"]) if _Counter is not None else None

# ---------------------------------------------------------------------------
# Ray metrics bridge
# ---------------------------------------------------------------------------

_ray_gauges: dict[str, Any] = {}
_ray_counters: dict[str, Any] = {}


def register_ray_metrics() -> None:
    """Create Ray-native metric mirrors. Safe to call multiple times."""
    try:
        from ray.util.metrics import Counter as RayCounter, Gauge as RayGauge
    except ImportError:
        return

    if _ray_gauges:
        return

    _ray_gauges["sim_mae"] = RayGauge("marl_sim_mae", description="Simulation MAE", tag_keys=("seed", "disturbed"))
    _ray_gauges["sim_rmse"] = RayGauge("marl_sim_rmse", description="Simulation RMSE", tag_keys=("seed", "disturbed"))
    _ray_gauges["sim_rounds"] = RayGauge("marl_sim_rounds", description="Rounds executed", tag_keys=("seed",))
    _ray_counters["rounds_total"] = RayCounter("marl_game_rounds_total", description="Total game rounds")
    _ray_counters["disturbance_total"] = RayCounter("marl_disturbance_injections_total", description="Disturbance injection count")


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------

def record_simulation_metrics(
    seed: int,
    disturbed: bool,
    mae_val: float,
    rmse_val: float,
    mape_val: float,
    worst: float,
    duration: float,
    rounds: int,
    scenario: str = "default",
) -> None:
    """Record end-of-simulation metrics to Prometheus gauges and Ray mirrors."""
    labels = {"seed": str(seed), "disturbed": str(disturbed).lower(), "scenario": scenario}
    if SIM_MAE is not None:
        SIM_MAE.labels(**labels).set(mae_val)
    if SIM_RMSE is not None:
        SIM_RMSE.labels(**labels).set(rmse_val)
    if SIM_MAPE is not None:
        SIM_MAPE.labels(**labels).set(mape_val)
    if SIM_WORST is not None:
        SIM_WORST.labels(**labels).set(worst)
    if SIM_DURATION is not None:
        SIM_DURATION.observe(duration)
    if SIM_ROUNDS is not None:
        SIM_ROUNDS.labels(seed=str(seed)).set(rounds)

    g = _ray_gauges.get("sim_mae")
    if g is not None:
        g.set(mae_val, tags=labels)
    g = _ray_gauges.get("sim_rmse")
    if g is not None:
        g.set(rmse_val, tags=labels)
    g = _ray_gauges.get("sim_rounds")
    if g is not None:
        g.set(rounds, tags={"seed": str(seed)})


def record_agent_metrics(actor: str, role: str, delta: float, reward: float) -> None:
    """Record per-agent delta and cumulative reward to Prometheus."""
    if AGENT_DELTA is not None:
        AGENT_DELTA.labels(actor=actor, role=role).observe(delta)
    if AGENT_REWARD is not None and reward > 0:
        AGENT_REWARD.labels(actor=actor).inc(reward)


def record_disturbance(injected: bool, increased_error: bool) -> None:
    """Increment disturbance injection and success counters."""
    if injected and DISTURBANCE_COUNT is not None:
        DISTURBANCE_COUNT.inc()
        c = _ray_counters.get("disturbance_total")
        if c is not None:
            c.inc()
    if injected and increased_error and DISTURBANCE_SUCCESS is not None:
        DISTURBANCE_SUCCESS.inc()


def record_alert(alert_type: str) -> None:
    """Increment the anomaly alert counter for the given *alert_type*."""
    if ALERT_ANOMALY is not None:
        ALERT_ANOMALY.labels(alert_type=alert_type).inc()


# ---------------------------------------------------------------------------
# OpenTelemetry tracing
# ---------------------------------------------------------------------------

_tracer: Any = None

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor

    _tp = _TracerProvider()
    # Enable OTLP exporter only when explicitly configured to avoid noisy connection
    # errors in local/CI environments without a collector.
    _otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if _otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as _OTLPExporter

            _tp.add_span_processor(_BatchSpanProcessor(_OTLPExporter(endpoint=_otlp_endpoint)))
        except Exception:
            pass
    _otel_trace.set_tracer_provider(_tp)

    _tracer = _otel_trace.get_tracer("marl_forecast_game")
except ImportError:
    _tracer = None


def get_tracer() -> Any | None:
    """Return the OpenTelemetry tracer, or None if tracing is unavailable."""
    return _tracer


@contextmanager
def create_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Context manager that yields an OpenTelemetry span, or a no-op if tracing is unavailable."""
    if _tracer is not None:
        with _tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span
    else:
        yield None


# ---------------------------------------------------------------------------
# Correlation context for structured logs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrelationContext:
    """Carries trace/span IDs and simulation seed for structured-log correlation."""

    trace_id: str = ""
    span_id: str = ""
    simulation_seed: int = 0


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameObserver:
    """Factory for structured or stdlib loggers used by ForecastGame."""

    logger_name: str = "forecast_game"

    def __post_init__(self) -> None:
        logging.basicConfig(level=logging.WARNING)

    def logger(self) -> Any:
        if _structlog is not None:
            return _structlog.get_logger(self.logger_name)
        return logging.getLogger(self.logger_name)


# ---------------------------------------------------------------------------
# Prometheus exposition
# ---------------------------------------------------------------------------

def export_prometheus_metrics() -> str:
    """Return all Prometheus metrics in text exposition format."""
    if _generate_latest is None:
        return ""
    return _generate_latest().decode("utf-8")


def start_metrics_server(port: int | None = None) -> bool:
    """Start a lightweight HTTP server on /metrics if prometheus_client is installed."""
    import os
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    resolved_port = port or int(os.getenv("METRICS_PORT", "0"))
    if resolved_port <= 0 or _generate_latest is None:
        return False

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/metrics":
                body = export_prometheus_metrics().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    try:
        server = HTTPServer(("0.0.0.0", resolved_port), MetricsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logging.getLogger("forecast_game").info("Metrics server started on port %d", resolved_port)
        return True
    except Exception:
        return False
