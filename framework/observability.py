from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable

try:
    import structlog as _structlog
except Exception:  # pragma: no cover
    _structlog = None  # type: ignore[assignment]

_Counter: Any = None
_Histogram: Any = None
_generate_latest: Callable[..., bytes] | None = None

try:
    from prometheus_client import Counter as _Counter  # type: ignore[assignment]
    from prometheus_client import Histogram as _Histogram  # type: ignore[assignment]
    from prometheus_client import generate_latest as _generate_latest  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pass


@dataclass(frozen=True)
class GameObserver:
    logger_name: str = "forecast_game"

    def __post_init__(self) -> None:
        logging.basicConfig(level=logging.WARNING)

    def logger(self) -> Any:
        if _structlog is not None:
            return _structlog.get_logger(self.logger_name)
        return logging.getLogger(self.logger_name)


ROUND_COUNTER = _Counter("marl_game_rounds_total", "Total game rounds") if _Counter is not None else None
ROUND_LATENCY = (
    _Histogram("marl_game_round_latency_seconds", "Round execution latency in seconds") if _Histogram is not None else None
)


def export_prometheus_metrics() -> str:
    if _generate_latest is None:
        return ""
    return _generate_latest().decode("utf-8")


def start_metrics_server(port: int | None = None) -> bool:
    """Start a lightweight HTTP server on /metrics if prometheus_client is installed.

    Controlled by METRICS_PORT env var or the port argument.
    Returns True if server started, False otherwise.
    """
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
