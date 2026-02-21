from __future__ import annotations

from dataclasses import dataclass
import logging

try:
    import structlog
except Exception:  # pragma: no cover
    structlog = None

try:
    from prometheus_client import Counter, Histogram, generate_latest
except Exception:  # pragma: no cover
    Counter = None
    Histogram = None
    generate_latest = None


@dataclass(frozen=True)
class GameObserver:
    logger_name: str = "forecast_game"

    def __post_init__(self) -> None:
        logging.basicConfig(level=logging.WARNING)

    def logger(self):
        if structlog is not None:
            return structlog.get_logger(self.logger_name)
        return logging.getLogger(self.logger_name)


ROUND_COUNTER = Counter("marl_game_rounds_total", "Total game rounds") if Counter is not None else None
ROUND_LATENCY = (
    Histogram("marl_game_round_latency_seconds", "Round execution latency in seconds") if Histogram is not None else None
)


def export_prometheus_metrics() -> str:
    if generate_latest is None:
        return ""
    return generate_latest().decode("utf-8")
