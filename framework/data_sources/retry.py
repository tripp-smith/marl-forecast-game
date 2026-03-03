"""Retry and rate-limiting utilities for HTTP data-source adapters.

Sub-phase: G (Data Pipeline Hardening).
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError

_logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    retryable_statuses: tuple[int, ...] = (429, 500, 502, 503),
) -> Callable[[F], F]:
    """Decorator that retries a function on :class:`HTTPError` with exponential backoff.

    Non-retryable HTTP errors (e.g. 4xx other than 429) are raised immediately.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except HTTPError as exc:
                    last_exc = exc
                    if exc.code not in retryable_statuses:
                        raise
                    wait = backoff_base * (2 ** attempt)
                    _logger.warning(
                        "Retry %d/%d for %s (HTTP %d), waiting %.1fs",
                        attempt + 1, max_attempts, fn.__name__, exc.code, wait,
                    )
                    time.sleep(wait)
                except Exception as exc:
                    last_exc = exc
                    wait = backoff_base * (2 ** attempt)
                    _logger.warning(
                        "Retry %d/%d for %s (%s), waiting %.1fs",
                        attempt + 1, max_attempts, fn.__name__, exc, wait,
                    )
                    time.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Call :meth:`acquire` before each HTTP request to enforce the configured
    ``calls_per_second`` ceiling.
    """

    def __init__(self, calls_per_second: float = 5.0) -> None:
        self._min_interval = 1.0 / max(0.01, calls_per_second)
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()
