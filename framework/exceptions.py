"""Custom exception hierarchy for the MARL Forecast Game framework.

All domain-specific exceptions inherit from :class:`MARLError` so that callers
can catch a single base class while retaining compatibility with built-in
exception handlers (``except Exception``).

Sub-phase: K (Technical Debt Cleanup).
"""
from __future__ import annotations


class MARLError(Exception):
    """Base exception for all MARL Forecast Game errors."""


class DataIngestionError(MARLError, ValueError):
    """Raised for data loading, parsing, or validation failures.

    Inherits from ``ValueError`` for backward compatibility with existing
    ``except ValueError`` callers.
    """


class PoisoningDetectedError(DataIngestionError):
    """Raised when statistical poisoning detection flags contaminated data."""


class AdapterFetchError(DataIngestionError):
    """Raised when a data-source adapter fails to retrieve data."""


class LLMUnavailableError(MARLError):
    """Raised when an LLM endpoint is unreachable or returns an error."""


class ConfigValidationError(MARLError, ValueError):
    """Raised when a configuration value is invalid.

    Inherits from ``ValueError`` for backward compatibility.
    """


class SimulationError(MARLError, RuntimeError):
    """Raised for errors during simulation execution.

    Inherits from ``RuntimeError`` for backward compatibility.
    """


class ConvergenceError(SimulationError):
    """Raised when the simulation fails to converge."""
