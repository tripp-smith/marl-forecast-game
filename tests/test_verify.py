from __future__ import annotations

from framework.verify import run_verification


def test_run_verification_multiprocessing_reports_parallel_runner_available() -> None:
    result = run_verification(backend="multiprocessing")
    assert result["checks"]["parallel_runner_available"] is True
