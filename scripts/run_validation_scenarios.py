#!/usr/bin/env python3
"""CLI runner for multi-scenario data validation.

Usage:
    python scripts/run_validation_scenarios.py --scenarios all
    python scripts/run_validation_scenarios.py --scenarios fred_cpi_clean,adversarial_gaussian
    python scripts/run_validation_scenarios.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.validation_scenarios import SCENARIO_REGISTRY, run_all_scenarios


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation scenarios for MARL Forecast Game")
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated scenario names, or 'all' (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    parser.add_argument(
        "--output-dir",
        default="planning",
        help="Directory for report output (default: planning)",
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'Name':<30} {'Description'}")
        print("-" * 80)
        for name, scenario in SCENARIO_REGISTRY.items():
            print(f"{name:<30} {scenario.description}")
        return 0

    names = None if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(",")]
    start = time.perf_counter()
    results = run_all_scenarios(names)
    total_duration = time.perf_counter() - start

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    print("=" * 80)
    print("VALIDATION SCENARIO RESULTS")
    print("=" * 80)
    print(f"{'Scenario':<35} {'Status':<10} {'Duration':<12} {'Errors'}")
    print("-" * 80)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        err_summary = "; ".join(r.errors[:2]) if r.errors else ""
        print(f"{r.name:<35} {status:<10} {r.duration_s:<12.4f} {err_summary}")
    print("-" * 80)
    print(f"Total: {passed} passed, {failed} failed, {total_duration:.2f}s elapsed")
    print("=" * 80)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_scenarios": len(results),
        "passed": passed,
        "failed": failed,
        "total_duration_s": round(total_duration, 4),
        "scenarios": [asdict(r) for r in results],
    }

    report_path = output_dir / "validation_scenarios_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report written to {report_path}")

    summary_path = output_dir / "validation_scenarios_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("scenario,passed,duration_s,errors\n")
        for r in results:
            err = "; ".join(r.errors).replace('"', "'")
            f.write(f'{r.name},{r.passed},{r.duration_s},"{err}"\n')
    print(f"Summary written to {summary_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
