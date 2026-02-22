#!/usr/bin/env python3
"""CLI for walk-forward backtesting on FRED or synthetic data.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --windows 10 --window-size 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.backtesting import WalkForwardBacktester
from framework.data import DataProfile, load_dataset
from framework.types import SimulationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward backtesting")
    parser.add_argument("--windows", type=int, default=10, help="Max windows")
    parser.add_argument("--window-size", type=int, default=60, help="Training window size")
    parser.add_argument("--step-size", type=int, default=20, help="Step size between windows")
    parser.add_argument("--output-dir", default="planning", help="Report output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = "fred_training" if os.getenv("FRED_API_KEY") else "sample_csv"
    print(f"[backtest] Data source: {source}")

    periods = args.window_size * args.windows + args.step_size * args.windows
    bundle = load_dataset(DataProfile(source=source, periods=max(240, periods)))
    all_rows = bundle.train + bundle.valid + bundle.test

    cfg = SimulationConfig(
        horizon=args.step_size,
        max_rounds=args.step_size * 2,
        disturbance_prob=0.2,
        disturbance_scale=1.2,
        adversarial_intensity=1.0,
        disturbance_model="gaussian",
        defense_model="ensemble",
        enable_refactor=True,
    )

    backtester = WalkForwardBacktester(
        config=cfg,
        window_size=args.window_size,
        step_size=args.step_size,
        seed=args.seed,
    )

    start = time.perf_counter()
    result = backtester.run(all_rows, max_windows=args.windows)
    elapsed = time.perf_counter() - start

    print(f"\n{'Window':<10} {'MAE':<12} {'RMSE':<12} {'Forecasts':<12}")
    print("-" * 46)
    for w in result.window_results:
        print(f"{w.window_idx:<10} {w.mae:<12.4f} {w.rmse:<12.4f} {w.n_forecasts:<12}")
    print("-" * 46)
    print(f"Aggregate MAE: {result.aggregate_mae:.4f}")
    print(f"Aggregate RMSE: {result.aggregate_rmse:.4f}")
    print(f"Windows: {result.n_windows}, elapsed: {elapsed:.2f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "backtest_report.json"

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "n_windows": result.n_windows,
        "aggregate_mae": result.aggregate_mae,
        "aggregate_rmse": result.aggregate_rmse,
        "elapsed_s": round(elapsed, 4),
        "windows": [
            {
                "idx": w.window_idx,
                "train_range": [w.train_start, w.train_end],
                "test_range": [w.test_start, w.test_end],
                "mae": w.mae,
                "rmse": w.rmse,
                "n_forecasts": w.n_forecasts,
            }
            for w in result.window_results
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
