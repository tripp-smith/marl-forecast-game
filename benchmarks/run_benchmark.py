#!/usr/bin/env python3
"""Reproducible benchmark and ablation harness for marl-forecast-game."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.stats import wilcoxon

from framework.data import DataProfile, load_csv, load_dataset, normalize_features
from framework.game import ForecastGame
from framework.metrics import interval_coverage, mae, mean_crps, rmse
from framework.types import ForecastState, SimulationConfig


def _approx_std(lower: float, upper: float) -> float:
    return max(1e-6, (upper - lower) / 3.92)


def _format_float(value: float) -> str:
    return f"{value:.4f}"


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    mae: float
    rmse: float
    crps: float
    coverage: float
    n_forecasts: int
    abs_errors: tuple[float, ...]


def _paired_wilcoxon(baseline: tuple[float, ...], candidate: tuple[float, ...]) -> float:
    pairs = [(left, right) for left, right in zip(baseline, candidate) if not math.isclose(left, right)]
    if len(pairs) < 5:
        return 1.0
    stat = wilcoxon([left for left, _ in pairs], [right for _, right in pairs], zero_method="wilcox")
    return float(stat.pvalue)


def _window_starts(n_rows: int, window_size: int, step_size: int, max_windows: int) -> list[int]:
    starts: list[int] = []
    start = 0
    while start + window_size + step_size <= n_rows and len(starts) < max_windows:
        starts.append(start)
        start += step_size
    return starts


def _evaluate_naive(rows: list[dict[str, float]], window_size: int, step_size: int, max_windows: int, *, k: int) -> BenchmarkResult:
    actuals: list[float] = []
    preds: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for start in _window_starts(len(rows), window_size, step_size, max_windows):
        train_rows = rows[start:start + window_size]
        test_rows = rows[start + window_size:start + window_size + step_size]
        anchor = [float(row["target"]) for row in train_rows[-k:]]
        prediction = mean(anchor)
        dispersion = max(0.5, max(anchor) - min(anchor))

        for row in test_rows:
            target = float(row["target"])
            actuals.append(target)
            preds.append(prediction)
            lowers.append(prediction - dispersion)
            uppers.append(prediction + dispersion)

    abs_errors = tuple(abs(actual - pred) for actual, pred in zip(actuals, preds))
    stds = [_approx_std(lo, hi) for lo, hi in zip(lowers, uppers)]
    return BenchmarkResult(
        name="naive_last" if k == 1 else f"moving_average_{k}",
        mae=mae(actuals, preds),
        rmse=rmse(actuals, preds),
        crps=mean_crps(actuals, preds, stds),
        coverage=interval_coverage(actuals, lowers, uppers),
        n_forecasts=len(preds),
        abs_errors=abs_errors,
    )


def _evaluate_game(
    rows: list[dict[str, float]],
    cfg: SimulationConfig,
    window_size: int,
    step_size: int,
    max_windows: int,
    *,
    disturbed: bool,
    label: str,
    seed: int,
) -> BenchmarkResult:
    actuals: list[float] = []
    preds: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for idx, start in enumerate(_window_starts(len(rows), window_size, step_size, max_windows)):
        train_rows = rows[start:start + window_size]
        test_rows = rows[start + window_size:start + window_size + step_size]
        init = ForecastState(
            t=0,
            value=float(train_rows[-1]["target"]),
            exogenous=0.0,
            hidden_shift=0.0,
        )
        out = ForecastGame(cfg, seed=seed + idx).run(init, rounds=len(test_rows), disturbed=disturbed)
        forecasts = out.forecasts[:len(test_rows)]
        confidence = out.confidence[:len(test_rows)]
        targets = [float(row["target"]) for row in test_rows[:len(forecasts)]]

        actuals.extend(targets)
        preds.extend(forecasts)
        lowers.extend(interval.lower for interval in confidence)
        uppers.extend(interval.upper for interval in confidence)

    abs_errors = tuple(abs(actual - pred) for actual, pred in zip(actuals, preds))
    stds = [_approx_std(lo, hi) for lo, hi in zip(lowers, uppers)]
    return BenchmarkResult(
        name=label,
        mae=mae(actuals, preds),
        rmse=rmse(actuals, preds),
        crps=mean_crps(actuals, preds, stds),
        coverage=interval_coverage(actuals, lowers, uppers),
        n_forecasts=len(preds),
        abs_errors=abs_errors,
    )


def _render_markdown(results: list[BenchmarkResult], p_values: dict[str, float]) -> str:
    lines = [
        "# Benchmark Summary",
        "",
        "| Model | MAE | RMSE | Approx. CRPS | 90% Coverage | Forecasts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {_format_float(result.mae)} | {_format_float(result.rmse)} | "
            f"{_format_float(result.crps)} | {_format_float(result.coverage)} | {result.n_forecasts} |"
        )

    lines.extend(
        [
            "",
            "## Significance Checks",
            "",
            "| Comparison | Wilcoxon p-value |",
            "|---|---:|",
        ]
    )
    for label, p_value in p_values.items():
        lines.append(f"| {label} | {_format_float(p_value)} |")
    return "\n".join(lines) + "\n"


def _load_rows(source: str, periods: int) -> list[dict[str, float]]:
    if source == "sample_csv":
        sample_path = ROOT / "data" / "sample_demand.csv"
        rows = load_csv(sample_path)[:periods]
        return normalize_features(rows)

    bundle = load_dataset(DataProfile(source=source, periods=periods, normalize=True))
    return bundle.train + bundle.valid + bundle.test


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible benchmarks and ablations.")
    parser.add_argument("--source", default="sample_csv", help="Dataset source for framework.data.load_dataset")
    parser.add_argument("--periods", type=int, default=240)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=12)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _load_rows(args.source, args.periods)

    clean_cfg = SimulationConfig(
        horizon=args.step_size,
        max_rounds=args.step_size,
        disturbance_prob=0.2,
        disturbance_scale=1.0,
        adversarial_intensity=1.0,
        disturbance_model="gaussian",
        defense_model="ensemble",
        enable_refactor=True,
    )
    undefended_cfg = SimulationConfig(
        horizon=args.step_size,
        max_rounds=args.step_size,
        disturbance_prob=0.2,
        disturbance_scale=1.0,
        adversarial_intensity=1.0,
        disturbance_model="gaussian",
        defense_model="identity",
        enable_refactor=True,
    )

    results = [
        _evaluate_naive(rows, args.window_size, args.step_size, args.windows, k=1),
        _evaluate_naive(rows, args.window_size, args.step_size, args.windows, k=5),
        _evaluate_game(
            rows,
            clean_cfg,
            args.window_size,
            args.step_size,
            args.windows,
            disturbed=False,
            label="game_clean_ensemble",
            seed=args.seed,
        ),
        _evaluate_game(
            rows,
            undefended_cfg,
            args.window_size,
            args.step_size,
            args.windows,
            disturbed=True,
            label="game_attack_identity",
            seed=args.seed,
        ),
        _evaluate_game(
            rows,
            clean_cfg,
            args.window_size,
            args.step_size,
            args.windows,
            disturbed=True,
            label="game_attack_ensemble",
            seed=args.seed,
        ),
    ]

    naive_errors = results[0].abs_errors
    p_values = {
        "game_attack_ensemble vs naive_last": _paired_wilcoxon(naive_errors, results[-1].abs_errors),
        "game_attack_ensemble vs game_attack_identity": _paired_wilcoxon(results[-2].abs_errors, results[-1].abs_errors),
    }

    payload = {
        "source": args.source,
        "seed": args.seed,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "windows": args.windows,
        "results": [
            {
                "name": result.name,
                "mae": result.mae,
                "rmse": result.rmse,
                "crps": result.crps,
                "coverage": result.coverage,
                "n_forecasts": result.n_forecasts,
            }
            for result in results
        ],
        "p_values": p_values,
    }

    output_dir = ROOT / "results" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_summary.json"
    md_path = output_dir / "benchmark_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(results, p_values), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
