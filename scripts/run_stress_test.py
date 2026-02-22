"""L.5: Stress test -- parallel simulations measuring throughput and stability."""
from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.distributed import ParallelGameRunner
from framework.types import ForecastState, SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test: parallel game simulations")
    parser.add_argument("--games", type=int, default=100, help="Number of concurrent games")
    parser.add_argument("--rounds", type=int, default=1000, help="Rounds per game")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    config = SimulationConfig(
        horizon=args.rounds,
        max_rounds=args.rounds,
        disturbance_prob=0.2,
        disturbance_model="gaussian",
        defense_model="ensemble",
    )
    init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
    seeds = list(range(args.games))

    runner = ParallelGameRunner(n_workers=args.workers)

    tracemalloc.start()
    start = time.perf_counter()

    results = runner.run_seeds(config, init, seeds=seeds, disturbed=True)

    elapsed = time.perf_counter() - start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_rounds = sum(r["convergence"]["rounds_executed"] for r in results)
    crashes = sum(1 for r in results if r["convergence"]["rounds_executed"] < args.rounds)
    throughput = total_rounds / elapsed if elapsed > 0 else 0

    print(f"Games:           {args.games}")
    print(f"Rounds/game:     {args.rounds}")
    print(f"Workers:         {args.workers}")
    print(f"Total rounds:    {total_rounds}")
    print(f"Elapsed:         {elapsed:.2f}s")
    print(f"Throughput:      {throughput:.0f} rounds/s")
    print(f"Peak memory:     {peak_mem / (1024 * 1024):.1f} MB")
    print(f"Crashes:         {crashes}")

    assert crashes == 0, f"{crashes} games did not complete all rounds"
    print("\nStress test PASSED")


if __name__ == "__main__":
    main()
