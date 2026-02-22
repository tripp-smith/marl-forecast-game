#!/usr/bin/env python3
"""CLI for training MARL agents on FRED or synthetic data.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --episodes 100 --horizon 40
    python scripts/run_training.py --algorithm wolf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.data import DataProfile, load_dataset
from framework.training import (
    DiscreteActionSpace,
    QTableAgent,
    RADversarialTrainer,
    TrainingLoop,
    WoLFPHCAgent,
)
from framework.types import ForecastState, SimulationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MARL agents")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes")
    parser.add_argument("--horizon", type=int, default=80, help="Rounds per episode")
    parser.add_argument("--algorithm", choices=["q", "wolf", "rarl"], default="wolf", help="Training algorithm")
    parser.add_argument("--output-dir", default="data/models", help="Output directory for Q-tables")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    source = "fred_training" if os.getenv("FRED_API_KEY") else "sample_csv"
    print(f"[training] Data source: {source}")
    bundle = load_dataset(DataProfile(source=source, periods=max(120, args.horizon * 2)))

    init_value = float(bundle.train[-1]["target"]) if bundle.train else 10.0
    init_state = ForecastState(t=0, value=init_value, exogenous=0.0, hidden_shift=0.0)

    cfg = SimulationConfig(
        horizon=args.horizon,
        max_rounds=args.horizon * 2,
        disturbance_prob=0.2,
        disturbance_scale=1.2,
        adversarial_intensity=1.0,
        disturbance_model="gaussian",
        defense_model="ensemble",
        enable_refactor=False,
    )

    action_space = DiscreteActionSpace(n_bins=21, max_delta=1.0)

    if args.algorithm == "wolf":
        forecaster = WoLFPHCAgent(action_space=action_space)
    else:
        forecaster = QTableAgent(action_space=action_space)

    if args.algorithm == "rarl":
        adversary = QTableAgent(action_space=action_space)
        trainer = RADversarialTrainer(config=cfg, total_epochs=args.episodes, seed=args.seed)
        result = trainer.train(forecaster, adversary, init_state)
        print(f"[training] RARL complete: {result['total_epochs']} epochs")
    else:
        loop = TrainingLoop(config=cfg, n_episodes=args.episodes, seed=args.seed)
        result = loop.train(forecaster, init_state=init_state)
        print(f"[training] Complete: {result['n_episodes']} episodes")
        print(f"[training] Mean reward (last 50): {result['mean_reward_last_50']:.4f}")
        print(f"[training] Mean TD error (last 100): {result['mean_td_error_last_100']:.4f}")
        print(f"[training] Final epsilon: {result['final_epsilon']:.4f}")

    output_dir = Path(args.output_dir)
    TrainingLoop.save_q_table(forecaster, output_dir / "forecaster_q.json")
    print(f"[training] Q-table saved to {output_dir / 'forecaster_q.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
