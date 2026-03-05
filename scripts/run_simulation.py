#!/usr/bin/env python3
"""Run a forecast simulation with configurable agent composition."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.agents import AgentRegistry, create_agent
from framework.game import ForecastGame
from framework.types import ForecastState, SimulationConfig


def _parse_agent_specs(raw: str, q_table: str | None, algorithm: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        if token == "qlearned-adversary":
            specs.append(
                {
                    "type": "qlearned",
                    "name": "qlearned-adversary",
                    "kwargs": {"role": "adversary", "q_table_path": q_table, "algorithm": algorithm},
                }
            )
            continue
        if token == "qlearned-forecaster":
            specs.append(
                {
                    "type": "qlearned",
                    "name": "qlearned-forecaster",
                    "kwargs": {"role": "forecaster", "q_table_path": q_table, "algorithm": algorithm},
                }
            )
            continue
        specs.append({"type": token, "name": token})
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live simulation with optional trained MARL agents")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disturbed", action="store_true", help="Enable disturbance/adversary effects")
    parser.add_argument("--agent-type", default="forecaster", help="Single forecaster type (e.g., forecaster, qlearned)")
    parser.add_argument("--name", default="forecaster", help="Name for single-agent mode")
    parser.add_argument("--q-table", default=None, help="Path to saved Q-table/policy")
    parser.add_argument("--algorithm", choices=["q", "wolf", "rarl"], default="q")
    parser.add_argument(
        "--agents",
        default="",
        help="Comma-separated composition, e.g. 'forecaster,qlearned-adversary,defender'",
    )
    args = parser.parse_args()

    cfg = SimulationConfig(horizon=args.horizon, max_rounds=max(args.horizon * 2, args.horizon))
    init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

    if args.agents:
        registry = AgentRegistry.from_config(
            {
                "agents": _parse_agent_specs(args.agents, q_table=args.q_table, algorithm=args.algorithm),
            }
        )
    else:
        single = create_agent(
            agent_type=args.agent_type,
            name=args.name,
            q_table_path=args.q_table,
            q_table=args.q_table,
            algorithm=args.algorithm,
        )
        registry = AgentRegistry(
            forecasters=(single,) if args.agent_type not in {"adversary", "wolfpack"} else (),
            adversaries=(single,) if args.agent_type in {"adversary", "wolfpack"} else (),
            defenders=(),
        )

    game = ForecastGame(cfg, seed=args.seed, registry=registry)
    out = game.run(init, disturbed=args.disturbed)
    print(f"rounds={len(out.steps)}")
    print(f"q_table_loaded={bool(args.q_table)}")
    print(f"agents={','.join([a.actor for a in out.steps[0].actions]) if out.steps else ''}")
    print(json.dumps(out.convergence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
