from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.verify import run_verification


def _load_config_from_yaml(path: str | Path) -> dict:
    """Load and validate a YAML config file against SimulationConfig fields."""
    import yaml
    from framework.types import SimulationConfig
    import dataclasses

    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping")

    valid_fields = {f.name for f in dataclasses.fields(SimulationConfig)}
    unknown = set(data.keys()) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config fields: {unknown}")

    SimulationConfig(**data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run verification checks")
    parser.add_argument(
        "--backend",
        choices=["auto", "ray", "multiprocessing"],
        default="auto",
        help="Parallel execution backend (default: auto)",
    )
    parser.add_argument(
        "--enable-qual",
        action="store_true",
        default=False,
        help="Include qualitative determinism verification checks",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file for SimulationConfig overrides",
    )
    args = parser.parse_args()

    if args.config:
        cfg_overrides = _load_config_from_yaml(args.config)
        print(f"Loaded config overrides from {args.config}: {list(cfg_overrides.keys())}")

    result = run_verification(backend=args.backend, enable_qual=args.enable_qual)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "verification_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"verification_report={report_path}")
    if not all(result["checks"].values()):
        raise SystemExit(1)
