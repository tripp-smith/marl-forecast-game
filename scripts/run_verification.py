from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.verify import run_verification


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
    args = parser.parse_args()

    result = run_verification(backend=args.backend, enable_qual=args.enable_qual)
    report_path = ROOT / "planning" / "verification_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"verification_report={report_path}")
    if not all(result["checks"].values()):
        raise SystemExit(1)
