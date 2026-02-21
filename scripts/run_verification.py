from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.verify import run_verification


if __name__ == "__main__":
    result = run_verification()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(result["checks"].values()):
        raise SystemExit(1)
