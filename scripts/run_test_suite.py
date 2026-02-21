from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.llm.ollama_interface import OllamaInterface
from framework.verify import run_verification


def main() -> int:
    started = datetime.utcnow().isoformat()
    pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
    pytest_proc = subprocess.run(pytest_cmd, cwd=ROOT, capture_output=True, text=True)

    verification = run_verification()
    checks_ok = all(verification["checks"].values())

    analysis = ""
    ollama = OllamaInterface()
    if ollama.is_available() and pytest_proc.returncode != 0:
        prompt = f"Summarize likely causes of this failing pytest output:\n{pytest_proc.stdout[-2000:]}\n{pytest_proc.stderr[-2000:]}"
        analysis = ollama.generate(prompt).get("response", "")

    report = {
        "started_at": started,
        "finished_at": datetime.utcnow().isoformat(),
        "pytest": {
            "cmd": " ".join(pytest_cmd),
            "returncode": pytest_proc.returncode,
            "stdout_tail": pytest_proc.stdout[-4000:],
            "stderr_tail": pytest_proc.stderr[-4000:],
        },
        "verification": verification,
        "ollama_failure_analysis": analysis,
    }

    out_dir = ROOT / "planning"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "test_suite_report.json"
    csv_path = out_dir / "test_suite_summary.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed"])
        writer.writeheader()
        for check, passed in verification["checks"].items():
            writer.writerow({"check": check, "passed": bool(passed)})

    print(f"report_json={json_path}")
    print(f"report_csv={csv_path}")
    return 0 if pytest_proc.returncode == 0 and checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
