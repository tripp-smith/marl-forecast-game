from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.agents import AdversaryAgent, AgentRegistry, DefenderAgent, ForecastingAgent, QLearnedAgent
from framework.game import ForecastGame
from framework.llm.ollama_interface import OllamaInterface
from framework.metrics import mae, mean_crps
from framework.training import QTableAgent, TrainingLoop
from framework.types import ForecastState, SimulationConfig
from framework.verify import run_verification


def _run_trained_vs_rulebased() -> dict[str, object]:
    model_path = ROOT / "data" / "models" / "test_qlearned.pkl"
    if not model_path.exists():
        TrainingLoop.save_q_table(QTableAgent(epsilon=0.0), model_path)

    cfg = SimulationConfig(horizon=10, max_rounds=20)
    init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

    rule_registry = AgentRegistry(
        forecasters=(ForecastingAgent(name="forecaster"),),
        adversaries=(AdversaryAgent(name="random-adversary"),),
        defenders=(DefenderAgent(name="defender"),),
    )
    trained_registry = AgentRegistry(
        forecasters=(ForecastingAgent(name="forecaster"),),
        adversaries=(QLearnedAgent(name="qlearned-adversary", q_table_path=str(model_path), algorithm="q"),),
        defenders=(DefenderAgent(name="defender"),),
    )

    rule_out = ForecastGame(cfg, seed=42, registry=rule_registry).run(init, disturbed=True)
    trained_out = ForecastGame(cfg, seed=42, registry=trained_registry).run(init, disturbed=True)

    def _stds(output: object) -> list[float]:
        return [max(1e-6, (s.confidence.upper - s.confidence.lower) / 3.92) for s in output.steps]  # type: ignore[attr-defined]

    rule_mae = mae(rule_out.targets, rule_out.forecasts)
    trained_mae = mae(trained_out.targets, trained_out.forecasts)
    rule_crps = mean_crps(rule_out.targets, rule_out.forecasts, _stds(rule_out))
    trained_crps = mean_crps(trained_out.targets, trained_out.forecasts, _stds(trained_out))

    passed = all(math.isfinite(v) for v in [rule_mae, trained_mae, rule_crps, trained_crps])
    return {
        "scenario": "trained_vs_rulebased",
        "passed": passed,
        "rulebased": {"mae": rule_mae, "crps": rule_crps},
        "trained_adversary": {"mae": trained_mae, "crps": trained_crps},
    }


def main() -> int:
    started = datetime.utcnow().isoformat()
    pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
    pytest_proc = subprocess.run(pytest_cmd, cwd=ROOT, capture_output=True, text=True)

    verification = run_verification()
    checks_ok = all(verification["checks"].values())
    marl_integration = _run_trained_vs_rulebased()

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
        "marl_integration": marl_integration,
        "ollama_failure_analysis": analysis,
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "test_suite_report.json"
    csv_path = out_dir / "test_suite_summary.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed"])
        writer.writeheader()
        for check, passed in verification["checks"].items():
            writer.writerow({"check": check, "passed": bool(passed)})
        writer.writerow({"check": "marl_integration", "passed": bool(marl_integration["passed"])})

    print(f"report_json={json_path}")
    print(f"report_csv={csv_path}")
    print(f"MARL integration: {'PASSED' if marl_integration['passed'] else 'FAILED'}")
    return 0 if pytest_proc.returncode == 0 and checks_ok and bool(marl_integration["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
