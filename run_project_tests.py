#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
MYPY_TARGETS = [
    "framework/types.py",
    "framework/game.py",
    "framework/metrics.py",
    "framework/observability.py",
    "framework/defenses.py",
    "framework/disturbances.py",
    "framework/distributed.py",
    "framework/ray_actors.py",
    "framework/rllib_env.py",
]


@dataclass
class StageResult:
    name: str
    command: str
    exit_code: int
    passed: bool
    duration_s: float
    log_path: str
    artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_report_dir() -> Path:
    return ROOT / "results" / "test-harness" / _timestamp()


def _cmd_display(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _write_log(log_path: Path, content: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


def _run_command(
    name: str,
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
) -> StageResult:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    duration = round(time.perf_counter() - started, 4)
    combined = []
    if proc.stdout:
        combined.append(proc.stdout)
    if proc.stderr:
        combined.append(proc.stderr)
    _write_log(log_path, "".join(combined))
    return StageResult(
        name=name,
        command=_cmd_display(cmd),
        exit_code=proc.returncode,
        passed=proc.returncode == 0,
        duration_s=duration,
        log_path=str(log_path),
    )


def _git_status_lines() -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git status failed")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _path_from_status(line: str) -> str:
    path = line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    return path


def _is_report_path(path: str, report_dir: Path) -> bool:
    try:
        rel_path = (ROOT / path).resolve().relative_to(report_dir.resolve())
        return rel_path is not None
    except ValueError:
        return False


def _filter_new_git_changes(before: list[str], after: list[str], report_dir: Path) -> list[str]:
    baseline = set(before)
    new_lines = [line for line in after if line not in baseline]
    return [line for line in new_lines if not _is_report_path(_path_from_status(line), report_dir)]


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _preflight_stage(log_path: Path, mode: str) -> tuple[StageResult, list[str]]:
    started = time.perf_counter()
    baseline = _git_status_lines()
    missing: list[str] = []
    required_modules = [
        "pytest",
        "hypothesis",
        "numpy",
        "networkx",
        "scipy",
        "sklearn",
        "yaml",
        "requests",
        "pydantic",
        "psutil",
    ]
    if mode == "full":
        required_modules.append("mypy")

    for module_name in required_modules:
        if not _module_available(module_name):
            missing.append(module_name)

    notes = [
        f"python={sys.version.split()[0]}",
        f"mode={mode}",
        f"baseline_git_entries={len(baseline)}",
    ]
    if sys.version_info < (3, 12):
        missing.append("python>=3.12")

    log_lines = [
        f"Python: {sys.version}",
        f"Mode: {mode}",
        "Baseline git status:",
        *baseline,
    ]
    if missing:
        log_lines.extend(["", "Missing requirements:", *missing])

    _write_log(log_path, "\n".join(log_lines) + "\n")
    duration = round(time.perf_counter() - started, 4)
    result = StageResult(
        name="preflight",
        command="internal: preflight",
        exit_code=0 if not missing else 1,
        passed=not missing,
        duration_s=duration,
        log_path=str(log_path),
        notes=notes + ([f"missing={','.join(missing)}"] if missing else []),
    )
    return result, baseline


def _ui_parse_stage(log_path: Path) -> StageResult:
    started = time.perf_counter()
    targets = [ROOT / "ui" / "app.py", *sorted((ROOT / "ui" / "pages").glob("*.py"))]
    notes: list[str] = []
    errors: list[str] = []
    for target in targets:
        try:
            ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
            notes.append(str(target.relative_to(ROOT)))
        except SyntaxError as exc:
            errors.append(f"{target}: {exc}")
    _write_log(log_path, "\n".join(notes + ([""] + errors if errors else [])) + "\n")
    duration = round(time.perf_counter() - started, 4)
    return StageResult(
        name="ui_parse",
        command="internal: ast.parse ui/app.py ui/pages/*.py",
        exit_code=0 if not errors else 1,
        passed=not errors,
        duration_s=duration,
        log_path=str(log_path),
        artifacts=[str(target) for target in targets],
        notes=errors or [f"parsed={len(targets)} files"],
    )


def _git_guard_stage(log_path: Path, before: list[str], report_dir: Path) -> StageResult:
    started = time.perf_counter()
    after = _git_status_lines()
    new_lines = _filter_new_git_changes(before, after, report_dir)
    lines = ["Post-run git status:", *after, "", "New changes outside report dir:", *new_lines]
    _write_log(log_path, "\n".join(lines) + "\n")
    duration = round(time.perf_counter() - started, 4)
    return StageResult(
        name="git_guard",
        command="internal: git status diff",
        exit_code=0 if not new_lines else 1,
        passed=not new_lines,
        duration_s=duration,
        log_path=str(log_path),
        notes=new_lines or ["no new git status entries outside report dir"],
    )


def _artifacts(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


def _stage_specs(mode: str, report_dir: Path, backend: str) -> list[tuple[str, list[str], list[Path], dict[str, str]]]:
    python = sys.executable
    cache_dir = report_dir / "cache"
    preference_dir = report_dir / "preferences"
    common_env = {
        "MFG_CACHE_DIR": str(cache_dir),
        "MFG_PREFERENCE_DIR": str(preference_dir),
    }
    verification_stage = (
        "verification",
        [
            python,
            "scripts/run_verification.py",
            "--backend",
            backend,
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(report_dir / "verification"),
        ],
        [report_dir / "verification" / "verification_report.json"],
        common_env,
    )
    reproducibility_stage = (
        "reproducibility",
        [
            python,
            "experiments/run_reproducibility_check.py",
            "--output-dir",
            str(report_dir / "experiments"),
        ],
        [report_dir / "experiments" / "reproducibility_check.json"],
        common_env,
    )
    training_wolf_stage = (
        "training_wolf",
        [
            python,
            "scripts/run_training.py",
            "--algorithm",
            "wolf",
            "--episodes",
            "10",
            "--horizon",
            "20",
            "--source",
            "sample_csv",
            "--output-dir",
            str(report_dir / "models" / "wolf"),
        ],
        [report_dir / "models" / "wolf" / "forecaster_q.json"],
        common_env,
    )
    simulation_rulebased_stage = (
        "simulation_rulebased",
        [
            python,
            "scripts/run_simulation.py",
            "--horizon",
            "5",
            "--disturbed",
            "--agents",
            "forecaster,adversary,defender",
        ],
        [],
        common_env,
    )
    backtest_stage = (
        "backtest",
        [
            python,
            "scripts/run_backtest.py",
            "--source",
            "sample_csv",
            "--windows",
            "3",
            "--window-size",
            "40",
            "--step-size",
            "15",
            "--output-dir",
            str(report_dir / "backtest"),
        ],
        [report_dir / "backtest" / "backtest_report.json"],
        common_env,
    )

    if mode == "quick":
        return [
            ("pytest", [python, "-m", "pytest", "-q"], [], common_env),
            verification_stage,
            reproducibility_stage,
            training_wolf_stage,
            simulation_rulebased_stage,
            backtest_stage,
        ]

    return [
        ("pytest", [python, "-m", "pytest", "-q"], [], common_env),
        verification_stage,
        (
            "validation_scenarios",
            [
                python,
                "scripts/run_validation_scenarios.py",
                "--scenarios",
                "all",
                "--cache-dir",
                str(cache_dir),
                "--output-dir",
                str(report_dir / "validation"),
            ],
            [
                report_dir / "validation" / "validation_scenarios_report.json",
                report_dir / "validation" / "validation_scenarios_summary.csv",
            ],
            common_env,
        ),
        reproducibility_stage,
        (
            "benchmark",
            [
                python,
                "benchmarks/run_benchmark.py",
                "--source",
                "sample_csv",
                "--windows",
                "4",
                "--window-size",
                "40",
                "--step-size",
                "10",
                "--output-dir",
                str(report_dir / "benchmarks"),
            ],
            [
                report_dir / "benchmarks" / "benchmark_summary.json",
                report_dir / "benchmarks" / "benchmark_summary.md",
            ],
            common_env,
        ),
        (
            "training_q",
            [
                python,
                "scripts/run_training.py",
                "--algorithm",
                "q",
                "--episodes",
                "10",
                "--horizon",
                "20",
                "--source",
                "sample_csv",
                "--output-dir",
                str(report_dir / "models" / "q"),
            ],
            [report_dir / "models" / "q" / "forecaster_q.json"],
            common_env,
        ),
        training_wolf_stage,
        (
            "training_rarl",
            [
                python,
                "scripts/run_training.py",
                "--algorithm",
                "rarl",
                "--episodes",
                "8",
                "--horizon",
                "20",
                "--source",
                "sample_csv",
                "--output-dir",
                str(report_dir / "models" / "rarl"),
            ],
            [report_dir / "models" / "rarl" / "forecaster_q.json"],
            common_env,
        ),
        (
            "training_mnpo",
            [
                python,
                "scripts/run_training.py",
                "--algorithm",
                "mnpo",
                "--episodes",
                "4",
                "--horizon",
                "20",
                "--source",
                "sample_csv",
                "--opponents",
                "3",
                "--preferences-dir",
                str(preference_dir),
                "--output-dir",
                str(report_dir / "models" / "mnpo"),
            ],
            [report_dir / "models" / "mnpo" / "forecaster_q.json"],
            common_env,
        ),
        simulation_rulebased_stage,
        (
            "simulation_qlearned",
            [
                python,
                "scripts/run_simulation.py",
                "--horizon",
                "5",
                "--disturbed",
                "--agents",
                "forecaster,qlearned-adversary,defender",
                "--q-table",
                str(report_dir / "models" / "q" / "forecaster_q.json"),
                "--algorithm",
                "q",
            ],
            [report_dir / "models" / "q" / "forecaster_q.json"],
            common_env,
        ),
        backtest_stage,
        (
            "stress_test",
            [
                python,
                "scripts/run_stress_test.py",
                "--games",
                "8",
                "--rounds",
                "100",
                "--workers",
                "2",
            ],
            [],
            common_env,
        ),
        (
            "mypy",
            [python, "-m", "mypy", *MYPY_TARGETS, "--ignore-missing-imports"],
            [],
            common_env,
        ),
    ]


def _summary_md(
    *,
    mode: str,
    report_dir: Path,
    started_at: str,
    finished_at: str,
    stages: list[StageResult],
    overall_passed: bool,
) -> str:
    lines = [
        "# Programmatic Test Harness",
        "",
        f"- Mode: `{mode}`",
        f"- Started: `{started_at}`",
        f"- Finished: `{finished_at}`",
        f"- Overall: `{'PASS' if overall_passed else 'FAIL'}`",
        f"- Report dir: `{report_dir}`",
        "",
        "| Stage | Status | Seconds | Command | Log |",
        "|---|---|---:|---|---|",
    ]
    for stage in stages:
        lines.append(
            f"| {stage.name} | {'PASS' if stage.passed else 'FAIL'} | {stage.duration_s:.2f} | "
            f"`{stage.command}` | `{Path(stage.log_path).name}` |"
        )

    failing = [stage for stage in stages if not stage.passed]
    if failing:
        lines.extend(["", "## Failures", ""])
        for stage in failing:
            note = "; ".join(stage.notes) if stage.notes else "See stage log."
            lines.append(f"- `{stage.name}`: {note}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the repo-wide programmatic test harness.")
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=["multiprocessing", "auto", "ray"], default="multiprocessing")
    args = parser.parse_args()

    report_dir = args.report_dir or _default_report_dir()
    stage_dir = report_dir / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    stages: list[StageResult] = []

    preflight, baseline = _preflight_stage(stage_dir / "preflight.log", args.mode)
    stages.append(preflight)

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"

    for name, cmd, expected_artifacts, extra_env in _stage_specs(args.mode, report_dir, args.backend):
        env = base_env.copy()
        env.update(extra_env)
        stage = _run_command(name, cmd, log_path=stage_dir / f"{name}.log", env=env)
        stage.artifacts = _artifacts(expected_artifacts)
        stages.append(stage)

    ui_stage = _ui_parse_stage(stage_dir / "ui_parse.log")
    stages.append(ui_stage)

    git_guard = _git_guard_stage(stage_dir / "git_guard.log", baseline, report_dir)
    stages.append(git_guard)

    finished_at = datetime.now(timezone.utc).isoformat()
    overall_passed = all(stage.passed for stage in stages)

    summary = {
        "mode": args.mode,
        "started_at": started_at,
        "finished_at": finished_at,
        "report_dir": str(report_dir),
        "overall_passed": overall_passed,
        "stages": [asdict(stage) for stage in stages],
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / "summary.md").write_text(
        _summary_md(
            mode=args.mode,
            report_dir=report_dir,
            started_at=started_at,
            finished_at=finished_at,
            stages=stages,
            overall_passed=overall_passed,
        ),
        encoding="utf-8",
    )

    print(f"report_dir={report_dir}")
    print(f"overall={'PASS' if overall_passed else 'FAIL'}")
    for stage in stages:
        print(f"{stage.name}: {'PASS' if stage.passed else 'FAIL'} ({stage.duration_s:.2f}s)")
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
