"""Findings Validator -- inspect and validate programmatic test harness runs."""
from __future__ import annotations

from pathlib import Path

import json
import streamlit as st

from ui.utils import discover_harness_summaries, validate_harness_data, validate_harness_summary

st.header("Findings Validator")
st.caption("Validate test-harness summaries, stage outcomes, logs, and linked artifacts.")

summary_paths = discover_harness_summaries()

with st.expander("Manual summary validation"):
    uploaded = st.file_uploader("Upload a harness summary.json", type=["json"], key="findings_upload")
    file_path = st.text_input("Or enter the path to a summary.json file", key="findings_path")

manual_reports: list[dict] = []
if uploaded is not None:
    manual_reports.append(validate_harness_data(json.loads(uploaded.getvalue().decode("utf-8")), label=uploaded.name))
elif file_path:
    manual_reports.append(validate_harness_summary(file_path))

reports = [validate_harness_summary(path) for path in summary_paths]
reports = manual_reports + reports if manual_reports else reports

if not reports:
    st.info("No harness summaries found. Run `python run_project_tests.py --mode full` first.")
    st.stop()

passing_runs = sum(1 for report in reports if report["summary"].get("overall_passed") is True)
failing_runs = len(reports) - passing_runs
stage_failures = sum(len(report["stage_failures"]) for report in reports)
validation_findings = sum(len(report["validation_findings"]) for report in reports)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Harness Runs", len(reports))
c2.metric("Passing Runs", passing_runs)
c3.metric("Failed Stages", stage_failures)
c4.metric("Validation Findings", validation_findings)

st.subheader("Run Inventory")
inventory_rows = []
for report in reports:
    summary = report["summary"]
    inventory_rows.append(
        {
            "Run": Path(report["path"]).parent.name,
            "Mode": summary.get("mode", "?"),
            "Overall": "PASS" if summary.get("overall_passed") else "FAIL",
            "Started": summary.get("started_at", "?"),
            "Stages": len(summary.get("stages", [])),
            "Failed stages": len(report["stage_failures"]),
            "Validation findings": len(report["validation_findings"]),
        }
    )
st.dataframe(inventory_rows, use_container_width=True, hide_index=True)

st.subheader("Per-Run Validation")
for report in reports:
    summary = report["summary"]
    run_name = Path(report["path"]).parent.name
    overall = "PASS" if summary.get("overall_passed") else "FAIL"
    badge = "PASS" if report["valid"] else "CHECK"
    with st.expander(f"{run_name} | {overall} | validator={badge}", expanded=(summary.get("overall_passed") is False)):
        top_cols = st.columns(4)
        top_cols[0].metric("Mode", summary.get("mode", "?"))
        top_cols[1].metric("Stages", len(summary.get("stages", [])))
        top_cols[2].metric("Failed stages", len(report["stage_failures"]))
        top_cols[3].metric("Validation findings", len(report["validation_findings"]))

        if report["validation_findings"]:
            st.warning("Validation findings")
            for finding in report["validation_findings"]:
                st.write(f"- {finding}")

        if report["stage_failures"]:
            st.error("Reported stage failures")
            for stage in report["stage_failures"]:
                notes = "; ".join(stage.get("notes", [])) or "No notes"
                st.write(f"- `{stage['name']}`: exit={stage['exit_code']} | {notes}")

        stage_rows = []
        for stage in summary.get("stages", []):
            stage_rows.append(
                {
                    "Stage": stage.get("name", "?"),
                    "Status": "PASS" if stage.get("passed") else "FAIL",
                    "Seconds": stage.get("duration_s", 0.0),
                    "Artifacts": len(stage.get("artifacts", [])),
                    "Log": stage.get("log_path", ""),
                }
            )
        st.dataframe(stage_rows, use_container_width=True, hide_index=True)

        if summary.get("stages"):
            selected_stage = st.selectbox(
                "Inspect stage",
                options=[stage["name"] for stage in summary["stages"]],
                key=f"stage_select_{run_name}",
            )
            stage = next(stage for stage in summary["stages"] if stage["name"] == selected_stage)
            st.code(stage.get("command", ""), language="bash")
            if stage.get("notes"):
                st.caption("Notes: " + "; ".join(stage["notes"]))

            log_path = Path(stage["log_path"])
            if not log_path.is_absolute():
                log_path = Path.cwd() / log_path
            if log_path.exists():
                with st.expander("Stage log tail"):
                    st.code(log_path.read_text(encoding="utf-8")[-4000:], language=None)

            if stage.get("artifacts"):
                st.caption("Artifacts")
                for artifact in stage["artifacts"]:
                    st.write(f"- `{artifact}`")

latest_report = reports[0]
st.subheader("Latest Summary JSON")
st.json(latest_report["summary"])
