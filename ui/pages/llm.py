"""LLM Inspection -- placeholder for DSPy-REPL / HaskellRLM integration."""
from __future__ import annotations

import streamlit as st

from ui.utils import load_trajectory_logs

st.header("LLM Inspection")

st.info(
    "LLM inspection will be available when DSPy-REPL or HaskellRLM "
    "integration is active. This page provides visibility into LLM-driven "
    "agent decisions when `enable_llm_refactor=True`."
)

st.subheader("Prompt History")
st.dataframe(
    {"Round": [], "Prompt": [], "Response": [], "Bias Adjustment": []},
    use_container_width=True,
)

st.subheader("Rationale Log")
uploaded = st.file_uploader("Upload simulation with LLM data", type=["json"], key="llm_upload")
if uploaded is not None:
    import json
    data = json.load(uploaded)
    logs = data if isinstance(data, list) else data.get("trajectory_logs", [])

    llm_entries = []
    for entry in logs:
        actions = entry.get("actions", [])
        for action in actions:
            if "rationale" in action or "bias_adjustment" in action:
                llm_entries.append({
                    "Round": entry.get("round_idx", "?"),
                    "Actor": action.get("actor", "?"),
                    "Bias Adjustment": action.get("bias_adjustment", "N/A"),
                    "Rationale": action.get("rationale", "N/A"),
                })

    if llm_entries:
        st.dataframe(llm_entries, use_container_width=True)
    else:
        st.info("No LLM rationale data found in this simulation output.")

st.subheader("Code Diff Viewer")
st.text_area(
    "Strategy refactoring diffs will appear here when the RecursiveStrategyRefiner is active.",
    value="(No diffs available)",
    disabled=True,
    height=200,
)
