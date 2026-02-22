"""Simulation Replay -- step-through visualization of game rounds."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from ui.utils import load_trajectory_logs, render_scenario_cards

st.header("Simulation Replay")

st.subheader("Select a Scenario")
selected_result = render_scenario_cards("replay")

with st.expander("Manual upload"):
    uploaded = st.file_uploader("Upload simulation output JSON", type=["json"])
    file_path = st.text_input("Or enter path to trajectory JSON file")

logs: list[dict] = []
if selected_result is not None:
    logs = load_trajectory_logs(selected_result)
elif uploaded is not None:
    import json
    logs = json.load(uploaded)
    if isinstance(logs, dict) and "trajectory_logs" in logs:
        logs = logs["trajectory_logs"]
elif file_path:
    logs = load_trajectory_logs(file_path)

if not logs:
    st.info("Select a scenario card above or upload a file to begin replay.")
    st.stop()

n_rounds = len(logs)
st.write(f"**{n_rounds} rounds** loaded")

round_idx = st.slider("Round", 0, n_rounds - 1, 0)
entry = logs[round_idx]

col1, col2 = st.columns(2)
with col1:
    st.subheader("State")
    state = entry.get("state", {})
    st.metric("t", state.get("t", "?"))
    st.metric("value", f"{state.get('value', 0):.4f}")
    st.metric("exogenous", f"{state.get('exogenous', 0):.4f}")
    st.metric("hidden_shift", f"{state.get('hidden_shift', 0):.4f}")

with col2:
    st.subheader("Round Summary")
    st.metric("Forecast", f"{entry.get('forecast', 0):.4f}")
    st.metric("Target", f"{entry.get('target', 0):.4f}")
    st.metric("Reward", f"{entry.get('reward', 0):.4f}")
    st.metric("Disturbance", f"{entry.get('disturbance', 0):.4f}")

st.subheader("Agent Deltas")
actions = entry.get("actions", [])
action_data = {a.get("actor", f"agent_{i}"): a.get("delta", 0) for i, a in enumerate(actions)}
st.table(action_data)

st.subheader("Forecast vs Target Over Time")
forecasts = [logs[i]["forecast"] for i in range(round_idx + 1)]
targets = [logs[i]["target"] for i in range(round_idx + 1)]
rounds = list(range(round_idx + 1))

fig = go.Figure()
fig.add_trace(go.Scatter(x=rounds, y=forecasts, mode="lines+markers", name="Forecast"))
fig.add_trace(go.Scatter(x=rounds, y=targets, mode="lines+markers", name="Target"))
fig.update_layout(xaxis_title="Round", yaxis_title="Value", height=400)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Component Breakdown")
disturbances = [logs[i].get("disturbance", 0) for i in range(round_idx + 1)]
deltas_f = [logs[i]["actions"][0]["delta"] if logs[i].get("actions") else 0 for i in range(round_idx + 1)]
deltas_a = [logs[i]["actions"][1]["delta"] if logs[i].get("actions") and len(logs[i]["actions"]) > 1 else 0 for i in range(round_idx + 1)]
deltas_d = [logs[i]["actions"][2]["delta"] if logs[i].get("actions") and len(logs[i]["actions"]) > 2 else 0 for i in range(round_idx + 1)]

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=rounds, y=deltas_f, name="Forecaster"))
fig2.add_trace(go.Bar(x=rounds, y=deltas_a, name="Adversary"))
fig2.add_trace(go.Bar(x=rounds, y=deltas_d, name="Defender"))
fig2.add_trace(go.Bar(x=rounds, y=disturbances, name="Disturbance"))
fig2.update_layout(barmode="stack", xaxis_title="Round", yaxis_title="Delta", height=400)
st.plotly_chart(fig2, use_container_width=True)
