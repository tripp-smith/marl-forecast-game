"""Agent Contributions -- per-agent delta and reward analysis."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from ui.utils import discover_result_files, load_trajectory_logs

st.header("Agent Contributions")

result_files = discover_result_files()
selected_result = None
if result_files:
    options = ["-- select --"] + [p.name for p in result_files]
    choice = st.selectbox("Load from results directory", options, key="agents_results")
    if choice != "-- select --":
        selected_result = next(p for p in result_files if p.name == choice)

uploaded = st.file_uploader("Upload simulation output JSON", type=["json"], key="agents_upload")
file_path = st.text_input("Or enter path to trajectory JSON file", key="agents_path")

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
    st.info("Load a simulation output file to view agent contributions.")
    st.stop()

n_rounds = len(logs)
rounds = list(range(n_rounds))

actor_names = []
if logs[0].get("actions"):
    actor_names = [a.get("actor", f"agent_{i}") for i, a in enumerate(logs[0]["actions"])]

delta_series: dict[str, list[float]] = {name: [] for name in actor_names}
cumulative_deltas: dict[str, list[float]] = {name: [] for name in actor_names}
reward_series: list[float] = []

for entry in logs:
    actions = entry.get("actions", [])
    reward_series.append(entry.get("reward", 0))
    for i, name in enumerate(actor_names):
        d = actions[i]["delta"] if i < len(actions) else 0
        delta_series[name].append(d)
        prev = cumulative_deltas[name][-1] if cumulative_deltas[name] else 0
        cumulative_deltas[name].append(prev + d)

st.subheader("Per-Round Deltas")
fig = go.Figure()
for name, deltas in delta_series.items():
    fig.add_trace(go.Scatter(x=rounds, y=deltas, mode="lines", name=name))
fig.update_layout(xaxis_title="Round", yaxis_title="Delta", height=400)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Cumulative Deltas (Stacked Area)")
fig2 = go.Figure()
for name, deltas in cumulative_deltas.items():
    fig2.add_trace(go.Scatter(x=rounds, y=deltas, mode="lines", stackgroup="one", name=name))
fig2.update_layout(xaxis_title="Round", yaxis_title="Cumulative Delta", height=400)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Reward History")
cumulative_reward = []
total = 0.0
for r in reward_series:
    total += r
    cumulative_reward.append(total)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=rounds, y=reward_series, mode="lines", name="Per-round Reward"))
fig3.add_trace(go.Scatter(x=rounds, y=cumulative_reward, mode="lines", name="Cumulative Reward"))
fig3.update_layout(xaxis_title="Round", yaxis_title="Reward", height=400)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Agent Delta Correlation Heatmap")
if len(actor_names) >= 2 and len(logs) > 1:
    matrix = np.array([delta_series[name] for name in actor_names])
    corr = np.corrcoef(matrix)
    fig4 = go.Figure(data=go.Heatmap(
        z=corr,
        x=actor_names,
        y=actor_names,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
    ))
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("At least 2 agents and 2 rounds needed for correlation heatmap.")
