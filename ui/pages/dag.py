"""Agent Action DAG -- visualize the agent interaction graph with per-round drill-down."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from ui.utils import load_simulation_result, render_scenario_cards

st.header("Agent Action DAG")

# -------------------------------------------------------------------
# Static canonical DAG
# -------------------------------------------------------------------
st.subheader("Canonical Agent Flow")
st.graphviz_chart("""
digraph agent_dag {
    rankdir=LR
    node [shape=box style="rounded,filled" fillcolor="#e8eaf6" fontname="Helvetica"]
    edge [fontname="Helvetica" fontsize=10]

    Forecasters [label="Forecaster(s)\\n+ BottomUp / TopDown"]
    Aggregator  [label="Ensemble\\nAggregator"]
    Adversary   [label="Adversary\\n(or Wolfpack)"]
    Defender    [label="Defender"]
    Refactor    [label="Refactoring\\nAgent"]
    Output      [label="Forecast\\nvalue + deltas + bias" shape=ellipse fillcolor="#c8e6c9"]

    Forecasters -> Aggregator [label="individual deltas"]
    Aggregator  -> Adversary  [label="f_delta"]
    Adversary   -> Defender   [label="a_delta"]
    Defender    -> Output     [label="d_delta"]
    Refactor    -> Output     [label="refactor_bias" style=dashed]
    Aggregator  -> Output     [label="f_delta"]
    Adversary   -> Output     [label="a_delta"]
}
""")

# -------------------------------------------------------------------
# Scenario selection
# -------------------------------------------------------------------
st.divider()
st.subheader("Per-Round Drill-Down")

selected = render_scenario_cards("dag")

with st.expander("Manual upload"):
    uploaded = st.file_uploader("Upload simulation JSON", type=["json"], key="dag_upload")
    file_path = st.text_input("Or enter path", key="dag_path")

result: dict = {}
if selected is not None:
    result = load_simulation_result(selected)
elif uploaded is not None:
    import json
    result = json.load(uploaded)
elif file_path:
    result = load_simulation_result(file_path)

logs = result.get("trajectory_logs", []) if isinstance(result, dict) else []
if not logs:
    st.info("Select a scenario above to drill into per-round agent actions.")
    st.stop()

# -------------------------------------------------------------------
# Convergence summary
# -------------------------------------------------------------------
convergence = result.get("convergence", {})
if convergence:
    st.subheader("Convergence Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds Executed", convergence.get("rounds_executed", "?"))
    c2.metric("Reason", convergence.get("reason", "?"))
    c3.metric("Attack Cost", f"{convergence.get('attack_cost_total', 0):.4f}")
    c4.metric("Defense Efficacy", f"{convergence.get('defense_efficacy_ratio', 0):.2%}")

    if convergence.get("accuracy_vs_cost", 0) > 0:
        st.caption(f"Accuracy vs Cost ratio: {convergence['accuracy_vs_cost']:.4f}")

# -------------------------------------------------------------------
# Per-round DAG with actual values
# -------------------------------------------------------------------
st.divider()
n_rounds = len(logs)
round_idx = st.slider("Round", 0, n_rounds - 1, 0, key="dag_round")
entry = logs[round_idx]

actions = entry.get("actions", [])
f_delta = actions[0]["delta"] if len(actions) > 0 else 0
a_delta = actions[1]["delta"] if len(actions) > 1 else 0
d_delta = actions[2]["delta"] if len(actions) > 2 else 0
forecast = entry.get("forecast", 0)
target = entry.get("target", 0)
error = target - forecast

def _color(val: float, threshold: float = 0.2) -> str:
    """Green for small absolute values, red for large."""
    if abs(val) < threshold * 0.5:
        return "#c8e6c9"
    if abs(val) < threshold:
        return "#fff9c4"
    return "#ffcdd2"

st.graphviz_chart(f"""
digraph round_dag {{
    rankdir=LR
    node [shape=box style="rounded,filled" fontname="Helvetica"]
    edge [fontname="Helvetica" fontsize=10]

    State  [label="State\\nvalue={entry.get('state', {{}}).get('value', 0):.3f}" fillcolor="#e3f2fd"]
    Fcst   [label="Forecaster\\ndelta={f_delta:.4f}" fillcolor="{_color(f_delta)}"]
    Adv    [label="Adversary\\ndelta={a_delta:.4f}" fillcolor="{_color(a_delta)}"]
    Def    [label="Defender\\ndelta={d_delta:.4f}" fillcolor="{_color(d_delta)}"]
    Result [label="Forecast={forecast:.4f}\\nTarget={target:.4f}\\nError={error:.4f}" shape=ellipse fillcolor="{_color(error, 0.5)}"]

    State -> Fcst
    Fcst  -> Adv  [label="{f_delta:+.4f}"]
    Adv   -> Def  [label="{a_delta:+.4f}"]
    Def   -> Result [label="{d_delta:+.4f}"]
}}
""")

# -------------------------------------------------------------------
# Message flow for selected round
# -------------------------------------------------------------------
messages = entry.get("messages", [])
if messages:
    st.subheader("Inter-Agent Messages")
    msg_data = []
    for m in messages:
        msg_data.append({
            "Sender": m.get("sender", "?"),
            "Receiver": m.get("receiver", "?"),
            "Payload": m.get("payload", ""),
        })
    st.dataframe(msg_data, use_container_width=True)

# -------------------------------------------------------------------
# Delta magnitude over all rounds
# -------------------------------------------------------------------
st.divider()
st.subheader("Delta Magnitude Over Time")

rounds_x = list(range(n_rounds))
f_deltas = [abs(logs[i]["actions"][0]["delta"]) if logs[i].get("actions") else 0 for i in range(n_rounds)]
a_deltas = [abs(logs[i]["actions"][1]["delta"]) if logs[i].get("actions") and len(logs[i]["actions"]) > 1 else 0 for i in range(n_rounds)]
d_deltas = [abs(logs[i]["actions"][2]["delta"]) if logs[i].get("actions") and len(logs[i]["actions"]) > 2 else 0 for i in range(n_rounds)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=rounds_x, y=f_deltas, mode="lines", name="Forecaster |delta|"))
fig.add_trace(go.Scatter(x=rounds_x, y=a_deltas, mode="lines", name="Adversary |delta|"))
fig.add_trace(go.Scatter(x=rounds_x, y=d_deltas, mode="lines", name="Defender |delta|"))
fig.add_vline(x=round_idx, line_dash="dash", line_color="gray", annotation_text="selected")
fig.update_layout(xaxis_title="Round", yaxis_title="|Delta|", height=350)
st.plotly_chart(fig, use_container_width=True)
