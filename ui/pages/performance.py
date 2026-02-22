"""Performance Dashboard -- runtime analysis, latency charts, convergence, BMA weights."""
from __future__ import annotations

import statistics

import streamlit as st
import plotly.graph_objects as go

from ui.utils import load_simulation_result, render_scenario_cards

st.header("Performance Dashboard")

# -------------------------------------------------------------------
# Scenario selection
# -------------------------------------------------------------------
selected = render_scenario_cards("perf")

with st.expander("Manual upload"):
    uploaded = st.file_uploader("Upload simulation JSON", type=["json"], key="perf_upload")
    file_path = st.text_input("Or enter path", key="perf_path")

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
    st.info("Select a scenario above to view performance data.")
    st.stop()

# -------------------------------------------------------------------
# Timing data
# -------------------------------------------------------------------
latencies = [entry.get("elapsed_s", 0) for entry in logs]
has_timing = any(l > 0 for l in latencies)
wall_clock = result.get("wall_clock_s", 0)

st.subheader("Timing Summary")

if has_timing:
    lat_ms = [l * 1000 for l in latencies]
    sorted_ms = sorted(lat_ms)
    p95_idx = int(len(sorted_ms) * 0.95)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Wall Clock", f"{wall_clock:.2f}s" if wall_clock else "--")
    c2.metric("Rounds", len(logs))
    c3.metric("Mean Latency", f"{statistics.mean(lat_ms):.2f}ms")
    c4.metric("P95 Latency", f"{sorted_ms[p95_idx] if p95_idx < len(sorted_ms) else sorted_ms[-1]:.2f}ms")
    c5.metric("Max Latency", f"{max(lat_ms):.2f}ms")

    # Per-round latency chart
    st.subheader("Per-Round Latency")
    fig_lat = go.Figure()
    fig_lat.add_trace(go.Scatter(
        x=list(range(len(lat_ms))), y=lat_ms,
        mode="lines", name="Round Latency",
        fill="tozeroy", fillcolor="rgba(100,149,237,0.15)",
    ))
    mean_val = statistics.mean(lat_ms)
    fig_lat.add_hline(y=mean_val, line_dash="dash", line_color="orange",
                      annotation_text=f"mean={mean_val:.2f}ms")
    fig_lat.update_layout(xaxis_title="Round", yaxis_title="Latency (ms)", height=350)
    st.plotly_chart(fig_lat, use_container_width=True)

    # Latency histogram
    st.subheader("Latency Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=lat_ms, nbinsx=30))
    fig_hist.update_layout(xaxis_title="Latency (ms)", yaxis_title="Count", height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Slowest rounds
    st.subheader("Top 10 Slowest Rounds")
    indexed = [(i, lat_ms[i]) for i in range(len(lat_ms))]
    indexed.sort(key=lambda x: x[1], reverse=True)
    slow_data = []
    for rank, (ridx, lat) in enumerate(indexed[:10], 1):
        entry = logs[ridx]
        slow_data.append({
            "Rank": rank,
            "Round": ridx,
            "Latency (ms)": f"{lat:.2f}",
            "Reward": f"{entry.get('reward', 0):.4f}",
            "Disturbance": f"{entry.get('disturbance', 0):.4f}",
        })
    st.dataframe(slow_data, use_container_width=True)
else:
    c1, c2 = st.columns(2)
    c1.metric("Total Wall Clock", f"{wall_clock:.2f}s" if wall_clock else "--")
    c2.metric("Rounds", len(logs))
    st.caption("Per-round timing data not available in this result file (pre-instrumentation).")

# -------------------------------------------------------------------
# Convergence analysis
# -------------------------------------------------------------------
convergence = result.get("convergence", {})
if convergence:
    st.divider()
    st.subheader("Convergence Analysis")

    conv_cols = st.columns(4)
    conv_cols[0].metric("Reason", convergence.get("reason", "?"))
    conv_cols[1].metric("Rounds Executed", convergence.get("rounds_executed", "?"))
    conv_cols[2].metric("Max Rounds", convergence.get("max_rounds", "?"))
    conv_cols[3].metric("Cap Hit", "Yes" if convergence.get("round_cap_hit") else "No")

    cost_cols = st.columns(3)
    cost_cols[0].metric("Attack Cost Total", f"{convergence.get('attack_cost_total', 0):.4f}")
    cost_cols[1].metric("Defense Efficacy", f"{convergence.get('defense_efficacy_ratio', 0):.2%}")
    acc_cost = convergence.get("accuracy_vs_cost", 0)
    cost_cols[2].metric("Accuracy / Cost", f"{acc_cost:.4f}" if acc_cost else "N/A")

    # Rolling error chart
    errors = [abs(entry.get("target", 0) - entry.get("forecast", 0)) for entry in logs]
    if errors:
        window = 20
        rolling = []
        for i in range(len(errors)):
            w = errors[max(0, i - window + 1):i + 1]
            rolling.append(sum(w) / len(w))

        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=list(range(len(errors))), y=errors,
            mode="lines", name="Absolute Error", opacity=0.3,
        ))
        fig_err.add_trace(go.Scatter(
            x=list(range(len(rolling))), y=rolling,
            mode="lines", name=f"Rolling MAE ({window}r)",
            line=dict(width=2),
        ))
        fig_err.update_layout(xaxis_title="Round", yaxis_title="Error", height=350)
        st.plotly_chart(fig_err, use_container_width=True)

# -------------------------------------------------------------------
# BMA Weight Evolution
# -------------------------------------------------------------------
bma_rounds = [(i, entry.get("bma_weights")) for i, entry in enumerate(logs) if entry.get("bma_weights")]
if bma_rounds:
    st.divider()
    st.subheader("BMA Weight Evolution")

    n_agents = len(bma_rounds[0][1])
    agent_labels = [f"Agent {j}" for j in range(n_agents)]

    fig_bma = go.Figure()
    for j in range(n_agents):
        fig_bma.add_trace(go.Scatter(
            x=[r[0] for r in bma_rounds],
            y=[r[1][j] for r in bma_rounds],
            mode="lines",
            name=agent_labels[j],
            stackgroup="bma",
        ))
    fig_bma.update_layout(
        xaxis_title="Round", yaxis_title="Weight",
        yaxis=dict(range=[0, 1]),
        height=350,
    )
    st.plotly_chart(fig_bma, use_container_width=True)

# -------------------------------------------------------------------
# LLM Call overhead
# -------------------------------------------------------------------
llm_calls = result.get("llm_calls", [])
if llm_calls:
    st.divider()
    st.subheader("LLM Call Overhead")
    total_llm_ms = sum(c.get("latency_ms", 0) for c in llm_calls)
    total_wall_ms = wall_clock * 1000 if wall_clock else 1
    pct = (total_llm_ms / total_wall_ms * 100) if total_wall_ms > 0 else 0

    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Total LLM Time", f"{total_llm_ms:.0f}ms")
    lc2.metric("Total Calls", len(llm_calls))
    lc3.metric("% of Wall Clock", f"{pct:.1f}%")
