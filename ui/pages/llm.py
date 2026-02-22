"""LLM Inspection -- full audit trail of all LLM calls with latency analysis."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from ui.utils import load_simulation_result, render_scenario_cards

st.header("LLM Inspection")

# -------------------------------------------------------------------
# Scenario selection
# -------------------------------------------------------------------
selected = render_scenario_cards("llm")

with st.expander("Manual upload"):
    uploaded = st.file_uploader("Upload simulation JSON", type=["json"], key="llm_upload")
    file_path = st.text_input("Or enter path", key="llm_path")

result: dict = {}
if selected is not None:
    result = load_simulation_result(selected)
elif uploaded is not None:
    import json
    result = json.load(uploaded)
elif file_path:
    result = load_simulation_result(file_path)

llm_calls: list[dict] = result.get("llm_calls", []) if isinstance(result, dict) else []

# -------------------------------------------------------------------
# Empty state
# -------------------------------------------------------------------
if not llm_calls:
    st.info(
        "No LLM call data found in this simulation.\n\n"
        "LLM calls are recorded when the simulation runs with Ollama enabled "
        "(`enable_llm_refactor=True` or a `ForecastingAgent` with an LLM REPL). "
        "If Ollama was not available during the simulation, calls may have failed "
        "silently and fallen back to deterministic logic."
    )

    logs = result.get("trajectory_logs", []) if isinstance(result, dict) else []
    if logs:
        llm_entries = []
        for entry in logs:
            for action in entry.get("actions", []):
                if "rationale" in action or "bias_adjustment" in action:
                    llm_entries.append({
                        "Round": entry.get("round_idx", "?"),
                        "Actor": action.get("actor", "?"),
                        "Bias Adj.": action.get("bias_adjustment", "N/A"),
                        "Rationale": action.get("rationale", "N/A"),
                    })
        if llm_entries:
            st.subheader("Rationale Log (from actions)")
            st.dataframe(llm_entries, use_container_width=True)
    st.stop()

# -------------------------------------------------------------------
# Summary metrics
# -------------------------------------------------------------------
st.subheader("Call Summary")

total_calls = len(llm_calls)
generate_calls = [c for c in llm_calls if c.get("call_type") == "generate"]
embed_calls = [c for c in llm_calls if c.get("call_type") == "embed"]
refactor_calls = [c for c in llm_calls if c.get("call_type") == "refactor"]
error_calls = [c for c in llm_calls if c.get("error")]
latencies = [c.get("latency_ms", 0) for c in llm_calls]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Calls", total_calls)
c2.metric("Generate", len(generate_calls))
c3.metric("Embed", len(embed_calls))
c4.metric("Errors", len(error_calls))
if latencies:
    sorted_lat = sorted(latencies)
    c5.metric("Median Latency", f"{sorted_lat[len(sorted_lat) // 2]:.1f}ms")

# -------------------------------------------------------------------
# Call Log Table
# -------------------------------------------------------------------
st.divider()
st.subheader("Call Log")

filter_type = st.multiselect(
    "Filter by type",
    options=["generate", "embed", "refactor"],
    default=["generate", "embed", "refactor"],
    key="llm_filter",
)

filtered = [c for c in llm_calls if c.get("call_type") in filter_type]

table_data = []
for i, c in enumerate(filtered):
    prompt_preview = str(c.get("prompt", ""))[:80]
    response_preview = str(c.get("response", ""))[:80]
    table_data.append({
        "#": i,
        "Round": c.get("round_idx", "?"),
        "Agent": c.get("agent", "?"),
        "Type": c.get("call_type", "?"),
        "Model": c.get("model", "?"),
        "Latency (ms)": f"{c.get('latency_ms', 0):.1f}",
        "Prompt": prompt_preview + ("..." if len(str(c.get("prompt", ""))) > 80 else ""),
        "Response": response_preview + ("..." if len(str(c.get("response", ""))) > 80 else ""),
        "Error": c.get("error") or "",
    })

st.dataframe(table_data, use_container_width=True, height=400)

# -------------------------------------------------------------------
# Prompt / Response Inspector
# -------------------------------------------------------------------
st.divider()
st.subheader("Prompt / Response Inspector")

if filtered:
    call_idx = st.number_input(
        "Select call # to inspect",
        min_value=0,
        max_value=len(filtered) - 1,
        value=0,
        key="llm_inspect_idx",
    )
    selected_call = filtered[int(call_idx)]

    meta_cols = st.columns(4)
    meta_cols[0].caption(f"Round: {selected_call.get('round_idx', '?')}")
    meta_cols[1].caption(f"Agent: {selected_call.get('agent', '?')}")
    meta_cols[2].caption(f"Type: {selected_call.get('call_type', '?')}")
    meta_cols[3].caption(f"Latency: {selected_call.get('latency_ms', 0):.1f}ms")

    if selected_call.get("error"):
        st.error(f"Error: {selected_call['error']}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Prompt**")
        st.code(str(selected_call.get("prompt", "")), language=None)
    with right:
        st.markdown("**Response**")
        st.code(str(selected_call.get("response", "")), language=None)

# -------------------------------------------------------------------
# Latency Distribution
# -------------------------------------------------------------------
if latencies:
    st.divider()
    st.subheader("Latency Distribution")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=latencies, nbinsx=30, name="All calls"))
    if generate_calls:
        fig_hist.add_trace(go.Histogram(
            x=[c.get("latency_ms", 0) for c in generate_calls],
            nbinsx=20, name="Generate", opacity=0.7,
        ))
    fig_hist.update_layout(
        xaxis_title="Latency (ms)", yaxis_title="Count",
        barmode="overlay", height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------------------------------------------
# Call Timeline
# -------------------------------------------------------------------
if filtered:
    rounds_with_calls = [c.get("round_idx", 0) or 0 for c in filtered]
    call_latencies = [c.get("latency_ms", 0) for c in filtered]
    call_types = [c.get("call_type", "?") for c in filtered]

    st.subheader("Call Timeline")
    fig_timeline = px.scatter(
        x=rounds_with_calls,
        y=call_latencies,
        color=call_types,
        size=[max(3, lat / 10) for lat in call_latencies],
        labels={"x": "Round", "y": "Latency (ms)", "color": "Type"},
        height=350,
    )
    fig_timeline.update_layout(xaxis_title="Round", yaxis_title="Latency (ms)")
    st.plotly_chart(fig_timeline, use_container_width=True)

# -------------------------------------------------------------------
# Fallback Tracker
# -------------------------------------------------------------------
if error_calls:
    st.divider()
    st.subheader("Fallback / Error Calls")
    error_data = []
    for c in error_calls:
        error_data.append({
            "Round": c.get("round_idx", "?"),
            "Agent": c.get("agent", "?"),
            "Type": c.get("call_type", "?"),
            "Error": c.get("error", ""),
            "Latency (ms)": f"{c.get('latency_ms', 0):.1f}",
        })
    st.dataframe(error_data, use_container_width=True)
