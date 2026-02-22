"""What-If Experimentation -- interactive parameter tweaking and re-simulation."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

st.header("What-If Experimentation")

with st.sidebar:
    st.subheader("Simulation Parameters")
    horizon = st.number_input("Horizon", min_value=1, max_value=50000, value=100)
    max_rounds = st.number_input("Max Rounds", min_value=1, max_value=50000, value=200)
    disturbance_prob = st.slider("Disturbance Probability", 0.0, 1.0, 0.1, 0.01)
    disturbance_scale = st.slider("Disturbance Scale", 0.0, 5.0, 1.0, 0.1)
    adversarial_intensity = st.slider("Adversarial Intensity", 0.0, 5.0, 1.0, 0.1)
    defense_model = st.selectbox("Defense Model", ["dampening", "clipping", "bias_guard", "dann_filter", "ensemble"])
    disturbance_model = st.selectbox("Disturbance Model", ["gaussian", "volatility", "regime_shift", "escalating", "targeted_volatility", "coordinated_regime", "wolfpack"])
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42)

    st.subheader("Initial State")
    initial_value = st.number_input("Value", value=10.0)
    initial_exogenous = st.number_input("Exogenous", value=0.0)
    initial_hidden_shift = st.number_input("Hidden Shift", value=0.0)

    disturbed = st.checkbox("Enable Disturbance", value=True)

col1, col2 = st.columns(2)

with col1:
    run_single = st.button("Run Simulation")
with col2:
    run_comparison = st.button("Run Comparison (Clean vs Attacked)")

if run_single or run_comparison:
    from ui.utils import run_simulation

    with st.spinner("Running simulation..."):
        result = run_simulation(
            horizon=horizon,
            max_rounds=max_rounds,
            disturbance_prob=disturbance_prob,
            disturbance_scale=disturbance_scale,
            adversarial_intensity=adversarial_intensity,
            defense_model=defense_model,
            disturbance_model=disturbance_model,
            seed=seed,
            initial_value=initial_value,
            initial_exogenous=initial_exogenous,
            initial_hidden_shift=initial_hidden_shift,
            disturbed=disturbed,
        )

    st.subheader("Results")
    m = result.get("metrics", {})
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("MAE", f"{m.get('mae', 0):.4f}")
    mc2.metric("RMSE", f"{m.get('rmse', 0):.4f}")
    mc3.metric("MAPE", f"{m.get('mape', 0):.2f}%")
    mc4.metric("Worst Case", f"{m.get('worst_case', 0):.4f}")

    conv = result.get("convergence", {})
    st.write(f"Rounds executed: **{conv.get('rounds_executed', '?')}** | "
             f"Reason: **{conv.get('reason', '?')}**")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=result["forecasts"], mode="lines", name="Forecast"))
    fig.add_trace(go.Scatter(y=result["targets"], mode="lines", name="Target"))
    fig.update_layout(xaxis_title="Round", yaxis_title="Value", height=400)
    st.plotly_chart(fig, use_container_width=True)

    if run_comparison:
        with st.spinner("Running clean scenario..."):
            clean_result = run_simulation(
                horizon=horizon,
                max_rounds=max_rounds,
                disturbance_prob=disturbance_prob,
                disturbance_scale=disturbance_scale,
                adversarial_intensity=adversarial_intensity,
                defense_model=defense_model,
                disturbance_model=disturbance_model,
                seed=seed,
                initial_value=initial_value,
                initial_exogenous=initial_exogenous,
                initial_hidden_shift=initial_hidden_shift,
                disturbed=False,
            )

        st.subheader("Comparison: Clean vs Attacked")
        cm = clean_result.get("metrics", {})
        am = result.get("metrics", {})
        comparison_metrics = ["mae", "rmse", "mape", "worst_case"]
        fig2 = go.Figure(data=[
            go.Bar(name="Clean", x=comparison_metrics, y=[cm.get(k, 0) for k in comparison_metrics]),
            go.Bar(name="Attacked", x=comparison_metrics, y=[am.get(k, 0) for k in comparison_metrics]),
        ])
        fig2.update_layout(barmode="group", height=400, yaxis_title="Value")
        st.plotly_chart(fig2, use_container_width=True)
