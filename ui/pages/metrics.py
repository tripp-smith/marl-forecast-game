"""Metric Decomposition -- error attribution and robustness analysis."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from ui.utils import load_simulation_result, render_scenario_cards

st.header("Metric Decomposition")

st.subheader("Select Clean Scenario")
clean_path = render_scenario_cards(
    "metrics_clean",
    filter_fn=lambda m: m.get("disturbed") is False or "clean" in str(m.get("label", "")).lower(),
)

st.subheader("Select Attacked Scenario")
attack_path = render_scenario_cards(
    "metrics_attack",
    filter_fn=lambda m: m.get("disturbed") is True or "attack" in str(m.get("label", "")).lower(),
)

if clean_path and attack_path:
    auto_clean_data = load_simulation_result(clean_path)
    auto_attack_data = load_simulation_result(attack_path)

    clean_logs = auto_clean_data.get("trajectory_logs", [])
    attack_logs = auto_attack_data.get("trajectory_logs", [])
    if clean_logs and attack_logs:
        st.subheader("Error Attribution Waterfall")
        n = min(len(clean_logs), len(attack_logs))
        trend_errors = []
        disturbance_contributions = []
        total_errors = []
        for i in range(n):
            c = clean_logs[i]
            a = attack_logs[i]
            clean_error = abs(c["target"] - c["forecast"])
            attack_error = abs(a["target"] - a["forecast"])
            dist_effect = abs(a.get("disturbance", 0))
            trend_errors.append(clean_error)
            disturbance_contributions.append(dist_effect)
            total_errors.append(attack_error)

        fig_auto = go.Figure(go.Waterfall(
            name="Error Attribution",
            orientation="v",
            x=["Trend Error", "Disturbance", "Total Attack Error"],
            y=[
                sum(trend_errors) / max(1, n),
                sum(disturbance_contributions) / max(1, n),
                sum(total_errors) / max(1, n),
            ],
            measure=["absolute", "relative", "total"],
        ))
        fig_auto.update_layout(height=400, yaxis_title="Mean Absolute Error")
        st.plotly_chart(fig_auto, use_container_width=True)

        cm = auto_clean_data.get("metrics", {})
        am = auto_attack_data.get("metrics", {})
        if cm and am:
            st.subheader("Side-by-Side Metrics")
            comparison_metrics = ["mae", "rmse", "mape", "worst_case"]
            fig_comp = go.Figure(data=[
                go.Bar(name="Clean", x=comparison_metrics, y=[cm.get(k, 0) for k in comparison_metrics]),
                go.Bar(name="Attacked", x=comparison_metrics, y=[am.get(k, 0) for k in comparison_metrics]),
            ])
            fig_comp.update_layout(barmode="group", height=400, yaxis_title="Value")
            st.plotly_chart(fig_comp, use_container_width=True)

st.divider()

with st.expander("Manual Upload"):
    col_clean, col_attack = st.columns(2)

    with col_clean:
        st.subheader("Clean Scenario")
        clean_file = st.file_uploader("Upload clean simulation JSON", type=["json"], key="clean")

    with col_attack:
        st.subheader("Attacked Scenario")
        attack_file = st.file_uploader("Upload attacked simulation JSON", type=["json"], key="attack")

    if clean_file is None and attack_file is None:
        single_file = st.file_uploader(
            "Or upload a single verification report JSON",
            type=["json"],
            key="single",
        )
        if single_file is not None:
            import json
            report = json.load(single_file)
            clean_data = report.get("clean", {})
            attack_data = report.get("attack", {})
            robustness = report.get("robustness", {})

            st.subheader("Side-by-Side Metrics")
            metric_names = ["mae", "rmse", "mape", "worst_case"]
            clean_vals = [clean_data.get(m, 0) for m in metric_names]
            attack_vals = [attack_data.get(m, 0) for m in metric_names]

            fig = go.Figure(data=[
                go.Bar(name="Clean", x=metric_names, y=clean_vals),
                go.Bar(name="Attacked", x=metric_names, y=attack_vals),
            ])
            fig.update_layout(barmode="group", height=400, yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Robustness")
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.metric("MAE Delta", f"{robustness.get('mae_delta', 0):.4f}")
            with r_col2:
                st.metric("MAE Ratio", f"{robustness.get('mae_ratio', 0):.4f}")

            sweep = report.get("scenario_sweep", {})
            if sweep:
                st.subheader("Intensity Sweep Robustness")
                intensities = sorted(sweep.keys(), key=float)
                deltas = [sweep[k].get("mae_delta", 0) for k in intensities]
                ratios = [sweep[k].get("mae_ratio", 0) for k in intensities]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=intensities, y=deltas, name="MAE Delta"))
                fig2.add_trace(go.Scatter(x=intensities, y=ratios, mode="lines+markers", name="MAE Ratio", yaxis="y2"))
                fig2.update_layout(
                    yaxis=dict(title="MAE Delta"),
                    yaxis2=dict(title="MAE Ratio", overlaying="y", side="right"),
                    height=400,
                )
                st.plotly_chart(fig2, use_container_width=True)

    if clean_file is not None and attack_file is not None:
        import json
        clean_logs_m = json.load(clean_file)
        attack_logs_m = json.load(attack_file)
        if isinstance(clean_logs_m, dict) and "trajectory_logs" in clean_logs_m:
            clean_logs_m = clean_logs_m["trajectory_logs"]
        if isinstance(attack_logs_m, dict) and "trajectory_logs" in attack_logs_m:
            attack_logs_m = attack_logs_m["trajectory_logs"]

        st.subheader("Error Attribution Waterfall")
        n = min(len(clean_logs_m), len(attack_logs_m))
        trend_errors = []
        disturbance_contributions = []
        total_errors = []
        for i in range(n):
            c = clean_logs_m[i]
            a = attack_logs_m[i]
            clean_error = abs(c["target"] - c["forecast"])
            attack_error = abs(a["target"] - a["forecast"])
            dist_effect = abs(a.get("disturbance", 0))
            trend_errors.append(clean_error)
            disturbance_contributions.append(dist_effect)
            total_errors.append(attack_error)

        fig = go.Figure(go.Waterfall(
            name="Error Attribution",
            orientation="v",
            x=["Trend Error", "Disturbance", "Total Attack Error"],
            y=[
                sum(trend_errors) / max(1, n),
                sum(disturbance_contributions) / max(1, n),
                sum(total_errors) / max(1, n),
            ],
            measure=["absolute", "relative", "total"],
        ))
        fig.update_layout(height=400, yaxis_title="Mean Absolute Error")
        st.plotly_chart(fig, use_container_width=True)
