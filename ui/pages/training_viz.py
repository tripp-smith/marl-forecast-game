"""Training Visualization -- reward curves, RAD epochs, and training metrics."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from ui.utils import RESULTS_DIR

st.header("Training Visualization")

# -------------------------------------------------------------------
# Locate training data
# -------------------------------------------------------------------
training_path = RESULTS_DIR / "training_results.json"
alt_path = Path("data/models/training_results.json")

training_data: dict = {}

if training_path.exists():
    with open(training_path) as f:
        training_data = json.load(f)
elif alt_path.exists():
    with open(alt_path) as f:
        training_data = json.load(f)

uploaded = st.file_uploader("Upload training results JSON", type=["json"], key="train_upload")
if uploaded is not None:
    training_data = json.load(uploaded)

if not training_data:
    st.info(
        "No training results found.\n\n"
        "Training results are generated during pipeline Phase 3 and saved to "
        "`results/training_results.json`. Run the pipeline or upload a training result file."
    )
    st.stop()

# -------------------------------------------------------------------
# Summary metrics
# -------------------------------------------------------------------
st.subheader("Training Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Algorithm", training_data.get("algorithm", "?"))
c2.metric("Episodes", training_data.get("episodes", training_data.get("n_episodes", "?")))

mean_reward = training_data.get("mean_reward_last_50")
if mean_reward is not None:
    c3.metric("Mean Reward (last 50)", f"{mean_reward:.4f}")
else:
    c3.metric("Mean Reward (last 50)", "--")

final_eps = training_data.get("final_epsilon")
if final_eps is not None:
    c4.metric("Final Epsilon", f"{final_eps:.4f}")
else:
    c4.metric("Final Epsilon", "--")

td_error = training_data.get("mean_td_error_last_100")
if td_error is not None:
    st.caption(f"Mean TD Error (last 100): {td_error:.6f}")

# -------------------------------------------------------------------
# Reward curve
# -------------------------------------------------------------------
rewards_history = training_data.get("rewards_history", [])
if rewards_history:
    st.divider()
    st.subheader("Reward Curve")

    fig_reward = go.Figure()
    fig_reward.add_trace(go.Scatter(
        x=list(range(len(rewards_history))),
        y=rewards_history,
        mode="lines",
        name="Episode Mean Reward",
        opacity=0.4,
    ))

    window = min(50, max(5, len(rewards_history) // 10))
    if len(rewards_history) >= window:
        smoothed = []
        for i in range(len(rewards_history)):
            w = rewards_history[max(0, i - window + 1):i + 1]
            smoothed.append(sum(w) / len(w))
        fig_reward.add_trace(go.Scatter(
            x=list(range(len(smoothed))),
            y=smoothed,
            mode="lines",
            name=f"Smoothed ({window}ep)",
            line=dict(width=2),
        ))

    fig_reward.update_layout(
        xaxis_title="Episode", yaxis_title="Mean Reward", height=400,
    )
    st.plotly_chart(fig_reward, use_container_width=True)

# -------------------------------------------------------------------
# RAD epoch results
# -------------------------------------------------------------------
epoch_results = training_data.get("epoch_results", [])
if epoch_results:
    st.divider()
    st.subheader("RAD Adversarial Training Epochs")

    epochs = [e.get("epoch", i) for i, e in enumerate(epoch_results)]
    rewards = [e.get("mean_reward", 0) for e in epoch_results]
    taus = [e.get("tau", 0) for e in epoch_results]
    training_who = [e.get("training", "?") for e in epoch_results]

    colors = ["#1f77b4" if t == "forecaster" else "#d62728" for t in training_who]

    fig_rad = go.Figure()
    fig_rad.add_trace(go.Bar(
        x=epochs, y=rewards, name="Mean Reward",
        marker_color=colors,
        hovertext=[f"Training: {t}" for t in training_who],
    ))
    fig_rad.add_trace(go.Scatter(
        x=epochs, y=taus, mode="lines+markers",
        name="Temperature (tau)", yaxis="y2",
        line=dict(color="orange", dash="dot"),
    ))
    fig_rad.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Mean Reward",
        yaxis2=dict(title="Temperature", overlaying="y", side="right"),
        height=400,
        legend=dict(x=0, y=1.15, orientation="h"),
    )
    st.plotly_chart(fig_rad, use_container_width=True)

    st.caption(
        "Blue bars = forecaster training epochs, "
        "Red bars = adversary training epochs. "
        "Orange line = Boltzmann temperature decay."
    )

    st.subheader("Epoch Details")
    st.dataframe(epoch_results, use_container_width=True, height=300)

# -------------------------------------------------------------------
# Pipeline training log
# -------------------------------------------------------------------
pipeline_log = RESULTS_DIR / "training_output.txt"
if pipeline_log.exists():
    with st.expander("Raw Pipeline Training Output"):
        st.code(pipeline_log.read_text()[-5000:], language=None)
