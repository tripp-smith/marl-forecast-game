"""MARL Forecast Game -- Explainability UI."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MARL Forecast Game", layout="wide")

replay = st.Page("pages/replay.py", title="Simulation Replay", icon="\u25b6")
agents_page = st.Page("pages/agents.py", title="Agent Contributions", icon="\U0001f916")
metrics_page = st.Page("pages/metrics.py", title="Metric Decomposition", icon="\U0001f4ca")
whatif_page = st.Page("pages/whatif.py", title="What-If", icon="\U0001f52c")
lineage_page = st.Page("pages/lineage.py", title="Data Lineage", icon="\U0001f5c2")
llm_page = st.Page("pages/llm.py", title="LLM Inspection", icon="\U0001f9e0")

nav = st.navigation([replay, agents_page, metrics_page, whatif_page, lineage_page, llm_page])

with st.sidebar:
    st.title("MARL Forecast Game")
    try:
        import ray
        if ray.is_initialized():
            st.success("Ray cluster connected")
        else:
            st.info("Ray not initialized")
    except ImportError:
        st.warning("Ray not installed")

nav.run()
