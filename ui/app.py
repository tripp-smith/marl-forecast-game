"""MARL Forecast Game -- Explainability UI."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MARL Forecast Game", layout="wide")

st.markdown("""
<style>
/* Scenario cards */
div[data-testid="stVerticalBlock"] div[data-testid="stContainer"] {
    transition: box-shadow 0.15s ease;
}
div[data-testid="stVerticalBlock"] div[data-testid="stContainer"]:hover {
    box-shadow: 0 2px 12px rgba(100, 100, 255, 0.15);
}

/* Home page cards */
div[data-testid="stContainer"] div[data-testid="stContainer"] {
    min-height: 180px;
}

/* Tighter metric spacing inside cards */
div[data-testid="stContainer"] div[data-testid="stMetric"] {
    padding: 0;
}
div[data-testid="stContainer"] div[data-testid="stMetric"] label {
    font-size: 0.75rem;
}
div[data-testid="stContainer"] div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

home_page = st.Page("pages/home.py", title="Home", icon="\U0001f3e0", default=True)
replay = st.Page("pages/replay.py", title="Simulation Replay", icon="\u25b6")
agents_page = st.Page("pages/agents.py", title="Agent Contributions", icon="\U0001f916")
metrics_page = st.Page("pages/metrics.py", title="Metric Decomposition", icon="\U0001f4ca")
whatif_page = st.Page("pages/whatif.py", title="What-If", icon="\U0001f52c")
lineage_page = st.Page("pages/lineage.py", title="Data Lineage", icon="\U0001f5c2")
llm_page = st.Page("pages/llm.py", title="LLM Inspection", icon="\U0001f9e0")

nav = st.navigation([home_page, replay, agents_page, metrics_page, whatif_page, lineage_page, llm_page])

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
