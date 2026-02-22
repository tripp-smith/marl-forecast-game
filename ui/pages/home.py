"""Home -- visual dashboard landing page."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.utils import RESULTS_DIR, discover_result_files

st.title("MARL Forecast Game")

pipeline_exit_file = RESULTS_DIR / ".pipeline_exit_code"
result_files = discover_result_files()

col_status, col_files, col_links = st.columns(3)

with col_status:
    if pipeline_exit_file.exists():
        code = pipeline_exit_file.read_text().strip()
        if code == "0":
            st.success("Pipeline: **passed**")
        else:
            st.warning(f"Pipeline: **completed** (exit {code})")
    else:
        st.info("Pipeline: **running or not started**")

with col_files:
    st.metric("Result files", len(result_files))

with col_links:
    st.markdown(
        "[Grafana :3000](http://localhost:3000) &nbsp; | &nbsp; "
        "[Prometheus :9090](http://localhost:9090)"
    )

st.divider()

PAGES = [
    {
        "icon": "\u25b6\ufe0f",
        "title": "Simulation Replay",
        "desc": "Step through game rounds with interactive charts",
        "key": "nav_replay",
    },
    {
        "icon": "\U0001f916",
        "title": "Agent Contributions",
        "desc": "Per-agent delta and reward analysis",
        "key": "nav_agents",
    },
    {
        "icon": "\U0001f4ca",
        "title": "Metric Decomposition",
        "desc": "Clean vs attacked error attribution",
        "key": "nav_metrics",
    },
    {
        "icon": "\U0001f52c",
        "title": "What-If",
        "desc": "Tweak parameters and re-run simulations",
        "key": "nav_whatif",
    },
    {
        "icon": "\U0001f5c2\ufe0f",
        "title": "Data Lineage",
        "desc": "Visualize the data pipeline and splits",
        "key": "nav_lineage",
    },
    {
        "icon": "\U0001f9e0",
        "title": "LLM Inspection",
        "desc": "Inspect LLM-driven agent decisions",
        "key": "nav_llm",
    },
]

for row_start in range(0, len(PAGES), 3):
    row = PAGES[row_start : row_start + 3]
    cols = st.columns(3)
    for col, page in zip(cols, row):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align:center;font-size:2.5rem;line-height:1.2'>"
                    f"{page['icon']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='text-align:center'><strong>{page['title']}</strong></div>",
                    unsafe_allow_html=True,
                )
                st.caption(page["desc"])
                st.page_link(
                    f"pages/{page['key'].replace('nav_', '')}.py",
                    label=f"Open {page['title']}",
                    use_container_width=True,
                )

if result_files:
    st.divider()
    st.subheader("Available Results")
    for f in result_files[:8]:
        st.text(f"\u2022 {f.name}")
    if len(result_files) > 8:
        st.caption(f"... and {len(result_files) - 8} more")
