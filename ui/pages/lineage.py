"""Data Lineage -- visualization of data pipeline and splits."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.header("Data Lineage")

st.subheader("Data Pipeline")
st.graphviz_chart("""
digraph pipeline {
    rankdir=LR
    node [shape=box style=rounded]
    API [label="External API\\n(FRED, IMF, etc.)"]
    Adapter [label="SourceAdapter.fetch()"]
    Cache [label="Cache\\ndata/cache/"]
    Validate [label="Schema +\\nChronological\\nValidation"]
    Poison [label="Poisoning\\nDetection"]
    Normalize [label="Normalization\\n(z-score)"]
    Split [label="Chronological\\nSplit"]
    Bundle [label="DatasetBundle\\n(train/valid/test)"]

    API -> Adapter -> Cache -> Validate -> Poison -> Normalize -> Split -> Bundle
}
""")

st.subheader("Dataset Configuration")
col1, col2 = st.columns(2)

with col1:
    source = st.selectbox(
        "Data Source",
        ["sample_csv", "fred", "imf", "polymarket", "bis", "gpr",
         "oecd_cli", "kaggle", "worldbank", "bea", "kalshi",
         "predictit", "eurostat", "hybrid"],
    )
    periods = st.number_input("Periods", min_value=10, max_value=10000, value=240)

with col2:
    train_ratio = st.slider("Train Ratio", 0.1, 0.9, 0.7, 0.05)
    valid_ratio = st.slider("Valid Ratio", 0.05, 0.5, 0.15, 0.05)
    normalize = st.checkbox("Normalize", value=True)

if st.button("Load Dataset"):
    try:
        from framework.data import DataProfile, load_dataset

        profile = DataProfile(
            source=source,
            periods=periods,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            normalize=normalize,
        )
        bundle = load_dataset(profile)

        st.success(f"Loaded: {len(bundle.train)} train, {len(bundle.valid)} valid, {len(bundle.test)} test rows")

        total = len(bundle.train) + len(bundle.valid) + len(bundle.test)
        fig = go.Figure(go.Bar(
            x=["Train", "Valid", "Test"],
            y=[len(bundle.train), len(bundle.valid), len(bundle.test)],
            text=[f"{len(bundle.train)/total*100:.1f}%", f"{len(bundle.valid)/total*100:.1f}%", f"{len(bundle.test)/total*100:.1f}%"],
            textposition="auto",
        ))
        fig.update_layout(yaxis_title="Rows", height=300)
        st.plotly_chart(fig, use_container_width=True)

        if bundle.train:
            train_targets = [r.get("target", 0) for r in bundle.train]
            valid_targets = [r.get("target", 0) for r in bundle.valid]
            test_targets = [r.get("target", 0) for r in bundle.test]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=train_targets, mode="lines", name="Train"))
            fig2.add_trace(go.Scatter(
                x=list(range(len(train_targets), len(train_targets) + len(valid_targets))),
                y=valid_targets, mode="lines", name="Valid",
            ))
            fig2.add_trace(go.Scatter(
                x=list(range(len(train_targets) + len(valid_targets), total)),
                y=test_targets, mode="lines", name="Test",
            ))
            fig2.update_layout(xaxis_title="Index", yaxis_title="Target", height=400)
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")

st.subheader("Cache Status")
cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache"
if cache_dir.exists():
    cache_files = sorted(cache_dir.glob("*.json"))
    if cache_files:
        cache_info = []
        for f in cache_files:
            stat = f.stat()
            cache_info.append({
                "File": f.name,
                "Size (KB)": f"{stat.st_size / 1024:.1f}",
            })
        st.table(cache_info)
    else:
        st.info("No cached files found.")
else:
    st.info("Cache directory does not exist.")
