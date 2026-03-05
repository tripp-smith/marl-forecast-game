"""Shared helpers for the Streamlit UI."""
from __future__ import annotations

import datetime
import csv
import json
import io
from pathlib import Path
from typing import Any

import streamlit as st

RESULTS_DIR = Path("/app/results")


def discover_result_files(
    results_dir: str | Path = RESULTS_DIR,
    suffix: str = ".json",
) -> list[Path]:
    """Scan a results directory for output files, sorted newest-first."""
    d = Path(results_dir)
    if not d.is_dir():
        return []
    files = sorted(d.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def get_result_metadata(path: Path) -> dict[str, Any]:
    """Extract display metadata from a result JSON without loading full trajectory."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {"path": path, "name": path.stem, "label": path.stem}

    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    meta: dict[str, Any] = {
        "path": path,
        "name": path.stem,
        "label": data.get("label", path.stem),
        "seed": data.get("seed", "?"),
        "disturbed": data.get("disturbed", "?"),
        "rounds": data.get("convergence", {}).get("rounds_executed", "?"),
        "mae": data.get("metrics", {}).get("mae"),
        "rmse": data.get("metrics", {}).get("rmse"),
        "timestamp": mtime.strftime("%H:%M:%S"),
    }
    return meta


def _format_metric(val: Any) -> str:
    if val is None or val == "?":
        return "--"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def render_scenario_cards(
    key_prefix: str,
    filter_fn: Any = None,
) -> Path | None:
    """Render visual cards for available result files. Returns selected path or None."""
    result_files = discover_result_files()
    if not result_files:
        return None

    metas = [get_result_metadata(p) for p in result_files]
    if filter_fn is not None:
        metas = [m for m in metas if filter_fn(m)]
    if not metas:
        return None

    state_key = f"_selected_{key_prefix}"

    cols_per_row = min(len(metas), 3)
    for row_start in range(0, len(metas), cols_per_row):
        row_metas = metas[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, meta in zip(cols, row_metas):
            with col:
                is_selected = st.session_state.get(state_key) == str(meta["path"])
                with st.container(border=True):
                    label = str(meta.get("label", meta["name"]))
                    disturbed = meta.get("disturbed", "?")
                    if disturbed is True:
                        badge = " \u26a0\ufe0f"
                    elif disturbed is False:
                        badge = " \u2705"
                    else:
                        badge = ""
                    st.markdown(f"**{label}**{badge}")

                    c1, c2 = st.columns(2)
                    c1.caption(f"Seed: {meta.get('seed', '?')}")
                    c2.caption(f"Rounds: {meta.get('rounds', '?')}")

                    mae_str = _format_metric(meta.get("mae"))
                    rmse_str = _format_metric(meta.get("rmse"))
                    if mae_str != "--" or rmse_str != "--":
                        m1, m2 = st.columns(2)
                        m1.metric("MAE", mae_str)
                        m2.metric("RMSE", rmse_str)

                    st.caption(f"\u23f0 {meta.get('timestamp', '')}")

                    if st.button(
                        "Selected" if is_selected else "Select",
                        key=f"{key_prefix}_{meta['name']}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state[state_key] = str(meta["path"])
                        st.rerun()

    selected = st.session_state.get(state_key)
    if selected:
        return Path(selected)
    return None


def load_trajectory_logs(path: str | Path) -> list[dict[str, Any]]:
    """Load trajectory_logs from a JSON file."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict) and "trajectory_logs" in data:
        return [dict(item) for item in data["trajectory_logs"]]
    return []


def load_simulation_result(path: str | Path) -> dict[str, Any]:
    """Load a full simulation result dict from JSON."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return dict(json.load(f))


def build_round_dataframe(result: dict[str, Any]) -> list[dict[str, float | int]]:
    logs = result.get("trajectory_logs", [])
    if logs:
        rows = []
        for idx, entry in enumerate(logs):
            rows.append(
                {
                    "round": idx,
                    "forecast": float(entry.get("forecast", 0.0)),
                    "target": float(entry.get("target", 0.0)),
                    "reward": float(entry.get("reward", 0.0)),
                    "disturbance": float(entry.get("disturbance", 0.0)),
                }
            )
        return rows
    forecasts = result.get("forecasts", [])
    targets = result.get("targets", [])
    n = min(len(forecasts), len(targets))
    return [
        {"round": idx, "forecast": float(forecasts[idx]), "target": float(targets[idx])}
        for idx in range(n)
    ]


def build_fan_chart_frame(result: dict[str, Any], *, window: int = 8) -> list[dict[str, float | int]]:
    frame = build_round_dataframe(result)
    if not frame:
        return frame
    residuals = [abs(float(row["forecast"]) - float(row["target"])) for row in frame]
    bands: list[dict[str, float | int]] = []
    for idx, row in enumerate(frame):
        start = max(0, idx - window + 1)
        rolling = sum(residuals[start : idx + 1]) / max(1, idx - start + 1)
        forecast = float(row["forecast"])
        bands.append(
            {
                **row,
                "band_50_low": forecast - 0.5 * rolling,
                "band_50_high": forecast + 0.5 * rolling,
                "band_90_low": forecast - 1.5 * rolling,
                "band_90_high": forecast + 1.5 * rolling,
            }
        )
    return bands


def export_result_csv(result: dict[str, Any]) -> bytes:
    rows = build_round_dataframe(result)
    if not rows:
        return b""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def export_result_pdf(result: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    metrics = result.get("metrics", {})
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle("MARL Forecast Report")
        pdf.drawString(72, 760, "MARL Forecast Game Report")
        y = 730
        for key in ["mae", "rmse", "mape", "worst_case"]:
            pdf.drawString(72, y, f"{key}: {metrics.get(key, 0)}")
            y -= 18
        pdf.save()
    except ImportError:
        payload = {
            "title": "MARL Forecast Game Report",
            "metrics": metrics,
            "rounds": int(result.get("convergence", {}).get("rounds_executed", 0)),
        }
        buffer.write(json.dumps(payload, indent=2).encode("utf-8"))
    return buffer.getvalue()


def run_simulation(
    horizon: int = 100,
    max_rounds: int = 200,
    disturbance_prob: float = 0.1,
    disturbance_scale: float = 1.0,
    adversarial_intensity: float = 1.0,
    defense_model: str = "dampening",
    disturbance_model: str = "gaussian",
    seed: int = 42,
    initial_value: float = 10.0,
    initial_exogenous: float = 0.0,
    initial_hidden_shift: float = 0.0,
    disturbed: bool = True,
) -> dict[str, Any]:
    """Run a simulation and return the output dict."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from framework.distributed import _game_outputs_to_dict
    from framework.game import ForecastGame
    from framework.metrics import mae, mape, rmse, worst_case_abs_error
    from framework.types import ForecastState, SimulationConfig

    config = SimulationConfig(
        horizon=horizon,
        max_rounds=max_rounds,
        disturbance_prob=disturbance_prob,
        disturbance_scale=disturbance_scale,
        adversarial_intensity=adversarial_intensity,
        defense_model=defense_model,
        disturbance_model=disturbance_model,
    )
    state = ForecastState(
        t=0,
        value=initial_value,
        exogenous=initial_exogenous,
        hidden_shift=initial_hidden_shift,
    )
    game = ForecastGame(config, seed=seed)
    out = game.run(state, disturbed=disturbed)
    result = _game_outputs_to_dict(out)
    result["seed"] = seed
    result["metrics"] = {
        "mae": mae(out.targets, out.forecasts),
        "rmse": rmse(out.targets, out.forecasts),
        "mape": mape(out.targets, out.forecasts),
        "worst_case": worst_case_abs_error(out.targets, out.forecasts),
    }
    return result
