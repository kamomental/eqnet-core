#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamlit viewer for EQNet telemetry logs.

Launch with:
    streamlit run tools/eqnet_telemetry_viewer.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from emot_terrain_lab.terrain.emotion import AXES as _EMOTION_AXES  # type: ignore
except Exception:  # pragma: no cover - viewer fallback
    _EMOTION_AXES = [
        "sensory",
        "temporal",
        "spatial",
        "affective",
        "cognitive",
        "social",
        "meta",
        "agency",
        "recursion",
    ]

LOG_DIR = Path("logs")
KPI_COLUMNS = ["body.R", "affect.love", "value.intent_trust", "fastpath.override_rate"]
AXES_COLUMNS = list(_EMOTION_AXES)


def load_jsonl(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                obj.setdefault("_source", source)
                rows.append(obj)
    return pd.DataFrame(rows)


def expand_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "metrics" not in df.columns:
        return df
    metrics = pd.json_normalize(df["metrics"]).add_prefix("")
    df = df.drop(columns=["metrics"]).reset_index(drop=True)
    return pd.concat([df, metrics], axis=1)


def prepare_df(df: pd.DataFrame, episode_id: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if "episode_id" in df.columns and episode_id is not None:
        df = df[df["episode_id"] == episode_id]
    return df


def ensure_time_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "step" in df.columns:
        return df, "step"
    df = df.reset_index().rename(columns={"index": "step"})
    return df, "step"


def _expand_axis_column(
    df: pd.DataFrame,
    column_name: str,
    axis_labels: Iterable[str],
) -> pd.DataFrame:
    if column_name not in df.columns:
        return df
    axis_labels = list(axis_labels)
    expanded: list[list[Optional[float]]] = []
    for value in df[column_name].tolist():
        if isinstance(value, list):
            vals = [float(v) if isinstance(v, (int, float)) else None for v in value]
        else:
            vals = []
        if len(vals) < len(axis_labels):
            vals = vals + [None] * (len(axis_labels) - len(vals))
        else:
            vals = vals[: len(axis_labels)]
        expanded.append(vals)
    columns = [f"{column_name}_{axis}" for axis in axis_labels]
    expanded_df = pd.DataFrame(expanded, columns=columns)
    df = df.drop(columns=[column_name]).reset_index(drop=True)
    return pd.concat([df, expanded_df], axis=1)


def main() -> None:
    st.title("EQNet Telemetry Viewer")
    st.sidebar.header("Settings")
    base_dir = Path(st.sidebar.text_input("logs dir", str(LOG_DIR)))

    df_kpi = expand_metrics(load_jsonl(base_dir / "kpi_rollup.jsonl", "kpi"))
    df_mcp = load_jsonl(base_dir / "mcp_actions.jsonl", "mcp")
    df_learner = load_jsonl(base_dir / "learner_hooks.jsonl", "learner")
    df_fast = load_jsonl(base_dir / "fastpath_state.jsonl", "fastpath")
    df_self = load_jsonl(base_dir / "self_report.jsonl", "self")
    df_narr = load_jsonl(base_dir / "narrative_log.jsonl", "narrative")
    df_moment = load_jsonl(base_dir / "moment_log.jsonl", "moment")
    df_risk = load_jsonl(base_dir / "future_risk.jsonl", "future_risk")

    if df_kpi.empty:
        st.warning("kpi_rollup.jsonl が見つからないか、空です。")
        return

    episodes = sorted(df_kpi.get("episode_id", pd.Series(dtype=str)).dropna().unique())
    episode_id = st.sidebar.selectbox("episode_id", episodes) if episodes else None

    dfk = prepare_df(df_kpi, episode_id)
    dfm = prepare_df(df_mcp, episode_id)
    dfl = prepare_df(df_learner, episode_id)
    dff = prepare_df(df_fast, episode_id)
    dfs = prepare_df(df_self, episode_id)
    dfmom = prepare_df(df_moment, episode_id)
    dfr = prepare_df(df_risk, episode_id)

    st.subheader(f"Episode: {episode_id or 'all'}")
    dfk, t_col = ensure_time_column(dfk)
    dfmom = expand_metrics(dfmom)
    dfmom = _expand_axis_column(dfmom, 'emotion_axes_sensor', AXES_COLUMNS)
    dfmom = _expand_axis_column(dfmom, 'emotion_axes_blended', AXES_COLUMNS)

    st.markdown("### KPI 時系列")
    for col in KPI_COLUMNS:
        if col in dfk.columns:
            fig = px.line(dfk, x=t_col, y=col, title=col)
            st.plotly_chart(fig, use_container_width=True)
    missing = [col for col in KPI_COLUMNS if col not in dfk.columns]
    if missing:
        st.info(f"Columns not found in KPIs: {', '.join(missing)}")

    st.markdown("### MCP Actions")
    if not dfm.empty:
        st.dataframe(dfm)
    else:
        st.info("MCP actions not recorded for this selection.")

    st.markdown("### Learner Hooks")
    if not dfl.empty:
        st.dataframe(dfl)
    else:
        st.info("Learner hooks not recorded for this selection.")

    st.markdown("### FAST-path State")
    st.markdown("### Future Risk")
    if dfr.empty:
        st.info("future_risk.jsonl not found or empty.")
    else:
        dfr_plot, t_risk = ensure_time_column(dfr.copy())
        fig_risk = px.line(dfr_plot, x=t_risk, y="future_risk_stress", title="Predicted future risk")
        st.plotly_chart(fig_risk, use_container_width=True)
        compare_cols = [col for col in ("body.R", "tension") if col in dfk.columns]
        if compare_cols:
            fig_actual = px.line(dfk, x=t_col, y=compare_cols, title="Actual body metrics")
            st.plotly_chart(fig_actual, use_container_width=True)

    st.markdown("### Sensor Trace")
    if dfmom.empty:
        st.info("moment_log.jsonl not found or empty.")
    else:
        dfmom_plot, t_moment = ensure_time_column(dfmom.copy())
        heart_cols = [col for col in ("heart_rate", "activity_level") if col in dfmom_plot.columns]
        if heart_cols:
            fig_hr = px.line(dfmom_plot, x=t_moment, y=heart_cols, title="Heart rate & activity")
            st.plotly_chart(fig_hr, use_container_width=True)
        hr_decomp = [col for col in ("heart_rate_motion", "heart_rate_emotion") if col in dfmom_plot.columns]
        if hr_decomp:
            fig_decomp = px.line(dfmom_plot, x=t_moment, y=hr_decomp, title="HR decomposition")
            st.plotly_chart(fig_decomp, use_container_width=True)
        sensor_cols = [col for col in dfmom_plot.columns if col.startswith("emotion_axes_sensor_")]
        if sensor_cols:
            fig_sensor = px.line(dfmom_plot, x=t_moment, y=sensor_cols, title="Sensor-derived axes")
            st.plotly_chart(fig_sensor, use_container_width=True)
        blended_cols = [col for col in dfmom_plot.columns if col.startswith("emotion_axes_blended_")]
        if blended_cols:
            fig_blend = px.line(dfmom_plot, x=t_moment, y=blended_cols, title="Blended emotion axes")
            st.plotly_chart(fig_blend, use_container_width=True)
    if not dff.empty:
        if "override_rate" in dff.columns:
            dff_plot, t_fast = ensure_time_column(dff.copy())
            fig = px.line(dff_plot, x=t_fast, y="override_rate", title="fastpath.override_rate")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dff)
    else:
        st.info("fastpath_state.jsonl に該当レコードがありません。")

    st.markdown("### Self Report")
    if not dfs.empty:
        entry = dfs.iloc[-1]
        st.write(f"Mood: {entry.get('mood', 'n/a')} / Tone: {entry.get('social_tone', 'n/a')}")
        st.write(entry.get('summary', ''))
        metrics_snapshot = entry.get('metrics')
        if isinstance(metrics_snapshot, dict):
            st.json(metrics_snapshot)
        else:
            st.dataframe(dfs)
    else:
        st.info("Self report not recorded for this selection.")

    st.markdown("### Narrative Context")
    df_narrative = pd.DataFrame()
    if episode_id and not df_narr.empty and 'episode_ids' in df_narr.columns:
        mask = df_narr['episode_ids'].apply(lambda value: isinstance(value, list) and episode_id in value)
        df_narrative = df_narr[mask]
    if not df_narrative.empty:
        row = df_narrative.iloc[-1]
        st.write(f"Theme: {row.get('theme', 'n/a')} / Emotional trend: {row.get('emotional_trend', 'n/a')}")
        st.write(row.get('description', ''))
        st.json(row.get('metrics', {}))
    else:
        st.info("No narrative entry covers this episode.")




if __name__ == "__main__":
    main()



