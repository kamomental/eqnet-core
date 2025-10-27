#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render field metric telemetry into quick-look plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

import matplotlib

matplotlib.use("Agg")

try:  # pragma: no cover - plotting is optional in CI
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

__all__ = [
    "load_events",
    "extract_series",
    "plot_timeseries",
    "plot_rho_scatter",
    "render_plots",
]


def load_events(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def extract_series(events: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    series: Dict[str, List[float]] = {"ts": [], "S": [], "H": [], "rho": [], "Ignition": []}
    for row in events:
        if row.get("event") != "field.metrics":
            continue
        data = row.get("data") or {}
        try:
            series["ts"].append(float(row.get("ts", 0.0)))
            series["S"].append(float(data.get("S", np.nan)))
            series["H"].append(float(data.get("H", np.nan)))
            series["rho"].append(float(data.get("rho", np.nan)))
            series["Ignition"].append(float(data.get("Ignition", np.nan)))
        except Exception:
            continue
    if not series["ts"]:
        return {}
    base = series["ts"][0]
    return {key: np.array(vals, dtype=float) - (base if key == "ts" else 0.0) for key, vals in series.items()}


def plot_timeseries(series: Mapping[str, np.ndarray], output: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    fig, ax_ts = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    time = series["ts"] / 60.0  # minutes
    ax_ts.plot(time, series["S"], label="S (entropy)", color="#FF7F50")
    ax_ts.plot(time, series["H"], label="H (enthalpy)", color="#1E90FF")
    ax_ts.plot(time, series["rho"], label="ρ (synchrony)", color="#2E8B57")
    ax_ts.plot(time, series["Ignition"], label="Ignition", color="#8A2BE2", linestyle="--")
    ax_ts.set_xlabel("Minutes since start")
    ax_ts.set_ylabel("Normalised value")
    ax_ts.set_ylim(-0.05, 1.05)
    ax_ts.legend(loc="best")
    ax_ts.set_title("Field metrics vs time")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_rho_scatter(series: Mapping[str, np.ndarray], output: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    fig, ax_heat = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    ax_heat.hist2d(
        series["rho"],
        series["Ignition"],
        bins=32,
        range=[[0.0, 1.0], [0.0, 1.0]],
        cmap="magma",
    )
    ax_heat.set_xlabel("ρ")
    ax_heat.set_ylabel("Ignition")
    ax_heat.set_title("ρ vs Ignition density")
    cbar = fig.colorbar(ax_heat.collections[0], ax=ax_heat)
    cbar.set_label("count")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_plots(log_path: Path, output_dir: Path) -> Dict[str, Path]:
    events = load_events(log_path)
    series = extract_series(events)
    if not series:
        raise RuntimeError(f"No field.metrics events found in {log_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_dir / "ignition_timeseries.png"
    scatter_path = output_dir / "rho_vs_I_scatter.png"
    plot_timeseries(series, timeseries_path)
    plot_rho_scatter(series, scatter_path)
    return {"timeseries": timeseries_path, "scatter": scatter_path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("logs/telemetry_events.jsonl"), help="Telemetry event log path")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/plots"), help="Output directory for figures")
    args = ap.parse_args()

    result = render_plots(args.log, args.out_dir)
    for name, path in result.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
