"""Rendering helpers for resonance analytics."""

from __future__ import annotations

import math
from pathlib import Path
import json
import math
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def _format_pair_name(agents: List[str]) -> str:
    return "_".join(agents)


def _plot_timeseries(time: np.ndarray, a: np.ndarray, b: np.ndarray, out_path: Path) -> None:
    if time.size == 0:
        return
    plt.figure(figsize=(6, 3))
    plt.plot(time, a, label="rho_A", linewidth=1.2)
    plt.plot(time, b, label="rho_B", linewidth=1.2)
    plt.xlabel("time")
    plt.ylabel("rho")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_cross_correlation(a: np.ndarray, b: np.ndarray, dt: float, out_path: Path) -> None:
    if a.size < 2 or b.size < 2:
        return
    a_center = a - np.mean(a)
    b_center = b - np.mean(b)
    denom = np.sqrt(np.sum(a_center ** 2) * np.sum(b_center ** 2))
    if denom <= 1e-12:
        return
    corr_full = np.correlate(a_center, b_center, mode="full") / denom
    lags = (np.arange(corr_full.size) - (b_center.size - 1)) * dt
    plt.figure(figsize=(5, 3))
    plt.plot(lags, corr_full, linewidth=1.1)
    plt.xlabel("lag (time units)")
    plt.ylabel("NCCF")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def render_resonance_plots(metrics: Dict[str, object], plots_dir: Path) -> List[str]:
    pairs = metrics.get("pairs") or []
    created: List[str] = []
    for idx, pair in enumerate(pairs):
        time = pair.get("time")
        rho_a = pair.get("rho_a")
        rho_b = pair.get("rho_b")
        if not (time and rho_a and rho_b):
            continue
        time_arr = np.asarray(time, dtype=float)
        a_arr = np.asarray(rho_a, dtype=float)
        b_arr = np.asarray(rho_b, dtype=float)
        if time_arr.size < 2:
            continue
        agents = pair.get("agents", [f"pair{idx}"])
        base_name = _format_pair_name(agents)
        ts_path = plots_dir / f"resonance_timeseries_{idx}_{base_name}.png"
        _plot_timeseries(time_arr, a_arr, b_arr, ts_path)
        if ts_path.exists():
            created.append(str(ts_path))
        dt = float(np.median(np.diff(time_arr))) if time_arr.size > 1 else 1.0
        xc_path = plots_dir / f"resonance_xcorr_{idx}_{base_name}.png"
        _plot_cross_correlation(a_arr, b_arr, dt, xc_path)
        if xc_path.exists():
            created.append(str(xc_path))
    return created


def plot_objective_history(history_path: Path, out_path: Path) -> None:
    if not history_path.exists():
        return
    ts: List[float] = []
    objs: List[float] = []
    try:
        for line in history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj = row.get("objective")
            if obj is None or not math.isfinite(obj):
                continue
            ts.append(float(row.get("ts", len(ts))))
            objs.append(float(obj))
    except Exception:
        return
    if not objs:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.plot(ts, objs, linewidth=1.2)
    plt.xlabel("timestamp")
    plt.ylabel("resonance objective")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


__all__ = ["render_resonance_plots", "plot_objective_history"]
