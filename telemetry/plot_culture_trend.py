"""Rendering helpers for culture-tagged trend visualisations."""

from __future__ import annotations

import collections
import json
import os
from typing import DefaultDict, List

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # type: ignore  # noqa: E402


def render_culture_trend(jsonl_path: str, out_path: str, *, min_count: int = 20) -> None:
    """Render multi-day trends for culture-tagged valence and rho metrics."""
    try:
        handle = open(jsonl_path, encoding="utf-8")
    except FileNotFoundError:
        return
    with handle:
        series: DefaultDict[str, dict[str, List[float]]] = collections.defaultdict(
            lambda: {"t": [], "v": [], "r": []}
        )
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = record.get("tag") or "unknown"
            count = record.get("count")
            try:
                count_val = int(count)
            except Exception:
                count_val = 0
            if count_val < max(0, min_count):
                continue
            ts = record.get("ts")
            valence = record.get("mean_valence")
            rho = record.get("mean_rho")
            if ts is None or valence is None or rho is None:
                continue
            try:
                series[tag]["t"].append(float(ts))
                series[tag]["v"].append(float(valence))
                series[tag]["r"].append(float(rho))
            except Exception:
                continue
    if not series:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (tag, values) in enumerate(sorted(series.items())):
        color = colors[idx % len(colors)] if colors else None
        ax1.plot(values["t"], values["v"], label=f"{tag} valence", color=color)
        ax2.plot(
            values["t"],
            values["r"],
            linestyle="--",
            label=f"{tag} rho",
            color=color,
        )

    ax1.set_xlabel("time")
    ax1.set_ylabel("mean valence")
    ax1.grid(True, alpha=0.2)
    ax2.set_ylabel("mean rho (right axis)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles1 or handles2:
        fig.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="lower center",
            ncol=max(1, len(handles1) + len(handles2)),
            fontsize=8,
        )
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


__all__ = ["render_culture_trend"]
