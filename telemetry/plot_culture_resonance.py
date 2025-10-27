"""Rendering helpers for culture-tagged affective statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
import numpy as np


def render_culture_resonance(
    culture_stats: Mapping[str, Mapping[str, float]],
    out_path: Path,
) -> None:
    """Render a bar/line chart summarising culture-tagged affective metrics."""
    if not culture_stats:
        return

    tags = list(culture_stats.keys())
    valence = np.array([float(culture_stats[tag].get("mean_valence", 0.0)) for tag in tags], dtype=float)
    arousal = np.array([float(culture_stats[tag].get("mean_arousal", 0.0)) for tag in tags], dtype=float)
    rho = np.array([float(culture_stats[tag].get("mean_rho", 0.0)) for tag in tags], dtype=float)
    counts = np.array([float(culture_stats[tag].get("count", 0.0)) for tag in tags], dtype=float)

    if valence.size == 0:
        return

    idx = np.arange(len(tags), dtype=float)
    width = 0.35

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    ax2 = ax1.twinx()

    bars_val = ax1.bar(idx - width / 2, valence, width=width, label="mean valence", color="#4C78A8")
    bars_ar = ax1.bar(idx + width / 2, arousal, width=width, label="mean arousal", color="#F58518")
    line_rho, = ax2.plot(idx, rho, marker="o", linestyle="-", color="#54A24B", label="mean rho")

    for i, count in enumerate(counts):
        if count > 0:
            ax1.text(
                idx[i],
                max(valence[i], arousal[i]) + 0.02,
                f"n={int(round(count))}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )

    ax1.set_xticks(idx)
    ax1.set_xticklabels(tags, rotation=30, ha="right")
    min_left = float(min(valence.min(), arousal.min()))
    max_left = float(max(valence.max(), arousal.max()))
    pad = max(0.1, (max_left - min_left) * 0.1)
    ax1.set_ylim(min(min_left - pad, -0.2), max(max_left + pad, 1.2))
    ax1.set_ylabel("valence / arousal")
    ax1.grid(True, axis="y", alpha=0.2)

    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("rho (synchrony)")

    handles = [bars_val, bars_ar, line_rho]
    labels = ["mean valence", "mean arousal", "mean rho"]
    ax1.legend(handles, labels, loc="upper left", fontsize=8)
    ax1.set_title("Culture-tagged affective summary")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


__all__ = ["render_culture_resonance"]
