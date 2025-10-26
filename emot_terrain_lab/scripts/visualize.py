# -*- coding: utf-8 -*-
"""
Visualization utilities for the 9D emotion/qualia terrain.

Outputs (written to the path given by --out):
  - trajectory_tsne.png                 : t-SNE projection of full trajectories
  - terrain_slice.png                   : heatmap + contour for selected 2 axes
  - terrain_slice_contour.png           : filled contour view of the same slice
  - terrain_surface.png                 : 3D surface (selected 2 axes + gradient)
  - trajectory_projection3d.png         : 3D scatter along chosen axes
  - top_emotion_components.png          : timeline of dominant components
  - (optional) bar chart race animation when --bar-race is provided
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import bar_chart_race as bcr
except ImportError:  # pragma: no cover - optional dependency
    bcr = None

from terrain.emotion import AXES, AXIS_BOUNDS
from terrain.system import EmotionalMemorySystem
from terrain.config import project_emotion

NEGATIVE_AXES = {axis for axis, bounds in AXIS_BOUNDS.items() if bounds[0] < 0}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_logs(path: Path, user_id: str) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows = [row for row in rows if row.get("user_id") == user_id]
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def axis_indices(selected: Sequence[str]) -> Tuple[int, ...]:
    return tuple(AXES.index(name) for name in selected)


def normalise_axis(axis: str, values: np.ndarray) -> np.ndarray:
    low, high = AXIS_BOUNDS[axis]
    if np.isclose(low, high):
        return np.zeros_like(values)
    normed = (values - low) / (high - low)
    return np.clip(normed, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_tsne_trajectory(points: np.ndarray, out_dir: Path) -> None:
    plt.figure()
    plt.plot(points[:, 0], points[:, 1], marker="o")
    for idx, (x, y) in enumerate(points):
        if idx % max(1, len(points) // 10) == 0:
            plt.text(x, y, str(idx))
    plt.title("Emotion trajectory (t-SNE)")
    plt.savefig(out_dir / "trajectory_tsne.png", dpi=160, bbox_inches="tight")
    plt.close()


def build_dense_slice(system: EmotionalMemorySystem, axes_pair: Tuple[str, str]) -> np.ndarray:
    terrain = system.terrain
    res = terrain.grid_res
    idx_pair = axis_indices(axes_pair)
    grid = np.zeros((res, res), dtype=float)
    counts = np.zeros((res, res), dtype=float)

    for coord, value in terrain.gradient_map.items():
        x = coord[idx_pair[0]]
        y = coord[idx_pair[1]]
        grid[x, y] += value
    for coord, value in terrain.visit_count.items():
        x = coord[idx_pair[0]]
        y = coord[idx_pair[1]]
        counts[x, y] += value

    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = np.where(counts > 0, grid / counts, grid)
    return averaged


def plot_terrain_slice(system: EmotionalMemorySystem, axes_pair: Tuple[str, str], out_dir: Path) -> np.ndarray:
    reduced = build_dense_slice(system, axes_pair)

    xs = np.linspace(0, 1, reduced.shape[0])
    ys = np.linspace(0, 1, reduced.shape[1])

    v_min = float(reduced.min())
    v_max = float(reduced.max())
    if np.isclose(v_min, v_max):
        contour_levels = 10
        contour_lines = None
    else:
        contour_levels = np.linspace(v_min, v_max, 12)
        contour_lines = contour_levels

    plt.figure()
    plt.imshow(reduced.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
    if contour_lines is not None:
        plt.contour(xs, ys, reduced.T, levels=contour_lines, colors="white", linewidths=0.6)
    plt.title(f"Terrain slice ({axes_pair[0]} vs {axes_pair[1]})")
    plt.xlabel(axes_pair[0])
    plt.ylabel(axes_pair[1])
    plt.colorbar()
    plt.savefig(out_dir / "terrain_slice.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    contour = plt.contourf(xs, ys, reduced.T, levels=contour_levels, cmap="viridis")
    if contour_lines is not None:
        plt.contour(xs, ys, reduced.T, levels=contour_lines, colors="black", linewidths=0.5)
    plt.title(f"Terrain contour ({axes_pair[0]} vs {axes_pair[1]})")
    plt.xlabel(axes_pair[0])
    plt.ylabel(axes_pair[1])
    plt.colorbar(contour)
    plt.savefig(out_dir / "terrain_slice_contour.png", dpi=160, bbox_inches="tight")
    plt.close()

    return reduced


def plot_terrain_surface(averaged: np.ndarray, axes_pair: Tuple[str, str], out_dir: Path) -> None:
    x_axis, y_axis = axes_pair
    xs = np.linspace(0, 1, averaged.shape[0])
    ys = np.linspace(0, 1, averaged.shape[1])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = averaged

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)

    z_min = float(Z.min())
    z_max = float(Z.max())
    if not np.isclose(z_min, z_max):
        contour_levels = np.linspace(z_min, z_max, 12)
        offset = z_min - 0.05 * (z_max - z_min + 1e-6)
        ax.contour(
            X,
            Y,
            Z,
            zdir="z",
            offset=offset,
            levels=contour_levels,
            cmap=cm.viridis,
            linewidths=0.7,
        )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel("gradient")
    ax.set_title(f"Terrain surface ({x_axis} vs {y_axis})")
    fig.colorbar(surface, shrink=0.6, aspect=12, pad=0.1)
    fig.savefig(out_dir / "terrain_surface.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_projection_3d(emotions: np.ndarray, axes_triplet: Tuple[str, str, str], out_dir: Path) -> None:
    idxs = axis_indices(axes_triplet)
    coords = np.stack([normalise_axis(axis, emotions[:, idx]) for axis, idx in zip(axes_triplet, idxs)], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker="o")
    ax.set_xlabel(axes_triplet[0])
    ax.set_ylabel(axes_triplet[1])
    ax.set_zlabel(axes_triplet[2])
    ax.set_title(f"Trajectory projection ({', '.join(axes_triplet)})")
    fig.savefig(out_dir / "trajectory_projection3d.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_top_component_timeline(emotions: np.ndarray, timestamps: Sequence[str], out_dir: Path) -> None:
    if emotions.size == 0:
        return
    top_indices = np.argmax(emotions, axis=1)
    values = np.take_along_axis(emotions, top_indices[:, None], axis=1).squeeze(axis=1)
    colors = plt.cm.tab20(top_indices / max(len(AXES) - 1, 1))

    positions = np.arange(len(top_indices))
    plt.figure()
    plt.bar(positions, values, color=colors)
    plt.xticks(positions, [ts.split("T")[0] for ts in timestamps], rotation=45, ha="right")
    legend_items = [
        plt.Line2D([0], [0], color=plt.cm.tab20(i / max(len(AXES) - 1, 1)), lw=4, label=axis)
        for i, axis in enumerate(AXES)
    ]
    plt.legend(handles=legend_items, title="Top component", ncol=2)
    plt.ylabel("value")
    plt.title("Dominant component over time")
    plt.tight_layout()
    plt.savefig(out_dir / "top_emotion_components.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_axis_trends(emotions: np.ndarray, timestamps: Sequence[str], out_dir: Path) -> None:
    if emotions.size == 0:
        return
    times = pd.to_datetime(timestamps)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    cmap = plt.cm.get_cmap("tab20", len(AXES))
    for idx, axis in enumerate(AXES):
        ax = axes[idx]
        ax.plot(times, emotions[:, idx], color=cmap(idx), linewidth=1.5)
        ax.set_title(axis)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_ylim(AXIS_BOUNDS[axis])
    for ax in axes[-3:]:
        ax.set_xlabel("time")
    fig.autofmt_xdate()
    plt.suptitle("Emotion axis trends")
    plt.tight_layout()
    plt.savefig(out_dir / "axis_trends.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_projected_path(emotions: np.ndarray, timestamps: Sequence[str], locale: str, out_dir: Path) -> None:
    if emotions.size == 0:
        return
    projected = np.array([project_emotion(vec, locale) for vec in emotions], dtype=float)
    times = pd.to_datetime(timestamps)

    plt.figure()
    plt.plot(projected[:, 0], projected[:, 1], marker="o", linewidth=1.5)
    for idx, (x, y) in enumerate(projected):
        if idx % max(1, len(projected) // 10) == 0:
            label = times[idx].strftime("%m-%d %H:%M")
            plt.text(x, y, label, fontsize=8)
    plt.title(f"Projected cultural trajectory ({locale})")
    plt.xlabel("culture axis 1")
    plt.ylabel("culture axis 2")
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "projection_path.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hexbin(projected[:, 0], projected[:, 1], gridsize=25, cmap="viridis")
    plt.title(f"Projected presence density ({locale})")
    plt.xlabel("culture axis 1")
    plt.ylabel("culture axis 2")
    plt.colorbar(label="counts")
    plt.savefig(out_dir / "projection_density.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_field_energy(field_snapshot: dict, out_dir: Path) -> None:
    energy = field_snapshot["energy"]
    plt.figure()
    plt.imshow(energy, origin="lower", cmap="magma")
    plt.title("Emotion field energy")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="energy")
    plt.savefig(out_dir / "field_energy.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_field_flow(field_snapshot: dict, out_dir: Path) -> None:
    energy = field_snapshot["energy"]
    flow_x = field_snapshot["flow_x"]
    flow_y = field_snapshot["flow_y"]
    magnitude = field_snapshot["magnitude"]
    xs = np.linspace(0, 1, energy.shape[1])
    ys = np.linspace(0, 1, energy.shape[0])
    plt.figure()
    strm = plt.streamplot(
        xs,
        ys,
        flow_x,
        flow_y,
        color=magnitude,
        cmap="viridis",
        density=1.0,
        linewidth=1.0,
    )
    plt.title("Emotion field flow")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(strm.lines, label="|flow|")
    plt.savefig(out_dir / "field_flow.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_field_phase(field_snapshot: dict, out_dir: Path) -> None:
    phase = field_snapshot["phase"]
    plt.figure()
    plt.imshow(phase, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.title("Emotion field phase")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="phase (rad)")
    plt.savefig(out_dir / "field_phase.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_field_metrics(log: list[dict], out_dir: Path) -> None:
    path = out_dir / "field_metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(log, handle, ensure_ascii=False, indent=2)
    if len(log) < 2:
        return
    df = pd.DataFrame(log)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        x = df["timestamp"]
    else:
        df = df.reset_index()
        x = df.index
    plt.figure()
    for column, label in (
        ("entropy", "entropy"),
        ("dissipation", "dissipation"),
        ("info_flux", "info flux"),
        ("enthalpy_mean", "enthalpy mean"),
    ):
        if column in df.columns:
            plt.plot(x, df[column], label=label)
    plt.legend()
    plt.title("Field thermodynamic metrics")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(alpha=0.3, linestyle="--")
    if "timestamp" in df.columns:
        plt.gcf().autofmt_xdate()
    plt.savefig(out_dir / "field_metrics.png", dpi=160, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# bar_chart_race support
# --------------------------------------------------------------------------- #

def adjusted_for_race(emotions: np.ndarray) -> np.ndarray:
    adjusted = emotions.copy()
    for idx, axis in enumerate(AXES):
        low, high = AXIS_BOUNDS[axis]
        if axis in NEGATIVE_AXES:
            adjusted[:, idx] = (adjusted[:, idx] - low) / (high - low)
        else:
            adjusted[:, idx] = (adjusted[:, idx] - low) / (high - low)
    return np.clip(adjusted, 0.0, 1.0)


def generate_bar_chart_race(emotions: np.ndarray, timestamps: Sequence[str], output_path: Path) -> None:
    if bcr is None:
        print("bar_chart_race is not installed; skipping race animation. Run `pip install bar_chart_race` to enable.")
        return

    values = adjusted_for_race(emotions) * 100.0  # percentage scale for readability
    df = pd.DataFrame(
        values,
        columns=AXES,
        index=pd.to_datetime(timestamps),
    )
    bcr.bar_chart_race(
        df=df,
        filename=str(output_path),
        title="Emotion Component Race",
        n_bars=len(AXES),
        steps_per_period=10,
        period_length=500,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="data/state")
    parser.add_argument("--in", dest="inp", type=str, required=True)
    parser.add_argument("--out", type=str, default="figures")
    parser.add_argument("--user", type=str, default="user_000")
    parser.add_argument(
        "--locale",
        type=str,
        default=os.getenv("CULTURE_LOCALE", "default"),
        help="Locale key for cultural projection (default: env CULTURE_LOCALE or 'default')",
    )
    parser.add_argument(
        "--slice-axes",
        type=str,
        default="affective,meta",
        help="Two comma-separated axes to use for terrain slice (default: affective,meta)",
    )
    parser.add_argument(
        "--projection-axes",
        type=str,
        default="affective,agency,recursion",
        help="Three comma-separated axes for 3D trajectory projection (default: affective,agency,recursion)",
    )
    parser.add_argument("--bar-race", type=str, help="Output path for bar chart race animation (mp4 / gif)")
    args = parser.parse_args()

    slice_axes = tuple(axis.strip() for axis in args.slice_axes.split(",") if axis.strip())
    if len(slice_axes) != 2:
        raise ValueError("slice-axes must specify exactly two axis names.")
    projection_axes = tuple(axis.strip() for axis in args.projection_axes.split(",") if axis.strip())
    if len(projection_axes) != 3:
        raise ValueError("projection-axes must specify exactly three axis names.")
    locale = args.locale

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    system = EmotionalMemorySystem(args.state)
    logs = load_logs(Path(args.inp), args.user)
    if len(logs) < 2:
        print("log entries are insufficient for visualization")
        return

    emotion_matrix = np.array([record["emotion_vec"] for record in logs], dtype=float)
    timestamps = [record["timestamp"] for record in logs]

    perplexity = min(30, max(5, len(emotion_matrix) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
        random_state=42,
    ).fit_transform(emotion_matrix)

    plot_tsne_trajectory(embedding, out_dir)
    averaged = plot_terrain_slice(system, slice_axes, out_dir)
    plot_terrain_surface(averaged, slice_axes, out_dir)
    plot_projection_3d(emotion_matrix, projection_axes, out_dir)
    plot_top_component_timeline(emotion_matrix, timestamps, out_dir)
    plot_axis_trends(emotion_matrix, timestamps, out_dir)
    plot_projected_path(emotion_matrix, timestamps, locale, out_dir)
    plot_field_metrics(system.field_metrics_state(), out_dir)
    field_snapshot = system.field.snapshot()
    plot_field_energy(field_snapshot, out_dir)
    plot_field_flow(field_snapshot, out_dir)
    plot_field_phase(field_snapshot, out_dir)
    with (out_dir / "membrane_state.json").open("w", encoding="utf-8") as mfile:
        json.dump(system.membrane_state(), mfile, ensure_ascii=False, indent=2)
    with (out_dir / "narrative_state.json").open("w", encoding="utf-8") as nfile:
        json.dump(system.narrative_state(), nfile, ensure_ascii=False, indent=2)
    with (out_dir / "story_graph.json").open("w", encoding="utf-8") as sgfile:
        json.dump(system.story_graph_state(), sgfile, ensure_ascii=False, indent=2)
    with (out_dir / "memory_palace.json").open("w", encoding="utf-8") as mpfile:
        json.dump(system.memory_palace_state(), mpfile, ensure_ascii=False, indent=2)
    with (out_dir / "consent_state.json").open("w", encoding="utf-8") as cfile:
        json.dump(system.ethics_state(), cfile, ensure_ascii=False, indent=2)
    with (out_dir / "bias_report.json").open("w", encoding="utf-8") as bref:
        json.dump(system.bias_report_state(), bref, ensure_ascii=False, indent=2)

    if args.bar_race:
        race_path = Path(args.bar_race)
        race_path.parent.mkdir(parents=True, exist_ok=True)
        generate_bar_chart_race(emotion_matrix, timestamps, race_path)


if __name__ == "__main__":
    main()
