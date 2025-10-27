#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render a coarse memory graph from telemetry logs."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

try:  # pragma: no cover - plotting optional
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

SECTOR_COLOR = {
    "emotional": "C3",
    "episodic": "C0",
    "semantic": "C2",
    "procedural": "C1",
    "reflective": "C4",
    "unknown": "C7",
}

__all__ = ["render_memory_graph"]


def _load_events(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _resolve_sector(data: Dict[str, any]) -> Tuple[str, str]:
    sector = str(
        data.get("sector")
        or data.get("stage")
        or data.get("field_source")
        or "unknown"
    )
    node_id = str(data.get("node_id") or sector)
    return node_id, sector


def _build_adjacency(events: Iterable[Dict]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str], Dict[str, int]]:
    adjacency: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    node_sector: Dict[str, str] = {}
    counts: Dict[str, int] = collections.Counter()
    prev_node: str | None = None
    for row in events:
        if row.get("event") != "field.metrics":
            continue
        data = row.get("data") or {}
        node, sector = _resolve_sector(data)
        node_sector[node] = sector
        counts[node] += 1
        if prev_node is not None:
            adjacency[prev_node][node] += 1
        prev_node = node
    return adjacency, node_sector, counts


def render_memory_graph(log_path: Path, output_path: Path) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required for memory graph plotting")
    events = _load_events(log_path)
    adjacency, node_sector, node_counts = _build_adjacency(events)
    nodes = list(node_sector.keys())
    if not nodes:
        raise RuntimeError("no field.metrics events found for memory graph")

    n = len(nodes)
    angles = [2 * math.pi * i / n for i in range(n)]
    coords = {
        node: (math.cos(angle), math.sin(angle))
        for node, angle in zip(nodes, angles)
    }

    plt.figure(figsize=(6, 6))
    max_count = max(node_counts.values()) if node_counts else 1
    for node, (x, y) in coords.items():
        sector = node_sector.get(node, "unknown")
        color = SECTOR_COLOR.get(sector, SECTOR_COLOR["unknown"])
        size = 200 + 400 * (node_counts.get(node, 1) / max_count)
        plt.scatter([x], [y], s=size, color=color, alpha=0.9)
        plt.text(x, y, node, ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    max_edge = max((w for targets in adjacency.values() for w in targets.values()), default=1)
    for src, targets in adjacency.items():
        x0, y0 = coords.get(src, (0.0, 0.0))
        for dst, weight in targets.items():
            x1, y1 = coords.get(dst, (0.0, 0.0))
            lw = 0.5 + 3.5 * (weight / max_edge)
            plt.plot([x0, x1], [y0, y1], linewidth=lw, alpha=0.5, color="#444444")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=color, markersize=8)
        for name, color in SECTOR_COLOR.items()
    ]
    plt.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=False, title="sector")
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", "--in", dest="log", type=Path, required=True)
    parser.add_argument("--out", "--out-dir", dest="out", type=Path, default=Path("reports/plots/memory_graph.png"))
    args = parser.parse_args()
    out = render_memory_graph(args.log, args.out)
    print(f"Saved memory graph: {out}")


if __name__ == "__main__":
    main()
