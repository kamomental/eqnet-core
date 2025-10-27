"""Simple valence/arousal scatter plot with summary stats for nightly reports."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _iter_affective_points(path: Path) -> Iterable[Tuple[float, float]]:
    if not path.exists():
        return []
    points = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                v = float(row.get("valence"))
                a = float(row.get("arousal"))
            except (TypeError, ValueError):
                continue
            points.append((v, a))
    except Exception:
        return []
    return points


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    series = list(values)
    if not series:
        return {"mean": 0.0, "std": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "count": 0}
    sorted_series = sorted(series)
    n = len(sorted_series)

    def quantile(p: float) -> float:
        if n == 1:
            return float(sorted_series[0])
        idx = p * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return float(sorted_series[lower] * (1 - weight) + sorted_series[upper] * weight)

    return {
        "mean": float(st.mean(sorted_series)),
        "std": float(st.pstdev(sorted_series)) if n > 1 else 0.0,
        "q25": quantile(0.25),
        "q50": quantile(0.50),
        "q75": quantile(0.75),
        "count": float(n),
    }


def render_affective_map(
    jsonl_path: Path | str,
    out_path: Path | str,
    *,
    json_out_path: Path | str | None = None,
) -> None:
    src = Path(jsonl_path)
    dst = Path(out_path)
    points = list(_iter_affective_points(src))
    if points:
        xs, ys = zip(*points)
        dst.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(5, 4))
        plt.scatter(xs, ys, s=8, alpha=0.45, color="tab:purple")
        plt.xlabel("valence")
        plt.ylabel("arousal")
        plt.xlim(-1.05, 1.05)
        plt.ylim(bottom=0.0)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(dst, dpi=160)
        plt.close()
    stats_payload = {
        "valence": _summary_stats((v for v, _ in points)),
        "arousal": _summary_stats((a for _, a in points)),
    }
    if json_out_path:
        stats_path = Path(json_out_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["render_affective_map"]
