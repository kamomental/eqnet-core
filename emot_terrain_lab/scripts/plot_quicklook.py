# -*- coding: utf-8 -*-
"""
Generate simple quick-look plots for entropy/enthalpy and rest activity.

Usage:
    python scripts/plot_quicklook.py --state data/state --out figures/sample/quicklook.png
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

try:  # pragma: no cover - allow running as script
    from scripts.export_timeseries import enrich_metrics, build_rest_index, load_json
except ImportError:
    from export_timeseries import enrich_metrics, build_rest_index, load_json


def prepare_series(state_dir: Path) -> Dict[str, List]:
    metrics_path = state_dir / "field_metrics.json"
    rest_path = state_dir / "rest_state.json"
    field_metrics = load_json(metrics_path)
    rest_payload = load_json(rest_path)
    rest_index = build_rest_index(rest_payload)
    rows = enrich_metrics(field_metrics, rest_index)
    xs = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    entropy = [row["entropy"] for row in rows]
    enthalpy = [row["enthalpy_mean"] for row in rows]
    rest_flags = [row["rest_flag"] for row in rows]
    return {
        "timestamp": xs,
        "entropy": entropy,
        "enthalpy": enthalpy,
        "rest_flag": rest_flags,
    }


def plot_quicklook(series: Dict[str, List], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(series["timestamp"], series["entropy"], label="Entropy", color="tab:blue")
    ax1.plot(series["timestamp"], series["enthalpy"], label="Enthalpy", color="tab:orange")
    ax1.set_ylabel("Value")
    ax1.set_xlabel("Timestamp")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(series["timestamp"], series["rest_flag"], where="post", label="Rest active", color="tab:red")
    ax2.set_ylabel("Rest flag")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc="upper right")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved quicklook plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, required=True, help="State directory")
    parser.add_argument("--out", type=str, required=True, help="Output image path")
    args = parser.parse_args()

    state_dir = Path(args.state)
    series = prepare_series(state_dir)
    plot_quicklook(series, Path(args.out))


if __name__ == "__main__":
    main()
