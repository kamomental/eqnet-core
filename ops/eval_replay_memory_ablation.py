# -*- coding: utf-8 -*-
"""
Replay memory ablation scaffold.

Runs synthetic trials with replay memory toggled on/off and reports simple KPIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random


def run_trials(n: int, *, seed: int = 42, memory_on: bool = True) -> list[dict[str, float | bool]]:
    rng = Random(seed)
    results: list[dict[str, float | bool]] = []
    for _ in range(n):
        success_prob = 0.60 + (0.02 if memory_on else 0.0)
        misfire_prob = 0.12 + (0.02 if not memory_on else 0.0)
        latency_base = 1.55 - (0.08 if memory_on else 0.0)
        success = rng.random() < success_prob
        misfire = rng.random() < misfire_prob
        latency = latency_base + 0.7 * rng.random()
        results.append(
            {
                "success": success,
                "misfire": misfire,
                "latency": latency,
            }
        )
    return results


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return float(ordered[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay memory ON/OFF ablation harness")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--out", type=str, default="reports/replay_memory_ablation.json")
    args = parser.parse_args()

    metrics_on = run_trials(args.trials, seed=42, memory_on=True)
    metrics_off = run_trials(args.trials, seed=43, memory_on=False)

    report = {
        "n": args.trials,
        "on": {
            "success": sum(1.0 for m in metrics_on if m["success"]) / len(metrics_on) if metrics_on else 0.0,
            "misfire": sum(1.0 for m in metrics_on if m["misfire"]) / len(metrics_on) if metrics_on else 0.0,
            "p95": p95([float(m["latency"]) for m in metrics_on]),
        },
        "off": {
            "success": sum(1.0 for m in metrics_off if m["success"]) / len(metrics_off) if metrics_off else 0.0,
            "misfire": sum(1.0 for m in metrics_off if m["misfire"]) / len(metrics_off) if metrics_off else 0.0,
            "p95": p95([float(m["latency"]) for m in metrics_off]),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"✓ saved {out_path}")


if __name__ == "__main__":
    main()
