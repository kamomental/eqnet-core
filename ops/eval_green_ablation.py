# -*- coding: utf-8 -*-
"""
Compare Green-function modulation ON vs OFF with synthetic trials.

This scaffolding mirrors the mood ablation harness: it runs N trials with
Green ON and OFF, sampling basic metrics plus explain_gain.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random


@dataclass
class TrialResult:
    success: bool
    misfire: bool
    latency: float
    explain_gain: float


def run_trial(rng: Random, green_on: bool) -> TrialResult:
    base_success = 0.65 if green_on else 0.63
    success = rng.random() < base_success
    misfire = rng.random() < (0.11 if green_on else 0.12)
    latency = 1.4 + 0.5 * rng.random() + (0.05 if not green_on else 0.0)
    explain_gain = 0.35 + (0.1 * rng.random()) + (0.05 if green_on else 0.0)
    return TrialResult(success, misfire, latency, explain_gain)


def aggregate(results: list[TrialResult]) -> dict[str, float]:
    if not results:
        return {"success": 0.0, "misfire": 0.0, "p95": 0.0, "explain_gain": 0.0}
    success = sum(r.success for r in results) / len(results)
    misfire = sum(r.misfire for r in results) / len(results)
    latencies = sorted(r.latency for r in results)
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    explain_gain = sum(r.explain_gain for r in results) / len(results)
    return {
        "success": success,
        "misfire": misfire,
        "p95": p95,
        "explain_gain": explain_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Green-function ON/OFF ablation")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="reports/green_ablation.json")
    args = parser.parse_args()

    rng_on = Random(args.seed)
    rng_off = Random(args.seed + 1)
    on_results = [run_trial(rng_on, True) for _ in range(args.trials)]
    off_results = [run_trial(rng_off, False) for _ in range(args.trials)]

    report = {
        "n": args.trials,
        "on": aggregate(on_results),
        "off": aggregate(off_results),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
