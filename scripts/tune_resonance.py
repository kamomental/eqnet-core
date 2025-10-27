#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore resonance gain (k_res) and record outcomes.

Example:
    python scripts/tune_resonance.py \
        --mode grid \
        --run-cmd "python scripts/run_quick_loop.py --steps {steps} --seed {seed}" \
        --nightly-cmd "python -m emot_terrain_lab.ops.nightly --telemetry_log telemetry/ignition-*.jsonl" \
        --logs telemetry/ignition-*.jsonl telemetry/resonance-*.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _run_command(template: str, **fmt) -> None:
    command = template.format(**fmt)
    subprocess.run(shlex.split(command), check=True)


def _update_resonance_cfg(cfg_path: Path, k_res: float, logs: Sequence[str] | None) -> None:
    cfg = _load_yaml(cfg_path)
    resonance = cfg.setdefault("resonance", {})
    resonance["k_res"] = float(k_res)
    if logs is not None:
        resonance["logs"] = list(logs)
    _dump_yaml(cfg_path, cfg)


def _restore_cfg(cfg_path: Path, snapshot: dict | None) -> None:
    if snapshot is None:
        return
    _dump_yaml(cfg_path, snapshot)


def _read_objective(nightly_path: Path) -> dict:
    if not nightly_path.exists():
        return {}
    report = json.loads(nightly_path.read_text(encoding="utf-8"))
    resonance = report.get("resonance") or {}
    summary = resonance.get("summary")
    if summary:
        base = summary.copy()
    else:
        pairs = resonance.get("pairs") or []
        base = pairs[0] if pairs else {}
    base["objective"] = base.get("objective", resonance.get("objective"))
    base["corr"] = base.get("corr", resonance.get("corr"))
    base["lag"] = base.get("lag", resonance.get("lag"))
    base["energy"] = base.get("energy", resonance.get("energy"))
    return base


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid/Random search for resonance gain (k_res).")
    parser.add_argument("--mode", choices=["grid", "random"], default="grid", help="Exploration mode.")
    parser.add_argument("--grid", default="0.00,0.02,0.04,0.06,0.08,0.10", help="Comma-separated k_res for grid mode.")
    parser.add_argument("--trials", type=int, default=6, help="Number of samples for random mode.")
    parser.add_argument("--steps", type=int, default=300, help="Steps for the run command.")
    parser.add_argument("--seed", type=int, default=12345, help="Seed offset.")
    parser.add_argument(
        "--run-cmd",
        default="python scripts/run_quick_loop.py --steps {steps} --seed {seed}",
        help="Command to generate telemetry (format placeholders: {steps}, {seed}, {k_res}).",
    )
    parser.add_argument(
        "--nightly-cmd",
        default="python -m emot_terrain_lab.ops.nightly --telemetry_log telemetry/ignition-*.jsonl",
        help="Command to produce nightly summary (placeholders: {k_res}).",
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help="Resonance log patterns to write into runtime.yaml (leave unset to keep existing).",
    )
    parser.add_argument("--config", default="config/runtime.yaml", help="Runtime config path.")
    parser.add_argument("--nightly", default="reports/nightly.json", help="Nightly JSON path.")
    parser.add_argument("--persist", action="store_true", help="Persist best k_res in config (default restores snapshot).")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_path = Path(args.config)
    nightly_path = Path(args.nightly)
    if not cfg_path.exists():
        print(f"[tune] config not found: {cfg_path}")
        return 1

    original_cfg = _load_yaml(cfg_path)
    logs = args.logs

    if args.mode == "grid":
        k_values = [float(x) for x in args.grid.split(",") if x]
    else:
        random.seed(args.seed)
        k_values = [random.uniform(0.0, 0.12) for _ in range(args.trials)]

    results: List[dict] = []
    try:
        for idx, k_res in enumerate(k_values):
            seed = args.seed + idx
            print(f"[tune] trial {idx+1}/{len(k_values)}  k_res={k_res:.4f}")
            _update_resonance_cfg(cfg_path, k_res, logs)
            _run_command(args.run_cmd, steps=args.steps, seed=seed, k_res=k_res)
            _run_command(args.nightly_cmd, k_res=k_res)
            metrics = _read_objective(nightly_path)
            metrics["k_res"] = k_res
            print(f"[tune] result {metrics}")
            results.append(metrics)
        if not results:
            print("[tune] no results produced.")
            return 0
        best = max(results, key=lambda r: (r.get("objective") if r.get("objective") is not None else float("-inf")))
        print(f"[tune] best => {best}")
        if nightly_path.exists():
            report = json.loads(nightly_path.read_text(encoding="utf-8"))
            suggestion = report.setdefault("tuning_suggestion", {})
            suggestion["k_res"] = best.get("k_res")
            nightly_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[tune] tuning_suggestion.k_res -> {best.get('k_res')}")
        if args.persist:
            _update_resonance_cfg(cfg_path, best.get("k_res", 0.0), logs)
        else:
            _restore_cfg(cfg_path, original_cfg)
    finally:
        if not args.persist:
            _restore_cfg(cfg_path, original_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
