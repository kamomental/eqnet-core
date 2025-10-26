#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to inspect and dry-run LoRA continual-learning pipelines.

Usage:
    python -m emot_terrain_lab.scripts.run_pipeline --config pipelines/lora_update.yaml

The script parses the YAML, prints a compact summary, and (optionally) attempts
to resolve brick modules to ensure dotted paths are valid. Actual training /
deployment is left to the brick implementations.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or dry-run a continual-learning pipeline YAML."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline YAML (e.g., pipelines/lora_update.yaml).",
    )
    parser.add_argument(
        "--resolve-modules",
        action="store_true",
        help="Attempt to import brick module paths for validation.",
    )
    parser.add_argument(
        "--print-flow",
        action="store_true",
        help="Render the pipeline flow edges after parsing.",
    )
    return parser.parse_args(argv)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Pipeline YAML must contain a top-level mapping.")
    return data


def _split_target(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Brick module spec must contain ':' separator: {spec}")
    module_name, attr = spec.split(":", 1)
    if not module_name or not attr:
        raise ValueError(f"Invalid brick module spec: {spec}")
    return module_name, attr


def resolve_brick(spec: str) -> Any:
    module_name, attr = _split_target(spec)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def summarize(data: Dict[str, Any]) -> None:
    pipeline = data.get("pipeline", {})
    name = pipeline.get("name", "<unnamed>")
    description = pipeline.get("description", "").strip().splitlines()[0:2]
    run_at = pipeline.get("run_at")
    timezone = pipeline.get("timezone", "UTC")

    print(f"Pipeline: {name}")
    if description:
        print("Description:")
        for line in description:
            print(f"  {line}")
    if run_at:
        print(f"Schedule : {run_at} ({timezone})")

    defaults = data.get("defaults", {})
    if defaults:
        print("Defaults:")
        for key, value in defaults.items():
            print(f"  {key}: {value}")

    bricks = data.get("bricks", {})
    print(f"Bricks ({len(bricks)}):")
    for name, cfg in bricks.items():
        target = cfg.get("module", "<missing>")
        print(f"  - {name}: {target}")

    deployer = bricks.get("deployer", {})
    stages = deployer.get("params", {}).get("stages")
    if stages:
        print("Canary stages:")
        for stage in stages:
            pct = stage.get("percentage")
            duration = stage.get("duration_hours", "n/a")
            pct_str = f"{pct:.2f}" if isinstance(pct, (int, float)) else str(pct)
            print(f"  * {pct_str} for {duration} h")


def print_flow(data: Dict[str, Any]) -> None:
    flow = data.get("flow", [])
    if not flow:
        print("Flow: (none defined)")
        return
    print("Flow edges:")
    for edge in flow:
        print(f"  {edge.get('from')} -> {edge.get('to')}")


def validate_modules(bricks: Dict[str, Any]) -> None:
    print("\nResolving brick modules:")
    for name, cfg in bricks.items():
        target = cfg.get("module")
        if not target:
            print(f"[warn] brick '{name}' missing 'module' entry")
            continue
        try:
            obj = resolve_brick(target)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {name}: {target} ({exc})")
        else:
            print(f"[ ok ] {name}: {target} → {obj}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config_path = Path(args.config)
    data = load_yaml(config_path)

    summarize(data)
    if args.print_flow:
        print_flow(data)
    if args.resolve_modules:
        bricks = data.get("bricks", {})
        validate_modules(bricks)
    else:
        print("\n(use --resolve-modules to attempt dynamic imports)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
