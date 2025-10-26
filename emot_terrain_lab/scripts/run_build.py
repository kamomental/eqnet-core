"""EQNet brick build launcher (metadata + validation)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

BUILD_DIR = Path("config/builds")


@dataclass
class BuildConfig:
    name: str
    description: str = ""
    wires: List[Dict[str, str]] = field(default_factory=list)
    guards: Dict[str, float] = field(default_factory=dict)
    policy: List[Dict[str, str]] = field(default_factory=list)
    nodes: List[str] = field(default_factory=list)
    eval: Dict[str, Dict[str, str]] = field(default_factory=dict)


def discover_builds() -> List[str]:
    if not BUILD_DIR.exists():
        return []
    return sorted(p.stem for p in BUILD_DIR.glob("*.yaml"))


def load_build(name: str) -> BuildConfig:
    path = Path(name)
    if not path.exists():
        path = BUILD_DIR / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Build YAML must be a mapping.")
    return BuildConfig(
        name=str(data.get("build", path.stem)),
        description=str(data.get("description", "")),
        wires=list(data.get("wires", [])),
        guards=dict(data.get("guards", {})),
        policy=list(data.get("policy", [])),
        nodes=list(data.get("nodes", [])),
        eval=dict(data.get("eval", {})),
    )


def validate_build(build: BuildConfig) -> List[str]:
    issues: List[str] = []
    for wire in build.wires:
        if "from" not in wire or "to" not in wire:
            issues.append(f"wire missing from/to: {wire}")
    for guard_key in build.guards:
        if not guard_key.lower().endswith("_max"):
            issues.append(f"Guard '{guard_key}' should end with '_max'")
    return issues


def summarize_build(build: BuildConfig, *, output_json: Optional[Path] = None) -> None:
    summary = {
        "name": build.name,
        "description": build.description,
        "nodes": build.nodes or ["default"],
        "wires": build.wires,
        "guards": build.guards,
        "policy": build.policy,
        "eval": build.eval,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if output_json:
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet brick build launcher")
    parser.add_argument("--list", action="store_true", help="List available builds.")
    parser.add_argument("--build", type=str, help="Name or path of build YAML.")
    parser.add_argument("--out", type=Path, help="Optional JSON summary output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        builds = discover_builds()
        if not builds:
            print("No builds found.")
            return 0
        for name in builds:
            print(name)
        return 0

    if not args.build:
        print("Specify --build NAME or --list.")
        return 1

    build = load_build(args.build)
    issues = validate_build(build)
    if issues:
        print("Build validation issues:")
        for issue in issues:
            print(f" - {issue}")
        return 1

    summarize_build(build, output_json=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
