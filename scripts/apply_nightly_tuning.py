#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Nightly tuning suggestions to runtime.yaml safely.

Usage:
    python scripts/apply_nightly_tuning.py --nightly reports/nightly.json --config config/runtime.yaml
    python scripts/apply_nightly_tuning.py --nightly reports/nightly.json --config config/runtime.yaml --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Nightly tuning suggestion to runtime.yaml.")
    parser.add_argument("--nightly", default="reports/nightly.json", help="Nightly JSON summary path.")
    parser.add_argument("--config", default="config/runtime.yaml", help="runtime.yaml path.")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    args = parser.parse_args()

    nightly_path = Path(args.nightly)
    config_path = Path(args.config)

    if not nightly_path.exists():
        print(f"[tuning] nightly summary not found: {nightly_path}")
        return 1
    if not config_path.exists():
        print(f"[tuning] runtime config not found: {config_path}")
        return 1

    report = load_json(nightly_path)
    suggestion = report.get("tuning_suggestion") or {}
    has_theta = {"theta_on", "theta_off"} <= suggestion.keys()
    has_k_res = "k_res" in suggestion
    policy_feedback = report.get("policy_feedback") or {}
    has_policy = bool(
        isinstance(policy_feedback, dict)
        and policy_feedback.get("enabled")
        and policy_feedback.get("politeness_after") is not None
        and policy_feedback.get("politeness_before") is not None
    )
    if not (has_theta or has_k_res or has_policy):
        print("[tuning] no suggestion found. nothing to do.")
        return 0

    cfg = load_yaml(config_path) or {}
    ignition_before = {}
    ignition_after = {}
    if has_theta:
        ignition = cfg.setdefault("ignition", {})
        ignition_before = {
            "theta_on": ignition.get("theta_on", 0.62),
            "theta_off": ignition.get("theta_off", 0.48),
        }
        ignition_after = {
            "theta_on": float(suggestion["theta_on"]),
            "theta_off": float(suggestion["theta_off"]),
        }
        if not (0.0 <= ignition_after["theta_on"] <= 1.0 and 0.0 <= ignition_after["theta_off"] <= 1.0):
            print("[tuning] suggested values outside [0,1]. abort.")
            return 2
        if not ignition_after["theta_on"] > ignition_after["theta_off"]:
            print("[tuning] theta_on must be greater than theta_off. abort.")
            return 3
        if abs(ignition_after["theta_on"] - ignition_before["theta_on"]) > 0.2:
            print("[tuning] change too large. abort.")
            return 4

    resonance_before = {}
    resonance_after = {}
    if has_k_res:
        resonance_cfg = cfg.setdefault("resonance", {})
        resonance_before["k_res"] = resonance_cfg.get("k_res")
        try:
            resonance_after["k_res"] = float(suggestion["k_res"])
        except (TypeError, ValueError):
            print("[tuning] invalid k_res suggestion.")
            return 5
        if resonance_after["k_res"] < 0.0:
            print("[tuning] k_res must be non-negative. abort.")
            return 6

    culture_before = {}
    culture_after = {}
    if has_policy:
        culture_cfg = cfg.setdefault("culture", {})
        culture_before["politeness"] = culture_cfg.get("politeness")
        try:
            target_value = float(policy_feedback["politeness_after"])
        except (TypeError, ValueError):
            print("[tuning] invalid policy feedback value.")
            return 7
        if not (0.0 <= target_value <= 1.0):
            print("[tuning] policy feedback politeness outside [0,1]. abort.")
            return 8
        culture_after["politeness"] = target_value

    if has_theta:
        print(f"[tuning] ignition current: {ignition_before}  suggested: {ignition_after}")
    if has_k_res:
        print(f"[tuning] resonance current: {resonance_before}  suggested: {resonance_after}")
    if has_policy:
        reason = policy_feedback.get("reason", "n/a")
        corr_val = policy_feedback.get("corr")
        print(
            "[tuning] culture politeness current: "
            f"{culture_before.get('politeness')}  suggested: {culture_after['politeness']} "
            f"(reason={reason}, corr={corr_val})"
        )

    if not args.apply:
        print("[tuning] dry-run mode. use --apply to persist changes.")
        return 0

    backup = config_path.with_suffix(config_path.suffix + f".{time.strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(config_path, backup)
    if has_theta:
        cfg.setdefault("ignition", {}).update(ignition_after)
    if has_k_res:
        cfg.setdefault("resonance", {})["k_res"] = resonance_after["k_res"]
    if has_policy:
        cfg.setdefault("culture", {})["politeness"] = culture_after["politeness"]
    dump_yaml(config_path, cfg)
    print(f"[tuning] runtime config updated. backup -> {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
