# -*- coding: utf-8 -*-
"""Apply nightly outputs to next-session overrides."""

from __future__ import annotations

import json
import pathlib
from typing import Dict

import yaml


def latest_report(report_dir: str = "reports/nightly") -> Dict[str, object]:
    path = pathlib.Path(report_dir)
    if not path.exists():
        return {}
    candidates = sorted(path.glob("*.json"))
    if not candidates:
        return {}
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_overrides(data: Dict[str, object], path: str = "config/overrides/autopilot.yaml") -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_bootstrap(report_dir: str = "reports/nightly") -> Dict[str, object]:
    report = latest_report(report_dir)
    overrides: Dict[str, object] = {}

    drift = report.get("drift", {}) if isinstance(report, dict) else {}
    action = drift.get("decision")
    if action == "switch_to_clean_CPT":
        overrides.setdefault("autopilot", {})["heartiness_start"] = 0.3
    elif action:
        overrides.setdefault("autopilot", {})["heartiness_start"] = 0.4

    if overrides:
        write_overrides(overrides)
    return overrides


__all__ = ["run_bootstrap", "latest_report", "write_overrides"]

