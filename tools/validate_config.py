# -*- coding: utf-8 -*-
"""Lightweight config validators for fast-path settings."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emot_terrain_lab.ops.task_profiles import TASK_PROFILES

FASTPATH_MODES = {"record_only", "soft_hint", "ab_test"}


def validate_fastpath(cfg: Mapping[str, Any]) -> bool:
    fast_cfg = cfg.get("fastpath") or {}
    enable = fast_cfg.get("enable", True)
    if not isinstance(enable, bool):
        raise ValueError("fastpath.enable must be bool")

    mode = str(fast_cfg.get("enforce_actions", "record_only")).lower()
    if mode not in FASTPATH_MODES:
        raise ValueError(f"fastpath.enforce_actions must be one of {sorted(FASTPATH_MODES)}")

    profiles = fast_cfg.get("profiles", ["cleanup"])
    if not isinstance(profiles, Iterable):
        raise ValueError("fastpath.profiles must be a list/sequence")
    for name in profiles:
        if name not in TASK_PROFILES:
            raise ValueError(f"fastpath profile '{name}' is not registered")

    fraction = float(fast_cfg.get("ab_test_fraction", 0.15))
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fastpath.ab_test_fraction must be between 0 and 1")

    scale = float(fast_cfg.get("ab_test_ttl_scale", 1.15))
    if scale < 1.0:
        raise ValueError("fastpath.ab_test_ttl_scale must be >= 1.0")

    fail_safe = fast_cfg.get("fail_safe") or {}
    if fail_safe:
        threshold = float(fail_safe.get("override_rate_threshold", 0.2))
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("fastpath.fail_safe.override_rate_threshold must be between 0 and 1")
        fallback = str(fail_safe.get("fallback_mode", "soft_hint")).lower()
        if fallback not in FASTPATH_MODES:
            raise ValueError("fastpath.fail_safe.fallback_mode must be a known fastpath mode")
        lookback = int(fail_safe.get("lookback_days", 3))
        if lookback < 1:
            raise ValueError("fastpath.fail_safe.lookback_days must be >= 1")

    return True


def _load_cfg(path: Path) -> Dict[str, Any]:
    data = path.read_text(encoding="utf-8")
    return yaml.safe_load(data) or {}


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python tools/validate_config.py <config.yaml>")
        return 1
    target = Path(argv[0])
    cfg = _load_cfg(target)
    validate_fastpath(cfg)
    print(f"[ok] fastpath section in {target} is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
