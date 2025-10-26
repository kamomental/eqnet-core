# -*- coding: utf-8 -*-
"""Helpers for loading fast-path configuration and overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


FASTPATH_CONFIG_PATH = Path("config/fastpath.yaml")
FASTPATH_OVERRIDE_PATH = Path("config/overrides/fastpath.yaml")


def _read_fastpath_block(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if isinstance(data, Mapping) and "fastpath" in data:
        block = data.get("fastpath") or {}
        return dict(block) if isinstance(block, Mapping) else {}
    return dict(data) if isinstance(data, Mapping) else {}


def load_fastpath_defaults(path: Path = FASTPATH_CONFIG_PATH) -> Dict[str, Any]:
    """Load the baseline fast-path configuration (no overrides applied)."""

    return _read_fastpath_block(path)


def load_fastpath_overrides(path: Path = FASTPATH_OVERRIDE_PATH) -> Dict[str, Any]:
    """Load the latest nighty/ops overrides for fast-path behaviour."""

    return _read_fastpath_block(path)


def load_fastpath_cfg(*, include_overrides: bool = True) -> Dict[str, Any]:
    """Return the merged fast-path configuration."""

    cfg = load_fastpath_defaults()
    if include_overrides:
        override_cfg = load_fastpath_overrides()
        if override_cfg:
            cfg.update(override_cfg)
    return cfg


def fail_safe_settings(cfg: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Expose fail-safe parameters with sensible defaults."""

    data = dict(cfg or load_fastpath_defaults())
    fail_safe = data.get("fail_safe") or {}
    threshold = float(fail_safe.get("override_rate_threshold", 0.2))
    threshold = min(1.0, max(0.0, threshold))
    fallback_mode = str(fail_safe.get("fallback_mode", "soft_hint")).lower()
    lookback = int(fail_safe.get("lookback_days", 3))
    return {
        "override_rate_threshold": threshold,
        "fallback_mode": fallback_mode,
        "lookback_days": max(1, lookback),
    }


__all__ = [
    "FASTPATH_CONFIG_PATH",
    "FASTPATH_OVERRIDE_PATH",
    "fail_safe_settings",
    "load_fastpath_cfg",
    "load_fastpath_defaults",
    "load_fastpath_overrides",
]
