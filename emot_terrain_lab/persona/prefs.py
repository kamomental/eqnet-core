# -*- coding: utf-8 -*-
"""Preference loader utilities for user customisation."""

from __future__ import annotations

import pathlib
from typing import Any, Dict

import yaml


def load_prefs(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    pref_path = pathlib.Path(path)
    if not pref_path.exists():
        return {}
    try:
        data = yaml.safe_load(pref_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data or {}


def apply_prefs_to_cfg(prefs: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cfg)
    if not prefs:
        return merged

    if "values" in prefs:
        vw = merged.setdefault("value_weights", {})
        vw.update({k: float(v) for k, v in prefs["values"].items()})

    if "style" in prefs:
        merged.setdefault("style", {}).update(prefs["style"])

    if "resonance" in prefs:
        merged.setdefault("resonance", {}).update(prefs["resonance"])

    if "safety" in prefs:
        merged.setdefault("safety", {}).update(prefs["safety"])

    return merged


__all__ = ["load_prefs", "apply_prefs_to_cfg"]

