# -*- coding: utf-8 -*-
"""Persona composer utilities for EQNet."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import yaml

BASE_DIR = Path(__file__).resolve().parent
CULTURE_DIR = BASE_DIR / "culture"
MODES_FILE = BASE_DIR / "modes.yaml"


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@lru_cache(maxsize=64)
def load_culture_pack(name: str) -> Dict[str, object]:
    path = CULTURE_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Culture pack not found: {name}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=8)
def load_modes() -> Dict[str, Dict[str, object]]:
    if not MODES_FILE.exists():
        raise FileNotFoundError("persona/modes.yaml not found.")
    with MODES_FILE.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("modes", {})


def load_mode(name: str) -> Dict[str, object]:
    modes = load_modes()
    if name not in modes:
        raise KeyError(f"Mode preset not found: {name}")
    return modes[name]


def _union_keys(*dicts: Mapping[str, float]) -> Iterable[str]:
    keys = set()
    for mapping in dicts:
        keys.update(mapping.keys())
    return keys


def compose_style(
    culture_axes: Mapping[str, float],
    mode_axes: Mapping[str, float],
    user_pref: Mapping[str, float],
    *,
    alpha: float,
    beta: float,
    safety: Optional[Mapping[str, object]] = None,
) -> Dict[str, float]:
    """Blend culture + mode + user preference into a style vector."""
    style: Dict[str, float] = {}
    alpha = float(_clamp(alpha, 0.0, 1.0))
    beta = float(_clamp(beta, 0.0, 1.0))
    safety = safety or {}
    disabled = set(safety.get("disabled_axes", []))

    for axis in _union_keys(culture_axes, mode_axes, user_pref):
        if axis in disabled:
            style[axis] = 0.0
            continue
        value = float(culture_axes.get(axis, 0.0))
        value += alpha * float(mode_axes.get(axis, 0.0))
        value += beta * float(user_pref.get(axis, 0.0))
        style[axis] = _clamp(value)
    return style


def apply_safety(style: MutableMapping[str, float], safety: Mapping[str, object]) -> None:
    """Apply safety overrides in place."""
    if safety.get("high_risk"):
        style["alpha"] = 0.0  # optional marker
        for axis in safety.get("suppress_axes", ["emoji_use", "sarcasm"]):
            style[axis] = 0.0
    for axis, limit in safety.get("max_abs", {}).items():
        if axis in style:
            style[axis] = _clamp(style[axis], -abs(limit), abs(limit))


def style_to_controls(
    style_axes: Mapping[str, float],
    base_controls: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Map style axes into concrete hub controls."""
    base = dict(base_controls or {})
    directness = base.get("directness", 0.0)
    hedging = style_axes.get("hedging", 0.0)
    directness += 0.12 * float(style_axes.get("directness", 0.0))
    directness -= 0.10 * hedging
    base["directness"] = _clamp(directness, -0.6, 0.6)

    pause_ms = float(base.get("pause_ms", 0.0))
    pause_ms += 120.0 * float(style_axes.get("rhythm_pause", 0.0))
    pause_ms = _clamp(pause_ms, 0.0, 600.0)
    base["pause_ms"] = pause_ms

    temperature = float(base.get("temp_mul", 1.0))
    temperature *= _clamp(1.0 - 0.10 * float(style_axes.get("formality", 0.0)), 0.6, 1.2)
    base["temp_mul"] = _clamp(temperature, 0.3, 1.5)

    base["emoji_bias"] = _clamp(float(style_axes.get("emoji_use", 0.0)), -1.0, 1.0)
    base["humor_bias"] = _clamp(float(style_axes.get("humor", 0.0)), -1.0, 1.0)
    base["figurative_bias"] = _clamp(float(style_axes.get("figurative", 0.0)), -1.0, 1.0)
    base["warmth_bias"] = _clamp(float(style_axes.get("warmth", 0.0)), -1.0, 1.0)
    base["tempo_bias"] = _clamp(float(style_axes.get("tempo", 0.0)), -1.0, 1.0)
    base["parasocial_bias"] = _clamp(float(style_axes.get("parasocial_responses", 0.0)), -1.0, 1.0)
    return base


@dataclass
class PersonaComposition:
    controls: Dict[str, float]
    style_axes: Dict[str, float]
    meta: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "controls": self.controls,
            "style_axes": self.style_axes,
            "meta": self.meta,
        }


def compose_controls(
    *,
    culture_name: str,
    mode_name: str,
    user_pref: Mapping[str, float],
    alpha: float,
    beta: float,
    safety: Optional[Mapping[str, object]] = None,
    base_controls: Optional[Mapping[str, float]] = None,
) -> PersonaComposition:
    """Convenience wrapper that loads packs and returns a PersonaComposition."""
    culture = load_culture_pack(culture_name)
    culture_axes = culture.get("style_axes", {})

    mode_data = load_mode(mode_name)
    mode_axes = mode_data.get("style_axes", {})
    if alpha is None:
        alpha = float(mode_data.get("alpha_default", 0.4))
    safety = safety or {}
    style = compose_style(
        culture_axes,
        mode_axes,
        user_pref,
        alpha=alpha,
        beta=beta,
        safety=safety,
    )
    apply_safety(style, safety)
    controls = style_to_controls(style, base_controls)
    meta = {
        "culture": culture.get("metadata", {}).get("name", culture_name),
        "mode": mode_name,
        "alpha": alpha,
        "beta": beta,
        "culture_metadata": culture.get("metadata", {}),
        "mode_metadata": {k: v for k, v in mode_data.items() if k != "style_axes"},
    }
    return PersonaComposition(controls=controls, style_axes=style, meta=meta)


__all__ = [
    "load_culture_pack",
    "load_mode",
    "compose_style",
    "style_to_controls",
    "compose_controls",
    "PersonaComposition",
]
