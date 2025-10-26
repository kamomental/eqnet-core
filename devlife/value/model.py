# -*- coding: utf-8 -*-
"""Value model utilities (weights + summary computation)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "value.yaml"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _sigmoid_centered(x: float, scale: float = 1.0) -> float:
    """Map real value to (0,1) with tunable slope."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x / scale, -20.0, 20.0))))


@lru_cache(maxsize=1)
def load_value_config() -> Dict[str, Dict[str, float]]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data = {}
    defaults = {
        "value_weights": {
            "extrinsic": 0.4,
            "novelty": 0.08,
            "social": 0.18,
            "coherence": 0.12,
            "homeostasis": 0.15,
            "qualia_fit": 0.07,
            "norm_penalty": 0.55,
        },
        "critical": {"lambda1_target": 0.05, "r_max": 0.8},
        "visualization": {"radar_hz": 1.5, "qualia_refresh_seconds": 15},
        "safety": {"hard_constraints": []},
    }
    for section, defaults_section in defaults.items():
        data.setdefault(section, {})
        for key, value in defaults_section.items():
            data[section].setdefault(key, value)
    return data


def compute_value_summary(
    *,
    extrinsic_signal: float = 0.0,
    novelty_signal: float = 0.0,
    social_alignment: float = 0.5,
    coherence_score: float = 0.5,
    homeostasis_error: float = 0.0,
    qualia_consistency: float = 0.5,
    norm_penalty: float = 0.0,
    metadata: Optional[Mapping[str, float]] = None,
    weights_override: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """Return weighted value summary given feature signals."""
    config = load_value_config()
    weights = dict(config["value_weights"])
    if weights_override:
        for key, value in weights_override.items():
            if key in weights:
                weights[key] = float(value)

    extrinsic = _sigmoid_centered(extrinsic_signal, scale=0.12)
    novelty = _sigmoid_centered(novelty_signal, scale=0.5)
    social = _clamp(social_alignment)
    coherence = _clamp(coherence_score)
    homeostasis = _clamp(1.0 - _sigmoid_centered(abs(homeostasis_error), scale=0.3))
    qualia = _clamp(qualia_consistency)
    norm_pen = _clamp(norm_penalty)

    components = {
        "extrinsic": extrinsic,
        "novelty": novelty,
        "social": social,
        "coherence": coherence,
        "homeostasis": homeostasis,
        "qualia_fit": qualia,
        "norm_penalty": norm_pen,
    }

    total = (
        weights["extrinsic"] * extrinsic
        + weights["novelty"] * novelty
        + weights["social"] * social
        + weights["coherence"] * coherence
        + weights["homeostasis"] * homeostasis
        + weights["qualia_fit"] * qualia
        - weights["norm_penalty"] * norm_pen
    )
    summary = {
        "total": float(max(-1.0, min(1.0, total))),
        "components": components,
        "weights": weights,
    }
    if metadata:
        summary["meta"] = dict(metadata)
    return summary


__all__ = ["load_value_config", "compute_value_summary"]
