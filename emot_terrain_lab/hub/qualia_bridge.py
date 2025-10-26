# -*- coding: utf-8 -*-
"""Language-to-sense reconstruction helpers."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, Tuple

from ..sense.envelope import SenseEnvelope, clamp_features


_LEXICON: Tuple[Tuple[re.Pattern[str], Dict[str, float]], ...] = (
    (re.compile(r"ぷるん|ぷるぷる|jiggle|wobbl", re.IGNORECASE), {"jiggle_hz": 0.85}),
    (re.compile(r"つや|ぴか|gloss|shiny", re.IGNORECASE), {"gloss": 0.8}),
    (re.compile(r"固め|かため|firm|dense", re.IGNORECASE), {"firmness": 0.8}),
    (re.compile(r"やわらか|soft|とろけ", re.IGNORECASE), {"firmness": 0.3}),
    (re.compile(r"カラメル|caramel", re.IGNORECASE), {"aftertaste": 0.7}),
    (re.compile(r"卵|egg", re.IGNORECASE), {"aftertaste": 0.4}),
    (re.compile(r"冷|chilled|cold", re.IGNORECASE), {"temp_curve": 0.2}),
    (re.compile(r"温|warm|hot", re.IGNORECASE), {"temp_curve": 0.7}),
)


def measure_language_loss(
    env: SenseEnvelope,
    utterance: str,
    metaphors_cfg: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """
    Reconstruct sense features from text and compute loss.

    Returns a payload with reconstructed feature map, coverage
    diagnostics, and a scalar loss in [0, 1].
    """

    utterance = utterance or ""
    reconstructed = infer_features_from_text(utterance)
    covered = sorted(reconstructed.keys())
    missing = [feat for feat in env.features.keys() if feat not in reconstructed]

    overlap = set(env.features) & set(reconstructed)
    if overlap:
        mse = sum(
            (float(env.features[k]) - float(reconstructed[k])) ** 2 for k in overlap
        ) / len(overlap)
    else:
        mse = 0.0
    loss = max(0.0, min(1.0, math.sqrt(mse)))

    return {
        "loss": float(loss),
        "reconstructed": reconstructed,
        "covered": covered,
        "missing": missing,
    }


def infer_features_from_text(text: str) -> Dict[str, float]:
    if not text:
        return {}
    accum: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for pattern, contrib in _LEXICON:
        if not pattern.search(text):
            continue
        for name, value in contrib.items():
            accum[name] += float(value)
            counts[name] += 1
    features = {
        name: (accum[name] / counts[name]) if counts[name] else 0.0 for name in accum
    }
    return clamp_features(features)


__all__ = ["measure_language_loss", "infer_features_from_text"]
