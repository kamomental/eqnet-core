# -*- coding: utf-8 -*-
"""Score reweighting helpers for retrieval hygiene."""

from __future__ import annotations


def reweight(score: float, *, junk_prob: float, trust: float, weight_min: float = 0.15) -> float:
    """Blend relevance score with hygiene metrics."""
    junk_prob = max(0.0, min(1.0, float(junk_prob)))
    trust = max(0.0, min(1.0, float(trust)))
    weight = float(score) * (0.4 + 0.6 * trust) * (1.0 - 0.7 * junk_prob)
    return max(weight_min, weight)


__all__ = ["reweight"]

