# -*- coding: utf-8 -*-
"""Eligibility-trace credit assignment helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable


def apply_credit_updates(
    rollout: Iterable[Dict[str, Any]],
    *,
    read_only_policy: bool,
    d_tau_step: float,
    eta: float = 0.8,
    lr: float = 0.05,
) -> Dict[str, Any]:
    """Return aggregate credit delta respecting read-only policies."""
    if read_only_policy:
        return {"applied": False, "delta": 0.0}

    eligibility = 0.0
    delta = 0.0

    for step in rollout:
        pred = float(step.get("predU", 0.0))
        target = float(step.get("targetU", 0.0))
        eligibility = math.exp(-eta * d_tau_step) * eligibility + 1.0
        delta += eligibility * (target - pred)

    return {"applied": True, "delta": delta * lr}


__all__ = ["apply_credit_updates"]
