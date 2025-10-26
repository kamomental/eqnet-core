# -*- coding: utf-8 -*-
"""Action selection heuristics bridging drives and value."""

from __future__ import annotations

from typing import Dict


def choose_action(overall_value: float, drives: Dict[str, float]) -> str:
    """Select a coarse action primitive from value estimates and drives."""
    thirst = float(drives.get("thirst", 0.0))
    rest = float(drives.get("rest", 0.0))
    social = float(drives.get("social", 0.0))

    if thirst > 0.6:
        return "suggest_hydration"
    if rest > 0.6:
        return "suggest_break"
    if social > 0.7:
        return "offer_check_in"
    if overall_value < 0.4:
        return "explain_options"
    return "normal_reply"


__all__ = ["choose_action"]
