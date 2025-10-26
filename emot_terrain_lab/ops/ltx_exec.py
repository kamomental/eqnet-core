"""Minimal runtime for executing Wave P3 rule snippets."""

from __future__ import annotations

import time
from typing import Dict


def exec_rules(state: Dict[str, float], now: float | None = None) -> Dict[str, float]:
    """Apply containment-style rules for network synchrony."""
    timestamp = time.time() if now is None else now
    inhibit_until = state.get("inhibit_until", 0.0)
    R = state.get("R", 0.0)
    if R > 0.78 and timestamp >= inhibit_until:
        state["warmth"] = max(0.0, state.get("warmth", 0.0) - 0.1)
        state["inhibit_until"] = timestamp + 8.0
    if state.get("bud_received_from", 0) >= 2:
        state["gain"] = state.get("gain", 1.0) * 0.7
    return state


__all__ = ["exec_rules"]
