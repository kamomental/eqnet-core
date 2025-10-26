# -*- coding: utf-8 -*-
"""Safety invariants and consent checks."""

from __future__ import annotations

from typing import Dict


def preflight_guard(action: str, ctx: Dict) -> Dict[str, object]:
    """Return guard decision for potentially risky actions."""
    risky_actions = {"execute_device", "schedule_payment", "apply_system_change"}
    if action not in risky_actions:
        return {"ok": True}
    description = ctx.get("action_description") or "impactful operation"
    return {
        "ok": False,
        "reason": "consent_required",
        "explain": f"{description} requires explicit consent.",
    }


__all__ = ["preflight_guard"]
