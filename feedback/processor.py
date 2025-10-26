# -*- coding: utf-8 -*-
"""Feedback application utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional


def apply_feedback(
    event: Dict[str, Any],
    *,
    persona_manager: Any,
    safety_orchestrator: Any,
    safety_ctx: Optional[Dict[str, Any]],
    default_mode: Optional[str],
) -> Dict[str, Any]:
    feedback_type = str(event.get("type", "")).lower()
    d_tau = float(event.get("d_tau", 0.1))
    mode = event.get("mode") or default_mode
    applied = False
    targets = []

    if feedback_type in {"thumbs_up", "thumbs_down"} and persona_manager is not None and mode:
        try:
            persona_manager.update(mode, success=(feedback_type == "thumbs_up"), d_tau=d_tau)
            applied = True
            targets.append("persona")
        except Exception:
            pass

    if feedback_type == "thumbs_down" and event.get("target") == "safety":
        if safety_orchestrator is not None and safety_ctx is not None:
            try:
                safety_orchestrator.update(safety_ctx, misfire=True, d_tau=d_tau)
                applied = True
                targets.append("safety")
            except Exception:
                pass

    return {
        "type": feedback_type,
        "mode": mode,
        "targets": targets,
        "applied": applied,
    }


__all__ = ["apply_feedback"]

