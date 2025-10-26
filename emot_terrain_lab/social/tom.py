# -*- coding: utf-8 -*-
"""Minimal Theory-of-Mind helpers."""

from __future__ import annotations

from typing import Dict


def peer_affect_hint(peer: Dict[str, float]) -> Dict[str, float]:
    """Return tiny valence/arousal adjustments based on peer stress."""
    stress = float(peer.get("stress", 0.0))
    comfort = float(peer.get("comfort", 0.0))
    return {
        "dv": -0.05 * max(0.0, stress - 0.5) + 0.03 * comfort,
        "da": -0.05 * stress,
    }


__all__ = ["peer_affect_hint"]
