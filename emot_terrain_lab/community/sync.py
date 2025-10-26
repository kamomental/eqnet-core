# -*- coding: utf-8 -*-
"""Lightweight synchrony helpers for community resonance."""

from __future__ import annotations

import cmath
import math
from typing import Dict, Iterable, List


def resonance_metrics(peers: Iterable[Dict]) -> Dict[str, float | None]:
    """Estimate Kuramoto-style synchrony magnitude r."""
    peers = list(peers)
    if not peers:
        return {"r": None, "norm_match": None}
    vec = [cmath.exp(1j * float(peer.get("phase", 0.0))) for peer in peers]
    r = abs(sum(vec) / len(vec))
    return {"r": round(r, 3), "norm_match": None}


def advance_phases(
    peers: Iterable[Dict],
    *,
    d_tau: float,
    coupling: float = 0.15,
) -> List[Dict]:
    """Advance peer phases by Δτ using a simple Kuramoto step."""
    peers = [dict(peer) for peer in peers]
    if not peers:
        return peers
    phases = [float(peer.get("phase", 0.0)) for peer in peers]
    omegas = [float(peer.get("omega", 0.0)) for peer in peers]
    n = len(peers)
    for i, peer in enumerate(peers):
        sum_term = 0.0
        for j in range(n):
            sum_term += math.sin(phases[j] - phases[i])
        dtheta = omegas[i] + coupling * sum_term / max(1, n)
        phase_new = (phases[i] + dtheta * d_tau) % (2 * math.pi)
        peer["phase"] = phase_new
    return peers


__all__ = ["resonance_metrics", "advance_phases"]
