# -*- coding: utf-8 -*-
"""Metamemory estimators (FOK/TOT) for EQNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math


@dataclass
class MetaOut:
    F: float
    R: float
    FOK: float
    TOT: bool
    clue: Dict[str, Any]


class MetaMemory:
    def __init__(self, cfg: Dict[str, Any], memory_index) -> None:
        self.theta_fok = float(cfg.get("theta_fok", 0.65))
        self.theta_rec = float(cfg.get("theta_rec", 0.30))
        self.cooldown_tau = float(cfg.get("cooldown_tau", 0.9))
        self.tot_mode = cfg.get("tot_mode", {}) or {}
        self._last_tau = -1e9
        self._memory = memory_index

    def estimate(self, cue: Dict[str, Any], tau_now: float) -> MetaOut:
        if (tau_now - self._last_tau) < self.cooldown_tau:
            return MetaOut(0.0, 0.0, 0.0, False, {})
        near = self._memory.topk(cue, k=16) if hasattr(self._memory, "topk") else []
        if not near:
            return MetaOut(0.0, 0.0, 0.0, False, {})
        sims = [n.get("sim", 0.0) for n in near]
        F = sum(sims) / max(1, len(sims))
        R = max(sims) if sims else 0.0
        partial = next((n.get("partial") for n in near if n.get("partial")), {})
        fluency = min(1.0, len(near) / 16.0)
        FOK = 1 / (1 + math.exp(-(0.6 * F + 0.25 * fluency + 0.15 * (1.0 if partial else 0.0) - 0.5)))
        TOT = (FOK > self.theta_fok) and (R < self.theta_rec)
        clue: Dict[str, Any] = {}
        if TOT and partial:
            for key in ("initial", "category", "year", "phonology"):
                if key in partial:
                    clue = {"type": key, "hint": partial[key]}
                    break
        if TOT:
            self._last_tau = tau_now
        return MetaOut(F=F, R=R, FOK=FOK, TOT=TOT, clue=clue)


__all__ = ["MetaMemory", "MetaOut"]
