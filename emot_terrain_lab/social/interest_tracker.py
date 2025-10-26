# -*- coding: utf-8 -*-
"""Track per-object visual interest across τ."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from emot_terrain_lab.sense.gaze import GazeSummary


@dataclass
class InterestReport:
    object_id: str
    interest: float
    phase: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_id": self.object_id,
            "interest": float(self.interest),
            "phase": self.phase,
        }


class InterestTracker:
    """Exponentially-weighted fixation strength tracker."""

    def __init__(self, cfg: Mapping[str, object] | None = None) -> None:
        cfg = cfg or {}
        self.tau_f = float(cfg.get("tau_f", 1.0))
        self.fix_ref_ms = float(cfg.get("fixation_ref_ms", 900.0))
        self.saccade_ref = float(cfg.get("saccade_ref_hz", 3.0))
        self.blink_ref = float(cfg.get("blink_ref_hz", 1.5))
        self.alpha = float(cfg.get("alpha", 0.6))
        self.beta = float(cfg.get("beta", 0.3))
        self.gamma = float(cfg.get("gamma", 0.25))
        self.eta = float(cfg.get("eta", 0.2))
        self.theta_focus = float(cfg.get("theta_focus", 0.55))
        self.hysteresis = float(cfg.get("hysteresis", 0.05))
        self.fatigue_blink = float(cfg.get("fatigue_blink_hz", 2.5))
        self.floor = float(cfg.get("decay_floor", 0.02))
        self._interest: Dict[str, float] = {}
        self._phase: Dict[str, str] = {}

    def update(self, summary: Optional[GazeSummary], d_tau: float) -> Optional[InterestReport]:
        decay = math.exp(-max(d_tau, 0.0) / max(self.tau_f, 1e-6))
        for obj in list(self._interest.keys()):
            self._interest[obj] *= decay
            if self._interest[obj] <= self.floor:
                self._interest.pop(obj, None)
                self._phase.pop(obj, None)

        if summary is None or summary.target_id is None:
            return None

        obj = summary.target_id
        current = self._interest.get(obj, 0.0)
        fix_term = summary.fixation_ms / max(self.fix_ref_ms, 1.0)
        sacc_term = summary.saccade_rate_hz / max(self.saccade_ref, 1e-6)
        blink_term = summary.blink_rate_hz / max(self.blink_ref, 1e-6)
        pupil_term = 0.5 + 0.5 * math.tanh(summary.pupil_z / 3.0)
        evidence = (
            self.alpha * min(fix_term, 1.25)
            + self.beta * pupil_term
            - self.gamma * min(sacc_term, 1.25)
            - self.eta * min(blink_term, 1.25)
        )
        updated = max(0.0, min(1.0, decay * current + (1.0 - decay) * max(evidence, 0.0)))

        phase = self._phase.get(obj, "orienting")
        if summary.blink_rate_hz >= self.fatigue_blink:
            phase = "fatigue"
        else:
            upper = self.theta_focus + self.hysteresis
            lower = self.theta_focus - self.hysteresis
            if updated >= upper:
                phase = "focusing"
            elif updated <= lower:
                phase = "orienting"
        self._interest[obj] = updated
        self._phase[obj] = phase
        return InterestReport(object_id=obj, interest=updated, phase=phase)

    def snapshot(self) -> Dict[str, float]:
        return dict(self._interest)


__all__ = ["InterestTracker", "InterestReport"]
