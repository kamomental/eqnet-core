# -*- coding: utf-8 -*-
"""Mode hysteresis helper to avoid rapid oscillations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModeDecision:
    new_mode: str
    reason: str
    thresholds: Dict[str, float]


class ModeHysteresis:
    def __init__(self, cfg: Dict[str, float]) -> None:
        self.enter = float(cfg.get("enter", 0.4))
        self.exit = float(cfg.get("exit", 0.25))

    def decide(self, current: str, risk_p: float) -> ModeDecision:
        curr = (current or "supportive").lower()
        thresholds = {"enter": self.enter, "exit": self.exit}
        if curr != "read_only" and risk_p >= self.enter:
            return ModeDecision("read_only", "enter_high_risk", thresholds)
        if curr == "read_only" and risk_p <= self.exit:
            return ModeDecision("supportive", "exit_low_risk", thresholds)
        return ModeDecision(current, "hold", thresholds)


__all__ = ["ModeHysteresis", "ModeDecision"]
