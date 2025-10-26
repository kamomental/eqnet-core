# -*- coding: utf-8 -*-
"""Aggregate multi-agent joint attention."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple


class GroupAttention:
    """Track group-level dwell time on shared objects."""

    def __init__(
        self,
        weights: Mapping[str, float] | None,
        theta: float,
        dwell_tau: float,
    ) -> None:
        self.weights = dict(weights or {"self": 1.0})
        self.theta = float(theta)
        self.dwell_tau = float(dwell_tau)
        self._state: Dict[str, Tuple[float, float]] = {}

    def update(
        self,
        interest_by_agent: Mapping[str, Mapping[str, float]],
        d_tau: float,
    ) -> Dict[str, Dict[str, float]]:
        reports: Dict[str, Dict[str, float]] = {}
        if not interest_by_agent:
            self._state.clear()
            return reports
        denom = sum(self.weights.get(agent, 1.0) for agent in interest_by_agent) or 1.0
        all_objects = set()
        for interest in interest_by_agent.values():
            all_objects.update(interest.keys())
        for obj in all_objects:
            numerator = 0.0
            for agent, interests in interest_by_agent.items():
                numerator += self.weights.get(agent, 1.0) * interests.get(obj, 0.0)
            joint = numerator / denom
            prev_joint, dwell = self._state.get(obj, (0.0, 0.0))
            if joint >= self.theta:
                dwell += max(d_tau, 0.0)
            else:
                dwell = 0.0
            event = joint >= self.theta and dwell >= self.dwell_tau
            reports[obj] = {
                "group_interest": round(joint, 4),
                "dwell_tau": round(dwell, 4),
                "event": bool(event),
            }
            self._state[obj] = (joint, dwell)
        # purge stale entries
        to_drop = [obj for obj in self._state if obj not in all_objects]
        for obj in to_drop:
            self._state.pop(obj, None)
        return reports


__all__ = ["GroupAttention"]
