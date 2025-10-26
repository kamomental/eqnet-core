# -*- coding: utf-8 -*-
"""Safety orchestration backed by Bayesian confidence bounds."""

from __future__ import annotations

from typing import Dict, Optional

from ..safety.bayes_gate import BayesSafetyGate


class SafetyOrchestrator:
    """Decide safety posture and maintain Bayesian posteriors."""

    def __init__(
        self,
        *,
        hard_constraints: Optional[Dict[str, object]] = None,
        bayes_gate: Optional[BayesSafetyGate] = None,
    ) -> None:
        self._hard_constraints = set(hard_constraints or [])
        self._bayes_gate = bayes_gate

    def evaluate(self, ctx: Dict[str, object], d_tau: float) -> Dict[str, object]:
        act_category = str(ctx.get("act_category") or ctx.get("domain") or "dialogue")
        domain_read_only = ctx.get("domain") in self._hard_constraints
        bayes_decision: Optional[str] = None
        risk_upper: Optional[float] = None
        if self._bayes_gate is not None:
            try:
                bayes_decision = self._bayes_gate.decide(act_category, d_tau=d_tau)
                risk_upper = self._bayes_gate.risk_upper(act_category)
            except Exception:
                bayes_decision = None
                risk_upper = None
        read_only = domain_read_only or (bayes_decision in {"READ_ONLY", "BLOCK"})
        return {
            "act_category": act_category,
            "bayes_decision": bayes_decision,
            "read_only": read_only,
            "domain_read_only": domain_read_only,
            "risk_upper": risk_upper,
        }

    def update(self, safety_ctx: Dict[str, object], *, misfire: bool, d_tau: float) -> None:
        if self._bayes_gate is None:
            return
        try:
            self._bayes_gate.update(
                str(safety_ctx.get("act_category")),
                misfire=misfire,
                d_tau=d_tau,
            )
        except Exception:
            pass

    def metrics(self) -> Dict[str, object]:
        if self._bayes_gate is None:
            return {}
        return self._bayes_gate.snapshot()


__all__ = ["SafetyOrchestrator"]
