# -*- coding: utf-8 -*-
"""Subjective-time aware Unified Replay Kernel."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..culture.norms import norms_penalty
from devlife.value.model import compute_value_summary


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _softmax(cands: Sequence["ReplayCandidate"], tau: float) -> "ReplayCandidate":
    tau = max(1e-3, tau)
    us = [cand.utility for cand in cands]
    m = max(us) if us else 0.0
    exps = [math.exp((u - m) / tau) for u in us]
    total = sum(exps) or 1.0
    r = random.random() * total
    acc = 0.0
    for cand, e in zip(cands, exps):
        acc += e
        if acc >= r:
            return cand
    return cands[-1] if cands else ReplayCandidate(action=None, state={}, qualia={}, external={}, utility=0.0, coherence=0.5, summary={})


class ReplayAdapter:
    """Minimal adapter when no domain-specific adapter provided."""

    def propose_actions(self, z0: Dict[str, Any], mood: Dict[str, float], norms: Dict[str, Any]) -> List[str]:
        return ["reflect", "reframe", "clarify"]

    def simulate(self, z0: Dict[str, Any], action: str, d_tau: float) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        # Dummy transition: carry forward state, fabricate qualia.
        qualia = {"tone": "soft" if action == "reflect" else "sharp", "sensation": 0.1, "meaning": 0.1}
        external = {"action": action}
        return z0, qualia, external

    def extrinsic_value(self, state: Dict[str, Any], external: Dict[str, Any]) -> float:
        return 0.0


@dataclass
class ReplayCandidate:
    action: Any
    state: Any
    qualia: Dict[str, Any]
    external: Dict[str, Any]
    utility: float
    coherence: float
    summary: Dict[str, Any]


class UnifiedReplayKernel:
    def __init__(self, adapter: Optional[ReplayAdapter] = None, *, c_ucb: float = 1.2) -> None:
        self.adapter = adapter or ReplayAdapter()
        self.c_ucb = float(c_ucb)

    def run(
        self,
        *,
        z0: Dict[str, Any],
        steps: int,
        d_tau_step: float,
        tau_rate: float,
        mood: Dict[str, float],
        norms: Dict[str, Any],
        weights: Dict[str, float],
        read_only_policy: bool = False,
        cand_cap: int = 64,
        kappa: float = 0.25,
        coherence_baseline: Optional[float] = None,
        coherence_cb: Optional[Any] = None,
        reverse_ratio: float = 0.5,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        steps = int(_clip(steps, 1, 6))
        d_tau_step = max(0.1, d_tau_step)
        tau_rate = _clip(tau_rate, 0.5, 1.5)
        reverse_ratio = max(0.0, min(1.0, reverse_ratio))

        candidates: List[ReplayCandidate] = []
        actions = self.adapter.propose_actions(z0, mood, norms)
        actions = actions or ["reflect"]
        reverse_steps = min(steps, int(round(steps * reverse_ratio)))

        for s in range(steps):
            direction = "reverse" if s < reverse_steps else "forward"
            if direction == "reverse":
                action_iter = list(reversed(actions))
                eff_step = steps - s
                discount = math.exp(-kappa * eff_step * d_tau_step)
            else:
                action_iter = actions
                discount = math.exp(-kappa * (s + 1) * d_tau_step)
            for action in action_iter:
                state, qualia, external = self.adapter.simulate(z0, action, d_tau_step)
                preview_coherence = None
                fallback_coherence = (
                    float(coherence_baseline)
                    if coherence_baseline is not None
                    else float(state.get("coherence", 0.5))
                )
                if coherence_cb is not None:
                    try:
                        preview_coherence = float(coherence_cb(action))
                    except Exception:
                        preview_coherence = None
                coherence_score = preview_coherence if preview_coherence is not None else fallback_coherence
                summary = compute_value_summary(
                    extrinsic_signal=self.adapter.extrinsic_value(state, external),
                    novelty_signal=float(state.get("novelty", 0.0)),
                    social_alignment=float(state.get("social_alignment", mood.get("s", 0.5))),
                    coherence_score=coherence_score,
                    homeostasis_error=float(state.get("homeostasis_error", 0.0)),
                    qualia_consistency=float(state.get("qualia_consistency", 0.5)),
                    norm_penalty=float(norms_penalty(action, norms)),
                    weights_override=weights,
                    metadata={"step": s, "discount": discount},
                )
                summary["direction"] = direction
                utility = summary["total"] * discount
                candidates.append(
                    ReplayCandidate(
                        action=action,
                        state=state,
                        qualia=qualia,
                        external=external,
                        utility=utility,
                        coherence=float(coherence_score),
                        summary=summary,
                    )
                )
                if len(candidates) >= cand_cap:
                    break
            if len(candidates) >= cand_cap:
                break

        if not candidates:
            dummy = {"a": None, "U": 0.0, "coherence": coherence_baseline or 0.5, "summary": {}, "read_only": read_only_policy}
            return dummy, []

        tau_soft = max(0.3, 1.2 - 0.4 * tau_rate - 0.3 * mood.get("a", 0.5))
        best_cand = _softmax(candidates, tau_soft)
        best = {
            "a": best_cand.action,
            "U": best_cand.utility,
            "coherence": best_cand.coherence,
            "summary": best_cand.summary,
            "read_only": read_only_policy,
        }
        cand_dump = [
            {
                "a": c.action,
                "U": round(c.utility, 4),
                "coherence": round(c.coherence, 3),
            }
            for c in candidates
        ]
        cand_dump.sort(key=lambda x: x["U"], reverse=True)
        return best, cand_dump


__all__ = ["UnifiedReplayKernel", "ReplayAdapter", "ReplayCandidate"]
