# -*- coding: utf-8 -*-
"""Interference-aware gating for replay traces."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

from devlife.mind.replay_memory import ReplayTrace


@dataclass
class _Signature:
    tokens: Set[str]
    expires_tau: float
    representative: bool


class InterferenceGate:
    """Detect highly similar traces and mask or drop them temporarily."""

    def __init__(self, cfg: Dict[str, object], *, timekeeper) -> None:
        config = dict(cfg or {})
        self.threshold = float(config.get("similarity_threshold", 0.8))
        self.mask_tau = float(config.get("mask_tau", 4.0))
        self.cooldown_tau = float(config.get("cooldown_tau", self.mask_tau))
        self.hold_tau = float(config.get("hold_tau", 12.0))
        self.ttl_override_tau = float(config.get("ttl_override_tau", 4.0))
        self.drop_on_mask = bool(config.get("drop_on_mask", False))
        self.keep_representative = bool(config.get("keep_representative", False))
        self.min_tokens = int(config.get("min_tokens", 1))
        self._tk = timekeeper
        self._registry: List[_Signature] = []

    def evaluate(self, trace: ReplayTrace) -> Dict[str, object]:
        """Return {'action': 'pass'|'mask'|'skip', ...}."""

        tokens = self._tokenize(trace)
        tau_now = self._current_tau()
        self._purge_expired(tau_now)
        if not tokens:
            self._register(tokens, tau_now)
            return {"action": "pass"}

        best_sim = 0.0
        for entry in self._registry:
            if not entry.tokens:
                continue
            union = tokens | entry.tokens
            if not union:
                continue
            sim = len(tokens & entry.tokens) / len(union)
            if sim > best_sim:
                best_sim = sim

        if best_sim >= self.threshold:
            mask_until = tau_now + self.cooldown_tau
            self._register(tokens, tau_now, representative=False, expires=mask_until)
            if self.drop_on_mask or self.keep_representative:
                return {
                    "action": "skip",
                    "reason": "interference",
                    "similarity": best_sim,
                }
            return {
                "action": "mask",
                "similarity": best_sim,
                "ttl_override_tau": self.ttl_override_tau,
                "mask_until_tau": mask_until,
            }

        self._register(tokens, tau_now, representative=True)
        return {"action": "pass"}

    def _register(self, tokens: Set[str], tau_now: float, representative: bool, expires: Optional[float] = None) -> None:
        expiry = expires if expires is not None else (tau_now + self.hold_tau)
        self._registry.append(_Signature(tokens=tokens, expires_tau=expiry, representative=representative))

    def _purge_expired(self, tau_now: float) -> None:
        self._registry = [sig for sig in self._registry if sig.expires_tau > tau_now]

    def _current_tau(self) -> float:
        tau_now = getattr(self._tk, "tau_now", None)
        if callable(tau_now):
            try:
                return float(tau_now())
            except Exception:
                return 0.0
        return float(getattr(self._tk, "tau", 0.0))

    def _tokenize(self, trace: ReplayTrace) -> Set[str]:
        imagined = trace.imagined or {}
        action = str(imagined.get("best_action") or imagined.get("excerpt") or "").lower()
        tokens = set(re.findall(r"\w+", action))
        if not tokens and action:
            tokens.add(action)
        domain = str(trace.meta.get("domain", "") if trace.meta else "")
        if domain:
            tokens.add(domain.lower())
        tokens = {tok for tok in tokens if tok}
        if len(tokens) < self.min_tokens and action:
            tokens.add(action[:8])
        return tokens


__all__ = ["InterferenceGate"]
