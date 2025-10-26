# -*- coding: utf-8 -*-
"""Memory TTL and decay management for replay traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class TTLConfig:
    ttl_tau_default: float
    ttl_tau_low: float
    halflife_tau: float
    ephemeral_tags: List[str]
    ephemeral_tau: float


class MemoryTTLManager:
    """Apply TTL and half-life decay to memory traces."""

    def __init__(self, cfg: Dict[str, Any], timekeeper) -> None:
        ttl_cfg = cfg.get("ttl_tau", {}) or {}
        self.cfg = TTLConfig(
            ttl_tau_default=float(ttl_cfg.get("default", 24.0)),
            ttl_tau_low=float(ttl_cfg.get("low_value", 6.0)),
            halflife_tau=float(cfg.get("halflife_tau", 24.0)),
            ephemeral_tags=list(cfg.get("ephemeral_tags", [])),
            ephemeral_tau=float(cfg.get("ephemeral_tau", 4.0)),
        )
        self._tk = timekeeper

    def _current_tau(self) -> float:
        tau_now = getattr(self._tk, "tau_now", None)
        if callable(tau_now):
            return float(tau_now())
        return float(getattr(self._tk, "tau", 0.0))

    def _age_tau(self, event: Dict[str, Any]) -> float:
        current_tau = self._current_tau()
        ts_tau = float(event.get("tau", event.get("ts_tau", current_tau)))
        return max(0.0, current_tau - ts_tau)

    def _ttl_for(self, event: Dict[str, Any]) -> float:
        tags = set(event.get("tags", []))
        if self.cfg.ephemeral_tags and tags.intersection(self.cfg.ephemeral_tags):
            return self.cfg.ephemeral_tau
        value = event.get("value", {}) or {}
        total = float(value.get("total", event.get("value", 0.0)))
        if abs(total) < 0.2:
            return self.cfg.ttl_tau_low
        ttl = self.cfg.ttl_tau_default
        meta = event.get("meta", {}) or {}
        ttl_override = meta.get("ttl_override_tau")
        if ttl_override is not None:
            try:
                return float(ttl_override)
            except Exception:
                return ttl
        ttl_scale = meta.get("ttl_scale")
        if ttl_scale is not None:
            try:
                ttl *= float(ttl_scale)
            except Exception:
                pass
        return ttl

    def should_drop(self, event: Dict[str, Any]) -> bool:
        age = self._age_tau(event)
        ttl = self._ttl_for(event)
        return age >= ttl

    def weight_after_decay(self, event: Dict[str, Any]) -> float:
        age = self._age_tau(event)
        weight = float(event.get("weight", 1.0))
        halflife = max(1e-6, self.cfg.halflife_tau)
        return weight * (2.0 ** (-age / halflife))

    def gc(self, events: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        kept: List[Dict[str, Any]] = []
        dropped = 0
        for event in events:
            if self.should_drop(event):
                dropped += 1
                continue
            event["weight"] = self.weight_after_decay(event)
            kept.append(event)
        stats = {"dropped": dropped, "kept": len(kept)}
        return kept, stats
