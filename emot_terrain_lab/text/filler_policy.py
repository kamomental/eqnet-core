# -*- coding: utf-8 -*-
"""Policy for inserting conversational fillers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass
class FillerDecision:
    enabled: bool
    entries: List[Dict[str, object]]
    damp_factor: float = 1.0
    probability: float = 0.0
    cooldown_ok: bool = True
    new_last_tau: float = float("-inf")
    max_added_chars: int = 40


class FillerPolicy:
    def __init__(self, bank: Mapping[str, object], *, rng: Optional[random.Random] = None) -> None:
        self.bank = dict(bank)
        self.rng = rng or random.Random()
        rules = self.bank.get("rules", {}) or {}
        self.cooldown_tau = float(rules.get("cooldown_tau", 0.8))
        self.base_max_per = int(rules.get("max_per_utterance", 2))
        self.base_ssml = rules.get("ssml_break_ms", {"soft": 200, "filler": 120})
        self.base_max_ssml = int(rules.get("max_ssml_break_ms", 260))
        self.base_max_added = int(rules.get("max_added_chars", 40))
        self.hard_off_domains: Sequence[str] = tuple(self.bank.get("hard_off_domains") or [])
        self.allow_modes: Sequence[str] = tuple(self.bank.get("allow_modes") or [])
        self.mode_overrides: Mapping[str, Mapping[str, object]] = self.bank.get("mode_overrides", {}) or {}

    def decide(
        self,
        *,
        mode: str,
        mood: Mapping[str, float],
        signals: Mapping[str, float],
        tau_info: Mapping[str, float],
        norms: Mapping[str, float],
        domain: str,
        heartiness: float,
        last_tau: Optional[float] = None,
    ) -> FillerDecision:
        prev_tau = float(last_tau if last_tau is not None else float("-inf"))
        if not self._mode_allowed(mode) or self._domain_blocked(domain):
            return FillerDecision(False, [], new_last_tau=prev_tau, max_added_chars=self.base_max_added)

        tau = float(tau_info.get("tau", 0.0))
        if (tau - prev_tau) < self.cooldown_tau:
            return FillerDecision(False, [], cooldown_ok=False, new_last_tau=prev_tau, max_added_chars=self.base_max_added)

        uncertainty = float(mood.get("u", 0.3))
        inflammation = float(signals.get("inflammation_global", 0.0))
        naturality = float(signals.get("naturality_residual", 0.0))
        politeness = float(norms.get("politeness", 0.0))

        base_prob = (
            0.05
            + 0.15 * uncertainty
            + 0.10 * float(heartiness)
            - 0.10 * inflammation
            - 0.05 * politeness
        )
        if mode == "professional":
            base_prob *= 0.5
        elif mode == "caregiver":
            base_prob *= 1.3

        override = self.mode_overrides.get(mode, {}) or {}
        base_prob *= float(override.get("prob_scale", 1.0))
        base_prob = max(0.0, min(0.5, base_prob))
        tot_flag = bool(signals.get("metamemory_tot", False))
        if tot_flag:
            base_prob *= float(self.bank.get("rules", {}).get("tot_prob_scale", 0.6))

        damp_factor = max(0.6, min(1.0, 1.0 - 0.8 * naturality))
        base_prob *= damp_factor
        base_prob = max(0.0, min(0.5, base_prob))
        final_prob = base_prob

        allowed_positions = tuple(override.get("allow_positions", ())) if override else ()
        max_per = int(override.get("max_per_utterance", self.base_max_per)) if override else self.base_max_per
        max_per = max(0, max_per)
        max_ssml = int(override.get("max_ssml_break_ms", self.base_max_ssml)) if override else self.base_max_ssml
        max_ssml = max(60, max_ssml)
        max_added = int(override.get("max_added_chars", self.base_max_added)) if override else self.base_max_added
        max_added = max(0, max_added)

        entries: List[Dict[str, object]] = []
        used_phrases: set[str] = set()
        prob = base_prob
        for _ in range(self.base_max_per):
            if len(entries) >= max_per:
                break
            if self.rng.random() >= prob:
                continue
            entry = self._sample_entry(
                mode=mode,
                tau_info=tau_info,
                signals=signals,
                allowed_positions=allowed_positions,
                damp_factor=damp_factor,
                max_ssml=max_ssml,
            )
            if entry is None or entry["phrase"] in used_phrases:
                continue
            entries.append(entry)
            used_phrases.add(entry["phrase"])
            prob *= 0.6

        if not entries:
            return FillerDecision(
                False,
                [],
                damp_factor=damp_factor,
                probability=final_prob,
                new_last_tau=prev_tau,
                max_added_chars=max_added,
            )

        return FillerDecision(
            True,
            entries,
            damp_factor=damp_factor,
            probability=final_prob,
            new_last_tau=tau,
            max_added_chars=max_added,
        )

    def sample_phrase(self, *, mode: str, kind: str) -> str:
        register = self.bank.get("register", {}) or {}
        mode_map = register.get(mode) or register.get("playful") or {}
        phrases = mode_map.get(kind, [])
        if not phrases:
            return ""
        return self.rng.choice(list(phrases))

    def _sample_entry(
        self,
        *,
        mode: str,
        tau_info: Mapping[str, float],
        signals: Mapping[str, float],
        allowed_positions: Sequence[str],
        damp_factor: float,
        max_ssml: int,
    ) -> Optional[Dict[str, object]]:
        kind = "filler" if self.rng.random() < 0.7 else "soft"
        if allowed_positions:
            pos = self.rng.choice(list(allowed_positions))
        else:
            pos = "sentence_start" if self.rng.random() < 0.6 else "clause"
        phrase = self.sample_phrase(mode=mode, kind=kind)
        if not phrase:
            return None

        tau_rate = float(tau_info.get("tau_rate", 1.0))
        inflammation = float(signals.get("inflammation_global", 0.0))
        tau_scale = 1.0 + 0.3 * max(0.0, 1.0 - tau_rate)
        infl_scale = 1.0 + 0.2 * max(0.0, inflammation)
        base_ms = int(self.base_ssml.get(kind, 120))
        ms = int(base_ms * (0.8 + 0.2 * damp_factor) * tau_scale * infl_scale)
        ms = max(60, min(max_ssml, ms))

        return {
            "kind": kind,
            "position": pos,
            "phrase": phrase,
            "ssml_break": ms,
            "tau": float(tau_info.get("tau", 0.0)),
        }

    def _domain_blocked(self, domain: str) -> bool:
        if not self.hard_off_domains:
            return False
        return domain in self.hard_off_domains

    def _mode_allowed(self, mode: str) -> bool:
        if not self.allow_modes:
            return True
        return mode in self.allow_modes


__all__ = ["FillerDecision", "FillerPolicy"]
