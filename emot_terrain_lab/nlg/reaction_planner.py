# -*- coding: utf-8 -*-
"""Persona-aware reaction planner driven by affective terrain."""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AffectSnapshot:
    valence: float = 0.0   # -1.0 .. +1.0
    arousal: float = 0.0   # -1.0 .. +1.0
    social: float = 0.0    #  0.0 .. +1.0 (connectedness / care)
    novelty: float = 0.0   #  0.0 .. +1.0
    certainty: float = 0.7 #  0.0 .. +1.0 (confidence in situation)


@dataclass
class ReactionChoice:
    text: str
    persona: str
    zone: str
    band: str
    family: str


class ReactionPlanner:
    """Map affect + heartiness to persona-specific micro reactions."""

    def __init__(
        self,
        persona_tag: str,
        persona_cfg: Dict[str, Any],
        *,
        seed_key: Optional[str] = None,
    ) -> None:
        self.persona = persona_tag
        self.cfg = persona_cfg or {}
        seed_src = seed_key or f"{persona_tag}::{time.strftime('%Y%m%d%H')}"
        seed_int = int(hashlib.sha1(seed_src.encode("utf-8")).hexdigest()[:16], 16)
        self.rng = random.Random(seed_int)

    # ------------------------------------------------------------------ planning

    def plan(
        self,
        affect: AffectSnapshot,
        *,
        heartiness: float,
        extras: Optional[Dict[str, Any]] = None,
        protected: bool = False,
    ) -> Optional[ReactionChoice]:
        zone = self._affect_to_zone(affect)
        band = self._affect_to_band(heartiness, affect)
        candidate = self._select_candidate(zone, band)
        if not candidate:
            return None
        text = self._render_text(candidate, extras or {}, heartiness, protected)
        if not text.strip():
            return None
        return ReactionChoice(
            text=text,
            persona=self.persona,
            zone=zone,
            band=band,
            family=str(candidate.get("family", "misc")),
        )

    # ------------------------------------------------------------------ helpers

    def _affect_to_zone(self, a: AffectSnapshot) -> str:
        v = max(-1.0, min(1.0, a.valence))
        ar = max(-1.0, min(1.0, a.arousal))
        social = max(0.0, min(1.0, a.social))
        novelty = max(0.0, min(1.0, a.novelty))
        uncertainty = max(0.0, min(1.0, 1.0 - a.certainty))

        if v > 0.4 and ar > 0.25:
            return "joy"
        if v > 0.25 and social > 0.5:
            return "tender"
        if abs(v) < 0.2 and ar < -0.25:
            return "calm"
        if novelty > 0.6 and ar > 0.1:
            return "curious"
        if uncertainty > 0.5 and ar > 0.25:
            return "anxious"
        if ar < -0.5:
            return "tired"
        if v < -0.25 and ar > 0.35:
            return "irritable"
        if abs(v) < 0.2 and abs(ar) < 0.2:
            return "focused"
        return "focused"

    def _affect_to_band(self, heartiness: float, a: AffectSnapshot) -> str:
        heart = max(0.0, min(1.0, heartiness))
        arousal = 0.5 * (max(-1.0, min(1.0, a.arousal)) + 1.0)  # 0..1
        level = 0.6 * heart + 0.4 * arousal
        if level < 0.33:
            return "low"
        if level < 0.66:
            return "mid"
        return "high"

    def _select_candidate(self, zone: str, band: str) -> Optional[Dict[str, Any]]:
        zones_cfg = self.cfg.get("zones", {})
        candidates = (
            zones_cfg.get(zone, {}).get(band)
            or zones_cfg.get(zone, {}).get(self.cfg.get("fallback_band", "mid"))
        )
        if not candidates:
            fallback_zone = self.cfg.get("fallback_zone", "focused")
            candidates = (
                zones_cfg.get(fallback_zone, {}).get(band)
                or zones_cfg.get(fallback_zone, {}).get(self.cfg.get("fallback_band", "mid"))
            )
        if not candidates:
            return None
        weights: List[float] = []
        for cand in candidates:
            w = float(cand.get("weight", cand.get("w", 1.0)) or 1.0)
            weights.append(max(1e-4, w))
        total = sum(weights)
        pick = self.rng.random() * total
        acc = 0.0
        for cand, w in zip(candidates, weights):
            acc += w
            if pick <= acc:
                return cand
        return candidates[-1]

    def _render_text(
        self,
        cand: Dict[str, Any],
        extras: Dict[str, Any],
        heartiness: float,
        protected: bool,
    ) -> str:
        texts: List[str] = []
        base = cand.get("text")
        if base:
            texts.append(str(base))
        variants = cand.get("variants") or []
        texts.extend(str(v) for v in variants if v)
        if not texts:
            return ""
        text = self.rng.choice(texts)
        text = self._safe_format(text, extras)
        if protected and cand.get("text_short"):
            text = self._safe_format(str(cand["text_short"]), extras)
        prefix = cand.get("prefix")
        if prefix:
            text = str(prefix) + text
        suffix = cand.get("suffix")
        if suffix:
            opts = suffix if isinstance(suffix, list) else [suffix]
            if opts:
                text = text + self.rng.choice(opts)
        emphasize = cand.get("emphasize")
        if emphasize:
            gain = 1.0 + 0.5 * heartiness
            mark = "!" if gain > 1.2 else "~"
            if not text.endswith(mark):
                text = f"{text}{mark}"
        return text

    @staticmethod
    def _safe_format(template: str, data: Dict[str, Any]) -> str:
        try:
            return template.format(**data)
        except Exception:
            return template
