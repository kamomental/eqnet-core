# -*- coding: utf-8 -*-
"""Self-model helpers for lightweight episode reflections."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class SelfReporter:
    """Consumes episode metrics and emits a simple self-report JSONL log."""

    log_path: Path | str = Path("logs/self_report.jsonl")

    def __post_init__(self) -> None:
        self.log_path = Path(self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public API
    def log_episode(
        self,
        episode: Mapping[str, Any],
        metrics: Mapping[str, Any],
        *,
        tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._build_payload(episode, metrics, tag=tag)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    # ------------------------------------------------------------------ helpers
    def _build_payload(
        self,
        episode: Mapping[str, Any],
        metrics: Mapping[str, Any],
        *,
        tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        love = float(metrics.get("affect.love", 0.0) or 0.0)
        trust = float(metrics.get("value.intent_trust", 0.0) or 0.0)
        override = float(metrics.get("fastpath.override_rate", 0.0) or 0.0)
        body_r = float(metrics.get("body.R", 1.0) or 1.0)
        tension = float(metrics.get("tension", 0.0) or 0.0)
        imagery_positive = bool(metrics.get("imagery_positive", False))

        mood, social_tone = self._classify_mood(love, trust, override)
        confidence = max(0.0, min(1.0, trust))
        stress_level = max(0.0, min(1.0, tension))
        summary = self._make_summary(mood, social_tone, love, trust, override)

        payload = {
            "ts": time.time(),
            "episode_id": episode.get("episode_id"),
            "stage": episode.get("stage"),
            "step": episode.get("step"),
            "mood": mood,
            "social_tone": social_tone,
            "confidence": confidence,
            "stress_level": stress_level,
            "summary": summary,
            "metrics": {
                "love": love,
                "intent_trust": trust,
                "fastpath_override": override,
                "body.R": body_r,
                "tension": tension,
            },
        }
        if imagery_positive:
            payload["imagery_positive"] = True
        if tag:
            payload["tag"] = tag
        if episode.get("timestamp"):
            payload["timestamp"] = episode.get("timestamp")
        return payload

    def _classify_mood(
        self, love: float, trust: float, override: float
    ) -> tuple[str, str]:
        if love >= 0.22 and trust >= 0.65:
            return "warm", "supportive"
        if trust < 0.4:
            return "guarded", "cautious"
        if override > 0.3:
            return "focused", "experimental"
        if love < 0.12:
            return "cautious", "soft"
        return "steady", "balanced"

    def _make_summary(
        self,
        mood: str,
        tone: str,
        love: float,
        trust: float,
        override: float,
    ) -> str:
        return (
            f"mood={mood}, tone={tone}. love={love:.2f}, intent_trust={trust:.2f}, "
            f"override={override:.2f}."
        )


__all__ = ["SelfReporter"]


