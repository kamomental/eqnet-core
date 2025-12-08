# -*- coding: utf-8 -*-
"""Gentle learner hook helpers for tone/style adjustments.

These hooks do not directly mutate policy weights; they emit structured
telemetry so that LoRA/MAP-Elites or fast-path controllers can replay the
adjustments inside safer jobs. In run_quick_loop we treat them as the
"learner" endpoint for Section 5's closed-loop demo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LearnerState:
    """Minimal state tracked by learner hooks during a loop run."""

    tone_bias: float = 0.0
    fastpath_weight: float = 0.0
    value_committee_weight: float = 1.0
    last_profile: Optional[str] = None


@dataclass
class LearnerHooks:
    """Collects adjustments and writes them to ``logs/learner_hooks.jsonl``."""

    log_path: Path = Path("logs/learner_hooks.jsonl")
    state: LearnerState = field(default_factory=LearnerState)
    experiment_tag: Optional[str] = None

    def _write(self, payload: Dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _base_event(self, kind: str, episode: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "type": kind,
            "episode_id": episode.get("episode_id"),
            "stage": episode.get("stage"),
            "step": episode.get("step"),
        }
        if self.experiment_tag:
            payload["tag"] = self.experiment_tag
        return payload

    def apply_love_softening(self, episode: Dict[str, Any], *, delta: float) -> None:
        """Nudge tone bias upward (gentler) when love_mode is low."""

        self.state.tone_bias = float(max(-0.5, min(0.5, self.state.tone_bias + delta)))
        event = self._base_event("tone.bias", episode)
        event.update({"delta": delta, "tone_bias": self.state.tone_bias})
        self._write(event)

    def apply_love_cooldown(self, episode: Dict[str, Any], *, delta: float) -> None:
        """Reduce tone bias when love_mode saturates to keep distance."""

        self.state.tone_bias = float(max(-0.5, min(0.5, self.state.tone_bias - delta)))
        event = self._base_event("tone.cooldown", episode)
        event.update({"delta": -delta, "tone_bias": self.state.tone_bias})
        self._write(event)

    def reinforce_value_weight(self, episode: Dict[str, Any], *, factor: float) -> None:
        """Adjust Value Committee weight when trust drops."""

        self.state.value_committee_weight = float(
            max(0.1, min(2.0, self.state.value_committee_weight * factor))
        )
        event = self._base_event("value.weight", episode)
        event.update({
            "factor": factor,
            "value_weight": self.state.value_committee_weight,
        })
        self._write(event)

    def apply_fastpath_style_override(
        self,
        episode: Dict[str, Any],
        *,
        profile: Optional[str],
        magnitude: float = 0.1,
    ) -> bool:
        """Log that a fast-path style override adjusted the learner.

        Returns ``True`` if the override was applied (profile provided),
        otherwise ``False``.
        """

        if not profile:
            return False
        self.state.fastpath_weight = float(
            max(0.0, min(1.0, self.state.fastpath_weight + magnitude))
        )
        self.state.last_profile = profile
        event = self._base_event("fastpath.style_override", episode)
        event.update(
            {
                "profile": profile,
                "magnitude": magnitude,
                "fastpath_weight": self.state.fastpath_weight,
            }
        )
        self._write(event)
        return True


__all__ = ["LearnerHooks", "LearnerState"]
