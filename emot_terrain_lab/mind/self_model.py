# -*- coding: utf-8 -*-
"""Self model providing a narrative-coherence heuristic."""

from __future__ import annotations

from typing import Optional

from .narrative import Narrative


class SelfModel:
    """Maintain lightweight self narrative and coherence estimates."""

    def __init__(self) -> None:
        self.narrative = Narrative()
        self._baseline = 0.5

    def note(self, event: dict) -> None:
        """Record an event into the self narrative."""
        self.narrative.add_event(event)

    def coherence(self) -> float:
        """Return present-time coherence estimate."""
        narrative_coh = self.narrative.coherence()
        if not self.narrative.events:
            return self._baseline
        return narrative_coh

    def preview_coherence(self, intent: Optional[str]) -> float:
        """
        Predict the coherence impact of choosing an action with a given intent.
        """
        current = self.coherence()
        if not self.narrative.events:
            delta = 0.05 if intent else 0.0
            return max(0.0, min(1.0, self._baseline + delta))
        last_intent = self.narrative.events[-1].get("intent")
        if last_intent is None or intent is None:
            return current
        delta = 0.05 if intent == last_intent else -0.05
        updated = current + delta
        return max(0.0, min(1.0, updated))

    def set_baseline(self, coherence: float) -> None:
        """Seed baseline coherence used before narrative history exists."""
        self._baseline = max(0.0, min(1.0, float(coherence)))


__all__ = ["SelfModel"]
