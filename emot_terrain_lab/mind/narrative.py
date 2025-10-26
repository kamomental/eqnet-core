# -*- coding: utf-8 -*-
"""Minimal narrative scaffold tracking intent relationships."""

from __future__ import annotations

from typing import Dict, List, Tuple


class Narrative:
    """
    Track events with lightweight edges indicating how intents relate.

    coherence = supports / (supports + contradicts + 1e-6)
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, object]] = []
        self.edges: List[Tuple[int, int, str]] = []

    def add_event(self, event: Dict[str, object]) -> int:
        """Add an event into the narrative timeline."""
        self.events.append(dict(event))
        idx = len(self.events) - 1
        if idx > 0:
            prev = self.events[idx - 1]
            intent = event.get("intent")
            prev_intent = prev.get("intent")
            if intent is not None and intent == prev_intent:
                self.edges.append((idx - 1, idx, "supports"))
            elif intent is not None and prev_intent is not None and intent != prev_intent:
                self.edges.append((idx - 1, idx, "contradicts"))
            if event.get("cause") == prev.get("id"):
                self.edges.append((idx - 1, idx, "causes"))
        return idx

    def coherence(self) -> float:
        """Return a bounded [0,1] coherence score."""
        supports = sum(1 for _, _, kind in self.edges if kind == "supports")
        contradicts = sum(1 for _, _, kind in self.edges if kind == "contradicts")
        total = supports + contradicts
        if total == 0:
            return 0.5
        raw = supports / float(total + 1e-6)
        tempered = 0.5 + 0.3 * (raw - 0.5)
        return max(0.0, min(1.0, tempered))


__all__ = ["Narrative"]
