"""Aesthetic guard brick enforcing taste constraints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GuardConfig:
    dislike_tags: Tuple[str, ...] = ("no_go",)
    like_tags: Tuple[str, ...] = ()
    threshold: float = -0.2


@dataclass
class AestheticGuard:
    config: GuardConfig = field(default_factory=GuardConfig)

    def evaluate(self, mood_v: float, content_tags: List[str]) -> Dict[str, float | int]:
        taste = mood_v
        if any(tag in self.config.dislike_tags for tag in content_tags):
            taste -= 0.5
        if any(tag in self.config.like_tags for tag in content_tags):
            taste += 0.2
        allow = int(taste >= self.config.threshold)
        return {"taste_score": taste, "allow": allow}

