from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PersonNode:
    person_id: str
    stable_traits: dict[str, float] = field(default_factory=dict)
    adaptive_traits: dict[str, float] = field(default_factory=dict)
    continuity_history: list[dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    ambiguity_flag: bool = True


@dataclass
class PersonRegistry:
    persons: dict[str, PersonNode] = field(default_factory=dict)
    uncertainty: float = 1.0


@dataclass
class SelfState:
    arousal: float = 0.0
    uncertainty: float = 1.0
    fatigue: float = 0.0
    trust: float = 0.0
    curiosity: float = 0.0
    task_load: float = 0.0
    social_tension: float = 0.0
    safety_margin: float = 0.5
