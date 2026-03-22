from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttentionState:
    salience_bias: float = 0.5
    uncertainty_tolerance: float = 0.5
    continuity_bias: float = 0.4
    affiliation_bias: float = 0.5
    caution_bias: float = 0.4
    community_bias: float = 0.5
    culture_bias: float = 0.5
    partner_style_relief: float = 0.0
    partner_style_caution: float = 0.0
    relational_future_pull: float = 0.0
    relational_reverence: float = 0.0
    relational_care: float = 0.0
    shared_world_pull: float = 0.0


@dataclass
class AccessCandidate:
    entity_id: str
    salience: float
    reasons: list[str] = field(default_factory=list)
    continuity_hint: str = ""


@dataclass
class ForegroundState:
    salient_entities: list[str] = field(default_factory=list)
    current_risks: list[str] = field(default_factory=list)
    active_goals: list[str] = field(default_factory=list)
    affective_summary: dict[str, float] = field(default_factory=dict)
    reportable_facts: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    candidates: list[AccessCandidate] = field(default_factory=list)
    selection_reasons: list[str] = field(default_factory=list)
    continuity_focus: list[str] = field(default_factory=list)
    reportability_scores: dict[str, float] = field(default_factory=dict)
    memory_candidates: list[str] = field(default_factory=list)
    memory_reasons: dict[str, list[str]] = field(default_factory=dict)
