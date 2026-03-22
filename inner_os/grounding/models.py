from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroundedEntity:
    entity_id: str
    label: str
    source: str
    confidence: float
    uncertainty: float
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class Affordance:
    action: str
    score: float
    constraints: list[str] = field(default_factory=list)
    risk: float = 0.0
    context_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SymbolGrounding:
    token: str
    percept_refs: list[str] = field(default_factory=list)
    affordance_refs: list[str] = field(default_factory=list)
    value_refs: list[str] = field(default_factory=list)
    action_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    context_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ObservationBundle:
    entities: list[GroundedEntity] = field(default_factory=list)
    affordances: dict[str, list[Affordance]] = field(default_factory=dict)
    symbol_groundings: dict[str, SymbolGrounding] = field(default_factory=dict)
    observation_uncertainty: float = 1.0
    notes: list[str] = field(default_factory=list)
