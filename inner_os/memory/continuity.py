from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..self_model.models import PersonNode, PersonRegistry


@dataclass
class ContinuityTrace:
    trace_id: str
    stable_themes: list[str] = field(default_factory=list)
    uncertainty: float = 1.0


@dataclass
class IdentityObservation:
    person_id_hint: str = ""
    summary: str = ""
    stable_traits: dict[str, float] = field(default_factory=dict)
    adaptive_traits: dict[str, float] = field(default_factory=dict)
    ambiguity: float = 1.0


@dataclass
class ContinuityUpdate:
    person_id: str
    confidence: float
    ambiguity: float
    corroborated_traits: list[str] = field(default_factory=list)
    history_entry: dict[str, str] = field(default_factory=dict)


def score_identity_continuity(
    node: PersonNode | None,
    observation: IdentityObservation,
) -> ContinuityUpdate:
    person_id = observation.person_id_hint or "unknown"
    if node is None:
        confidence = 0.35 if person_id != "unknown" else 0.0
        corroborated_traits: list[str] = []
    else:
        shared_traits = set(node.stable_traits) & set(observation.stable_traits)
        corroborated_traits = sorted(shared_traits)
        confidence = min(
            1.0,
            max(node.confidence, 0.3)
            + 0.15 * len(corroborated_traits)
            + 0.1 * max(0.0, 1.0 - observation.ambiguity),
        )
    ambiguity = max(0.0, min(1.0, observation.ambiguity))
    return ContinuityUpdate(
        person_id=person_id,
        confidence=confidence,
        ambiguity=ambiguity,
        corroborated_traits=corroborated_traits,
        history_entry={
            "observation": observation.summary,
            "ambiguity": f"{ambiguity:.2f}",
        },
    )


def update_person_registry(
    person_registry: PersonRegistry,
    observations: Mapping[str, object],
    context: Mapping[str, object],
) -> PersonRegistry:
    registry = dict(person_registry.persons)
    observation = IdentityObservation(
        person_id_hint=str(context.get("person_id") or ""),
        summary=str(observations.get("summary") or ""),
        stable_traits=dict(observations.get("stable_traits") or {}),
        adaptive_traits=dict(observations.get("adaptive_traits") or {}),
        ambiguity=float(observations.get("ambiguity", 1.0)),
    )
    person_id = observation.person_id_hint or "unknown"
    node = registry.get(person_id)
    update = score_identity_continuity(node, observation)
    base_node = node or PersonNode(person_id=update.person_id)
    merged_stable = dict(base_node.stable_traits)
    merged_stable.update(observation.stable_traits)
    merged_adaptive = dict(base_node.adaptive_traits)
    merged_adaptive.update(observation.adaptive_traits)
    registry[update.person_id] = PersonNode(
        person_id=update.person_id,
        stable_traits=merged_stable,
        adaptive_traits=merged_adaptive,
        continuity_history=list(base_node.continuity_history) + [update.history_entry],
        confidence=update.confidence,
        ambiguity_flag=update.ambiguity > 0.5,
    )
    return PersonRegistry(
        persons=registry,
        uncertainty=min(
            1.0,
            max(0.0, 0.2 + 0.6 * update.ambiguity),
        ),
    )
