from __future__ import annotations

from typing import Mapping, Any

from ..memory.continuity import update_person_registry as update_continuity_registry
from ..world_model.models import WorldState
from .models import PersonNode, PersonRegistry, SelfState


def update_self_state(
    prev_self_state: SelfState,
    world_state: WorldState,
    events: Mapping[str, Any],
) -> SelfState:
    uncertainty = max(world_state.uncertainty, float(events.get("uncertainty", prev_self_state.uncertainty)))
    task_load = min(1.0, prev_self_state.task_load * 0.8 + len(world_state.task_states) * 0.1)
    return SelfState(
        arousal=min(1.0, prev_self_state.arousal * 0.7 + float(events.get("arousal_delta", 0.0))),
        uncertainty=uncertainty,
        fatigue=min(1.0, prev_self_state.fatigue * 0.85 + float(events.get("fatigue_delta", 0.0))),
        trust=prev_self_state.trust,
        curiosity=min(1.0, prev_self_state.curiosity * 0.8 + 0.1),
        task_load=round(task_load, 4),
        social_tension=prev_self_state.social_tension,
        safety_margin=max(0.0, min(1.0, 1.0 - uncertainty * 0.5)),
    )


def update_person_registry(
    person_registry: PersonRegistry,
    observations: Mapping[str, Any],
    context: Mapping[str, Any],
) -> PersonRegistry:
    return update_continuity_registry(person_registry, observations, context)


def person_registry_from_snapshot(snapshot: Mapping[str, Any] | None) -> PersonRegistry:
    if not isinstance(snapshot, Mapping):
        return PersonRegistry()
    persons_payload = snapshot.get("persons")
    persons: dict[str, PersonNode] = {}
    if isinstance(persons_payload, Mapping):
        for person_id, payload in persons_payload.items():
            if not isinstance(payload, Mapping):
                continue
            persons[str(person_id)] = PersonNode(
                person_id=str(payload.get("person_id") or person_id),
                stable_traits=dict(payload.get("stable_traits") or {}),
                adaptive_traits=dict(payload.get("adaptive_traits") or {}),
                continuity_history=list(payload.get("continuity_history") or []),
                confidence=float(payload.get("confidence", 0.0) or 0.0),
                ambiguity_flag=bool(payload.get("ambiguity_flag", True)),
            )
    return PersonRegistry(
        persons=persons,
        uncertainty=float(snapshot.get("uncertainty", 1.0) or 1.0),
    )
