from __future__ import annotations

from ..grounding.models import ObservationBundle, Affordance
from .models import WorldState


def update_world_state(
    prev_world_state: WorldState,
    observation: ObservationBundle,
    affordances: dict[str, list[Affordance]],
) -> WorldState:
    scene_graph = dict(prev_world_state.scene_graph)
    object_states = dict(prev_world_state.object_states)
    social_relation_graph = dict(prev_world_state.social_relation_graph)
    for entity in observation.entities:
        scene_graph[entity.entity_id] = [aff.action for aff in affordances.get(entity.entity_id, [])]
        object_states[entity.entity_id] = entity.label
        person_id_hint = str(entity.attributes.get("person_id_hint") or "").strip()
        if person_id_hint:
            relations = list(social_relation_graph.get(entity.entity_id, []))
            person_ref = f"person:{person_id_hint}"
            if person_ref not in relations:
                relations.append(person_ref)
            social_relation_graph[entity.entity_id] = relations
    return WorldState(
        scene_graph=scene_graph,
        spatial_map=dict(prev_world_state.spatial_map),
        object_states=object_states,
        task_states=dict(prev_world_state.task_states),
        social_relation_graph=social_relation_graph,
        uncertainty=max(prev_world_state.uncertainty * 0.8, observation.observation_uncertainty),
    )
