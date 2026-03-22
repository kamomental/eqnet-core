from __future__ import annotations

from ..self_model.models import PersonRegistry, SelfState
from ..world_model.models import WorldState
from .emotional_dft import TerrainSnapshot
from .models import ValueState


def compute_value_state(
    world_state: WorldState,
    self_state: SelfState,
    person_registry: PersonRegistry,
) -> ValueState:
    overload = min(1.0, self_state.task_load * 0.6 + self_state.fatigue * 0.4)
    exploration = max(0.0, self_state.curiosity * 0.7 - world_state.uncertainty * 0.2)
    trust = max(0.0, self_state.trust * 0.7 + (1.0 - person_registry.uncertainty) * 0.2)
    terrain_energy = min(
        1.0,
        self_state.arousal * 0.25
        + self_state.social_tension * 0.2
        + overload * 0.2
        + world_state.uncertainty * 0.2
        + max(0.0, 1.0 - trust) * 0.15,
    )
    gradient_norm = min(
        1.0,
        world_state.uncertainty * 0.4
        + overload * 0.35
        + self_state.social_tension * 0.25,
    )
    max_curvature = min(
        1.0,
        world_state.uncertainty * 0.5
        + max(0.0, overload - exploration) * 0.3
        + max(0.0, 1.0 - trust) * 0.2,
    )
    return ValueState(
        danger_score=round(world_state.uncertainty * 0.5, 4),
        trust_preservation_score=round(trust, 4),
        overload_risk=round(overload, 4),
        exploration_drive=round(exploration, 4),
        repair_urgency=round(max(0.0, self_state.social_tension * 0.5), 4),
        terrain_snapshot=TerrainSnapshot(
            state_energy=round(terrain_energy, 4),
            gradient_norm=round(gradient_norm, 4),
            max_curvature=round(max_curvature, 4),
        ),
        terrain_energy=round(terrain_energy, 4),
        value_axes={
            "danger": round(world_state.uncertainty * 0.5, 4),
            "trust": round(trust, 4),
            "exploration": round(exploration, 4),
            "terrain_energy": round(terrain_energy, 4),
        },
    )
