from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SceneState:
    place_mode: str = "unspecified"
    privacy_level: float = 0.5
    social_topology: str = "one_to_one"
    task_phase: str = "ongoing"
    temporal_phase: str = "ongoing"
    norm_pressure: float = 0.0
    safety_margin: float = 0.5
    environmental_load: float = 0.0
    mobility_context: str = "stationary"
    scene_family: str = "co_present"
    scene_tags: tuple[str, ...] = ()


def derive_scene_state(
    *,
    place_mode: str = "unspecified",
    privacy_level: float = 0.5,
    social_topology: str = "one_to_one",
    task_phase: str = "ongoing",
    temporal_phase: str = "ongoing",
    norm_pressure: float = 0.0,
    safety_margin: float = 0.5,
    environmental_load: float = 0.0,
    mobility_context: str = "stationary",
    current_risks: Sequence[str] = (),
    active_goals: Sequence[str] = (),
) -> SceneState:
    privacy_level = _clamp01(privacy_level)
    norm_pressure = _clamp01(norm_pressure)
    safety_margin = _clamp01(safety_margin)
    environmental_load = _clamp01(environmental_load)

    tags: list[str] = []
    if privacy_level <= 0.28:
        tags.append("public")
    elif privacy_level >= 0.72:
        tags.append("private")

    if norm_pressure >= 0.62:
        tags.append("high_norm")
    if environmental_load >= 0.58:
        tags.append("high_load")
    if safety_margin <= 0.34:
        tags.append("low_safety")
    if social_topology in {"group_present", "hierarchical", "public_visible"}:
        tags.append("socially_exposed")
    if task_phase:
        tags.append(f"task:{task_phase}")
    if temporal_phase:
        tags.append(f"time:{temporal_phase}")
    if mobility_context and mobility_context != "stationary":
        tags.append(f"mobility:{mobility_context}")
    for risk in current_risks:
        tags.append(f"risk:{risk}")
    for goal in active_goals:
        tags.append(f"goal:{goal}")

    scene_family = "co_present"
    if "danger" in current_risks or safety_margin <= 0.24:
        scene_family = "guarded_boundary"
    elif "repair" in active_goals:
        scene_family = "repair_window"
    elif task_phase in {"coordination", "co_work", "shared_task"}:
        scene_family = "shared_world"
    elif norm_pressure >= 0.62 or privacy_level <= 0.28:
        scene_family = "reverent_distance"
    elif privacy_level >= 0.72 and safety_margin >= 0.56:
        scene_family = "attuned_presence"

    return SceneState(
        place_mode=place_mode,
        privacy_level=round(privacy_level, 4),
        social_topology=social_topology,
        task_phase=task_phase,
        temporal_phase=temporal_phase,
        norm_pressure=round(norm_pressure, 4),
        safety_margin=round(safety_margin, 4),
        environmental_load=round(environmental_load, 4),
        mobility_context=mobility_context,
        scene_family=scene_family,
        scene_tags=tuple(sorted(set(tags))),
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
