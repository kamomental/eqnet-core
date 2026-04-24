from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SubjectiveSceneState:
    """観測空間を自己基準の lived space として束ねた主観場。"""

    dominant_zone: str = "ambient_field"
    anchor_frame: str = "ambient_margin"
    egocentric_closeness: float = 0.0
    workspace_proximity: float = 0.0
    frontal_alignment: float = 0.0
    motion_salience: float = 0.0
    self_related_salience: float = 0.0
    shared_scene_potential: float = 0.0
    familiarity: float = 0.0
    comfort: float = 0.0
    curiosity: float = 0.0
    tension: float = 0.0
    uncertainty: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_zone": self.dominant_zone,
            "anchor_frame": self.anchor_frame,
            "egocentric_closeness": round(self.egocentric_closeness, 4),
            "workspace_proximity": round(self.workspace_proximity, 4),
            "frontal_alignment": round(self.frontal_alignment, 4),
            "motion_salience": round(self.motion_salience, 4),
            "self_related_salience": round(self.self_related_salience, 4),
            "shared_scene_potential": round(self.shared_scene_potential, 4),
            "familiarity": round(self.familiarity, 4),
            "comfort": round(self.comfort, 4),
            "curiosity": round(self.curiosity, 4),
            "tension": round(self.tension, 4),
            "uncertainty": round(self.uncertainty, 4),
        }


def derive_subjective_scene_state(
    *,
    previous_state: Mapping[str, Any] | SubjectiveSceneState | None = None,
    camera_observation: Mapping[str, Any] | None = None,
    world_state: Mapping[str, Any] | None = None,
    self_state: Mapping[str, Any] | None = None,
    external_field_state: Mapping[str, Any] | None = None,
) -> SubjectiveSceneState:
    previous = coerce_subjective_scene_state(previous_state)
    observation = dict(camera_observation or {})
    world = dict(world_state or {})
    self_payload = dict(self_state or {})
    external_field = dict(external_field_state or {})

    if not observation and not world and not self_payload and not external_field:
        return previous

    egocentric_closeness = _clamp01(
        _float01(_pick(observation, "egocentric_closeness", "distance_closeness", "distance_score")) * 0.58
        + _float01(_pick(observation, "distance_band_closeness", "near_score")) * 0.22
        + _float01(_pick(observation, "visual_centering", "centrality")) * 0.12
        + _float01(_pick(world, "spatial_nearness", "proximity")) * 0.08
    )
    workspace_proximity = _clamp01(
        _float01(_pick(observation, "workspace_proximity", "workspace_overlap", "task_surface_overlap")) * 0.68
        + egocentric_closeness * 0.18
        + _float01(self_payload.get("task_load")) * 0.06
        + _float01(_pick(world, "task_surface_salience", "task_relevance")) * 0.08
    )
    frontal_alignment = _clamp01(
        _float01(_pick(observation, "frontal_alignment", "frontality", "gaze_alignment")) * 0.66
        + _float01(_pick(observation, "visual_centering", "centrality")) * 0.16
        + _float01(_pick(observation, "perspective_alignment", "perspective_match")) * 0.18
    )
    motion_salience = _clamp01(
        _float01(_pick(observation, "motion_salience", "movement_score", "temporal_change")) * 0.7
        + _float01(_pick(observation, "audio_shift_salience", "source_shift")) * 0.12
        + _float01(external_field.get("novelty")) * 0.1
        + _float01(_pick(world, "scene_change_salience", "change_pressure")) * 0.08
    )
    self_related_salience = _clamp01(
        _float01(_pick(observation, "self_reference_score", "self_related_score", "body_relevance")) * 0.62
        + workspace_proximity * 0.12
        + frontal_alignment * 0.08
        + _float01(_pick(world, "self_related_affordance", "ownership_hint")) * 0.1
        + _float01(self_payload.get("curiosity")) * 0.08
    )
    shared_scene_potential = _clamp01(
        _float01(_pick(observation, "shared_reference_score", "co_presence_hint", "shared_attention_hint")) * 0.54
        + frontal_alignment * 0.14
        + workspace_proximity * 0.08
        + motion_salience * 0.08
        + _float01(external_field.get("continuity_pull")) * 0.08
        + _float01(_pick(world, "social_scene_pull", "social_relation_pull")) * 0.08
    )

    familiarity = _clamp01(
        _float01(_pick(observation, "familiarity_hint", "known_object_score")) * 0.52
        + self_related_salience * 0.12
        + _float01(_pick(world, "scene_familiarity", "continuity_hint")) * 0.18
        + _float01(external_field.get("continuity_pull")) * 0.18
    )
    comfort = _clamp01(
        _float01(_pick(observation, "comfort_hint", "safety_hint")) * 0.46
        + familiarity * 0.16
        + _float01(self_payload.get("safety_margin")) * 0.2
        + _float01(external_field.get("safety_envelope")) * 0.18
    )
    curiosity = _clamp01(
        _float01(self_payload.get("curiosity")) * 0.42
        + motion_salience * 0.18
        + max(0.0, 1.0 - familiarity) * 0.16
        + _float01(external_field.get("novelty")) * 0.16
        + shared_scene_potential * 0.08
    )
    tension = _clamp01(
        _float01(_pick(observation, "tension_hint", "intrusion_hint")) * 0.38
        + max(0.0, 1.0 - comfort) * 0.18
        + _float01(self_payload.get("social_tension")) * 0.18
        + _float01(external_field.get("social_pressure")) * 0.14
        + max(0.0, 1.0 - workspace_proximity) * 0.04
        + max(0.0, 1.0 - frontal_alignment) * 0.08
    )
    uncertainty = _clamp01(
        _float01(_pick(observation, "uncertainty", "scene_uncertainty")) * 0.54
        + _float01(_pick(world, "uncertainty")) * 0.18
        + _float01(self_payload.get("uncertainty")) * 0.18
        + max(0.0, 1.0 - shared_scene_potential) * 0.1
    )

    egocentric_closeness = _carry(previous.egocentric_closeness, egocentric_closeness, previous_state, 0.18)
    workspace_proximity = _carry(previous.workspace_proximity, workspace_proximity, previous_state, 0.2)
    frontal_alignment = _carry(previous.frontal_alignment, frontal_alignment, previous_state, 0.2)
    motion_salience = _carry(previous.motion_salience, motion_salience, previous_state, 0.16)
    self_related_salience = _carry(previous.self_related_salience, self_related_salience, previous_state, 0.2)
    shared_scene_potential = _carry(previous.shared_scene_potential, shared_scene_potential, previous_state, 0.18)
    familiarity = _carry(previous.familiarity, familiarity, previous_state, 0.24)
    comfort = _carry(previous.comfort, comfort, previous_state, 0.22)
    curiosity = _carry(previous.curiosity, curiosity, previous_state, 0.18)
    tension = _carry(previous.tension, tension, previous_state, 0.22)
    uncertainty = _carry(previous.uncertainty, uncertainty, previous_state, 0.2)

    dominant_zone = _dominant_zone(
        egocentric_closeness=egocentric_closeness,
        workspace_proximity=workspace_proximity,
        motion_salience=motion_salience,
        shared_scene_potential=shared_scene_potential,
    )
    anchor_frame = _anchor_frame(
        self_related_salience=self_related_salience,
        shared_scene_potential=shared_scene_potential,
        frontal_alignment=frontal_alignment,
    )
    return SubjectiveSceneState(
        dominant_zone=dominant_zone,
        anchor_frame=anchor_frame,
        egocentric_closeness=egocentric_closeness,
        workspace_proximity=workspace_proximity,
        frontal_alignment=frontal_alignment,
        motion_salience=motion_salience,
        self_related_salience=self_related_salience,
        shared_scene_potential=shared_scene_potential,
        familiarity=familiarity,
        comfort=comfort,
        curiosity=curiosity,
        tension=tension,
        uncertainty=uncertainty,
    )


def coerce_subjective_scene_state(
    value: Mapping[str, Any] | SubjectiveSceneState | None,
) -> SubjectiveSceneState:
    if isinstance(value, SubjectiveSceneState):
        return value
    payload = dict(value or {})
    return SubjectiveSceneState(
        dominant_zone=_text(payload.get("dominant_zone")) or "ambient_field",
        anchor_frame=_text(payload.get("anchor_frame")) or "ambient_margin",
        egocentric_closeness=_float01(payload.get("egocentric_closeness")),
        workspace_proximity=_float01(payload.get("workspace_proximity")),
        frontal_alignment=_float01(payload.get("frontal_alignment")),
        motion_salience=_float01(payload.get("motion_salience")),
        self_related_salience=_float01(payload.get("self_related_salience")),
        shared_scene_potential=_float01(payload.get("shared_scene_potential")),
        familiarity=_float01(payload.get("familiarity")),
        comfort=_float01(payload.get("comfort")),
        curiosity=_float01(payload.get("curiosity")),
        tension=_float01(payload.get("tension")),
        uncertainty=_float01(payload.get("uncertainty", 1.0)),
    )


def _dominant_zone(
    *,
    egocentric_closeness: float,
    workspace_proximity: float,
    motion_salience: float,
    shared_scene_potential: float,
) -> str:
    if workspace_proximity >= 0.58:
        return "near_working_field"
    if egocentric_closeness >= 0.6:
        return "near_personal_space"
    if shared_scene_potential >= 0.56:
        return "shared_front_space"
    if motion_salience >= 0.54:
        return "moving_edge"
    return "ambient_field"


def _anchor_frame(
    *,
    self_related_salience: float,
    shared_scene_potential: float,
    frontal_alignment: float,
) -> str:
    if self_related_salience >= 0.56:
        return "self_margin"
    if shared_scene_potential >= 0.54:
        return "shared_margin"
    if frontal_alignment >= 0.52:
        return "front_field"
    return "ambient_margin"


def _pick(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _carry(
    previous_value: float,
    current_value: float,
    previous_state: Mapping[str, Any] | SubjectiveSceneState | None,
    carry_ratio: float,
) -> float:
    if previous_state is None:
        return _clamp01(current_value)
    if isinstance(previous_state, Mapping) and not previous_state:
        return _clamp01(current_value)
    return _clamp01(previous_value * carry_ratio + current_value * (1.0 - carry_ratio))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return float(numeric)


def _float01(value: Any, default: float = 0.0) -> float:
    return _clamp01(_float(value, default))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
