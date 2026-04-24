from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SharedPresenceState:
    """同じ場に居る感覚を主観空間から束ねた共在状態。"""

    dominant_mode: str = "ambient_presence"
    co_presence: float = 0.0
    shared_attention: float = 0.0
    shared_scene_salience: float = 0.0
    self_projection_strength: float = 0.0
    other_projection_receptivity: float = 0.0
    boundary_stability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_mode": self.dominant_mode,
            "co_presence": round(self.co_presence, 4),
            "shared_attention": round(self.shared_attention, 4),
            "shared_scene_salience": round(self.shared_scene_salience, 4),
            "self_projection_strength": round(self.self_projection_strength, 4),
            "other_projection_receptivity": round(self.other_projection_receptivity, 4),
            "boundary_stability": round(self.boundary_stability, 4),
        }


def derive_shared_presence_state(
    *,
    previous_state: Mapping[str, Any] | SharedPresenceState | None = None,
    subjective_scene_state: Mapping[str, Any] | None = None,
    self_other_attribution_state: Mapping[str, Any] | None = None,
    joint_state: Mapping[str, Any] | None = None,
    organism_state: Mapping[str, Any] | None = None,
    external_field_state: Mapping[str, Any] | None = None,
) -> SharedPresenceState:
    previous = coerce_shared_presence_state(previous_state)
    subjective_scene = dict(subjective_scene_state or {})
    attribution = dict(self_other_attribution_state or {})
    joint = dict(joint_state or {})
    organism = dict(organism_state or {})
    external_field = dict(external_field_state or {})

    if not subjective_scene and not attribution and not joint and not organism and not external_field:
        return previous

    co_presence = _clamp01(
        _float01(attribution.get("shared_likelihood")) * 0.34
        + _float01(subjective_scene.get("shared_scene_potential")) * 0.24
        + _float01(joint.get("joint_attention")) * 0.1
        + _float01(joint.get("common_ground")) * 0.08
        + _float01(external_field.get("continuity_pull")) * 0.08
        + _float01(subjective_scene.get("frontal_alignment")) * 0.08
        + _float01(subjective_scene.get("workspace_proximity")) * 0.08
    )
    shared_attention = _clamp01(
        _float01(subjective_scene.get("shared_scene_potential")) * 0.28
        + _float01(subjective_scene.get("frontal_alignment")) * 0.18
        + _float01(subjective_scene.get("motion_salience")) * 0.12
        + _float01(attribution.get("perspective_match")) * 0.16
        + _float01(joint.get("joint_attention")) * 0.14
        + _float01(joint.get("common_ground")) * 0.12
    )
    shared_scene_salience = _clamp01(
        _float01(subjective_scene.get("shared_scene_potential")) * 0.3
        + _float01(subjective_scene.get("workspace_proximity")) * 0.12
        + _float01(subjective_scene.get("motion_salience")) * 0.18
        + _float01(attribution.get("shared_likelihood")) * 0.16
        + _float01(subjective_scene.get("familiarity")) * 0.1
        + _float01(subjective_scene.get("comfort")) * 0.14
    )
    self_projection_strength = _clamp01(
        _float01(subjective_scene.get("self_related_salience")) * 0.28
        + _float01(attribution.get("self_likelihood")) * 0.18
        + _float01(attribution.get("perspective_match")) * 0.14
        + _float01(attribution.get("sensorimotor_consistency")) * 0.18
        + _float01(organism.get("attunement")) * 0.12
        + co_presence * 0.1
    )
    other_projection_receptivity = _clamp01(
        _float01(attribution.get("shared_likelihood")) * 0.2
        + _float01(attribution.get("other_likelihood")) * 0.14
        + _float01(subjective_scene.get("comfort")) * 0.18
        + _float01(external_field.get("safety_envelope")) * 0.14
        + _float01(joint.get("mutual_room")) * 0.14
        + _float01(organism.get("attunement")) * 0.1
        + co_presence * 0.1
    )
    boundary_stability = _clamp01(
        _float01(external_field.get("safety_envelope")) * 0.24
        + _float01(organism.get("grounding")) * 0.18
        + _float01(attribution.get("attribution_confidence")) * 0.18
        + _float01(subjective_scene.get("comfort")) * 0.12
        + max(0.0, 1.0 - _float01(attribution.get("unknown_likelihood"))) * 0.12
        + max(0.0, 1.0 - _float01(subjective_scene.get("tension"))) * 0.16
    )

    co_presence = _carry(previous.co_presence, co_presence, previous_state, 0.18)
    shared_attention = _carry(previous.shared_attention, shared_attention, previous_state, 0.16)
    shared_scene_salience = _carry(previous.shared_scene_salience, shared_scene_salience, previous_state, 0.18)
    self_projection_strength = _carry(previous.self_projection_strength, self_projection_strength, previous_state, 0.18)
    other_projection_receptivity = _carry(previous.other_projection_receptivity, other_projection_receptivity, previous_state, 0.18)
    boundary_stability = _carry(previous.boundary_stability, boundary_stability, previous_state, 0.2)

    dominant_mode = _dominant_mode(
        co_presence=co_presence,
        shared_attention=shared_attention,
        self_projection_strength=self_projection_strength,
        other_projection_receptivity=other_projection_receptivity,
        boundary_stability=boundary_stability,
    )
    return SharedPresenceState(
        dominant_mode=dominant_mode,
        co_presence=co_presence,
        shared_attention=shared_attention,
        shared_scene_salience=shared_scene_salience,
        self_projection_strength=self_projection_strength,
        other_projection_receptivity=other_projection_receptivity,
        boundary_stability=boundary_stability,
    )


def coerce_shared_presence_state(
    value: Mapping[str, Any] | SharedPresenceState | None,
) -> SharedPresenceState:
    if isinstance(value, SharedPresenceState):
        return value
    payload = dict(value or {})
    return SharedPresenceState(
        dominant_mode=_text(payload.get("dominant_mode")) or "ambient_presence",
        co_presence=_float01(payload.get("co_presence")),
        shared_attention=_float01(payload.get("shared_attention")),
        shared_scene_salience=_float01(payload.get("shared_scene_salience")),
        self_projection_strength=_float01(payload.get("self_projection_strength")),
        other_projection_receptivity=_float01(payload.get("other_projection_receptivity")),
        boundary_stability=_float01(payload.get("boundary_stability")),
    )


def _dominant_mode(
    *,
    co_presence: float,
    shared_attention: float,
    self_projection_strength: float,
    other_projection_receptivity: float,
    boundary_stability: float,
) -> str:
    if co_presence >= 0.5 and shared_attention >= 0.46 and boundary_stability >= 0.44:
        return "inhabited_shared_space"
    if self_projection_strength >= 0.46 and other_projection_receptivity >= 0.42:
        return "soft_projection"
    if boundary_stability <= 0.34:
        return "guarded_boundary"
    return "ambient_presence"


def _carry(
    previous_value: float,
    current_value: float,
    previous_state: Mapping[str, Any] | SharedPresenceState | None,
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
