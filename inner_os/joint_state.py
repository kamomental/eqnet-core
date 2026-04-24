from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class JointStateFrame:
    """自己と相手のあいだで立ち上がる共同状態の短い履歴。"""

    step: int = 0
    dominant_mode: str = "ambient"
    shared_tension: float = 0.0
    shared_delight: float = 0.0
    repair_readiness: float = 0.0
    common_ground: float = 0.0
    joint_attention: float = 0.0
    mutual_room: float = 0.0
    coupling_strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "dominant_mode": self.dominant_mode,
            "shared_tension": round(self.shared_tension, 4),
            "shared_delight": round(self.shared_delight, 4),
            "repair_readiness": round(self.repair_readiness, 4),
            "common_ground": round(self.common_ground, 4),
            "joint_attention": round(self.joint_attention, 4),
            "mutual_room": round(self.mutual_room, 4),
            "coupling_strength": round(self.coupling_strength, 4),
        }


@dataclass(frozen=True)
class JointState:
    """shared_tension / shared_delight / common_ground を束ねる canonical latent。"""

    shared_tension: float = 0.0
    shared_delight: float = 0.0
    repair_readiness: float = 0.0
    common_ground: float = 0.0
    joint_attention: float = 0.0
    mutual_room: float = 0.0
    coupling_strength: float = 0.0
    dominant_mode: str = "ambient"
    trace: tuple[JointStateFrame, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_tension": round(self.shared_tension, 4),
            "shared_delight": round(self.shared_delight, 4),
            "repair_readiness": round(self.repair_readiness, 4),
            "common_ground": round(self.common_ground, 4),
            "joint_attention": round(self.joint_attention, 4),
            "mutual_room": round(self.mutual_room, 4),
            "coupling_strength": round(self.coupling_strength, 4),
            "dominant_mode": self.dominant_mode,
            "trace": [frame.to_dict() for frame in self.trace],
        }

    def to_packet_axes(
        self,
        previous: Mapping[str, Any] | "JointState" | None = None,
    ) -> dict[str, dict[str, float]]:
        previous_state = coerce_joint_state(previous)
        current_axes = _axis_values(self)
        previous_axes = _axis_values(previous_state)
        return {
            axis_name: {
                "value": round(axis_value, 4),
                "delta": round(axis_value - previous_axes.get(axis_name, 0.0), 4),
            }
            for axis_name, axis_value in current_axes.items()
        }


def derive_joint_state(
    *,
    previous_state: Mapping[str, Any] | JointState | None = None,
    shared_moment_state: Mapping[str, Any] | None = None,
    listener_action_state: Mapping[str, Any] | None = None,
    live_engagement_state: Mapping[str, Any] | None = None,
    meaning_update_state: Mapping[str, Any] | None = None,
    organism_state: Mapping[str, Any] | None = None,
    external_field_state: Mapping[str, Any] | None = None,
    terrain_dynamics_state: Mapping[str, Any] | None = None,
    memory_dynamics_state: Mapping[str, Any] | None = None,
    subjective_scene_state: Mapping[str, Any] | None = None,
    self_other_attribution_state: Mapping[str, Any] | None = None,
    shared_presence_state: Mapping[str, Any] | None = None,
    trace_limit: int = 8,
) -> JointState:
    previous = coerce_joint_state(previous_state)
    shared_moment = dict(shared_moment_state or {})
    listener_action = dict(listener_action_state or {})
    live_engagement = dict(live_engagement_state or {})
    meaning_update = dict(meaning_update_state or {})
    organism = dict(organism_state or {})
    external_field = dict(external_field_state or {})
    terrain_dynamics = dict(terrain_dynamics_state or {})
    memory_dynamics = dict(memory_dynamics_state or {})
    subjective_scene = dict(subjective_scene_state or {})
    self_other_attribution = dict(self_other_attribution_state or {})
    shared_presence = dict(shared_presence_state or {})

    if (
        not shared_moment
        and not listener_action
        and not live_engagement
        and not meaning_update
        and not organism
        and not external_field
        and not terrain_dynamics
        and not memory_dynamics
        and not subjective_scene
        and not self_other_attribution
        and not shared_presence
    ):
        return previous

    moment_kind = _text(shared_moment.get("moment_kind"))
    relation_update = _text(meaning_update.get("relation_update"))
    world_update = _text(meaning_update.get("world_update"))
    preserve_guard = _text(meaning_update.get("preserve_guard"))
    listener_mode = _text(listener_action.get("state"))
    live_mode = _text(live_engagement.get("state"))
    dominant_relation_type = _text(memory_dynamics.get("dominant_relation_type"))
    relation_generation_mode = _text(memory_dynamics.get("relation_generation_mode"))
    dominant_causal_type = _text(memory_dynamics.get("dominant_causal_type"))
    relation_meta_type = ""
    raw_meta_relations = memory_dynamics.get("meta_relations")
    if isinstance(raw_meta_relations, list) and raw_meta_relations and isinstance(raw_meta_relations[0], Mapping):
        relation_meta_type = _text(raw_meta_relations[0].get("meta_type"))
    dominant_attribution = _text(self_other_attribution.get("dominant_attribution"))

    shared_delight = _clamp01(
        _float01(shared_moment.get("score")) * 0.26
        + _float01(shared_moment.get("jointness")) * 0.18
        + _float01(shared_moment.get("afterglow")) * 0.18
        + _float01(listener_action.get("laughter_room")) * 0.16
        + _float01(listener_action.get("acknowledgement_room")) * 0.08
        + _float01(organism.get("play_window")) * 0.08
        + _float01(organism.get("attunement")) * 0.06
        + (0.08 if moment_kind == "laugh" else 0.05 if moment_kind == "relief" else 0.0)
        + (0.08 if relation_update == "shared_smile_window" else 0.0)
        + (0.06 if dominant_relation_type == "same_anchor" else 0.0)
        + (0.05 if dominant_causal_type in {"reframed_by", "amplified_by"} else 0.0)
        + (0.04 if relation_meta_type == "reinforces" else 0.0)
        + (0.04 if listener_mode == "warm_laugh_ack" else 0.0)
        + _float01(subjective_scene.get("comfort")) * 0.04
        + _float01(shared_presence.get("co_presence")) * 0.04
        + _float01(shared_presence.get("shared_scene_salience")) * 0.04
        + (0.04 if dominant_attribution == "shared" else 0.0)
    )
    shared_tension = _clamp01(
        _float01(organism.get("protective_tension")) * 0.28
        + _float01(external_field.get("social_pressure")) * 0.18
        + _float01(external_field.get("ambiguity_load")) * 0.12
        + _float01(terrain_dynamics.get("barrier_height")) * 0.16
        + _float01(memory_dynamics.get("memory_tension")) * 0.12
        + (0.08 if preserve_guard else 0.0)
        + (0.08 if dominant_relation_type == "unfinished_carry" else 0.0)
        + (0.08 if dominant_causal_type == "reopened_by" else 0.0)
        + (0.06 if dominant_causal_type == "suppressed_by" else 0.0)
        + (0.06 if relation_meta_type == "competes_with" else 0.0)
        + _float01(subjective_scene.get("tension")) * 0.08
        + max(0.0, 1.0 - _float01(shared_presence.get("boundary_stability"))) * 0.08
        + _float01(self_other_attribution.get("unknown_likelihood")) * 0.06
        - _float01(shared_moment.get("afterglow")) * 0.1
        - _float01(listener_action.get("acknowledgement_room")) * 0.06
    )
    repair_readiness = _clamp01(
        _float01(organism.get("attunement")) * 0.22
        + _float01(listener_action.get("acknowledgement_room")) * 0.16
        + _float01(listener_action.get("filler_room")) * 0.06
        + _float01(live_engagement.get("comment_pickup_room")) * 0.1
        + _float01(live_engagement.get("riff_room")) * 0.08
        + _float01(terrain_dynamics.get("recovery_gradient")) * 0.18
        + _float01(external_field.get("safety_envelope")) * 0.08
        + (0.1 if relation_update in {"shared_smile_window", "repair_window", "repairing_contact"} else 0.0)
        + (0.08 if dominant_relation_type == "unfinished_carry" else 0.0)
        + (0.08 if dominant_causal_type in {"enabled_by", "reopened_by"} else 0.0)
        + (0.06 if live_mode in {"pickup_comment", "riff_with_comment"} else 0.0)
        + _float01(shared_presence.get("other_projection_receptivity")) * 0.1
        + _float01(subjective_scene.get("comfort")) * 0.04
    )
    common_ground = _clamp01(
        _float01(shared_moment.get("jointness")) * 0.18
        + _float01(organism.get("relation_pull")) * 0.18
        + _float01(organism.get("attunement")) * 0.08
        + _float01(external_field.get("continuity_pull")) * 0.18
        + _float01(terrain_dynamics.get("basin_pull")) * 0.14
        + _float01(memory_dynamics.get("monument_salience")) * 0.08
        + _float01(memory_dynamics.get("activation_confidence")) * 0.06
        + (0.08 if relation_update else 0.0)
        + (0.08 if world_update else 0.0)
        + (0.08 if dominant_relation_type in {"same_anchor", "recurrent_association"} else 0.0)
        + (0.08 if dominant_causal_type in {"enabled_by", "reframed_by"} else 0.0)
        + (0.06 if relation_meta_type == "reinforces" else 0.0)
        + _float01(shared_presence.get("co_presence")) * 0.12
        + _float01(subjective_scene.get("shared_scene_potential")) * 0.08
        + _float01(self_other_attribution.get("shared_likelihood")) * 0.08
        + (0.06 if dominant_attribution == "shared" else 0.0)
    )
    joint_attention = _clamp01(
        _float01(shared_moment.get("jointness")) * 0.22
        + _float01(live_engagement.get("comment_pickup_room")) * 0.18
        + _float01(live_engagement.get("topic_seed_room")) * 0.08
        + _float01(organism.get("attunement")) * 0.16
        + _float01(listener_action.get("acknowledgement_room")) * 0.14
        + _float01(external_field.get("continuity_pull")) * 0.08
        + (0.08 if dominant_relation_type == "cross_context_bridge" else 0.0)
        + (0.06 if dominant_causal_type == "triggered_by" else 0.0)
        + (0.04 if relation_meta_type == "specializes" else 0.0)
        + (0.06 if live_mode in {"pickup_comment", "riff_with_comment", "seed_topic"} else 0.0)
        + _float01(shared_presence.get("shared_attention")) * 0.18
        + _float01(subjective_scene.get("frontal_alignment")) * 0.08
        + _float01(subjective_scene.get("motion_salience")) * 0.04
    )
    mutual_room = _clamp01(
        _float01(listener_action.get("filler_room")) * 0.2
        + _float01(listener_action.get("acknowledgement_room")) * 0.16
        + _float01(live_engagement.get("riff_room")) * 0.16
        + _float01(live_engagement.get("comment_pickup_room")) * 0.1
        + _float01(organism.get("expressive_readiness")) * 0.16
        + _float01(organism.get("play_window")) * 0.1
        + _float01(external_field.get("safety_envelope")) * 0.06
        + _float01(shared_presence.get("other_projection_receptivity")) * 0.12
        + _float01(subjective_scene.get("workspace_proximity")) * 0.06
        - _float01(organism.get("protective_tension")) * 0.14
        - _float01(external_field.get("social_pressure")) * 0.08
        - _float01(subjective_scene.get("tension")) * 0.06
    )
    coupling_strength = _clamp01(
        common_ground * 0.28
        + joint_attention * 0.2
        + repair_readiness * 0.14
        + shared_delight * 0.16
        + _float01(organism.get("attunement")) * 0.14
        + _float01(terrain_dynamics.get("recovery_gradient")) * 0.06
        + _float01(memory_dynamics.get("activation_confidence")) * 0.06
        + (0.06 if relation_generation_mode == "ignited" else 0.0)
        + _float01(shared_presence.get("co_presence")) * 0.12
        + _float01(shared_presence.get("self_projection_strength")) * 0.08
        + _float01(self_other_attribution.get("shared_likelihood")) * 0.06
        + _float01(shared_presence.get("boundary_stability")) * 0.04
        - shared_tension * 0.12
    )

    shared_delight = _carry(previous.shared_delight, shared_delight, previous_state, 0.18)
    shared_tension = _carry(previous.shared_tension, shared_tension, previous_state, 0.24)
    repair_readiness = _carry(previous.repair_readiness, repair_readiness, previous_state, 0.2)
    common_ground = _carry(previous.common_ground, common_ground, previous_state, 0.22)
    joint_attention = _carry(previous.joint_attention, joint_attention, previous_state, 0.2)
    mutual_room = _carry(previous.mutual_room, mutual_room, previous_state, 0.18)
    coupling_strength = _carry(previous.coupling_strength, coupling_strength, previous_state, 0.2)

    dominant_mode = _dominant_mode(
        shared_tension=shared_tension,
        shared_delight=shared_delight,
        repair_readiness=repair_readiness,
        common_ground=common_ground,
        joint_attention=joint_attention,
        mutual_room=mutual_room,
    )
    step = previous.trace[-1].step + 1 if previous.trace else 1
    trace = list(previous.trace[-max(0, trace_limit - 1) :]) if trace_limit > 0 else []
    trace.append(
        JointStateFrame(
            step=step,
            dominant_mode=dominant_mode,
            shared_tension=shared_tension,
            shared_delight=shared_delight,
            repair_readiness=repair_readiness,
            common_ground=common_ground,
            joint_attention=joint_attention,
            mutual_room=mutual_room,
            coupling_strength=coupling_strength,
        )
    )
    return JointState(
        shared_tension=shared_tension,
        shared_delight=shared_delight,
        repair_readiness=repair_readiness,
        common_ground=common_ground,
        joint_attention=joint_attention,
        mutual_room=mutual_room,
        coupling_strength=coupling_strength,
        dominant_mode=dominant_mode,
        trace=tuple(trace[-trace_limit:] if trace_limit > 0 else ()),
    )


def coerce_joint_state(
    value: Mapping[str, Any] | JointState | None,
) -> JointState:
    if isinstance(value, JointState):
        return value
    payload = dict(value or {})
    trace_items: list[JointStateFrame] = []
    for item in payload.get("trace") or ():
        if isinstance(item, JointStateFrame):
            trace_items.append(item)
        elif isinstance(item, Mapping):
            trace_items.append(
                JointStateFrame(
                    step=int(_float(item.get("step"), 0.0)),
                    dominant_mode=_text(item.get("dominant_mode")) or "ambient",
                    shared_tension=_float01(item.get("shared_tension")),
                    shared_delight=_float01(item.get("shared_delight")),
                    repair_readiness=_float01(item.get("repair_readiness")),
                    common_ground=_float01(item.get("common_ground")),
                    joint_attention=_float01(item.get("joint_attention")),
                    mutual_room=_float01(item.get("mutual_room")),
                    coupling_strength=_float01(item.get("coupling_strength")),
                )
            )
    return JointState(
        shared_tension=_float01(payload.get("shared_tension")),
        shared_delight=_float01(payload.get("shared_delight")),
        repair_readiness=_float01(payload.get("repair_readiness")),
        common_ground=_float01(payload.get("common_ground")),
        joint_attention=_float01(payload.get("joint_attention")),
        mutual_room=_float01(payload.get("mutual_room")),
        coupling_strength=_float01(payload.get("coupling_strength")),
        dominant_mode=_text(payload.get("dominant_mode")) or "ambient",
        trace=tuple(trace_items[-8:]),
    )


def _axis_values(state: JointState) -> dict[str, float]:
    return {
        "delight": _clamp01(state.shared_delight),
        "tension": _clamp01(state.shared_tension),
        "repair": _clamp01(state.repair_readiness),
        "ground": _clamp01(state.common_ground),
        "attention": _clamp01(state.joint_attention),
        "coupling": _clamp01(state.coupling_strength),
    }


def _carry(
    previous_value: float,
    current_value: float,
    previous_state: Mapping[str, Any] | JointState | None,
    carry_ratio: float,
) -> float:
    if previous_state is None:
        return _clamp01(current_value)
    if isinstance(previous_state, Mapping) and not previous_state:
        return _clamp01(current_value)
    return _clamp01(previous_value * carry_ratio + current_value * (1.0 - carry_ratio))


def _dominant_mode(
    *,
    shared_tension: float,
    shared_delight: float,
    repair_readiness: float,
    common_ground: float,
    joint_attention: float,
    mutual_room: float,
) -> str:
    if shared_delight >= max(shared_tension, repair_readiness, common_ground, 0.54) and mutual_room >= 0.38:
        return "delighted_jointness"
    if repair_readiness >= max(shared_tension, shared_delight, common_ground, 0.5):
        return "repair_attunement"
    if shared_tension >= max(shared_delight, repair_readiness, 0.54):
        return "strained_jointness"
    if common_ground >= 0.46 or joint_attention >= 0.48:
        return "shared_attention"
    return "ambient"


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


def _float01(value: Any) -> float:
    return _clamp01(_float(value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
