from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return bool(value)


@dataclass(frozen=True)
class SurfaceContextPacket(MappingABC[str, Any]):
    conversation_phase: str = ""
    shared_core: dict[str, Any] = field(default_factory=dict)
    response_role: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    surface_profile: dict[str, Any] = field(default_factory=dict)
    source_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_phase": self.conversation_phase,
            "shared_core": dict(self.shared_core),
            "response_role": dict(self.response_role),
            "constraints": dict(self.constraints),
            "surface_profile": dict(self.surface_profile),
            "source_state": dict(self.source_state),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def coerce_surface_context_packet(
    value: Mapping[str, Any] | SurfaceContextPacket | None,
) -> SurfaceContextPacket:
    if isinstance(value, SurfaceContextPacket):
        return value
    packet = dict(value or {})
    return SurfaceContextPacket(
        conversation_phase=_clean_text(packet.get("conversation_phase")),
        shared_core=dict(packet.get("shared_core") or {}),
        response_role=dict(packet.get("response_role") or {}),
        constraints=dict(packet.get("constraints") or {}),
        surface_profile=dict(packet.get("surface_profile") or {}),
        source_state=dict(packet.get("source_state") or {}),
    )


def build_surface_context_packet(
    *,
    recent_dialogue_state: Mapping[str, Any] | None = None,
    discussion_thread_state: Mapping[str, Any] | None = None,
    issue_state: Mapping[str, Any] | None = None,
    turn_delta: Mapping[str, Any] | None = None,
    interaction_constraints: Mapping[str, Any] | None = None,
    boundary_transform: Mapping[str, Any] | None = None,
    residual_reflection: Mapping[str, Any] | None = None,
    surface_profile: Mapping[str, Any] | None = None,
    contact_reflection_state: Mapping[str, Any] | None = None,
    green_kernel_composition: Mapping[str, Any] | None = None,
    dialogue_context: Mapping[str, Any] | None = None,
    live_engagement_state: Mapping[str, Any] | None = None,
    lightness_budget_state: Mapping[str, Any] | None = None,
    heartbeat_structure_state: Mapping[str, Any] | None = None,
    shared_moment_state: Mapping[str, Any] | None = None,
    listener_action_state: Mapping[str, Any] | None = None,
    appraisal_state: Mapping[str, Any] | None = None,
    meaning_update_state: Mapping[str, Any] | None = None,
    joint_state: Mapping[str, Any] | None = None,
    utterance_reason_packet: Mapping[str, Any] | None = None,
    interaction_policy_packet: Mapping[str, Any] | None = None,
    action_posture: Mapping[str, Any] | None = None,
    actuation_plan: Mapping[str, Any] | None = None,
    organism_state: Mapping[str, Any] | None = None,
    external_field_state: Mapping[str, Any] | None = None,
    terrain_dynamics_state: Mapping[str, Any] | None = None,
    memory_dynamics_state: Mapping[str, Any] | None = None,
) -> SurfaceContextPacket:
    recent = dict(recent_dialogue_state or {})
    discussion = dict(discussion_thread_state or {})
    issue = dict(issue_state or {})
    delta = dict(turn_delta or {})
    constraints = dict(interaction_constraints or {})
    boundary = dict(boundary_transform or {})
    residual = dict(residual_reflection or {})
    profile = dict(surface_profile or {})
    contact_reflection = dict(contact_reflection_state or {})
    green = dict((green_kernel_composition or {}).get("field") or {})
    dialogue = dict(dialogue_context or {})
    live_engagement = dict(live_engagement_state or {})
    lightness_budget = dict(lightness_budget_state or {})
    heartbeat = dict(heartbeat_structure_state or {})
    shared_moment = dict(shared_moment_state or {})
    listener_action = dict(listener_action_state or {})
    appraisal = dict(appraisal_state or {})
    meaning_update = dict(meaning_update_state or {})
    joint = dict(joint_state or {})
    utterance_reason = dict(utterance_reason_packet or {})
    interaction_policy = dict(interaction_policy_packet or {})
    posture = dict(action_posture or {})
    actuation = dict(actuation_plan or {})
    organism = dict(organism_state or {})
    external_field = dict(external_field_state or {})
    terrain_dynamics = dict(terrain_dynamics_state or {})
    memory_dynamics = dict(memory_dynamics_state or {})

    conversation_phase = (
        _clean_text(delta.get("kind"))
        or _clean_text(recent.get("state"))
        or _clean_text(discussion.get("state"))
        or _clean_text(issue.get("state"))
    )
    anchor = (
        _clean_text(delta.get("anchor_hint"))
        or _clean_text(discussion.get("topic_anchor"))
        or _clean_text(recent.get("recent_anchor"))
        or _clean_text(issue.get("issue_anchor"))
    )
    shared_core = {
        "anchor": anchor,
        "already_shared": _clean_list(
            [
                anchor,
                _clean_text(dialogue.get("user_text")),
                _clean_text(shared_moment.get("cue_text")),
            ]
        ),
        "not_yet_shared": _clean_list(
            [
                _clean_text(residual.get("focus")),
                *_clean_list(residual.get("reasons")),
            ]
        ),
    }
    primary_act = _clean_text(delta.get("preferred_act"))
    reflection_style = _clean_text(contact_reflection.get("reflection_style"))
    response_role = {
        "primary": primary_act,
        "secondary": reflection_style,
    }
    max_questions = 0
    if primary_act.startswith("gentle_question") or reflection_style == "reflect_then_question":
        max_questions = 1
    constraints_payload = {
        "no_generic_clarification": True,
        "no_advice": _bool_value(constraints.get("avoid_obvious_advice"), default=True),
        "max_questions": max_questions,
        "keep_thread_visible": _bool_value(constraints.get("keep_thread_visible"), default=False),
        "prefer_return_point": _bool_value(constraints.get("prefer_return_point"), default=False),
        "boundary_style": _clean_text(boundary.get("surface_mode")),
    }
    surface_payload = {
        "response_length": _clean_text(profile.get("response_length")),
        "cultural_register": _clean_text(profile.get("cultural_register")),
        "group_register": _clean_text(profile.get("group_register")),
        "sentence_temperature": _clean_text(profile.get("sentence_temperature")),
        "surface_mode": _clean_text(profile.get("surface_mode")),
        "voice_texture": _clean_text(profile.get("voice_texture")),
        "brightness": _clean_text(live_engagement.get("state") or lightness_budget.get("state")),
        "playfulness": round(float(lightness_budget.get("banter_room") or 0.0), 4),
        "tempo": round(float(live_engagement.get("score") or 0.0), 4),
        "heartbeat_reaction": _clean_text(heartbeat.get("dominant_reaction")),
        "heartbeat_pulse_band": _clean_text(heartbeat.get("pulse_band")),
        "heartbeat_tempo": round(float(heartbeat.get("response_tempo") or 0.0), 4),
        "shared_moment_kind": _clean_text(shared_moment.get("moment_kind")),
        "shared_moment_afterglow": round(float(shared_moment.get("afterglow") or 0.0), 4),
        "listener_action": _clean_text(listener_action.get("state")),
        "listener_filler_mode": _clean_text(listener_action.get("filler_mode")),
        "listener_token_profile": _clean_text(listener_action.get("token_profile")),
        "joint_mode": _clean_text(joint.get("dominant_mode")),
        "joint_shared_delight": round(float(joint.get("shared_delight") or 0.0), 4),
        "joint_shared_tension": round(float(joint.get("shared_tension") or 0.0), 4),
        "joint_repair_readiness": round(float(joint.get("repair_readiness") or 0.0), 4),
        "joint_common_ground": round(float(joint.get("common_ground") or 0.0), 4),
        "joint_attention": round(float(joint.get("joint_attention") or 0.0), 4),
        "joint_mutual_room": round(float(joint.get("mutual_room") or 0.0), 4),
        "joint_coupling_strength": round(float(joint.get("coupling_strength") or 0.0), 4),
        "appraisal_event": _clean_text(appraisal.get("moment_event")),
        "appraisal_shared_shift": _clean_text(appraisal.get("shared_shift")),
        "appraisal_relation_type": _clean_text(appraisal.get("dominant_relation_type")),
        "appraisal_causal_type": _clean_text(appraisal.get("dominant_causal_type")),
        "meaning_update_relation": _clean_text(meaning_update.get("relation_update")),
        "meaning_update_relation_frame": _clean_text(meaning_update.get("relation_frame")),
        "meaning_update_causal_frame": _clean_text(meaning_update.get("causal_frame")),
        "utterance_reason_offer": _clean_text(utterance_reason.get("offer")),
        "utterance_reason_relation_frame": _clean_text(utterance_reason.get("relation_frame")),
        "utterance_reason_causal_frame": _clean_text(utterance_reason.get("causal_frame")),
        "utterance_reason_memory_frame": _clean_text(utterance_reason.get("memory_frame")),
        "organism_posture": _clean_text(organism.get("dominant_posture")),
        "organism_attunement": round(float(organism.get("attunement") or 0.0), 4),
        "organism_grounding": round(float(organism.get("grounding") or 0.0), 4),
        "organism_protective_tension": round(float(organism.get("protective_tension") or 0.0), 4),
        "organism_expressive_readiness": round(float(organism.get("expressive_readiness") or 0.0), 4),
        "organism_play_window": round(float(organism.get("play_window") or 0.0), 4),
        "organism_relation_pull": round(float(organism.get("relation_pull") or 0.0), 4),
        "external_field_dominant": _clean_text(external_field.get("dominant_field")),
        "external_field_safety": round(float(external_field.get("safety_envelope") or 0.0), 4),
        "external_field_continuity": round(float(external_field.get("continuity_pull") or 0.0), 4),
        "terrain_dominant_basin": _clean_text(terrain_dynamics.get("dominant_basin")),
        "terrain_dominant_flow": _clean_text(terrain_dynamics.get("dominant_flow")),
        "terrain_energy": round(float(terrain_dynamics.get("terrain_energy") or 0.0), 4),
        "terrain_entropy": round(float(terrain_dynamics.get("entropy") or 0.0), 4),
        "terrain_ignition_pressure": round(float(terrain_dynamics.get("ignition_pressure") or 0.0), 4),
        "memory_dynamics_mode": _clean_text(memory_dynamics.get("dominant_mode")),
        "memory_palace_mode": _clean_text(memory_dynamics.get("palace_mode")),
        "memory_ignition_mode": _clean_text(memory_dynamics.get("ignition_mode")),
        "memory_recall_anchor": _clean_text(memory_dynamics.get("recall_anchor")),
        "memory_activation_confidence": round(float(memory_dynamics.get("activation_confidence") or 0.0), 4),
    }
    source_state = {
        "recent_dialogue_state": _clean_text(recent.get("state")),
        "discussion_thread_state": _clean_text(discussion.get("state")),
        "issue_state": _clean_text(issue.get("state")),
        "turn_delta_kind": _clean_text(delta.get("kind")),
        "interaction_policy_strategy": _clean_text(
            interaction_policy.get("response_strategy")
        ),
        "interaction_policy_opening_move": _clean_text(
            interaction_policy.get("opening_move")
        ),
        "interaction_policy_followup_move": _clean_text(
            interaction_policy.get("followup_move")
        ),
        "interaction_policy_closing_move": _clean_text(
            interaction_policy.get("closing_move")
        ),
        "scene_family": _clean_text(interaction_policy.get("scene_family")),
        "green_guardedness": float(green.get("guardedness") or 0.0),
        "green_reopening_pull": float(green.get("reopening_pull") or 0.0),
        "green_affective_charge": float(green.get("affective_charge") or 0.0),
        "voice_texture": _clean_text(profile.get("voice_texture")),
        "live_engagement_state": _clean_text(live_engagement.get("state")),
        "live_primary_move": _clean_text(live_engagement.get("primary_move")),
        "lightness_budget_state": _clean_text(lightness_budget.get("state")),
        "lightness_banter_room": round(float(lightness_budget.get("banter_room") or 0.0), 4),
        "shared_moment_state": _clean_text(shared_moment.get("state")),
        "shared_moment_kind": _clean_text(shared_moment.get("moment_kind")),
        "shared_moment_score": round(float(shared_moment.get("score") or 0.0), 4),
        "shared_moment_jointness": round(float(shared_moment.get("jointness") or 0.0), 4),
        "listener_action_state": _clean_text(listener_action.get("state")),
        "listener_filler_mode": _clean_text(listener_action.get("filler_mode")),
        "listener_token_profile": _clean_text(listener_action.get("token_profile")),
        "listener_action_score": round(float(listener_action.get("score") or 0.0), 4),
        "joint_state": _clean_text(joint.get("dominant_mode")),
        "joint_shared_delight": round(float(joint.get("shared_delight") or 0.0), 4),
        "joint_shared_tension": round(float(joint.get("shared_tension") or 0.0), 4),
        "joint_repair_readiness": round(float(joint.get("repair_readiness") or 0.0), 4),
        "joint_common_ground": round(float(joint.get("common_ground") or 0.0), 4),
        "joint_attention": round(float(joint.get("joint_attention") or 0.0), 4),
        "joint_mutual_room": round(float(joint.get("mutual_room") or 0.0), 4),
        "joint_coupling_strength": round(float(joint.get("coupling_strength") or 0.0), 4),
        "appraisal_state": _clean_text(appraisal.get("state")),
        "appraisal_background_state": _clean_text(appraisal.get("background_state")),
        "appraisal_event": _clean_text(appraisal.get("moment_event")),
        "appraisal_shared_shift": _clean_text(appraisal.get("shared_shift")),
        "appraisal_dominant_relation_type": _clean_text(appraisal.get("dominant_relation_type")),
        "appraisal_dominant_relation_key": _clean_text(appraisal.get("dominant_relation_key")),
        "appraisal_relation_meta_type": _clean_text(appraisal.get("relation_meta_type")),
        "appraisal_dominant_causal_type": _clean_text(appraisal.get("dominant_causal_type")),
        "appraisal_dominant_causal_key": _clean_text(appraisal.get("dominant_causal_key")),
        "appraisal_memory_mode": _clean_text(appraisal.get("memory_mode")),
        "appraisal_memory_anchor": _clean_text(appraisal.get("recall_anchor")),
        "appraisal_memory_resonance": round(float(appraisal.get("memory_resonance") or 0.0), 4),
        "appraisal_easing_shift": round(float(appraisal.get("easing_shift") or 0.0), 4),
        "meaning_update_state": _clean_text(meaning_update.get("state")),
        "meaning_update_self": _clean_text(meaning_update.get("self_update")),
        "meaning_update_relation": _clean_text(meaning_update.get("relation_update")),
        "meaning_update_relation_frame": _clean_text(meaning_update.get("relation_frame")),
        "meaning_update_relation_key": _clean_text(meaning_update.get("relation_key")),
        "meaning_update_relation_meta_type": _clean_text(meaning_update.get("relation_meta_type")),
        "meaning_update_causal_frame": _clean_text(meaning_update.get("causal_frame")),
        "meaning_update_causal_key": _clean_text(meaning_update.get("causal_key")),
        "meaning_update_world": _clean_text(meaning_update.get("world_update")),
        "meaning_update_memory": _clean_text(meaning_update.get("memory_update")),
        "meaning_update_memory_anchor": _clean_text(meaning_update.get("recall_anchor")),
        "meaning_update_memory_resonance": round(float(meaning_update.get("memory_resonance") or 0.0), 4),
        "meaning_update_preserve_guard": _clean_text(meaning_update.get("preserve_guard")),
        "utterance_reason_state": _clean_text(utterance_reason.get("state")),
        "utterance_reason_target": _clean_text(utterance_reason.get("reaction_target")),
        "utterance_reason_frame": _clean_text(utterance_reason.get("reason_frame")),
        "utterance_reason_relation_frame": _clean_text(utterance_reason.get("relation_frame")),
        "utterance_reason_relation_key": _clean_text(utterance_reason.get("relation_key")),
        "utterance_reason_causal_frame": _clean_text(utterance_reason.get("causal_frame")),
        "utterance_reason_causal_key": _clean_text(utterance_reason.get("causal_key")),
        "utterance_reason_memory_frame": _clean_text(utterance_reason.get("memory_frame")),
        "utterance_reason_memory_anchor": _clean_text(utterance_reason.get("memory_anchor")),
        "utterance_reason_offer": _clean_text(utterance_reason.get("offer")),
        "utterance_reason_preserve": _clean_text(utterance_reason.get("preserve")),
        "utterance_reason_question_policy": _clean_text(utterance_reason.get("question_policy")),
        "utterance_reason_tone_hint": _clean_text(utterance_reason.get("tone_hint")),
        "action_posture_mode": _clean_text(posture.get("mode")),
        "actuation_execution_mode": _clean_text(actuation.get("execution_mode")),
        "actuation_primary_action": _clean_text(actuation.get("primary_action")),
        "actuation_response_channel": _clean_text(actuation.get("response_channel")),
        "organism_posture": _clean_text(organism.get("dominant_posture")),
        "organism_relation_focus": _clean_text(organism.get("relation_focus")),
        "organism_social_mode": _clean_text(organism.get("social_mode")),
        "organism_attunement": round(float(organism.get("attunement") or 0.0), 4),
        "organism_coherence": round(float(organism.get("coherence") or 0.0), 4),
        "organism_grounding": round(float(organism.get("grounding") or 0.0), 4),
        "organism_protective_tension": round(float(organism.get("protective_tension") or 0.0), 4),
        "organism_expressive_readiness": round(float(organism.get("expressive_readiness") or 0.0), 4),
        "organism_play_window": round(float(organism.get("play_window") or 0.0), 4),
        "organism_relation_pull": round(float(organism.get("relation_pull") or 0.0), 4),
        "organism_social_exposure": round(float(organism.get("social_exposure") or 0.0), 4),
        "external_field_dominant": _clean_text(external_field.get("dominant_field")),
        "external_field_social_mode": _clean_text(external_field.get("social_mode")),
        "external_field_thread_mode": _clean_text(external_field.get("thread_mode")),
        "external_field_environmental_load": round(float(external_field.get("environmental_load") or 0.0), 4),
        "external_field_social_pressure": round(float(external_field.get("social_pressure") or 0.0), 4),
        "external_field_continuity_pull": round(float(external_field.get("continuity_pull") or 0.0), 4),
        "external_field_ambiguity_load": round(float(external_field.get("ambiguity_load") or 0.0), 4),
        "external_field_safety_envelope": round(float(external_field.get("safety_envelope") or 0.0), 4),
        "external_field_novelty": round(float(external_field.get("novelty") or 0.0), 4),
        "terrain_dominant_basin": _clean_text(terrain_dynamics.get("dominant_basin")),
        "terrain_dominant_flow": _clean_text(terrain_dynamics.get("dominant_flow")),
        "terrain_energy": round(float(terrain_dynamics.get("terrain_energy") or 0.0), 4),
        "terrain_entropy": round(float(terrain_dynamics.get("entropy") or 0.0), 4),
        "terrain_ignition_pressure": round(float(terrain_dynamics.get("ignition_pressure") or 0.0), 4),
        "terrain_barrier_height": round(float(terrain_dynamics.get("barrier_height") or 0.0), 4),
        "terrain_recovery_gradient": round(float(terrain_dynamics.get("recovery_gradient") or 0.0), 4),
        "terrain_basin_pull": round(float(terrain_dynamics.get("basin_pull") or 0.0), 4),
        "memory_dynamics_mode": _clean_text(memory_dynamics.get("dominant_mode")),
        "memory_dominant_relation_type": _clean_text(memory_dynamics.get("dominant_relation_type")),
        "memory_relation_generation_mode": _clean_text(memory_dynamics.get("relation_generation_mode")),
        "memory_dominant_causal_type": _clean_text(memory_dynamics.get("dominant_causal_type")),
        "memory_causal_generation_mode": _clean_text(memory_dynamics.get("causal_generation_mode")),
        "memory_palace_mode": _clean_text(memory_dynamics.get("palace_mode")),
        "memory_monument_mode": _clean_text(memory_dynamics.get("monument_mode")),
        "memory_ignition_mode": _clean_text(memory_dynamics.get("ignition_mode")),
        "memory_reconsolidation_mode": _clean_text(memory_dynamics.get("reconsolidation_mode")),
        "memory_recall_anchor": _clean_text(memory_dynamics.get("recall_anchor")),
        "memory_monument_salience": round(float(memory_dynamics.get("monument_salience") or 0.0), 4),
        "memory_activation_confidence": round(float(memory_dynamics.get("activation_confidence") or 0.0), 4),
        "memory_tension": round(float(memory_dynamics.get("memory_tension") or 0.0), 4),
        "heartbeat_pulse_band": _clean_text(heartbeat.get("pulse_band")),
        "heartbeat_phase_window": _clean_text(heartbeat.get("phase_window")),
        "heartbeat_reaction": _clean_text(heartbeat.get("dominant_reaction")),
        "heartbeat_activation_drive": round(float(heartbeat.get("activation_drive") or 0.0), 4),
        "heartbeat_containment_bias": round(float(heartbeat.get("containment_bias") or 0.0), 4),
        "heartbeat_bounce_room": round(float(heartbeat.get("bounce_room") or 0.0), 4),
        "heartbeat_response_tempo": round(float(heartbeat.get("response_tempo") or 0.0), 4),
    }
    return SurfaceContextPacket(
        conversation_phase=conversation_phase,
        shared_core=shared_core,
        response_role=response_role,
        constraints=constraints_payload,
        surface_profile=surface_payload,
        source_state=source_state,
    )
