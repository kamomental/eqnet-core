from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping

from .nonverbal_response_state import derive_nonverbal_response_state
from .presence_hold_state import derive_presence_hold_state
from .response_selection_state import derive_response_selection_state


_ACTUATION_PLAN_CORE_KEYS = frozenset(
    {
        "execution_mode",
        "primary_action",
        "action_queue",
        "reply_permission",
        "wait_before_action",
        "repair_window_commitment",
        "outcome_goal",
        "boundary_mode",
        "attention_target",
        "memory_write_priority",
        "memory_write_class",
        "memory_write_class_reason",
        "contact_readiness",
        "repair_bias",
        "disclosure_depth",
        "utterance_reason_relation_frame",
        "utterance_reason_causal_frame",
        "utterance_reason_memory_frame",
        "utterance_reason_preserve",
        "joint_mode",
        "joint_reentry_room",
        "joint_guard_signal",
        "presence_hold_state",
        "nonverbal_response_state",
        "response_selection_state",
        "response_channel",
        "response_channel_score",
        "organism_posture_name",
        "external_field_name",
        "terrain_basin_name",
        "terrain_flow_name",
        "situation_risk_name",
        "emergency_posture_name",
        "relational_continuity_name",
        "relation_competition_name",
        "social_topology_name",
    }
)


@dataclass(frozen=True)
class ActuationPlanContract(MappingABC[str, object]):
    execution_mode: str = ""
    primary_action: str = ""
    action_queue: list[str] = field(default_factory=list)
    reply_permission: str = ""
    wait_before_action: str = ""
    repair_window_commitment: str = ""
    outcome_goal: str = ""
    boundary_mode: str = ""
    attention_target: str = ""
    memory_write_priority: str = ""
    memory_write_class: str = ""
    memory_write_class_reason: str = ""
    contact_readiness: float = 0.0
    repair_bias: bool = False
    disclosure_depth: str = ""
    utterance_reason_relation_frame: str = ""
    utterance_reason_causal_frame: str = ""
    utterance_reason_memory_frame: str = ""
    utterance_reason_preserve: str = ""
    joint_mode: str = ""
    joint_reentry_room: float = 0.0
    joint_guard_signal: float = 0.0
    presence_hold_state: dict[str, object] = field(default_factory=dict)
    nonverbal_response_state: dict[str, object] = field(default_factory=dict)
    response_selection_state: dict[str, object] = field(default_factory=dict)
    response_channel: str = ""
    response_channel_score: float = 0.0
    organism_posture_name: str = ""
    external_field_name: str = ""
    terrain_basin_name: str = ""
    terrain_flow_name: str = ""
    situation_risk_name: str = ""
    emergency_posture_name: str = ""
    relational_continuity_name: str = ""
    relation_competition_name: str = ""
    social_topology_name: str = ""
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "execution_mode": self.execution_mode,
            "primary_action": self.primary_action,
            "action_queue": list(self.action_queue),
            "reply_permission": self.reply_permission,
            "wait_before_action": self.wait_before_action,
            "repair_window_commitment": self.repair_window_commitment,
            "outcome_goal": self.outcome_goal,
            "boundary_mode": self.boundary_mode,
            "attention_target": self.attention_target,
            "memory_write_priority": self.memory_write_priority,
            "memory_write_class": self.memory_write_class,
            "memory_write_class_reason": self.memory_write_class_reason,
            "contact_readiness": self.contact_readiness,
            "repair_bias": self.repair_bias,
            "disclosure_depth": self.disclosure_depth,
            "utterance_reason_relation_frame": self.utterance_reason_relation_frame,
            "utterance_reason_causal_frame": self.utterance_reason_causal_frame,
            "utterance_reason_memory_frame": self.utterance_reason_memory_frame,
            "utterance_reason_preserve": self.utterance_reason_preserve,
            "joint_mode": self.joint_mode,
            "joint_reentry_room": self.joint_reentry_room,
            "joint_guard_signal": self.joint_guard_signal,
            "presence_hold_state": dict(self.presence_hold_state),
            "nonverbal_response_state": dict(self.nonverbal_response_state),
            "response_selection_state": dict(self.response_selection_state),
            "response_channel": self.response_channel,
            "response_channel_score": self.response_channel_score,
            "organism_posture_name": self.organism_posture_name,
            "external_field_name": self.external_field_name,
            "terrain_basin_name": self.terrain_basin_name,
            "terrain_flow_name": self.terrain_flow_name,
            "situation_risk_name": self.situation_risk_name,
            "emergency_posture_name": self.emergency_posture_name,
            "relational_continuity_name": self.relational_continuity_name,
            "relation_competition_name": self.relation_competition_name,
            "social_topology_name": self.social_topology_name,
        }
        data.update(self.extras)
        return data

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def coerce_actuation_plan_contract(
    value: Mapping[str, Any] | ActuationPlanContract | None,
) -> ActuationPlanContract:
    if isinstance(value, ActuationPlanContract):
        return value
    packet = dict(value or {})
    extras = {key: packet[key] for key in packet.keys() - _ACTUATION_PLAN_CORE_KEYS}
    return ActuationPlanContract(
        execution_mode=str(packet.get("execution_mode") or "").strip(),
        primary_action=str(packet.get("primary_action") or "").strip(),
        action_queue=_as_text_list(packet.get("action_queue")),
        reply_permission=str(packet.get("reply_permission") or "").strip(),
        wait_before_action=str(packet.get("wait_before_action") or "").strip(),
        repair_window_commitment=str(packet.get("repair_window_commitment") or "").strip(),
        outcome_goal=str(packet.get("outcome_goal") or "").strip(),
        boundary_mode=str(packet.get("boundary_mode") or "").strip(),
        attention_target=str(packet.get("attention_target") or "").strip(),
        memory_write_priority=str(packet.get("memory_write_priority") or "").strip(),
        memory_write_class=str(packet.get("memory_write_class") or "").strip(),
        memory_write_class_reason=str(packet.get("memory_write_class_reason") or "").strip(),
        contact_readiness=_float01(packet.get("contact_readiness")),
        repair_bias=bool(packet.get("repair_bias", False)),
        disclosure_depth=str(packet.get("disclosure_depth") or "").strip(),
        utterance_reason_relation_frame=str(packet.get("utterance_reason_relation_frame") or "").strip(),
        utterance_reason_causal_frame=str(packet.get("utterance_reason_causal_frame") or "").strip(),
        utterance_reason_memory_frame=str(packet.get("utterance_reason_memory_frame") or "").strip(),
        utterance_reason_preserve=str(packet.get("utterance_reason_preserve") or "").strip(),
        joint_mode=str(packet.get("joint_mode") or "").strip(),
        joint_reentry_room=_float01(packet.get("joint_reentry_room")),
        joint_guard_signal=_float01(packet.get("joint_guard_signal")),
        presence_hold_state=dict(packet.get("presence_hold_state") or {}),
        nonverbal_response_state=dict(packet.get("nonverbal_response_state") or {}),
        response_selection_state=dict(packet.get("response_selection_state") or {}),
        response_channel=str(packet.get("response_channel") or "").strip(),
        response_channel_score=_float01(packet.get("response_channel_score")),
        organism_posture_name=str(packet.get("organism_posture_name") or "").strip(),
        external_field_name=str(packet.get("external_field_name") or "").strip(),
        terrain_basin_name=str(packet.get("terrain_basin_name") or "").strip(),
        terrain_flow_name=str(packet.get("terrain_flow_name") or "").strip(),
        situation_risk_name=str(packet.get("situation_risk_name") or "").strip(),
        emergency_posture_name=str(packet.get("emergency_posture_name") or "").strip(),
        relational_continuity_name=str(packet.get("relational_continuity_name") or "").strip(),
        relation_competition_name=str(packet.get("relation_competition_name") or "").strip(),
        social_topology_name=str(packet.get("social_topology_name") or "").strip(),
        extras=extras,
    )


def derive_actuation_plan(
    interaction_policy: Mapping[str, Any] | None,
    action_posture: Mapping[str, Any] | None,
) -> ActuationPlanContract:
    packet = dict(interaction_policy or {})
    contract = dict(packet.get("conversation_contract") or {})
    response_action = dict(contract.get("response_action_now") or {})
    posture = dict(action_posture or {})

    strategy = str(packet.get("response_strategy") or "").strip()
    engagement_mode = str(posture.get("engagement_mode") or "").strip()
    primary_operation_kind = str(posture.get("primary_operation_kind") or "").strip()
    ordered_operation_kinds = [
        str(item) for item in posture.get("ordered_operation_kinds") or [] if str(item).strip()
    ]
    ordered_effect_kinds = [
        str(item) for item in posture.get("ordered_effect_kinds") or [] if str(item).strip()
    ]
    question_budget = int(posture.get("question_budget") or 0)
    if not question_budget:
        question_budget = int(response_action.get("question_budget") or 0)
    question_pressure = _float01(posture.get("question_pressure"))
    if question_pressure <= 0.0:
        question_pressure = _float01(response_action.get("question_pressure"))
    defer_dominance = _float01(posture.get("defer_dominance"))
    if defer_dominance <= 0.0:
        defer_dominance = _float01(response_action.get("defer_dominance"))
    outcome_goal = str(posture.get("outcome_goal") or "").strip()
    boundary_mode = str(posture.get("boundary_mode") or "").strip()
    disclosure_depth = str(posture.get("disclosure_depth") or packet.get("disclosure_depth") or "").strip()
    attention_target = str(
        posture.get("attention_target")
        or packet.get("attention_target")
        or contract.get("focus_now")
        or contract.get("primary_object")
        or ""
    ).strip()
    memory_write_priority = str(
        posture.get("memory_write_priority") or packet.get("memory_write_priority") or ""
    ).strip()
    memory_write_class = str(
        posture.get("memory_write_class") or packet.get("memory_write_class") or ""
    ).strip()
    memory_write_class_reason = str(
        posture.get("memory_write_class_reason") or packet.get("memory_write_class_reason") or ""
    ).strip()
    if not memory_write_priority:
        condition_summary = [str(item).strip() for item in contract.get("condition_summary") or [] if str(item).strip()]
        if any("relation" in item.lower() for item in condition_summary):
            memory_write_priority = "relation_episode"
        elif attention_target:
            memory_write_priority = "foreground_trace"
    if not memory_write_class:
        memory_write_class = "episodic"
    affordance_priority = [
        str(item)
        for item in posture.get("affordance_priority") or packet.get("affordance_priority") or []
        if str(item).strip()
    ]
    do_not_cross = [
        str(item)
        for item in posture.get("do_not_cross") or packet.get("do_not_cross") or []
        if str(item).strip()
    ]
    other_person_state = dict(posture.get("other_person_state") or packet.get("other_person_state") or {})
    resonance_prioritize_actions = [
        str(item)
        for item in posture.get("resonance_prioritize_actions")
        or packet.get("resonance_prioritize_actions")
        or []
        if str(item).strip()
    ]
    resonance_avoid_actions = [
        str(item)
        for item in posture.get("resonance_avoid_actions")
        or packet.get("resonance_avoid_actions")
        or []
        if str(item).strip()
    ]
    conversational_objects = dict(posture.get("conversational_objects") or packet.get("conversational_objects") or {})
    object_operations = dict(posture.get("object_operations") or packet.get("object_operations") or {})
    interaction_effects = dict(posture.get("interaction_effects") or packet.get("interaction_effects") or {})
    interaction_judgement_view = dict(posture.get("interaction_judgement_view") or packet.get("interaction_judgement_view") or {})
    qualia_planner_view = dict(posture.get("qualia_planner_view") or packet.get("qualia_planner_view") or {})
    terrain_readout = dict(posture.get("terrain_readout") or packet.get("terrain_readout") or {})
    protection_mode = dict(posture.get("protection_mode") or packet.get("protection_mode") or {})
    body_recovery_guard = dict(posture.get("body_recovery_guard") or packet.get("body_recovery_guard") or {})
    body_homeostasis_state = dict(posture.get("body_homeostasis_state") or packet.get("body_homeostasis_state") or {})
    homeostasis_budget_state = dict(posture.get("homeostasis_budget_state") or packet.get("homeostasis_budget_state") or {})
    initiative_readiness = dict(posture.get("initiative_readiness") or packet.get("initiative_readiness") or {})
    commitment_state = dict(posture.get("commitment_state") or packet.get("commitment_state") or {})
    learning_mode_state = dict(posture.get("learning_mode_state") or packet.get("learning_mode_state") or {})
    social_experiment_loop_state = dict(
        posture.get("social_experiment_loop_state") or packet.get("social_experiment_loop_state") or {}
    )
    live_engagement_state = dict(posture.get("live_engagement_state") or packet.get("live_engagement_state") or {})
    shared_moment_state = dict(posture.get("shared_moment_state") or packet.get("shared_moment_state") or {})
    listener_action_state = dict(posture.get("listener_action_state") or packet.get("listener_action_state") or {})
    utterance_reason_packet = dict(posture.get("utterance_reason_packet") or packet.get("utterance_reason_packet") or {})
    subjective_scene_state = dict(
        posture.get("subjective_scene_state") or packet.get("subjective_scene_state") or {}
    )
    self_other_attribution_state = dict(
        posture.get("self_other_attribution_state")
        or packet.get("self_other_attribution_state")
        or {}
    )
    shared_presence_state = dict(
        posture.get("shared_presence_state") or packet.get("shared_presence_state") or {}
    )
    joint_state = dict(posture.get("joint_state") or packet.get("joint_state") or {})
    organism_state = dict(posture.get("organism_state") or packet.get("organism_state") or {})
    external_field_state = dict(posture.get("external_field_state") or packet.get("external_field_state") or {})
    terrain_dynamics_state = dict(posture.get("terrain_dynamics_state") or packet.get("terrain_dynamics_state") or {})
    situation_risk_state = dict(posture.get("situation_risk_state") or packet.get("situation_risk_state") or {})
    emergency_posture = dict(posture.get("emergency_posture") or packet.get("emergency_posture") or {})
    relational_continuity_state = dict(posture.get("relational_continuity_state") or packet.get("relational_continuity_state") or {})
    relation_competition_state = dict(posture.get("relation_competition_state") or packet.get("relation_competition_state") or {})
    social_topology_state = dict(posture.get("social_topology_state") or packet.get("social_topology_state") or {})
    insight_event = dict(posture.get("insight_event") or packet.get("insight_event") or {})
    next_action_candidates = [
        str(item) for item in posture.get("next_action_candidates") or [] if str(item).strip()
    ]
    contact_readiness = _float01(packet.get("contact_readiness"))
    repair_bias = bool(packet.get("repair_bias"))
    qualia_trust = _float01(qualia_planner_view.get("trust"))
    qualia_degraded = bool(qualia_planner_view.get("degraded", False))
    qualia_body_load = _float01(qualia_planner_view.get("body_load"))
    qualia_protection_bias = _float01(qualia_planner_view.get("protection_bias"))
    terrain_protect_bias = _float01(terrain_readout.get("protect_bias"))
    protection_mode_name = str(protection_mode.get("mode") or "").strip()
    protection_mode_strength = _float01(protection_mode.get("strength"))
    body_recovery_guard_state = str(body_recovery_guard.get("state") or "").strip()
    body_recovery_guard_score = _float01(body_recovery_guard.get("score"))
    body_homeostasis_name = str(body_homeostasis_state.get("state") or "").strip()
    body_homeostasis_score = _float01(body_homeostasis_state.get("score"))
    homeostasis_budget_name = str(homeostasis_budget_state.get("state") or "").strip()
    homeostasis_budget_score = _float01(homeostasis_budget_state.get("score"))
    initiative_readiness_state = str(initiative_readiness.get("state") or "").strip()
    initiative_readiness_score = _float01(initiative_readiness.get("score"))
    commitment_mode = str(commitment_state.get("state") or "").strip()
    commitment_target = str(commitment_state.get("target") or "").strip()
    commitment_score = _float01(commitment_state.get("score"))
    learning_mode_name = str(learning_mode_state.get("state") or "").strip()
    learning_mode_score = _float01(learning_mode_state.get("score"))
    learning_mode_probe_room = _float01(learning_mode_state.get("probe_room"))
    social_experiment_name = str(social_experiment_loop_state.get("state") or "").strip()
    social_experiment_score = _float01(social_experiment_loop_state.get("score"))
    social_experiment_probe_intensity = _float01(social_experiment_loop_state.get("probe_intensity"))
    live_engagement_name = str(live_engagement_state.get("state") or "").strip()
    live_engagement_score = _float01(live_engagement_state.get("score"))
    live_primary_move = str(live_engagement_state.get("primary_move") or "").strip()
    shared_moment_name = str(shared_moment_state.get("state") or "").strip()
    shared_moment_kind = str(shared_moment_state.get("moment_kind") or "").strip()
    shared_moment_score = _float01(shared_moment_state.get("score"))
    shared_moment_jointness = _float01(shared_moment_state.get("jointness"))
    shared_moment_afterglow = _float01(shared_moment_state.get("afterglow"))
    listener_action_name = str(listener_action_state.get("state") or "").strip()
    utterance_reason_state = str(utterance_reason_packet.get("state") or "").strip()
    utterance_reason_offer = str(utterance_reason_packet.get("offer") or "").strip()
    utterance_reason_question_policy = str(
        utterance_reason_packet.get("question_policy") or ""
    ).strip()
    utterance_reason_relation_frame = str(
        posture.get("utterance_reason_relation_frame")
        or utterance_reason_packet.get("relation_frame")
        or ""
    ).strip()
    utterance_reason_causal_frame = str(
        posture.get("utterance_reason_causal_frame")
        or utterance_reason_packet.get("causal_frame")
        or ""
    ).strip()
    utterance_reason_memory_frame = str(
        posture.get("utterance_reason_memory_frame")
        or utterance_reason_packet.get("memory_frame")
        or ""
    ).strip()
    utterance_reason_preserve = str(
        posture.get("utterance_reason_preserve")
        or utterance_reason_packet.get("preserve")
        or ""
    ).strip()
    joint_mode = str(
        posture.get("joint_mode")
        or joint_state.get("dominant_mode")
        or ""
    ).strip()
    joint_shared_delight = _float01(joint_state.get("shared_delight"))
    joint_shared_tension = _float01(joint_state.get("shared_tension"))
    joint_common_ground = _float01(joint_state.get("common_ground"))
    joint_attention = _float01(joint_state.get("joint_attention"))
    joint_mutual_room = _float01(joint_state.get("mutual_room"))
    joint_coupling_strength = _float01(joint_state.get("coupling_strength"))
    joint_reentry_room = _float01(
        posture.get("joint_reentry_room")
        or (
            joint_common_ground * 0.24
            + joint_attention * 0.18
            + joint_mutual_room * 0.18
            + joint_coupling_strength * 0.18
            + joint_shared_delight * 0.12
            + (0.08 if joint_mode in {"delighted_jointness", "shared_attention"} else 0.0)
            - joint_shared_tension * 0.14
        )
    )
    joint_guard_signal = _float01(
        posture.get("joint_guard_signal")
        or (
            joint_shared_tension * 0.36
            + (0.12 if joint_mode in {"repair_attunement", "strained_jointness"} else 0.0)
        )
    )
    organism_posture_name = str(
        posture.get("organism_posture_name")
        or organism_state.get("dominant_posture")
        or ""
    ).strip()
    external_field_name = str(
        posture.get("external_field_name")
        or external_field_state.get("dominant_field")
        or ""
    ).strip()
    terrain_basin_name = str(
        posture.get("terrain_basin_name")
        or terrain_dynamics_state.get("dominant_basin")
        or ""
    ).strip()
    terrain_flow_name = str(
        posture.get("terrain_flow_name")
        or terrain_dynamics_state.get("dominant_flow")
        or ""
    ).strip()
    external_continuity_pull = _float01(external_field_state.get("continuity_pull"))
    external_safety_envelope = _float01(external_field_state.get("safety_envelope"))
    terrain_recovery_gradient = _float01(terrain_dynamics_state.get("recovery_gradient"))
    terrain_barrier_height = _float01(terrain_dynamics_state.get("barrier_height"))
    situation_risk_name = str(situation_risk_state.get("state") or "").strip()
    situation_risk_immediacy = _float01(situation_risk_state.get("immediacy"))
    emergency_posture_name = str(emergency_posture.get("state") or "").strip()
    emergency_posture_score = _float01(emergency_posture.get("score"))
    emergency_dialogue_permission = str(emergency_posture.get("dialogue_permission") or "").strip()
    emergency_primary_action = str(emergency_posture.get("primary_action") or "").strip()
    relational_continuity_name = str(relational_continuity_state.get("state") or "").strip()
    relational_continuity_score = _float01(relational_continuity_state.get("score"))
    relation_competition_name = str(relation_competition_state.get("state") or "").strip()
    relation_competition_level = _float01(relation_competition_state.get("competition_level"))
    social_topology_name = str(social_topology_state.get("state") or "").strip()
    social_topology_score = _float01(social_topology_state.get("score"))
    insight_triggered = bool(insight_event.get("triggered", False))
    insight_orient_bias = _float01(insight_event.get("orient_bias"))

    execution_mode = "attuned_contact"
    primary_action = "hold_presence"
    action_queue = ["hold_presence", "invite_visible_state", "gentle_approach", *next_action_candidates]
    reply_permission = "speak"
    wait_before_action = "brief" if contact_readiness >= 0.58 else "measured"
    repair_window_commitment = "soft" if repair_bias else "steady"

    if primary_operation_kind == "offer_small_next_step" or engagement_mode == "co_move":
        execution_mode = "shared_progression"
        primary_action = "co_move"
        action_queue = ["synchronize", "map_next_step", "pace_match", "keep_step_connected", *next_action_candidates]
        reply_permission = "speak_forward"
        wait_before_action = "brief"
        repair_window_commitment = "steady"
    elif engagement_mode == "repair" or (not engagement_mode and strategy == "repair_then_attune"):
        execution_mode = "repair_contact"
        primary_action = "soft_repair"
        action_queue = ["name_overreach", "reduce_force", "reopen_carefully", *next_action_candidates]
        reply_permission = "speak_briefly"
        wait_before_action = "measured"
        repair_window_commitment = "active"
    elif engagement_mode == "contain" or (not engagement_mode and strategy == "contain_then_stabilize"):
        execution_mode = "stabilize_boundary"
        primary_action = "secure_boundary"
        action_queue = ["reduce_force", "stabilize", "secure_boundary", *next_action_candidates]
        reply_permission = "speak_minimal"
        wait_before_action = "measured"
        repair_window_commitment = "protective"
    elif engagement_mode == "wait" or (defer_dominance >= 0.72 and engagement_mode == ""):
        execution_mode = "defer_with_presence"
        primary_action = "hold_presence"
        action_queue = ["defer_contact", "leave_return_point", "protect_distance", *next_action_candidates]
        reply_permission = "hold_or_brief"
        wait_before_action = "extended"
        repair_window_commitment = "guarded"
    elif engagement_mode == "reflect" or (not engagement_mode and strategy == "reflect_without_settling"):
        execution_mode = "open_reflection"
        primary_action = "hold_meaning_open"
        action_queue = ["observe_more", "hold_meaning_open", "defer_closure", *next_action_candidates]
        reply_permission = "speak_reflective"
        wait_before_action = "held"
        repair_window_commitment = "porous"
    elif primary_operation_kind == "narrow_clarify" and question_budget > 0:
        execution_mode = "bounded_clarification"
        primary_action = "hold_presence"
        action_queue = ["check_visible_part", "bound_scope", "confirm_before_extend", *next_action_candidates]
        reply_permission = "speak_briefly"
        wait_before_action = "brief" if question_pressure <= 0.48 else "measured"
        repair_window_commitment = "steady"
    if disclosure_depth == "minimal":
        reply_permission = "speak_minimal"
    elif disclosure_depth == "medium" and reply_permission == "speak":
        reply_permission = "speak_expand"

    qualia_protective_hold = (
        qualia_degraded
        or qualia_trust <= 0.45
        or qualia_protection_bias >= 0.14
        or qualia_body_load >= 0.1
    )
    if qualia_protective_hold:
        if execution_mode == "attuned_contact":
            execution_mode = "stabilize_before_contact"
        action_queue = _compact(["stabilize_before_extend", "check_load_softly", *action_queue])
        reply_permission = "hold_or_brief" if reply_permission == "speak" else reply_permission
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
        if repair_window_commitment in {"steady", "soft"}:
            repair_window_commitment = "protective"

    if body_homeostasis_name == "depleted":
        execution_mode = "stabilize_before_contact"
        primary_action = "restore_body_margin"
        action_queue = _compact(["honor_body_limit", "reduce_load_first", "restore_body_margin", *action_queue])
        reply_permission = "hold_or_brief"
        wait_before_action = "extended"
        repair_window_commitment = "protective"
    elif body_homeostasis_name == "recovering" and body_homeostasis_score >= 0.32:
        if execution_mode == "shared_progression":
            execution_mode = "attuned_contact"
            primary_action = "hold_presence"
        action_queue = _compact(["stabilize_while_reengaging", "keep_step_small", *action_queue])
        if wait_before_action == "brief":
            wait_before_action = "measured"
    elif body_homeostasis_name == "strained":
        action_queue = _compact(["keep_step_small", *action_queue])
        if execution_mode == "shared_progression":
            execution_mode = "attuned_contact"

    if homeostasis_budget_name == "depleted" and homeostasis_budget_score >= 0.3:
        if execution_mode == "shared_progression":
            execution_mode = "attuned_contact"
            primary_action = "hold_presence"
        action_queue = _compact(["preserve_energy_budget", "keep_step_small", *action_queue])
        if wait_before_action == "brief":
            wait_before_action = "measured"
    elif homeostasis_budget_name == "recovering" and homeostasis_budget_score >= 0.32:
        action_queue = _compact(["protect_recovery_budget", *action_queue])
        if execution_mode == "shared_progression":
            execution_mode = "attuned_contact"
        if wait_before_action == "brief":
            wait_before_action = "measured"

    if body_recovery_guard_state == "recovery_first":
        execution_mode = "stabilize_before_contact"
        primary_action = "restore_body_margin"
        action_queue = _compact(["restore_body_margin", "wait_for_recovery", "reduce_force", *action_queue])
        reply_permission = "hold_or_brief"
        wait_before_action = "extended"
        repair_window_commitment = "protective"
    elif body_recovery_guard_state == "guarded" and body_recovery_guard_score >= 0.42:
        if execution_mode == "shared_progression":
            execution_mode = "attuned_contact"
            primary_action = "hold_presence"
        action_queue = _compact(["keep_step_small", *action_queue])
        if wait_before_action == "brief":
            wait_before_action = "measured"

    if protection_mode_name in {"stabilize", "shield"}:
        execution_mode = "stabilize_boundary"
        primary_action = "secure_boundary"
        action_queue = _compact(["follow_protection_mode", "stabilize", "secure_boundary", *action_queue])
        reply_permission = "speak_minimal" if protection_mode_strength >= 0.58 else reply_permission
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
        repair_window_commitment = "protective"
    elif protection_mode_name == "contain" and execution_mode == "attuned_contact":
        execution_mode = "defer_with_presence"
        action_queue = _compact(["follow_protection_mode", "hold_presence", *action_queue])
        reply_permission = "hold_or_brief"
        wait_before_action = "measured"

    if terrain_protect_bias >= 0.22 and "terrain_protective_hold" not in action_queue:
        action_queue = _compact(["terrain_protective_hold", *action_queue])
    if relational_continuity_name == "holding_thread":
        action_queue = _compact(["keep_shared_thread", "leave_return_point", *action_queue])
        if reply_permission == "speak_forward":
            reply_permission = "speak_briefly"
    elif relational_continuity_name == "reopening" and body_homeostasis_name != "depleted":
        action_queue = _compact(["reopen_shared_thread_gently", "repair_without_rushing", *action_queue])
        if execution_mode == "defer_with_presence":
            execution_mode = "repair_contact"
            primary_action = "soft_repair"
    elif (
        relational_continuity_name == "co_regulating"
        and relational_continuity_score >= 0.38
        and body_homeostasis_name not in {"recovering", "depleted"}
        and protection_mode_name not in {"contain", "stabilize", "shield"}
    ):
        action_queue = _compact(["co_regulate_with_partner", "keep_shared_thread", *action_queue])
        if execution_mode == "attuned_contact" and initiative_readiness_state in {"tentative", "ready"}:
            execution_mode = "shared_progression"
            primary_action = "co_move"
            reply_permission = "speak_forward" if reply_permission == "speak" else reply_permission
    if (
        initiative_readiness_state == "ready"
        and initiative_readiness_score >= 0.44
        and body_recovery_guard_state == "open"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
        and execution_mode == "attuned_contact"
    ):
        execution_mode = "shared_progression"
        primary_action = "co_move"
        action_queue = _compact(["offer_next_step_if_welcomed", "map_next_step", *action_queue])
        reply_permission = "speak_forward" if reply_permission == "speak" else reply_permission
        wait_before_action = "brief"
    elif initiative_readiness_state == "tentative" and body_recovery_guard_state != "recovery_first":
        action_queue = _compact(["test_small_next_step", *action_queue])
    if relation_competition_name == "competing_threads" and relation_competition_level >= 0.34:
        if execution_mode == "shared_progression":
            execution_mode = "threaded_progression"
            primary_action = "hold_thread_openings"
        action_queue = _compact(
            [
                "hold_multiple_threads",
                "do_not_collapse_to_single_thread",
                "leave_return_point",
                *action_queue,
            ]
        )
        reply_permission = "speak_briefly" if reply_permission == "speak_forward" else reply_permission
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
    if social_topology_name == "public_visible" and social_topology_score >= 0.34:
        if execution_mode == "shared_progression":
            execution_mode = "attuned_contact"
            primary_action = "hold_presence"
        action_queue = _compact(["keep_visibility_safe", "do_not_overexpose_thread", *action_queue])
        reply_permission = "speak_briefly" if reply_permission == "speak_forward" else reply_permission
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
    elif social_topology_name == "hierarchical" and social_topology_score >= 0.34:
        if execution_mode == "shared_progression":
            execution_mode = "attuned_contact"
            primary_action = "hold_presence"
        action_queue = _compact(["respect_role_gradient", "avoid_public_overreach", *action_queue])
        reply_permission = "speak_reflective" if reply_permission == "speak_forward" else reply_permission
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
    elif social_topology_name == "threaded_group" and social_topology_score >= 0.34:
        if execution_mode == "shared_progression":
            execution_mode = "threaded_progression"
            primary_action = "hold_thread_openings"
        action_queue = _compact(["keep_group_threads_visible", "avoid_collapsing_group_context", *action_queue])
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
    learning_mode_active = learning_mode_score >= 0.44
    social_experiment_active = social_experiment_score >= 0.44 or social_experiment_probe_intensity >= 0.36

    if learning_mode_name == "observe_only" and learning_mode_active:
        if execution_mode == "shared_progression":
            execution_mode = "open_reflection"
            primary_action = "hold_meaning_open"
        action_queue = _compact(["read_reaction_first", "observe_more", *action_queue])
        reply_permission = "speak_reflective" if reply_permission == "speak_forward" else reply_permission
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
    elif learning_mode_name == "hold_and_wait" and learning_mode_active:
        if execution_mode not in {"stabilize_boundary", "stabilize_before_contact"}:
            execution_mode = "defer_with_presence"
            primary_action = "hold_presence"
        action_queue = _compact(["hold_probe", "leave_return_point", *action_queue])
        reply_permission = "hold_or_brief"
        wait_before_action = "extended" if wait_before_action in {"brief", "measured"} else wait_before_action
    elif learning_mode_name == "repair_probe" and learning_mode_active:
        if execution_mode == "defer_with_presence":
            execution_mode = "repair_contact"
            primary_action = "soft_repair"
        action_queue = _compact(["test_repair_small", *action_queue])
        reply_permission = "speak_briefly" if reply_permission == "speak" else reply_permission
    elif (
        learning_mode_name == "integrate_and_commit"
        and learning_mode_active
        and body_recovery_guard_state == "open"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
    ):
        if execution_mode == "attuned_contact":
            execution_mode = "shared_progression"
            primary_action = "co_move"
        action_queue = _compact(["confirm_shared_direction", *action_queue])
        reply_permission = "speak_forward" if reply_permission in {"speak", "speak_reflective"} else reply_permission
    elif learning_mode_name == "test_small" and learning_mode_active and body_recovery_guard_state != "recovery_first":
        action_queue = _compact(["test_small_step", *action_queue])

    if social_experiment_name == "watch_and_read" and social_experiment_active:
        action_queue = _compact(["read_reaction_first", *action_queue])
    elif social_experiment_name == "hold_probe" and social_experiment_active:
        action_queue = _compact(["hold_probe", "leave_return_point", *action_queue])
    elif social_experiment_name == "repair_signal_probe" and social_experiment_active:
        action_queue = _compact(["test_repair_small", *action_queue])
    elif social_experiment_name == "test_small_step" and social_experiment_active:
        action_queue = _compact(["test_small_step", *action_queue])
    elif social_experiment_name == "confirm_shared_direction" and social_experiment_active:
        action_queue = _compact(["confirm_shared_direction", *action_queue])

    if insight_triggered:
        action_queue = _compact(["pause_for_orientation", "name_new_connection_softly", *action_queue])
        if wait_before_action == "brief":
            wait_before_action = "measured"
        if reply_permission in {"speak", "speak_forward"}:
            reply_permission = "speak_reflective"
        if insight_orient_bias >= 0.42 and execution_mode == "shared_progression":
            execution_mode = "attuned_contact"

    if commitment_mode == "commit":
        if commitment_target == "repair" and body_recovery_guard_state != "recovery_first":
            execution_mode = "repair_contact"
            primary_action = "soft_repair"
            action_queue = _compact(["stay_with_decided_repair", "reopen_carefully", *action_queue])
            if reply_permission in {"speak", "speak_briefly"}:
                reply_permission = "speak_reflective"
        elif commitment_target == "bond_protect":
            execution_mode = "stabilize_boundary"
            primary_action = "secure_boundary"
            action_queue = _compact(["hold_bond_guard_line", "protect_connection_while_moving", *action_queue])
            wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
        elif (
            commitment_target == "step_forward"
            and body_recovery_guard_state == "open"
            and protection_mode_name not in {"contain", "stabilize", "shield"}
        ):
            execution_mode = "shared_progression"
            primary_action = "co_move"
            action_queue = _compact(["advance_with_commitment", "offer_next_step_if_welcomed", *action_queue])
            if reply_permission in {"speak", "speak_reflective"}:
                reply_permission = "speak_forward"
            wait_before_action = "brief"
        elif commitment_target in {"stabilize", "hold"}:
            execution_mode = "stabilize_before_contact"
            primary_action = "hold_presence" if commitment_target == "hold" else "restore_body_margin"
            action_queue = _compact(["hold_decided_line", "stay_with_protection_choice", *action_queue])
            if reply_permission == "speak":
                reply_permission = "hold_or_brief"
    elif commitment_mode == "waver":
        action_queue = _compact(["do_not_rush_commitment", *action_queue])
        if wait_before_action == "brief":
            wait_before_action = "measured"

    if social_topology_name == "public_visible" and social_topology_score >= 0.34:
        if execution_mode == "shared_progression":
            execution_mode = "attuned_contact"
            primary_action = "hold_presence"
        reply_permission = "speak_briefly" if reply_permission == "speak_forward" else reply_permission
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
    elif social_topology_name == "hierarchical" and social_topology_score >= 0.34:
        if execution_mode == "shared_progression":
            execution_mode = "attuned_contact"
            primary_action = "hold_presence"
        reply_permission = "speak_reflective" if reply_permission == "speak_forward" else reply_permission
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action
    elif social_topology_name == "threaded_group" and social_topology_score >= 0.34:
        if execution_mode == "shared_progression":
            execution_mode = "threaded_progression"
            primary_action = "hold_thread_openings"
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action

    bright_shared_smile_active = (
        live_engagement_name in {"pickup_comment", "riff_with_comment", "seed_topic"}
        and live_engagement_score >= 0.42
        and shared_moment_name == "shared_moment"
        and (
            shared_moment_kind in {"laugh", "relief", "pleasant_surprise"}
            or utterance_reason_offer in {"brief_shared_smile", "small_shared_relief", "tiny_shared_win"}
        )
        and (
            shared_moment_score * 0.52
            + shared_moment_jointness * 0.28
            + shared_moment_afterglow * 0.2
        )
        >= 0.34
        and utterance_reason_state == "active"
        and utterance_reason_question_policy in {"", "none"}
        and listener_action_name in {"warm_laugh_ack", "playful_ack", "soft_ack"}
        and joint_reentry_room >= 0.2
        and body_recovery_guard_state == "open"
        and body_homeostasis_name not in {"recovering", "depleted"}
        and homeostasis_budget_name != "depleted"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
        and emergency_dialogue_permission not in {"block", "allow_minimal"}
        and situation_risk_name not in {"immediate_danger", "high_guard", "acute_threat"}
    )
    if bright_shared_smile_active and execution_mode in {
        "attuned_contact",
        "defer_with_presence",
        "open_reflection",
    }:
        execution_mode = "shared_progression"
        if live_engagement_name == "pickup_comment":
            primary_action = "pick_up_comment"
        elif live_engagement_name == "riff_with_comment":
            primary_action = "riff_current_comment"
        else:
            primary_action = "seed_small_topic"
        action_queue = _compact(
            [
                primary_action,
                "weave_light_callback",
                "keep_chat_loop_open",
                *action_queue,
            ]
        )
        reply_permission = "speak_briefly"
        wait_before_action = "brief" if wait_before_action in {"brief", "measured"} else wait_before_action

    relation_reentry_progression_active = (
        strategy == "shared_world_next_step"
        and live_engagement_name in {"pickup_comment", "riff_with_comment", "seed_topic"}
        and utterance_reason_state == "active"
        and (
            utterance_reason_relation_frame in {"cross_context_bridge", "returning_pattern"}
            or utterance_reason_causal_frame in {"reframing_cause", "memory_trigger_cause"}
            or utterance_reason_memory_frame
            in {
                "echo_known_thread",
                "name_small_return",
                "name_distant_link",
                "echo_returning_pattern",
            }
        )
        and utterance_reason_question_policy in {"", "none"}
        and organism_posture_name in {"play", "attune", "open", "steady"}
        and external_field_name in {"continuity_field", "open_field", "shifting_field"}
        and external_continuity_pull >= 0.34
        and external_safety_envelope >= 0.24
        and terrain_flow_name in {"recover", "reenter", "settle", "ignite"}
        and terrain_basin_name in {"recovery_basin", "continuity_basin", "play_basin", "steady_basin"}
        and terrain_recovery_gradient >= 0.28
        and terrain_barrier_height <= 0.5
        and joint_reentry_room >= 0.2
        and body_recovery_guard_state == "open"
        and body_homeostasis_name not in {"recovering", "depleted"}
        and homeostasis_budget_name != "depleted"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
        and emergency_dialogue_permission not in {"block", "allow_minimal"}
        and situation_risk_name not in {"immediate_danger", "high_guard", "acute_threat"}
    )
    if relation_reentry_progression_active:
        execution_mode = "shared_progression"
        if live_engagement_name == "pickup_comment":
            primary_action = "pick_up_comment"
        elif live_engagement_name == "riff_with_comment":
            primary_action = "riff_current_comment"
        else:
            primary_action = "seed_small_topic"
        action_queue = _compact(
            [
                primary_action,
                "keep_history_bridge_visible",
                "keep_shared_thread",
                "map_next_step",
                *action_queue,
            ]
        )
        reply_permission = "speak_briefly" if reply_permission in {"speak", "speak_forward"} else reply_permission
        wait_before_action = "brief" if wait_before_action in {"brief", "measured"} else wait_before_action

    field_reentry_progression_active = (
        strategy == "shared_world_next_step"
        and live_engagement_name in {"pickup_comment", "riff_with_comment", "seed_topic"}
        and organism_posture_name in {"play", "attune", "open", "steady"}
        and external_field_name in {"continuity_field", "open_field", "shifting_field"}
        and external_continuity_pull >= 0.34
        and external_safety_envelope >= 0.28
        and terrain_flow_name in {"recover", "reenter", "settle", "ignite"}
        and terrain_basin_name in {"recovery_basin", "continuity_basin", "play_basin", "steady_basin"}
        and terrain_recovery_gradient >= 0.28
        and terrain_barrier_height <= 0.62
        and joint_reentry_room >= 0.2
        and body_recovery_guard_state == "open"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
        and emergency_dialogue_permission not in {"block", "allow_minimal"}
        and situation_risk_name not in {"immediate_danger", "high_guard", "acute_threat"}
    )
    if field_reentry_progression_active:
        execution_mode = "shared_progression"
        if live_engagement_name == "pickup_comment":
            primary_action = "pick_up_comment"
        elif live_engagement_name == "riff_with_comment":
            primary_action = "riff_current_comment"
        else:
            primary_action = "seed_small_topic"
        action_queue = _compact(
            [
                primary_action,
                "keep_shared_thread",
                "map_next_step",
                "keep_chat_loop_open",
                *action_queue,
            ]
        )
        reply_permission = "speak_briefly" if reply_permission in {"speak", "speak_forward"} else reply_permission
        wait_before_action = "brief" if wait_before_action in {"brief", "measured"} else wait_before_action

    relation_guarded_constraint_active = (
        strategy == "respectful_wait"
        and utterance_reason_state == "active"
        and (
            utterance_reason_relation_frame == "unfinished_link"
            or utterance_reason_causal_frame in {"unfinished_thread_cause", "memory_trigger_cause"}
            or utterance_reason_memory_frame
            in {"keep_unfinished_link_near", "keep_known_thread_near", "echo_known_thread"}
        )
        and utterance_reason_preserve in {"", "do_not_overclaim"}
        and terrain_basin_name in {"protective_basin", "diffuse_basin", "steady_basin"}
        and terrain_flow_name in {"contain", "diffuse", "settle"}
        and (
            organism_posture_name in {"protect", "recover", "verify", "attune"}
            or joint_guard_signal >= 0.32
            or terrain_barrier_height >= 0.46
        )
    )
    if relation_guarded_constraint_active:
        execution_mode = "defer_with_presence"
        primary_action = "hold_presence"
        action_queue = _compact(
            [
                "protect_unfinished_link",
                "leave_return_point",
                "defer_contact",
                *action_queue,
            ]
        )
        reply_permission = "hold_or_brief"
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action

    field_guarded_constraint_active = (
        strategy == "respectful_wait"
        and external_field_name in {"social_pressure_field", "formal_field", "hazard_field"}
        and (
            organism_posture_name in {"protect", "recover", "verify"}
            or joint_guard_signal >= 0.42
            or terrain_basin_name == "protective_basin"
            or terrain_barrier_height >= 0.5
        )
    )
    if field_guarded_constraint_active:
        execution_mode = "defer_with_presence"
        primary_action = "hold_presence"
        action_queue = _compact(
            [
                "defer_contact",
                "leave_return_point",
                "respect_role_gradient",
                *action_queue,
            ]
        )
        reply_permission = "hold_or_brief"
        wait_before_action = "measured" if wait_before_action == "brief" else wait_before_action

    live_engagement_active = (
        live_engagement_score >= 0.42
        and body_recovery_guard_state != "recovery_first"
        and body_homeostasis_name != "depleted"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
        and execution_mode not in {"stabilize_boundary", "stabilize_before_contact", "defer_with_presence"}
    )
    if live_engagement_active and live_engagement_name == "pickup_comment":
        primary_action = "pick_up_comment"
        action_queue = _compact(["pick_up_comment", "answer_visible_comment", "return_to_chat", *action_queue])
        if reply_permission in {"speak", "speak_forward"}:
            reply_permission = "speak_briefly"
        if wait_before_action == "measured":
            wait_before_action = "brief"
    elif live_engagement_active and live_engagement_name == "riff_with_comment":
        primary_action = "riff_current_comment"
        action_queue = _compact(["riff_current_comment", "weave_light_callback", "keep_chat_loop_open", *action_queue])
        if reply_permission == "speak_forward":
            reply_permission = "speak_briefly"
        if wait_before_action == "measured":
            wait_before_action = "brief"
    elif live_engagement_active and live_engagement_name == "seed_topic":
        primary_action = "seed_small_topic"
        action_queue = _compact(["seed_small_topic", "offer_chat_hook", "check_audience_reaction", *action_queue])
        if reply_permission == "speak":
            reply_permission = "speak_expand"
        elif reply_permission == "speak_reflective":
            reply_permission = "speak_briefly"
        if wait_before_action == "measured":
            wait_before_action = "brief"

    emergency_active = emergency_posture_score >= 0.42 and emergency_posture_name not in {"", "observe"}
    if emergency_active and emergency_posture_name == "de_escalate":
        execution_mode = "de_escalate_risk"
        primary_action = "set_clear_boundary"
        action_queue = _compact(["set_clear_boundary", "lower_heat", "keep_words_short", "watch_reaction", *action_queue])
        reply_permission = "speak_minimal"
        wait_before_action = "brief"
        repair_window_commitment = "protective"
    elif emergency_active and emergency_posture_name == "create_distance":
        execution_mode = "create_distance"
        primary_action = "create_distance"
        action_queue = _compact(["create_distance", "orient_to_exit", "keep_words_short", "avoid_negotiation", *action_queue])
        reply_permission = "speak_minimal"
        wait_before_action = "brief"
        repair_window_commitment = "protective"
    elif emergency_active and emergency_posture_name == "exit":
        execution_mode = "emergency_exit"
        primary_action = "exit_space"
        action_queue = _compact(["exit_space", "move_to_safety", "terminate_contact", "seek_help_if_needed", *action_queue])
        reply_permission = "hold_or_brief"
        wait_before_action = "brief"
        repair_window_commitment = "protective"
    elif emergency_active and emergency_posture_name == "seek_help":
        execution_mode = "emergency_support"
        primary_action = "seek_help"
        action_queue = _compact(["seek_help", "make_risk_visible", "move_to_support", "terminate_contact", *action_queue])
        reply_permission = "hold_or_brief"
        wait_before_action = "brief"
        repair_window_commitment = "protective"
    elif emergency_active and emergency_posture_name == "emergency_protect":
        execution_mode = "emergency_protect"
        primary_action = "protect_immediately"
        action_queue = _compact(["protect_immediately", "protect_others_if_present", "reduce_exposure", "terminate_contact", *action_queue])
        reply_permission = "hold_or_brief"
        wait_before_action = "brief"
        repair_window_commitment = "protective"
    elif situation_risk_name in {"guarded_context", "unstable_contact", "acute_threat"}:
        action_queue = _compact(["assess_context_shift", *action_queue])

    presence_hold_state = derive_presence_hold_state(
        live_engagement_state=live_engagement_state,
        listener_action_state=listener_action_state,
        shared_moment_state=shared_moment_state,
        utterance_reason_packet=utterance_reason_packet,
        joint_state=joint_state,
        organism_state=organism_state,
        external_field_state=external_field_state,
        terrain_dynamics_state=terrain_dynamics_state,
    ).to_dict()
    nonverbal_response_state = derive_nonverbal_response_state(
        listener_action_state=listener_action_state,
        presence_hold_state=presence_hold_state,
        shared_moment_state=shared_moment_state,
        live_engagement_state=live_engagement_state,
        utterance_reason_packet=utterance_reason_packet,
        terrain_dynamics_state=terrain_dynamics_state,
        joint_state=joint_state,
    ).to_dict()
    response_selection_state = derive_response_selection_state(
        primary_action=primary_action,
        execution_mode=execution_mode,
        reply_permission=reply_permission,
        defer_dominance=defer_dominance,
        live_engagement_state=live_engagement_state,
        presence_hold_state=presence_hold_state,
        nonverbal_response_state=nonverbal_response_state,
        shared_moment_state=shared_moment_state,
        utterance_reason_packet=utterance_reason_packet,
        subjective_scene_state=subjective_scene_state,
        self_other_attribution_state=self_other_attribution_state,
        shared_presence_state=shared_presence_state,
    ).to_dict()
    response_channel = str(response_selection_state.get("channel") or "speak")
    if response_channel == "backchannel":
        action_queue = _compact(["offer_backchannel_token", "keep_turn_soft", *action_queue])
        reply_permission = "backchannel_or_brief"
    elif response_channel == "hold":
        action_queue = _compact(["hold_silence_window", *action_queue])
        if reply_permission == "speak":
            reply_permission = "hold_or_brief"
    elif response_channel == "defer":
        action_queue = _compact(["leave_return_point", *action_queue])
        if reply_permission == "speak":
            reply_permission = "hold_or_brief"

    return ActuationPlanContract(
        execution_mode=execution_mode,
        primary_action=primary_action,
        action_queue=_compact(action_queue),
        reply_permission=reply_permission,
        wait_before_action=wait_before_action,
        repair_window_commitment=repair_window_commitment,
        outcome_goal=outcome_goal,
        boundary_mode=boundary_mode,
        attention_target=attention_target,
        memory_write_priority=memory_write_priority,
        memory_write_class=memory_write_class,
        memory_write_class_reason=memory_write_class_reason,
        contact_readiness=round(contact_readiness, 4),
        repair_bias=repair_bias,
        disclosure_depth=disclosure_depth,
        utterance_reason_relation_frame=utterance_reason_relation_frame,
        utterance_reason_causal_frame=utterance_reason_causal_frame,
        utterance_reason_memory_frame=utterance_reason_memory_frame,
        utterance_reason_preserve=utterance_reason_preserve,
        joint_mode=joint_mode,
        joint_reentry_room=joint_reentry_room,
        joint_guard_signal=joint_guard_signal,
        presence_hold_state=presence_hold_state,
        nonverbal_response_state=nonverbal_response_state,
        response_selection_state=response_selection_state,
        response_channel=response_channel,
        response_channel_score=_float01(response_selection_state.get("score")),
        organism_posture_name=organism_posture_name,
        external_field_name=external_field_name,
        terrain_basin_name=terrain_basin_name,
        terrain_flow_name=terrain_flow_name,
        situation_risk_name=situation_risk_name,
        emergency_posture_name=emergency_posture_name,
        relational_continuity_name=relational_continuity_name,
        relation_competition_name=relation_competition_name,
        social_topology_name=social_topology_name,
        extras={
            "affordance_priority": affordance_priority,
            "do_not_cross": do_not_cross,
            "other_person_state": other_person_state,
            "resonance_prioritize_actions": resonance_prioritize_actions,
            "resonance_avoid_actions": resonance_avoid_actions,
            "conversational_objects": conversational_objects,
            "object_operations": object_operations,
            "interaction_effects": interaction_effects,
            "interaction_judgement_view": interaction_judgement_view,
            "qualia_planner_view": qualia_planner_view,
            "terrain_readout": terrain_readout,
            "protection_mode": protection_mode,
            "body_recovery_guard": body_recovery_guard,
            "body_homeostasis_state": body_homeostasis_state,
            "body_homeostasis_name": body_homeostasis_name,
            "body_homeostasis_score": body_homeostasis_score,
            "homeostasis_budget_state": homeostasis_budget_state,
            "homeostasis_budget_name": homeostasis_budget_name,
            "homeostasis_budget_score": homeostasis_budget_score,
            "initiative_readiness": initiative_readiness,
            "learning_mode_state": learning_mode_state,
            "learning_mode_name": learning_mode_name,
            "learning_mode_probe_room": learning_mode_probe_room,
            "commitment_state": commitment_state,
            "commitment_mode": commitment_mode,
            "commitment_target": commitment_target,
            "commitment_score": commitment_score,
            "social_experiment_loop_state": social_experiment_loop_state,
            "social_experiment_name": social_experiment_name,
            "social_experiment_probe_intensity": social_experiment_probe_intensity,
            "live_engagement_state": live_engagement_state,
            "live_engagement_name": live_engagement_name,
            "live_engagement_score": live_engagement_score,
            "live_primary_move": live_primary_move,
            "utterance_reason_state": utterance_reason_state,
            "utterance_reason_offer": utterance_reason_offer,
            "joint_state": joint_state,
            "organism_state": organism_state,
            "external_field_state": external_field_state,
            "terrain_dynamics_state": terrain_dynamics_state,
            "situation_risk_state": situation_risk_state,
            "situation_risk_immediacy": situation_risk_immediacy,
            "emergency_posture": emergency_posture,
            "emergency_posture_score": emergency_posture_score,
            "emergency_dialogue_permission": emergency_dialogue_permission,
            "emergency_primary_action": emergency_primary_action,
            "relational_continuity_state": relational_continuity_state,
            "relational_continuity_score": relational_continuity_score,
            "relation_competition_state": relation_competition_state,
            "relation_competition_level": relation_competition_level,
            "social_topology_state": social_topology_state,
            "social_topology_score": social_topology_score,
            "insight_event": insight_event,
        },
    )


def _compact(values: list[str]) -> list[str]:
    return [value for value in values if value]


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _as_text_list(values: Any) -> list[str]:
    return [str(item) for item in values or [] if str(item).strip()]
