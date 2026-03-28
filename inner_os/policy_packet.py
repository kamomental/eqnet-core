from __future__ import annotations

from typing import Any, Mapping, Sequence

from .agenda_state import derive_agenda_state
from .agenda_window_state import derive_agenda_window_state
from .commitment_state import derive_commitment_state
from .cultural_conversation_state import derive_cultural_conversation_state
from .expressive_style_state import derive_expressive_style_state
from .emergency_posture import derive_emergency_posture
from .learning_mode_state import derive_learning_mode_state
from .lightness_budget_state import derive_lightness_budget_state
from .live_engagement_state import derive_live_engagement_state
from .persona_memory_fragment import build_persona_memory_fragments
from .persona_memory_selector import derive_persona_memory_selection
from .relation_competition import derive_relation_competition_state
from .relational_style_memory import derive_relational_style_memory_state
from .social_experiment_loop import derive_social_experiment_loop_state
from .social_topology_state import coerce_social_topology_label, derive_social_topology_state
from .situation_risk_state import derive_situation_risk_state
from .temperament_estimate import derive_temperament_estimate


def derive_interaction_policy_packet(
    *,
    dialogue_act: str,
    current_focus: str,
    current_risks: Sequence[str],
    reportable_facts: Sequence[str],
    relation_bias_strength: float,
    related_person_ids: Sequence[str],
    partner_address_hint: str,
    partner_timing_hint: str,
    partner_stance_hint: str,
    partner_social_interpretation: str = "",
    recent_strain: float = 0.0,
    orchestration: Mapping[str, Any],
    surface_profile: Mapping[str, Any],
    live_regulation: Any,
    scene_state: Mapping[str, Any] | None = None,
    interaction_option_candidates: Sequence[Mapping[str, Any]] | None = None,
    affect_blend_state: Mapping[str, Any] | None = None,
    constraint_field: Mapping[str, Any] | None = None,
    conscious_workspace: Mapping[str, Any] | None = None,
    resonance_evaluation: Mapping[str, Any] | None = None,
    conversational_objects: Mapping[str, Any] | None = None,
    object_operations: Mapping[str, Any] | None = None,
    interaction_effects: Mapping[str, Any] | None = None,
    interaction_judgement_view: Mapping[str, Any] | None = None,
    qualia_planner_view: Mapping[str, Any] | None = None,
    affective_position: Mapping[str, Any] | None = None,
    terrain_readout: Mapping[str, Any] | None = None,
    protection_mode: Mapping[str, Any] | None = None,
    insight_event: Mapping[str, Any] | None = None,
    insight_reframing_bias: float = 0.0,
    insight_class_focus: str = "",
    association_reweighting_focus: str = "",
    association_reweighting_reason: str = "",
    insight_terrain_shape_target: str = "",
    self_state: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    orchestration_mode = str(orchestration.get("orchestration_mode") or "attune")
    dominant_driver = str(orchestration.get("dominant_driver") or "shared_attention")
    contact_readiness = _clamp01(float(orchestration.get("contact_readiness", 0.0) or 0.0))
    coherence_score = _clamp01(float(orchestration.get("coherence_score", 0.0) or 0.0))
    human_presence_signal = _clamp01(float(orchestration.get("human_presence_signal", 0.0) or 0.0))
    distance_strategy = str(orchestration.get("distance_strategy") or getattr(live_regulation, "distance_expectation", "holding_space"))
    opening_pace = str(surface_profile.get("opening_pace_windowed") or "ready")
    gaze_return = str(surface_profile.get("return_gaze_expectation") or "soft_return")
    repair_bias = bool(orchestration.get("repair_bias") or getattr(live_regulation, "repair_window_open", False))
    strained_pause = _clamp01(float(getattr(live_regulation, "strained_pause", 0.0) or 0.0))
    future_pull = _clamp01(float(getattr(live_regulation, "future_loop_pull", 0.0) or 0.0))
    fantasy_pull = _clamp01(float(getattr(live_regulation, "fantasy_loop_pull", 0.0) or 0.0))
    recent_strain = _clamp01(recent_strain)
    scene = dict(scene_state or {})
    scene_family = str(scene.get("scene_family") or "").strip()
    scene_tags = [str(item) for item in scene.get("scene_tags") or [] if str(item).strip()]
    option_candidates = [dict(item) for item in interaction_option_candidates or [] if isinstance(item, Mapping)]
    top_option_family = str(option_candidates[0].get("family_id") or "").strip() if option_candidates else ""
    affect_blend = dict(affect_blend_state or {})
    constraint = dict(constraint_field or {})
    workspace = dict(conscious_workspace or {})
    resonance = dict(resonance_evaluation or {})
    conversational_objects_payload = dict(conversational_objects or {})
    object_operations_payload = dict(object_operations or {})
    interaction_effects_payload = dict(interaction_effects or {})
    interaction_judgement_payload = dict(interaction_judgement_view or {})
    qualia_planner_payload = dict(qualia_planner_view or {})
    affective_position_payload = dict(affective_position or {})
    terrain_readout_payload = dict(terrain_readout or {})
    protection_mode_payload = dict(protection_mode or {})
    insight_event_payload = dict(insight_event or {})
    self_state_payload = dict(self_state or {})
    identity_arc_kind = str(self_state_payload.get("identity_arc_kind") or "").strip()
    identity_arc_phase = str(self_state_payload.get("identity_arc_phase") or "").strip()
    identity_arc_summary = str(self_state_payload.get("identity_arc_summary") or "").strip()
    identity_arc_open_tension = str(self_state_payload.get("identity_arc_open_tension") or "").strip()
    identity_arc_stability = _clamp01(float(self_state_payload.get("identity_arc_stability", 0.0) or 0.0))
    discussion_thread_registry_snapshot = {
        key: value
        for key, value in dict(self_state_payload.get("discussion_thread_registry_snapshot") or {}).items()
    }
    temperament_estimate = derive_temperament_estimate(self_state_payload).to_dict()
    relation_competition_state = derive_relation_competition_state(
        self_state=self_state_payload,
        related_person_ids=related_person_ids,
        dominant_hint_person_id=str(self_state_payload.get("related_person_id") or "").strip(),
    ).to_dict()
    active_relation_table = {
        "entries": list(relation_competition_state.get("entries") or []),
        "dominant_person_id": str(relation_competition_state.get("dominant_person_id") or "").strip(),
        "top_person_ids": [
            str(item)
            for item in relation_competition_state.get("top_person_ids") or []
            if str(item).strip()
        ],
        "total_people": int(relation_competition_state.get("total_people") or 0),
    }
    social_topology_state = derive_social_topology_state(
        scene_state=scene,
        relation_competition_state=relation_competition_state,
        related_person_ids=related_person_ids,
        self_state=self_state_payload,
    ).to_dict()
    social_topology = coerce_social_topology_label(str(social_topology_state.get("state") or "ambient"))
    association_reweighting_focus = str(association_reweighting_focus or "").strip()
    association_reweighting_reason = str(association_reweighting_reason or "").strip()
    insight_terrain_shape_target = str(insight_terrain_shape_target or "").strip()
    object_operation_items = [
        dict(item)
        for item in object_operations_payload.get("operations") or []
        if isinstance(item, Mapping)
    ]
    primary_operation_id = str(object_operations_payload.get("primary_operation_id") or "").strip()
    primary_object_operation = next(
        (
            item
            for item in object_operation_items
            if str(item.get("operation_id") or "").strip() == primary_operation_id
        ),
        {},
    )
    operation_kinds = {
        str(item.get("operation_kind") or "").strip()
        for item in object_operation_items
        if str(item.get("operation_kind") or "").strip()
    }
    question_budget = int(object_operations_payload.get("question_budget") or 0)
    question_pressure = _clamp01(float(object_operations_payload.get("question_pressure", 0.0) or 0.0))
    defer_dominance = _clamp01(float(object_operations_payload.get("defer_dominance", 0.0) or 0.0))
    interaction_effect_items = [
        dict(item)
        for item in interaction_effects_payload.get("effects") or []
        if isinstance(item, Mapping)
    ]
    primary_effect_ids = {
        str(item)
        for item in interaction_effects_payload.get("primary_effect_ids") or []
        if str(item).strip()
    }
    effect_kinds = {
        str(item.get("effect_kind") or "").strip()
        for item in interaction_effect_items
        if str(item.get("effect_kind") or "").strip()
    }
    conversational_object_items = [
        dict(item)
        for item in conversational_objects_payload.get("objects") or []
        if isinstance(item, Mapping)
    ]
    primary_conversational_object_id = str(conversational_objects_payload.get("primary_object_id") or "").strip()
    primary_conversational_object = next(
        (
            item
            for item in conversational_object_items
            if str(item.get("object_id") or "").strip() == primary_conversational_object_id
        ),
        {},
    )
    primary_conversational_object_label = _normalize_focus_label(str(
        primary_conversational_object.get("label")
        or primary_conversational_object.get("explicit_text")
        or ""
    ).strip())
    deferred_object_ids = {
        str(item)
        for item in conversational_objects_payload.get("deferred_object_ids") or []
        if str(item).strip()
    }
    deferred_object_labels = [
        str(item.get("label") or "").strip()
        for item in conversational_object_items
        if str(item.get("object_id") or "").strip() in deferred_object_ids
        and str(item.get("label") or "").strip()
    ]
    ordered_operation_kinds = _ordered_operation_kinds(
        primary_operation_id=primary_operation_id,
        operation_items=object_operation_items,
    )
    ordered_effect_kinds = _ordered_effect_kinds(
        primary_effect_ids=primary_effect_ids,
        effect_items=interaction_effect_items,
    )
    conflict_level = _clamp01(float(affect_blend.get("conflict_level", 0.0) or 0.0))
    residual_tension = _clamp01(float(affect_blend.get("residual_tension", 0.0) or 0.0))
    dominant_blend_mode = str(affect_blend.get("dominant_mode") or "").strip()
    body_cost = _clamp01(float(constraint.get("body_cost", 0.0) or 0.0))
    boundary_pressure = _clamp01(float(constraint.get("boundary_pressure", 0.0) or 0.0))
    repair_pressure = _clamp01(float(constraint.get("repair_pressure", 0.0) or 0.0))
    shared_world_pressure = _clamp01(float(constraint.get("shared_world_pressure", 0.0) or 0.0))
    constraint_disclosure_limit = str(constraint.get("disclosure_limit") or "").strip()
    reportability_limit = str(constraint.get("reportability_limit") or "").strip()
    workspace_mode = str(workspace.get("workspace_mode") or "").strip()
    workspace_stability = _clamp01(float(workspace.get("workspace_stability", 0.0) or 0.0))
    reportable_slice = [str(item) for item in workspace.get("reportable_slice") or [] if str(item).strip()]
    withheld_slice = [str(item) for item in workspace.get("withheld_slice") or [] if str(item).strip()]
    actionable_slice = [str(item) for item in workspace.get("actionable_slice") or [] if str(item).strip()]
    reportability_gate = dict(workspace.get("reportability_gate") or {})
    reportability_gate_mode = str(reportability_gate.get("gate_mode") or reportability_limit or "").strip()
    workspace_decision = {
        "workspace_mode": workspace_mode,
        "ignition_phase": str(workspace.get("ignition_phase") or "").strip(),
        "winner_margin": round(_clamp01(float(workspace.get("winner_margin", 0.0) or 0.0)), 4),
        "slot_scores": {
            str(key): round(_clamp01(float(value or 0.0)), 4)
            for key, value in dict(workspace.get("slot_scores") or {}).items()
        },
        "dominant_inputs": [
            str(item)
            for item in workspace.get("dominant_inputs") or []
            if str(item).strip()
        ],
    }
    other_person_state = dict(resonance.get("estimated_other_person_state") or {})
    other_detail_room = str(other_person_state.get("detail_room_level") or "").strip()
    other_acknowledgement_need = str(other_person_state.get("acknowledgement_need_level") or "").strip()
    other_pressure_sensitivity = str(other_person_state.get("pressure_sensitivity_level") or "").strip()
    other_next_step_room = str(other_person_state.get("next_step_room_level") or "").strip()
    resonance_prioritize = [str(item) for item in resonance.get("prioritize_actions") or [] if str(item).strip()]
    resonance_avoid = [str(item) for item in resonance.get("avoid_actions") or [] if str(item).strip()]
    resonance_expected = [str(item) for item in resonance.get("expected_effects") or [] if str(item).strip()]
    respectful_context = (
        partner_timing_hint == "delayed"
        or partner_stance_hint == "respectful"
        or scene_family == "reverent_distance"
        or "high_norm" in scene_tags
    )
    future_open_context = "future_open" in partner_social_interpretation
    repair_context = (
        orchestration_mode == "repair"
        or scene_family == "repair_window"
        or top_option_family == "repair"
        or repair_bias
        or repair_pressure >= 0.46
        or strained_pause >= 0.44
        or (recent_strain >= 0.3 and (strained_pause >= 0.3 or contact_readiness <= 0.62))
    )
    contain_context = (
        "danger" in current_risks
        or orchestration_mode == "contain"
        or scene_family == "guarded_boundary"
        or top_option_family == "contain"
        or body_cost >= 0.66
        or boundary_pressure >= 0.68
    )
    advance_context = (
        not respectful_context
        and not repair_context
        and not contain_context
        and strained_pause <= 0.34
        and reportability_limit != "withhold"
        and (
            (
                future_open_context
                and (orchestration_mode == "advance" or future_pull >= 0.44 or top_option_family == "co_move" or scene_family == "shared_world" or shared_world_pressure >= 0.5)
                and contact_readiness >= 0.48
                and coherence_score >= 0.45
                and human_presence_signal >= 0.45
            )
            or (
                (orchestration_mode == "advance" or future_pull >= 0.58 or top_option_family == "co_move" or scene_family == "shared_world" or shared_world_pressure >= 0.58)
                and contact_readiness >= 0.64
                and coherence_score >= 0.58
                and human_presence_signal >= 0.56
            )
        )
    )
    clarify_context = top_option_family == "clarify" and not contain_context and not repair_context
    withdraw_context = top_option_family == "withdraw" and not repair_context and not advance_context

    response_strategy = "attune_then_extend"
    opening_move = "stay_with_visible"
    followup_move = "invite_visible_state"
    closing_move = "hold_space"

    resolved_dialogue_act = dialogue_act
    if contain_context:
        response_strategy = "contain_then_stabilize"
        opening_move = "reduce_force_and_secure_boundary"
        followup_move = "name_stable_next_anchor"
        closing_move = "do_not_overextend"
    elif repair_context:
        response_strategy = "repair_then_attune"
        opening_move = "name_overreach_and_reduce_force"
        followup_move = "invite_visible_state"
        closing_move = "hold_space"
    elif respectful_context:
        response_strategy = "respectful_wait"
        opening_move = "acknowledge_and_wait"
        followup_move = "offer_return_point"
        closing_move = "leave_space"
    elif advance_context:
        response_strategy = "shared_world_next_step"
        opening_move = "synchronize_then_propose"
        followup_move = "map_next_step"
        closing_move = "keep_pace_mutual"
    elif clarify_context:
        response_strategy = "attune_then_extend"
        resolved_dialogue_act = "clarify"
        opening_move = "narrow_scope_before_extend"
        followup_move = "invite_visible_state"
        closing_move = "leave_room_for_correction"
    elif withdraw_context:
        response_strategy = "respectful_wait"
        opening_move = "acknowledge_and_wait"
        followup_move = "protect_distance"
        closing_move = "leave_space"
    elif fantasy_pull >= 0.42 or orchestration_mode == "reflect" or top_option_family == "reflect":
        response_strategy = "reflect_without_settling"
        opening_move = "stay_with_visible"
        followup_move = "hold_meaning_open"
        closing_move = "avoid_false_closure"
    elif top_option_family == "wait":
        response_strategy = "respectful_wait"
        opening_move = "acknowledge_and_wait"
        followup_move = "offer_return_point"
        closing_move = "leave_space"
    elif top_option_family == "attune":
        response_strategy = "attune_then_extend"
        opening_move = "stay_with_visible"
        followup_move = "invite_visible_state"
        closing_move = "hold_space"

    object_driven_moves = _derive_object_driven_moves(
        primary_operation=primary_object_operation,
        operation_kinds=operation_kinds,
        effect_kinds=effect_kinds,
        question_budget=question_budget,
        question_pressure=question_pressure,
        defer_dominance=defer_dominance,
        repair_context=repair_context,
        respectful_context=respectful_context,
        advance_context=advance_context,
    )
    if object_driven_moves:
        opening_move = str(object_driven_moves.get("opening_move") or opening_move)
        followup_move = str(object_driven_moves.get("followup_move") or followup_move)
        closing_move = str(object_driven_moves.get("closing_move") or closing_move)
    if not primary_object_operation:
        primary_object_operation = _fallback_primary_operation(
            response_strategy=response_strategy,
            opening_move=opening_move,
            followup_move=followup_move,
            primary_target_label=primary_conversational_object_label or _normalize_focus_label(str(current_focus)),
        )
    if not ordered_operation_kinds:
        ordered_operation_kinds = _fallback_ordered_operations(
            response_strategy=response_strategy,
            primary_operation=primary_object_operation,
            closing_move=closing_move,
        )
        operation_kinds = set(ordered_operation_kinds)
    if not ordered_effect_kinds:
        ordered_effect_kinds = _fallback_ordered_effects(
            response_strategy=response_strategy,
            closing_move=closing_move,
        )
        effect_kinds = set(ordered_effect_kinds)

    disclosure_depth = "light"
    if response_strategy in {"shared_world_next_step", "attune_then_extend"} and contact_readiness >= 0.52:
        disclosure_depth = "medium"
    if response_strategy == "contain_then_stabilize" or strained_pause >= 0.56 or withdraw_context:
        disclosure_depth = "minimal"
    if constraint_disclosure_limit in {"minimal", "light", "medium"}:
        disclosure_depth = constraint_disclosure_limit

    attention_target = current_focus or "ambient"
    if related_person_ids:
        attention_target = f"person:{related_person_ids[0]}"
    elif primary_conversational_object:
        attention_target = str(
            primary_conversational_object.get("label")
            or primary_conversational_object.get("explicit_text")
            or attention_target
        )
    elif reportable_slice:
        attention_target = str(reportable_slice[0])
    elif actionable_slice:
        attention_target = str(actionable_slice[0])
    elif reportable_facts:
        attention_target = str(reportable_facts[0])

    memory_write_priority = "ambient"
    if related_person_ids and relation_bias_strength >= 0.28:
        memory_write_priority = "relation_episode"
    elif current_risks:
        memory_write_priority = "stability_trace"
    elif future_pull >= 0.46:
        memory_write_priority = "prospective_trace"
    if workspace_mode == "guarded_foreground" and memory_write_priority == "ambient":
        memory_write_priority = "stability_trace"
    elif workspace_stability >= 0.56 and reportable_slice and memory_write_priority == "ambient":
        memory_write_priority = "foreground_trace"
    elif actionable_slice and memory_write_priority == "ambient":
        memory_write_priority = "action_trace"
    qualia_memory_bias = _derive_qualia_memory_bias(qualia_planner_payload)
    qualia_memory_bias_applied = False
    qualia_memory_priority = str(qualia_memory_bias.get("priority") or "").strip()
    if qualia_memory_priority == "stability_trace" and memory_write_priority in {
        "ambient",
        "foreground_trace",
        "action_trace",
    }:
        memory_write_priority = "stability_trace"
        qualia_memory_bias_applied = True
    elif qualia_memory_priority == "foreground_trace" and memory_write_priority == "ambient":
        memory_write_priority = "foreground_trace"
        qualia_memory_bias_applied = True
    protection_mode_name = str(protection_mode_payload.get("mode") or "").strip()
    terrain_protect_bias = _clamp01(float(terrain_readout_payload.get("protect_bias", 0.0) or 0.0))
    terrain_approach_bias = _clamp01(float(terrain_readout_payload.get("approach_bias", 0.0) or 0.0))
    if (
        memory_write_priority in {"ambient", "foreground_trace", "action_trace"}
        and (
            protection_mode_name in {"contain", "stabilize", "shield"}
            or terrain_protect_bias >= 0.32
        )
    ):
        memory_write_priority = "stability_trace"
        qualia_memory_bias_applied = True
    qualia_memory_bias["applied"] = qualia_memory_bias_applied
    body_recovery_guard = _derive_body_recovery_guard(
        self_state=self_state_payload,
        temperament_estimate=temperament_estimate,
        qualia_planner_view=qualia_planner_payload,
        terrain_readout=terrain_readout_payload,
        protection_mode=protection_mode_payload,
        contact_readiness=contact_readiness,
    )
    body_homeostasis_state = _derive_body_homeostasis_state(
        self_state=self_state_payload,
        temperament_estimate=temperament_estimate,
        qualia_planner_view=qualia_planner_payload,
        terrain_readout=terrain_readout_payload,
        protection_mode=protection_mode_payload,
        body_recovery_guard=body_recovery_guard,
        contact_readiness=contact_readiness,
    )
    homeostasis_budget_state = _derive_homeostasis_budget_state(
        self_state=self_state_payload,
        temperament_estimate=temperament_estimate,
        body_homeostasis_state=body_homeostasis_state,
        body_recovery_guard=body_recovery_guard,
        protection_mode=protection_mode_payload,
        contact_readiness=contact_readiness,
    )
    initiative_followup_bias = {
        "state": str(self_state_payload.get("initiative_followup_state") or "hold").strip() or "hold",
        "score": round(_clamp01(float(self_state_payload.get("initiative_followup_bias", 0.0) or 0.0)), 4),
    }
    body_homeostasis_carry = {
        "focus": str(self_state_payload.get("body_homeostasis_focus") or "").strip(),
        "carry_bias": round(_clamp01(float(self_state_payload.get("body_homeostasis_carry_bias", 0.0) or 0.0)), 4),
    }
    homeostasis_budget_carry = {
        "focus": str(self_state_payload.get("homeostasis_budget_focus") or "").strip(),
        "carry_bias": round(_clamp01(float(self_state_payload.get("homeostasis_budget_bias", 0.0) or 0.0)), 4),
    }
    commitment_carry = {
        "target_focus": str(self_state_payload.get("commitment_target_focus") or "").strip(),
        "state_focus": str(self_state_payload.get("commitment_state_focus") or "waver").strip() or "waver",
        "carry_bias": round(_clamp01(float(self_state_payload.get("commitment_carry_bias", 0.0) or 0.0)), 4),
        "followup_focus": str(self_state_payload.get("commitment_followup_focus") or "").strip(),
        "mode_focus": str(self_state_payload.get("commitment_mode_focus") or "").strip(),
        "carry_reason": str(self_state_payload.get("commitment_carry_reason") or "").strip(),
    }
    agenda_carry = {
        "focus": str(self_state_payload.get("agenda_focus") or "").strip(),
        "carry_bias": round(_clamp01(float(self_state_payload.get("agenda_bias", 0.0) or 0.0)), 4),
        "reason": str(self_state_payload.get("agenda_reason") or "").strip(),
    }
    learning_mode_carry = {
        "focus": str(self_state_payload.get("learning_mode_focus") or "").strip(),
        "carry_bias": round(_clamp01(float(self_state_payload.get("learning_mode_carry_bias", 0.0) or 0.0)), 4),
    }
    social_experiment_carry = {
        "focus": str(self_state_payload.get("social_experiment_focus") or "").strip(),
        "carry_bias": round(_clamp01(float(self_state_payload.get("social_experiment_carry_bias", 0.0) or 0.0)), 4),
    }
    temperament_carry = {
        "focus": str(self_state_payload.get("temperament_focus") or "").strip(),
        "forward_bias": round(_clamp01(float(self_state_payload.get("temperament_forward_bias", 0.0) or 0.0)), 4),
        "guard_bias": round(_clamp01(float(self_state_payload.get("temperament_guard_bias", 0.0) or 0.0)), 4),
        "bond_bias": round(_clamp01(float(self_state_payload.get("temperament_bond_bias", 0.0) or 0.0)), 4),
        "recovery_bias": round(_clamp01(float(self_state_payload.get("temperament_recovery_bias", 0.0) or 0.0)), 4),
    }
    relational_continuity_carry = {
        "focus": str(self_state_payload.get("relational_continuity_focus") or "").strip(),
        "carry_bias": round(_clamp01(float(self_state_payload.get("relational_continuity_carry_bias", 0.0) or 0.0)), 4),
    }
    expressive_style_carry = {
        "focus": str(self_state_payload.get("expressive_style_focus") or "").strip(),
        "carry_bias": round(_clamp01(float(self_state_payload.get("expressive_style_carry_bias", 0.0) or 0.0)), 4),
    }
    expressive_style_history_carry = {
        "focus": str(self_state_payload.get("expressive_style_history_focus") or "").strip(),
        "carry_bias": round(_clamp01(float(self_state_payload.get("expressive_style_history_bias", 0.0) or 0.0)), 4),
    }
    banter_style_carry = {
        "focus": str(self_state_payload.get("banter_style_focus") or "").strip(),
        "carry_bias": round(_clamp01(float(self_state_payload.get("lexical_variation_carry_bias", 0.0) or 0.0)), 4),
    }
    initiative_readiness = _derive_initiative_readiness(
        self_state=self_state_payload,
        temperament_estimate=temperament_estimate,
        qualia_planner_view=qualia_planner_payload,
        terrain_readout=terrain_readout_payload,
        protection_mode=protection_mode_payload,
        body_recovery_guard=body_recovery_guard,
        contact_readiness=contact_readiness,
        coherence_score=coherence_score,
        human_presence_signal=human_presence_signal,
    )
    memory_write_class_payload = _derive_memory_write_class(
        memory_write_priority=memory_write_priority,
        qualia_planner_view=qualia_planner_payload,
        affective_position=affective_position_payload,
        terrain_readout=terrain_readout_payload,
        protection_mode=protection_mode_payload,
        workspace_mode=workspace_mode,
        deferred_object_labels=deferred_object_labels,
        relation_bias_strength=relation_bias_strength,
        related_person_ids=related_person_ids,
        insight_reframing_bias=insight_reframing_bias,
        insight_class_focus=insight_class_focus,
    )
    commitment_state = derive_commitment_state(
        self_state=self_state_payload,
        qualia_planner_view=qualia_planner_payload,
        terrain_readout=terrain_readout_payload,
        protection_mode=protection_mode_payload,
        body_recovery_guard=body_recovery_guard,
        initiative_readiness=initiative_readiness,
        initiative_followup_bias=initiative_followup_bias,
        temperament_estimate=temperament_estimate,
        memory_write_class=str(memory_write_class_payload.get("memory_class") or "episodic"),
        memory_write_class_reason=str(memory_write_class_payload.get("reason") or ""),
        insight_event=insight_event_payload,
    ).to_dict()
    agenda_state = derive_agenda_state(
        self_state=self_state_payload,
        body_recovery_guard=body_recovery_guard,
        body_homeostasis_state=body_homeostasis_state,
        homeostasis_budget_state=homeostasis_budget_state,
        initiative_readiness=initiative_readiness,
        initiative_followup_bias=initiative_followup_bias,
        commitment_state=commitment_state,
        protection_mode=protection_mode_payload,
        memory_write_class=str(memory_write_class_payload.get("memory_class") or "episodic"),
        insight_event=insight_event_payload,
    ).to_dict()
    relational_continuity_state = _derive_relational_continuity_state(
        self_state=self_state_payload,
        temperament_estimate=temperament_estimate,
        relation_bias_strength=relation_bias_strength,
        related_person_ids=related_person_ids,
        relation_competition_state=relation_competition_state,
        social_topology_state=social_topology_state,
        partner_timing_hint=partner_timing_hint,
        partner_stance_hint=partner_stance_hint,
        partner_social_interpretation=partner_social_interpretation,
        contact_readiness=contact_readiness,
        coherence_score=coherence_score,
        human_presence_signal=human_presence_signal,
        conscious_workspace=workspace,
        resonance_evaluation=resonance,
        conversational_objects=conversational_objects_payload,
        object_operations=object_operations_payload,
        interaction_effects=interaction_effects_payload,
        terrain_readout=terrain_readout_payload,
        protection_mode=protection_mode_payload,
        body_homeostasis_state=body_homeostasis_state,
    )
    attention_regulation_state = _derive_attention_regulation_state(
        current_focus=current_focus,
        reportable_slice=reportable_slice,
        withheld_slice=withheld_slice,
        current_risks=current_risks,
        primary_conversational_object_label=primary_conversational_object_label,
        body_recovery_guard=body_recovery_guard,
        body_homeostasis_state=body_homeostasis_state,
        terrain_readout=terrain_readout_payload,
        protection_mode=protection_mode_payload,
        relation_competition_state=relation_competition_state,
        social_topology_state=social_topology_state,
    )
    relational_style_memory_state = derive_relational_style_memory_state(
        self_state=self_state_payload,
        relation_bias_strength=relation_bias_strength,
        related_person_ids=related_person_ids,
        relation_competition_state=relation_competition_state,
        social_topology_state=social_topology_state,
        body_recovery_guard=body_recovery_guard,
        body_homeostasis_state=body_homeostasis_state,
    ).to_dict()
    cultural_conversation_state = derive_cultural_conversation_state(
        self_state=self_state_payload,
        social_topology_state=social_topology_state,
        relation_competition_state=relation_competition_state,
        relational_style_memory_state=relational_style_memory_state,
        body_recovery_guard=body_recovery_guard,
    ).to_dict()
    agenda_window_state = derive_agenda_window_state(
        self_state=self_state_payload,
        agenda_state=agenda_state,
        body_recovery_guard=body_recovery_guard,
        body_homeostasis_state=body_homeostasis_state,
        homeostasis_budget_state=homeostasis_budget_state,
        initiative_followup_bias=initiative_followup_bias,
        commitment_state=commitment_state,
        relational_continuity_state=relational_continuity_state,
        relation_competition_state=relation_competition_state,
        social_topology_state=social_topology_state,
        cultural_conversation_state=cultural_conversation_state,
        related_person_ids=related_person_ids,
        relation_bias_strength=relation_bias_strength,
        scene_family=scene_family,
    ).to_dict()
    grice_guard_state = _derive_grice_guard_state(
        self_state=self_state_payload,
        current_focus=current_focus,
        attention_target=attention_target,
        primary_conversational_object_label=primary_conversational_object_label,
        reportable_slice=reportable_slice,
        withheld_slice=withheld_slice,
        actionable_slice=actionable_slice,
        relation_bias_strength=relation_bias_strength,
        response_strategy=response_strategy,
        question_budget=question_budget,
        question_pressure=question_pressure,
        relational_continuity_state=relational_continuity_state,
        relational_style_memory_state=relational_style_memory_state,
        cultural_conversation_state=cultural_conversation_state,
        body_recovery_guard=body_recovery_guard,
        protection_mode=protection_mode_payload,
        attention_regulation_state=attention_regulation_state,
    )
    learning_mode_state = derive_learning_mode_state(
        self_state=self_state_payload,
        body_recovery_guard=body_recovery_guard,
        body_homeostasis_state=body_homeostasis_state,
        homeostasis_budget_state=homeostasis_budget_state,
        protection_mode=protection_mode_payload,
        initiative_readiness=initiative_readiness,
        agenda_state=agenda_state,
        agenda_window_state=agenda_window_state,
        commitment_state=commitment_state,
        attention_regulation_state=attention_regulation_state,
        grice_guard_state=grice_guard_state,
        relational_continuity_state=relational_continuity_state,
        social_topology_state=social_topology_state,
        insight_event=insight_event_payload,
        identity_arc_kind=identity_arc_kind,
        identity_arc_phase=identity_arc_phase,
    ).to_dict()
    social_experiment_loop_state = derive_social_experiment_loop_state(
        learning_mode_state=learning_mode_state,
        commitment_state=commitment_state,
        agenda_state=agenda_state,
        agenda_window_state=agenda_window_state,
        body_recovery_guard=body_recovery_guard,
        protection_mode=protection_mode_payload,
        grice_guard_state=grice_guard_state,
        relational_continuity_state=relational_continuity_state,
        social_topology_state=social_topology_state,
        self_state=self_state_payload,
        identity_arc_kind=identity_arc_kind,
    ).to_dict()
    persona_memory_fragment_models = build_persona_memory_fragments(
        self_state=self_state_payload,
        relation_bias_strength=relation_bias_strength,
        related_person_ids=related_person_ids,
        social_topology_state=social_topology_state,
        relational_style_memory_state=relational_style_memory_state,
        cultural_conversation_state=cultural_conversation_state,
        protection_mode=protection_mode_payload,
        grice_guard_state=grice_guard_state,
    )
    persona_memory_fragments = [
        fragment.to_dict()
        for fragment in persona_memory_fragment_models
    ]
    persona_memory_selection = derive_persona_memory_selection(
        fragments=persona_memory_fragment_models,
        current_focus=current_focus,
        reportable_facts=reportable_facts,
        current_risks=current_risks,
        relation_bias_strength=relation_bias_strength,
        agenda_window_state=agenda_window_state,
        social_topology_state=social_topology_state,
        grice_guard_state=grice_guard_state,
    ).to_dict()
    expressive_style_state = derive_expressive_style_state(
        self_state=self_state_payload,
        temperament_estimate=temperament_estimate,
        body_recovery_guard=body_recovery_guard,
        body_homeostasis_state=body_homeostasis_state,
        homeostasis_budget_state=homeostasis_budget_state,
        initiative_readiness=initiative_readiness,
        commitment_state=commitment_state,
        relational_continuity_state=relational_continuity_state,
        relational_style_memory_state=relational_style_memory_state,
        cultural_conversation_state=cultural_conversation_state,
        social_topology_state=social_topology_state,
        attention_regulation_state=attention_regulation_state,
        grice_guard_state=grice_guard_state,
        protection_mode=protection_mode_payload,
        contact_readiness=contact_readiness,
        coherence_score=coherence_score,
        human_presence_signal=human_presence_signal,
    ).to_dict()
    lightness_budget_state = derive_lightness_budget_state(
        body_recovery_guard=body_recovery_guard,
        body_homeostasis_state=body_homeostasis_state,
        homeostasis_budget_state=homeostasis_budget_state,
        protection_mode=protection_mode_payload,
        attention_regulation_state=attention_regulation_state,
        grice_guard_state=grice_guard_state,
        social_topology_state=social_topology_state,
        relation_competition_state=relation_competition_state,
        expressive_style_state=expressive_style_state,
        relational_style_memory_state=relational_style_memory_state,
        cultural_conversation_state=cultural_conversation_state,
    ).to_dict()
    live_engagement_state = derive_live_engagement_state(
        self_state=self_state_payload,
        initiative_readiness=initiative_readiness,
        initiative_followup_bias=initiative_followup_bias,
        relational_style_memory_state=relational_style_memory_state,
        lightness_budget_state=lightness_budget_state,
        social_topology_state=social_topology_state,
        body_recovery_guard=body_recovery_guard,
        body_homeostasis_state=body_homeostasis_state,
        homeostasis_budget_state=homeostasis_budget_state,
    ).to_dict()
    situation_risk_state = derive_situation_risk_state(
        current_risks=current_risks,
        scene_state=scene,
        self_state=self_state_payload,
    ).to_dict()
    emergency_posture = derive_emergency_posture(
        situation_risk_state=situation_risk_state,
        constraint_field=constraint,
        protection_mode=protection_mode_payload,
        body_recovery_guard=body_recovery_guard,
        body_homeostasis_state=body_homeostasis_state,
        homeostasis_budget_state=homeostasis_budget_state,
    ).to_dict()
    effective_question_budget = min(
        question_budget,
        int(grice_guard_state.get("question_budget_cap") or question_budget),
    )
    protection_mode_decision = {
        "mode": protection_mode_name,
        "scores": {
            str(key): round(_clamp01(float(value or 0.0)), 4)
            for key, value in (protection_mode_payload.get("scores") or {}).items()
        },
        "winner_margin": round(_clamp01(float(protection_mode_payload.get("winner_margin", 0.0) or 0.0)), 4),
        "dominant_inputs": [
            str(item)
            for item in protection_mode_payload.get("dominant_inputs") or []
            if str(item).strip()
        ],
        "reasons": [
            str(item)
            for item in protection_mode_payload.get("reasons") or []
            if str(item).strip()
        ],
    }
    overnight_bias_roles = {
        "association_reweighting_focus": association_reweighting_focus,
        "association_reweighting_reason": association_reweighting_reason,
        "insight_terrain_shape_target": insight_terrain_shape_target,
        "insight_class_focus": insight_class_focus,
        "insight_reframing_bias": round(insight_reframing_bias, 4),
        "initiative_followup_state": initiative_followup_bias["state"],
        "initiative_followup_bias": initiative_followup_bias["score"],
        "body_homeostasis_focus": body_homeostasis_carry["focus"],
        "body_homeostasis_carry_bias": body_homeostasis_carry["carry_bias"],
        "homeostasis_budget_focus": homeostasis_budget_carry["focus"],
        "homeostasis_budget_bias": homeostasis_budget_carry["carry_bias"],
        "homeostasis_budget_state": str(homeostasis_budget_state.get("state") or "steady"),
        "agenda_focus": agenda_carry["focus"],
        "agenda_bias": agenda_carry["carry_bias"],
        "agenda_reason": agenda_carry["reason"],
        "agenda_window_focus": str(self_state_payload.get("agenda_window_focus") or ""),
        "agenda_window_bias": round(float(self_state_payload.get("agenda_window_bias", 0.0) or 0.0), 4),
        "agenda_window_reason": str(self_state_payload.get("agenda_window_reason") or ""),
        "commitment_carry_bias": commitment_carry["carry_bias"],
        "commitment_target_focus": commitment_carry["target_focus"],
        "commitment_followup_focus": commitment_carry["followup_focus"],
        "commitment_mode_focus": commitment_carry["mode_focus"],
        "relational_continuity_focus": relational_continuity_carry["focus"],
        "relational_continuity_carry_bias": relational_continuity_carry["carry_bias"],
        "expressive_style_focus": expressive_style_carry["focus"],
        "expressive_style_carry_bias": expressive_style_carry["carry_bias"],
        "expressive_style_history_focus": expressive_style_history_carry["focus"],
        "expressive_style_history_bias": expressive_style_history_carry["carry_bias"],
        "relational_style_memory_state": str(relational_style_memory_state.get("state") or "grounded_gentle"),
        "relational_style_playful_ceiling": round(float(relational_style_memory_state.get("playful_ceiling") or 0.0), 4),
        "relational_style_advice_tolerance": round(float(relational_style_memory_state.get("advice_tolerance") or 0.0), 4),
        "relational_style_banter_style": str(relational_style_memory_state.get("banter_style") or "grounded_companion"),
        "relational_style_lexical_variation_bias": round(float(relational_style_memory_state.get("lexical_variation_bias") or 0.0), 4),
        "cultural_conversation_state": str(cultural_conversation_state.get("state") or "careful_polite"),
        "cultural_directness_ceiling": round(float(cultural_conversation_state.get("directness_ceiling") or 0.0), 4),
        "cultural_joke_ratio_ceiling": round(float(cultural_conversation_state.get("joke_ratio_ceiling") or 0.0), 4),
        "lightness_budget_state": str(lightness_budget_state.get("state") or "grounded_only"),
        "lightness_budget_banter_room": round(float(lightness_budget_state.get("banter_room") or 0.0), 4),
        "banter_style_focus": banter_style_carry["focus"],
        "lexical_variation_carry_bias": banter_style_carry["carry_bias"],
        "temperament_leader_tendency": round(float(temperament_estimate.get("leader_tendency", 0.0) or 0.0), 4),
        "temperament_hero_tendency": round(float(temperament_estimate.get("hero_tendency", 0.0) or 0.0), 4),
        "temperament_focus": temperament_carry["focus"],
        "temperament_forward_bias": temperament_carry["forward_bias"],
        "temperament_guard_bias": temperament_carry["guard_bias"],
        "temperament_bond_bias": temperament_carry["bond_bias"],
        "temperament_recovery_bias": temperament_carry["recovery_bias"],
        "commitment_state": str(commitment_state.get("state") or "waver"),
        "commitment_target": str(commitment_state.get("target") or "hold"),
        "learning_mode_state": str(learning_mode_state.get("state") or "observe_only"),
        "learning_mode_probe_room": round(float(learning_mode_state.get("probe_room", 0.0) or 0.0), 4),
        "learning_mode_focus": learning_mode_carry["focus"],
        "learning_mode_carry_bias": learning_mode_carry["carry_bias"],
        "social_experiment_state": str(social_experiment_loop_state.get("state") or "watch_and_read"),
        "social_experiment_probe_intensity": round(float(social_experiment_loop_state.get("probe_intensity", 0.0) or 0.0), 4),
        "social_experiment_focus": social_experiment_carry["focus"],
        "social_experiment_carry_bias": social_experiment_carry["carry_bias"],
        "identity_arc_kind": identity_arc_kind,
        "identity_arc_phase": identity_arc_phase,
        "identity_arc_open_tension": identity_arc_open_tension,
        "identity_arc_stability": round(identity_arc_stability, 4),
    }
    reaction_vs_overnight_bias = {
        "same_turn": {
            "protection_mode": protection_mode_name,
            "protection_mode_winner_margin": protection_mode_decision["winner_margin"],
            "memory_write_class": str(memory_write_class_payload.get("memory_class") or "episodic"),
            "memory_write_class_reason": str(memory_write_class_payload.get("reason") or ""),
            "memory_write_class_winner_margin": round(float(memory_write_class_payload.get("winner_margin", 0.0) or 0.0), 4),
            "terrain_active_patch_label": str(terrain_readout_payload.get("active_patch_label") or "").strip(),
            "insight_triggered": bool(insight_event_payload.get("triggered", False)),
            "body_recovery_guard_state": str(body_recovery_guard.get("state") or "open"),
            "body_recovery_guard_winner_margin": round(float(body_recovery_guard.get("winner_margin", 0.0) or 0.0), 4),
            "body_homeostasis_state": str(body_homeostasis_state.get("state") or "steady"),
            "body_homeostasis_winner_margin": round(float(body_homeostasis_state.get("winner_margin", 0.0) or 0.0), 4),
            "homeostasis_budget_state": str(homeostasis_budget_state.get("state") or "steady"),
            "homeostasis_budget_winner_margin": round(float(homeostasis_budget_state.get("winner_margin", 0.0) or 0.0), 4),
            "initiative_readiness_state": str(initiative_readiness.get("state") or "hold"),
            "initiative_readiness_score": round(float(initiative_readiness.get("score", 0.0) or 0.0), 4),
            "agenda_state": str(agenda_state.get("state") or "hold"),
            "agenda_reason": str(agenda_state.get("reason") or ""),
            "agenda_winner_margin": round(float(agenda_state.get("winner_margin", 0.0) or 0.0), 4),
            "agenda_window_state": str(agenda_window_state.get("state") or "long_hold"),
            "agenda_window_reason": str(agenda_window_state.get("reason") or ""),
            "agenda_window_deferral_budget": round(float(agenda_window_state.get("deferral_budget", 0.0) or 0.0), 4),
            "agenda_window_carry_target": str(agenda_window_state.get("carry_target") or ""),
            "temperament_risk_tolerance": round(float(temperament_estimate.get("risk_tolerance", 0.0) or 0.0), 4),
            "temperament_leader_tendency": round(float(temperament_estimate.get("leader_tendency", 0.0) or 0.0), 4),
            "temperament_hero_tendency": round(float(temperament_estimate.get("hero_tendency", 0.0) or 0.0), 4),
            "commitment_state": str(commitment_state.get("state") or "waver"),
            "commitment_target": str(commitment_state.get("target") or "hold"),
            "commitment_winner_margin": round(float(commitment_state.get("winner_margin", 0.0) or 0.0), 4),
            "commitment_accepted_cost": round(float(commitment_state.get("accepted_cost", 0.0) or 0.0), 4),
            "learning_mode_state": str(learning_mode_state.get("state") or "observe_only"),
            "learning_mode_probe_room": round(float(learning_mode_state.get("probe_room", 0.0) or 0.0), 4),
            "learning_mode_winner_margin": round(float(learning_mode_state.get("winner_margin", 0.0) or 0.0), 4),
            "learning_mode_focus": learning_mode_carry["focus"],
            "learning_mode_carry_bias": learning_mode_carry["carry_bias"],
            "social_experiment_state": str(social_experiment_loop_state.get("state") or "watch_and_read"),
            "social_experiment_probe_intensity": round(float(social_experiment_loop_state.get("probe_intensity", 0.0) or 0.0), 4),
            "social_experiment_winner_margin": round(float(social_experiment_loop_state.get("winner_margin", 0.0) or 0.0), 4),
            "social_experiment_focus": social_experiment_carry["focus"],
            "social_experiment_carry_bias": social_experiment_carry["carry_bias"],
            "identity_arc_kind": identity_arc_kind,
            "identity_arc_phase": identity_arc_phase,
            "identity_arc_summary": identity_arc_summary,
            "identity_arc_open_tension": identity_arc_open_tension,
            "identity_arc_stability": round(identity_arc_stability, 4),
            "relational_continuity_state": str(relational_continuity_state.get("state") or "distant"),
            "relational_continuity_winner_margin": round(float(relational_continuity_state.get("winner_margin", 0.0) or 0.0), 4),
            "relation_competition_state": str(relation_competition_state.get("state") or "ambient"),
            "relation_competition_level": round(float(relation_competition_state.get("competition_level", 0.0) or 0.0), 4),
            "relation_competition_winner_margin": round(float(relation_competition_state.get("winner_margin", 0.0) or 0.0), 4),
            "relation_competition_dominant_person_id": str(relation_competition_state.get("dominant_person_id") or ""),
            "social_topology": social_topology,
            "social_topology_state": str(social_topology_state.get("state") or "ambient"),
            "social_topology_winner_margin": round(float(social_topology_state.get("winner_margin", 0.0) or 0.0), 4),
            "relational_style_memory_state": str(relational_style_memory_state.get("state") or "grounded_gentle"),
            "relational_style_playful_ceiling": round(float(relational_style_memory_state.get("playful_ceiling") or 0.0), 4),
            "relational_style_advice_tolerance": round(float(relational_style_memory_state.get("advice_tolerance") or 0.0), 4),
            "relational_style_banter_room": round(float(relational_style_memory_state.get("banter_room") or 0.0), 4),
            "relational_style_banter_style": str(relational_style_memory_state.get("banter_style") or "grounded_companion"),
            "relational_style_lexical_variation_bias": round(float(relational_style_memory_state.get("lexical_variation_bias") or 0.0), 4),
            "cultural_conversation_state": str(cultural_conversation_state.get("state") or "careful_polite"),
            "cultural_directness_ceiling": round(float(cultural_conversation_state.get("directness_ceiling") or 0.0), 4),
            "cultural_joke_ratio_ceiling": round(float(cultural_conversation_state.get("joke_ratio_ceiling") or 0.0), 4),
            "expressive_style_state": str(expressive_style_state.get("state") or "grounded_gentle"),
            "expressive_style_winner_margin": round(float(expressive_style_state.get("winner_margin", 0.0) or 0.0), 4),
            "expressive_lightness_room": round(float(expressive_style_state.get("lightness_room", 0.0) or 0.0), 4),
            "expressive_continuity_weight": round(float(expressive_style_state.get("continuity_weight", 0.0) or 0.0), 4),
            "lightness_budget_state": str(lightness_budget_state.get("state") or "grounded_only"),
            "lightness_budget_banter_room": round(float(lightness_budget_state.get("banter_room") or 0.0), 4),
            "lightness_budget_suppression": round(float(lightness_budget_state.get("suppression") or 0.0), 4),
            "live_engagement_state": str(live_engagement_state.get("state") or "hold"),
            "live_engagement_score": round(float(live_engagement_state.get("score", 0.0) or 0.0), 4),
            "live_engagement_winner_margin": round(float(live_engagement_state.get("winner_margin", 0.0) or 0.0), 4),
            "live_primary_move": str(live_engagement_state.get("primary_move") or "hold_presence"),
            "situation_risk_state": str(situation_risk_state.get("state") or "ordinary_context"),
            "situation_risk_immediacy": round(float(situation_risk_state.get("immediacy", 0.0) or 0.0), 4),
            "situation_risk_intent_clarity": round(float(situation_risk_state.get("intent_clarity", 0.0) or 0.0), 4),
            "situation_risk_escape_room": round(float(situation_risk_state.get("escape_room", 0.0) or 0.0), 4),
            "situation_risk_relation_break": round(float(situation_risk_state.get("relation_break", 0.0) or 0.0), 4),
            "emergency_posture": str(emergency_posture.get("state") or "observe"),
            "emergency_dialogue_permission": str(emergency_posture.get("dialogue_permission") or "allow_short"),
            "emergency_primary_action": str(emergency_posture.get("primary_action") or "observe_context_shift"),
        },
        "overnight": {
            "association_reweighting_focus": association_reweighting_focus,
            "association_reweighting_reason": association_reweighting_reason,
            "insight_terrain_shape_target": insight_terrain_shape_target,
            "insight_class_focus": insight_class_focus,
            "insight_reframing_bias": round(insight_reframing_bias, 4),
            "initiative_followup_state": initiative_followup_bias["state"],
            "initiative_followup_bias": initiative_followup_bias["score"],
            "body_homeostasis_focus": body_homeostasis_carry["focus"],
            "body_homeostasis_carry_bias": body_homeostasis_carry["carry_bias"],
            "homeostasis_budget_focus": homeostasis_budget_carry["focus"],
            "homeostasis_budget_bias": homeostasis_budget_carry["carry_bias"],
            "agenda_focus": agenda_carry["focus"],
            "agenda_bias": agenda_carry["carry_bias"],
            "agenda_reason": agenda_carry["reason"],
            "agenda_window_focus": str(self_state_payload.get("agenda_window_focus") or ""),
            "agenda_window_bias": round(float(self_state_payload.get("agenda_window_bias", 0.0) or 0.0), 4),
            "agenda_window_reason": str(self_state_payload.get("agenda_window_reason") or ""),
            "learning_mode_focus": learning_mode_carry["focus"],
            "learning_mode_carry_bias": learning_mode_carry["carry_bias"],
            "social_experiment_focus": social_experiment_carry["focus"],
            "social_experiment_carry_bias": social_experiment_carry["carry_bias"],
            "commitment_carry_bias": commitment_carry["carry_bias"],
            "commitment_target_focus": commitment_carry["target_focus"],
            "commitment_followup_focus": commitment_carry["followup_focus"],
            "commitment_mode_focus": commitment_carry["mode_focus"],
            "relational_continuity_focus": relational_continuity_carry["focus"],
            "relational_continuity_carry_bias": relational_continuity_carry["carry_bias"],
            "expressive_style_focus": expressive_style_carry["focus"],
            "expressive_style_carry_bias": expressive_style_carry["carry_bias"],
            "expressive_style_history_focus": expressive_style_history_carry["focus"],
            "expressive_style_history_bias": expressive_style_history_carry["carry_bias"],
            "banter_style_focus": banter_style_carry["focus"],
            "lexical_variation_carry_bias": banter_style_carry["carry_bias"],
            "temperament_leader_tendency": round(float(temperament_estimate.get("leader_tendency", 0.0) or 0.0), 4),
            "temperament_hero_tendency": round(float(temperament_estimate.get("hero_tendency", 0.0) or 0.0), 4),
            "commitment_carry": str(commitment_state.get("state") or "waver"),
            "temperament_focus": temperament_carry["focus"],
            "temperament_forward_bias": temperament_carry["forward_bias"],
            "temperament_guard_bias": temperament_carry["guard_bias"],
            "temperament_bond_bias": temperament_carry["bond_bias"],
            "temperament_recovery_bias": temperament_carry["recovery_bias"],
            "identity_arc_kind": identity_arc_kind,
            "identity_arc_phase": identity_arc_phase,
            "identity_arc_open_tension": identity_arc_open_tension,
            "identity_arc_stability": round(identity_arc_stability, 4),
        },
    }

    affordance_priority: list[str] = []
    if response_strategy == "shared_world_next_step":
        affordance_priority.extend(["co_plan", "co_move", "pace_match"])
    elif response_strategy == "respectful_wait":
        affordance_priority.extend(["wait", "defer", "leave_return_point"])
    elif response_strategy == "repair_then_attune":
        affordance_priority.extend(["repair", "soften", "re-open-carefully"])
    elif clarify_context:
        affordance_priority.extend(["clarify", "check_visible", "confirm_scope"])
    else:
        affordance_priority.extend(["attune", "clarify", "stay_visible"])
    if bool(insight_event_payload.get("triggered", False)):
        affordance_priority = ["orient", "connect_dots", *affordance_priority]

    do_not_cross: list[str] = []
    if disclosure_depth == "minimal":
        do_not_cross.append("overinterpret")
    if partner_stance_hint == "respectful":
        do_not_cross.append("flatten_distance")
    if partner_timing_hint == "delayed":
        do_not_cross.append("force_immediacy")
    if "danger" in current_risks:
        do_not_cross.append("increase_pressure")
    for item in constraint.get("do_not_cross") or []:
        text = str(item).strip()
        if text:
            do_not_cross.append(text)
    if reportability_limit == "withhold":
        do_not_cross.append("force_reportability")

    content_plan = _derive_content_plan(
        primary_object_label=primary_conversational_object_label or _normalize_focus_label(str(attention_target)),
        ordered_operation_kinds=ordered_operation_kinds,
        ordered_effect_kinds=ordered_effect_kinds,
        deferred_object_labels=deferred_object_labels,
    )
    dialogue_order = _derive_dialogue_order(
        opening_move=opening_move,
        followup_move=followup_move,
        closing_move=closing_move,
        primary_object_label=primary_conversational_object_label or _normalize_focus_label(str(attention_target)),
        ordered_operation_kinds=ordered_operation_kinds,
        ordered_effect_kinds=ordered_effect_kinds,
        deferred_object_labels=deferred_object_labels,
    )
    shell_guidance = _compact(
        [
            f"driver:{dominant_driver}",
            f"scene:{scene_family}" if scene_family else "",
            f"top_option:{top_option_family}" if top_option_family else "",
            f"workspace:{workspace_mode}" if workspace_mode else "",
            f"reportability:{reportability_gate_mode}" if reportability_gate_mode else "",
            f"blend:{dominant_blend_mode}" if dominant_blend_mode else "",
            f"opening:{opening_pace}",
            f"return:{gaze_return}",
            f"distance:{distance_strategy}",
            f"voice:{str(expressive_style_state.get('state') or '').strip()}" if str(expressive_style_state.get("state") or "").strip() else "",
            f"address:{partner_address_hint}" if partner_address_hint else "",
            f"timing:{partner_timing_hint}" if partner_timing_hint else "",
            f"stance:{partner_stance_hint}" if partner_stance_hint else "",
            "insight:connection" if bool(insight_event_payload.get("triggered", False)) else "",
        ]
    )
    wanted_effect_on_other = [
        {
            "effect": str(item.get("effect_kind") or "").strip(),
            "target": str(item.get("target_label") or "").strip(),
            "intensity": round(float(item.get("intensity", 0.0) or 0.0), 4),
        }
        for item in interaction_effect_items[:4]
        if str(item.get("effect_kind") or "").strip()
    ]
    if not wanted_effect_on_other:
        wanted_effect_on_other = [
            {
                "effect": effect_kind,
                "target": primary_conversational_object_label or _normalize_focus_label(str(attention_target)),
                "intensity": 0.6,
            }
            for effect_kind in ordered_effect_kinds[:3]
            if effect_kind
        ]
    response_action_now = {
        "primary_operation": str(primary_object_operation.get("operation_kind") or "").strip(),
        "operation_target": str(primary_object_operation.get("target_label") or "").strip(),
        "ordered_operations": ordered_operation_kinds,
        "question_budget": question_budget,
        "effective_question_budget": effective_question_budget,
        "question_pressure": round(question_pressure, 4),
        "defer_dominance": round(defer_dominance, 4),
    }

    return {
        "dialogue_act": resolved_dialogue_act,
        "response_strategy": response_strategy,
        "opening_move": opening_move,
        "followup_move": followup_move,
        "closing_move": closing_move,
        "disclosure_depth": disclosure_depth,
        "attention_target": attention_target,
        "repair_bias": repair_bias,
        "contact_readiness": round(contact_readiness, 4),
        "coherence_score": round(coherence_score, 4),
        "human_presence_signal": round(human_presence_signal, 4),
        "distance_strategy": distance_strategy,
        "opening_pace": opening_pace,
        "gaze_return_expectation": gaze_return,
        "memory_write_priority": memory_write_priority,
        "memory_write_class": str(memory_write_class_payload.get("memory_class") or "episodic"),
        "memory_write_class_reason": str(memory_write_class_payload.get("reason") or ""),
        "memory_write_class_bias": memory_write_class_payload,
        "protection_mode_decision": protection_mode_decision,
        "body_recovery_guard": body_recovery_guard,
        "body_homeostasis_state": body_homeostasis_state,
        "homeostasis_budget_state": homeostasis_budget_state,
        "body_homeostasis_carry": body_homeostasis_carry,
        "homeostasis_budget_carry": homeostasis_budget_carry,
        "initiative_followup_bias": initiative_followup_bias,
        "initiative_readiness": initiative_readiness,
        "agenda_carry": agenda_carry,
        "learning_mode_carry": learning_mode_carry,
        "social_experiment_carry": social_experiment_carry,
        "agenda_state": agenda_state,
        "agenda_window_state": agenda_window_state,
        "commitment_carry": commitment_carry,
        "commitment_state": commitment_state,
        "learning_mode_state": learning_mode_state,
        "social_experiment_loop_state": social_experiment_loop_state,
        "identity_arc_kind": identity_arc_kind,
        "identity_arc_phase": identity_arc_phase,
        "identity_arc_summary": identity_arc_summary,
        "identity_arc_open_tension": identity_arc_open_tension,
        "identity_arc_stability": round(identity_arc_stability, 4),
        "persona_memory_fragments": persona_memory_fragments,
        "persona_memory_selection": persona_memory_selection,
        "attention_regulation_state": attention_regulation_state,
        "grice_guard_state": grice_guard_state,
        "relational_style_memory_state": relational_style_memory_state,
        "cultural_conversation_state": cultural_conversation_state,
        "expressive_style_state": expressive_style_state,
        "lightness_budget_state": lightness_budget_state,
        "live_engagement_state": live_engagement_state,
        "situation_risk_state": situation_risk_state,
        "emergency_posture": emergency_posture,
        "expressive_style_history_focus": expressive_style_history_carry["focus"],
        "expressive_style_history_bias": expressive_style_history_carry["carry_bias"],
        "banter_style_focus": banter_style_carry["focus"],
        "lexical_variation_carry_bias": banter_style_carry["carry_bias"],
        "relational_continuity_state": relational_continuity_state,
        "discussion_thread_registry_snapshot": discussion_thread_registry_snapshot,
        "relation_competition_state": relation_competition_state,
        "active_relation_table": active_relation_table,
        "social_topology": social_topology,
        "social_topology_state": social_topology_state,
        "related_person_ids": list(related_person_ids),
        "relational_continuity_carry": relational_continuity_carry,
        "temperament_estimate": temperament_estimate,
        "affordance_priority": affordance_priority,
        "do_not_cross": do_not_cross,
        "content_plan": content_plan,
        "dialogue_order": dialogue_order,
        "shell_guidance": shell_guidance,
        "scene_state": scene,
        "scene_family": scene_family,
        "interaction_option_candidates": option_candidates,
        "top_interaction_option_family": top_option_family,
        "affect_blend_state": affect_blend,
        "constraint_field": constraint,
        "conscious_workspace": workspace,
        "workspace_decision": workspace_decision,
        "workspace_mode": workspace_mode,
        "reportability_gate_mode": reportability_gate_mode,
        "conversational_objects": conversational_objects_payload,
        "primary_conversational_object_id": primary_conversational_object_id,
        "primary_conversational_object_label": primary_conversational_object_label,
        "deferred_object_labels": deferred_object_labels,
        "object_operations": object_operations_payload,
        "interaction_effects": interaction_effects_payload,
        "interaction_judgement_view": interaction_judgement_payload,
        "resonance_evaluation": resonance,
        "other_person_state": other_person_state,
        "other_person_detail_room": other_detail_room,
        "other_person_acknowledgement_need": other_acknowledgement_need,
        "other_person_pressure_sensitivity": other_pressure_sensitivity,
        "other_person_next_step_room": other_next_step_room,
        "resonance_prioritize_actions": resonance_prioritize,
        "resonance_avoid_actions": resonance_avoid,
        "resonance_expected_effects": resonance_expected,
        "focus_now": primary_conversational_object_label or _normalize_focus_label(str(attention_target)),
        "leave_closed_for_now": deferred_object_labels,
        "response_action_now": response_action_now,
        "wanted_effect_on_other": wanted_effect_on_other,
        "ordered_effects": ordered_effect_kinds,
        "reportable_slice": reportable_slice,
        "withheld_slice": withheld_slice,
        "actionable_slice": actionable_slice,
        "primary_object_operation": primary_object_operation,
        "object_operation_kinds": sorted(operation_kinds),
        "ordered_operation_kinds": ordered_operation_kinds,
        "interaction_effect_kinds": sorted(effect_kinds),
        "ordered_effect_kinds": ordered_effect_kinds,
        "question_budget": question_budget,
        "effective_question_budget": effective_question_budget,
        "question_pressure": round(question_pressure, 4),
        "defer_dominance": round(defer_dominance, 4),
        "blend_conflict_level": round(conflict_level, 4),
        "blend_residual_tension": round(residual_tension, 4),
        "qualia_planner_view": qualia_planner_payload,
        "qualia_memory_bias": qualia_memory_bias,
        "affective_position": affective_position_payload,
        "terrain_readout": terrain_readout_payload,
        "protection_mode": protection_mode_payload,
        "insight_event": insight_event_payload,
        "association_reweighting_focus": association_reweighting_focus,
        "association_reweighting_reason": association_reweighting_reason,
        "insight_terrain_shape_target": insight_terrain_shape_target,
        "overnight_bias_roles": overnight_bias_roles,
        "reaction_vs_overnight_bias": reaction_vs_overnight_bias,
    }


def _compact(values: Sequence[str]) -> list[str]:
    return [value for value in values if value]


def _derive_object_driven_moves(
    *,
    primary_operation: Mapping[str, Any],
    operation_kinds: set[str],
    effect_kinds: set[str],
    question_budget: int,
    question_pressure: float,
    defer_dominance: float,
    repair_context: bool,
    respectful_context: bool,
    advance_context: bool,
) -> dict[str, str]:
    operation_kind = str(primary_operation.get("operation_kind") or "").strip()
    if operation_kind == "offer_small_next_step":
        return {
            "opening_move": "anchor_visible_part",
            "followup_move": "offer_one_small_next_step",
            "closing_move": "keep_choice_with_other_person",
        }
    if operation_kind == "narrow_clarify":
        return {
            "opening_move": "narrow_scope_first",
            "followup_move": "ask_one_bounded_part",
            "closing_move": "leave_room_for_no_answer",
        }
    if repair_context and operation_kind in {"hold_without_probe", "acknowledge"}:
        return {
            "opening_move": "name_overreach_and_reduce_force",
            "followup_move": "protect_talking_room",
            "closing_move": "leave_return_point",
        }
    if respectful_context and operation_kind in {"hold_without_probe", "acknowledge"}:
        return {
            "opening_move": "acknowledge_without_probe",
            "followup_move": "protect_talking_room",
            "closing_move": "leave_return_point",
        }
    if operation_kind == "hold_without_probe":
        closing_move = "leave_return_point" if "keep_return_point" in operation_kinds else "hold_space"
        if "protect_unfinished_part" in operation_kinds or defer_dominance >= 0.6 or question_budget == 0:
            closing_move = "leave_unfinished_part_closed_for_now"
        return {
            "opening_move": "acknowledge_without_probe",
            "followup_move": "protect_talking_room",
            "closing_move": closing_move,
        }
    if operation_kind == "acknowledge":
        followup_move = "keep_shared_thread_visible" if "anchor_shared_thread" in operation_kinds else "invite_visible_state"
        if question_budget == 0 or question_pressure >= 0.68:
            followup_move = "protect_talking_room"
        closing_move = "keep_choice_with_other_person" if "preserve_self_pacing" in effect_kinds else "hold_space"
        return {
            "opening_move": "acknowledge_named_state",
            "followup_move": followup_move,
            "closing_move": closing_move,
        }
    if advance_context and "anchor_next_step_in_theme" in operation_kinds:
        return {
            "opening_move": "anchor_shared_thread",
            "followup_move": "offer_one_small_next_step",
            "closing_move": "keep_choice_with_other_person",
        }
    return {}


def _fallback_primary_operation(
    *,
    response_strategy: str,
    opening_move: str,
    followup_move: str,
    primary_target_label: str,
) -> dict[str, str]:
    operation_kind = "acknowledge"
    if followup_move in {"offer_one_small_next_step", "map_next_step"} or opening_move in {
        "anchor_visible_part",
        "anchor_shared_thread",
        "synchronize_then_propose",
    }:
        operation_kind = "offer_small_next_step"
    elif opening_move in {"narrow_scope_first", "narrow_scope_before_extend"} or followup_move == "ask_one_bounded_part":
        operation_kind = "narrow_clarify"
    elif opening_move in {"acknowledge_and_wait", "acknowledge_without_probe", "name_overreach_and_reduce_force"}:
        operation_kind = "hold_without_probe"
    elif opening_move == "reduce_force_and_secure_boundary":
        operation_kind = "protect_unfinished_part"
    elif response_strategy == "shared_world_next_step":
        operation_kind = "offer_small_next_step"
    elif response_strategy in {"respectful_wait", "repair_then_attune"}:
        operation_kind = "hold_without_probe"
    elif response_strategy == "contain_then_stabilize":
        operation_kind = "protect_unfinished_part"
    return {
        "operation_kind": operation_kind,
        "target_label": primary_target_label,
    }


def _fallback_ordered_operations(
    *,
    response_strategy: str,
    primary_operation: Mapping[str, Any],
    closing_move: str,
) -> list[str]:
    primary_operation_kind = str(primary_operation.get("operation_kind") or "").strip()
    operations = [primary_operation_kind] if primary_operation_kind else []
    if response_strategy == "shared_world_next_step":
        operations.extend(["anchor_next_step_in_theme"])
    elif response_strategy in {"respectful_wait", "repair_then_attune"}:
        operations.extend(["keep_return_point"])
    elif response_strategy == "contain_then_stabilize":
        operations.extend(["protect_unfinished_part"])
    elif response_strategy == "attune_then_extend":
        operations.extend(["anchor_shared_thread"])
    if closing_move in {"leave_space", "hold_space", "keep_choice_with_other_person"}:
        operations.append("keep_return_point")
    return _compact(list(dict.fromkeys(item for item in operations if item)))


def _fallback_ordered_effects(
    *,
    response_strategy: str,
    closing_move: str,
) -> list[str]:
    effects: list[str] = []
    if response_strategy == "shared_world_next_step":
        effects.extend(["enable_small_next_step", "keep_next_step_connected"])
    elif response_strategy == "respectful_wait":
        effects.extend(["preserve_self_pacing", "keep_connection_open"])
    elif response_strategy == "repair_then_attune":
        effects.extend(["feel_received", "preserve_self_pacing"])
    elif response_strategy == "contain_then_stabilize":
        effects.extend(["protect_boundary", "avoid_forced_reopening"])
    elif response_strategy == "attune_then_extend":
        effects.extend(["feel_received", "keep_connection_open"])
    elif response_strategy == "reflect_without_settling":
        effects.extend(["preserve_continuity", "keep_connection_open"])
    if closing_move == "keep_pace_mutual":
        effects.append("keep_next_step_connected")
    elif closing_move in {"leave_space", "hold_space"}:
        effects.append("keep_connection_open")
    return _compact(list(dict.fromkeys(item for item in effects if item)))


def _ordered_operation_kinds(
    *,
    primary_operation_id: str,
    operation_items: Sequence[Mapping[str, Any]],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    primary_kind = ""
    scored_items: list[tuple[float, str]] = []
    for item in operation_items:
        operation_id = str(item.get("operation_id") or "").strip()
        operation_kind = str(item.get("operation_kind") or "").strip()
        if not operation_kind:
            continue
        if operation_id == primary_operation_id:
            primary_kind = operation_kind
        score = float(item.get("priority", 0.0) or 0.0) + float(item.get("operation_strength", 0.0) or 0.0)
        scored_items.append((score, operation_kind))
    if primary_kind:
        seen.add(primary_kind)
        ordered.append(primary_kind)
    for _, operation_kind in sorted(scored_items, key=lambda item: item[0], reverse=True):
        if operation_kind in seen:
            continue
        seen.add(operation_kind)
        ordered.append(operation_kind)
    return ordered


def _ordered_effect_kinds(
    *,
    primary_effect_ids: set[str],
    effect_items: Sequence[Mapping[str, Any]],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    primary_items: list[tuple[float, str]] = []
    secondary_items: list[tuple[float, str]] = []
    for item in effect_items:
        effect_id = str(item.get("effect_id") or "").strip()
        effect_kind = str(item.get("effect_kind") or "").strip()
        if not effect_kind:
            continue
        score = float(item.get("intensity", 0.0) or 0.0)
        if effect_id in primary_effect_ids:
            primary_items.append((score, effect_kind))
        else:
            secondary_items.append((score, effect_kind))
    for group in (primary_items, secondary_items):
        for _, effect_kind in sorted(group, key=lambda item: item[0], reverse=True):
            if effect_kind in seen:
                continue
            seen.add(effect_kind)
            ordered.append(effect_kind)
    return ordered


def _derive_content_plan(
    *,
    primary_object_label: str,
    ordered_operation_kinds: Sequence[str],
    ordered_effect_kinds: Sequence[str],
    deferred_object_labels: Sequence[str],
) -> list[str]:
    plan: list[str] = []
    if primary_object_label:
        plan.append(f"focus:{primary_object_label}")
    for operation_kind in ordered_operation_kinds[:3]:
        if operation_kind:
            plan.append(f"operate:{operation_kind}")
    for effect_kind in ordered_effect_kinds[:2]:
        if effect_kind:
            plan.append(f"effect:{effect_kind}")
    if deferred_object_labels:
        plan.append("defer:unfinished_part")
    return _compact(plan)


def _derive_dialogue_order(
    *,
    opening_move: str,
    followup_move: str,
    closing_move: str,
    primary_object_label: str,
    ordered_operation_kinds: Sequence[str],
    ordered_effect_kinds: Sequence[str],
    deferred_object_labels: Sequence[str],
) -> list[str]:
    order: list[str] = []
    if opening_move:
        order.append(f"open:{opening_move}")
    if primary_object_label:
        order.append(f"focus:{primary_object_label}")
    for operation_kind in ordered_operation_kinds[:3]:
        if operation_kind:
            order.append(f"operate:{operation_kind}")
    if followup_move:
        order.append(f"follow:{followup_move}")
    for effect_kind in ordered_effect_kinds[:2]:
        if effect_kind:
            order.append(f"effect:{effect_kind}")
    if deferred_object_labels:
        order.append("defer:unfinished_part")
    if closing_move:
        order.append(f"close:{closing_move}")
    return _compact(order)


def _normalize_focus_label(label: str) -> str:
    normalized = str(label or "").strip()
    if not normalized:
        return ""
    if normalized.lower() in {"social", "person", "ambient", "meaning", "place"}:
        return "what is here right now"
    return normalized


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _derive_qualia_memory_bias(qualia_planner_view: Mapping[str, Any]) -> dict[str, object]:
    payload = dict(qualia_planner_view or {})
    trust = _clamp01(float(payload.get("trust", 1.0) or 0.0))
    degraded = bool(payload.get("degraded", False))
    dominant_axis = str(payload.get("dominant_axis") or "").strip()
    body_load = _clamp01(float(payload.get("body_load", 0.0) or 0.0))
    protection_bias = _clamp01(float(payload.get("protection_bias", 0.0) or 0.0))
    felt_energy = _clamp01(float(payload.get("felt_energy", 0.0) or 0.0))

    priority = "ambient"
    reason = ""
    if protection_bias >= 0.12 or body_load >= 0.08:
        priority = "stability_trace"
        reason = "protective_felt_bias"
    elif felt_energy >= 0.16 and trust >= 0.45 and not degraded:
        priority = "foreground_trace"
        reason = "felt_energy_bias"

    return {
        "priority": priority,
        "reason": reason,
        "trust": round(trust, 4),
        "degraded": degraded,
        "dominant_axis": dominant_axis,
        "body_load": round(body_load, 4),
        "protection_bias": round(protection_bias, 4),
        "felt_energy": round(felt_energy, 4),
    }


def _derive_memory_write_class(
    *,
    memory_write_priority: str,
    qualia_planner_view: Mapping[str, Any],
    affective_position: Mapping[str, Any],
    terrain_readout: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    workspace_mode: str,
    deferred_object_labels: Sequence[str],
    relation_bias_strength: float,
    related_person_ids: Sequence[str],
    insight_reframing_bias: float = 0.0,
    insight_class_focus: str = "",
) -> dict[str, object]:
    qualia = dict(qualia_planner_view or {})
    position = dict(affective_position or {})
    terrain = dict(terrain_readout or {})
    protection = dict(protection_mode or {})
    source_weights = dict(position.get("source_weights") or {})

    trust = _clamp01(float(qualia.get("trust", 1.0) or 0.0))
    degraded = bool(qualia.get("degraded", False))
    body_load = _clamp01(float(qualia.get("body_load", 0.0) or 0.0))
    protection_bias = _clamp01(float(qualia.get("protection_bias", 0.0) or 0.0))
    felt_energy = _clamp01(float(qualia.get("felt_energy", 0.0) or 0.0))
    position_confidence = _clamp01(float(position.get("confidence", 0.0) or 0.0))
    position_memory_weight = _clamp01(float(source_weights.get("memory", 0.0) or 0.0))
    terrain_protect_bias = _clamp01(float(terrain.get("protect_bias", 0.0) or 0.0))
    terrain_approach_bias = _clamp01(float(terrain.get("approach_bias", 0.0) or 0.0))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_mode_strength = _clamp01(float(protection.get("strength", 0.0) or 0.0))
    relation_active = relation_bias_strength >= 0.28 or bool(related_person_ids)
    insight_reframing_bias = _clamp01(float(insight_reframing_bias or 0.0))
    insight_class_focus = str(insight_class_focus or "").strip()
    insight_prior = _derive_insight_memory_prior(
        insight_reframing_bias=insight_reframing_bias,
        insight_class_focus=insight_class_focus,
        relation_active=relation_active,
        protection_mode_name=protection_mode_name,
        terrain_approach_bias=terrain_approach_bias,
        terrain_protect_bias=terrain_protect_bias,
        protection_bias=protection_bias,
        trust=trust,
        degraded=degraded,
        body_load=body_load,
    )
    same_turn_scores = _derive_memory_class_same_turn_scores(
        workspace_mode=workspace_mode,
        deferred_object_labels=deferred_object_labels,
        relation_active=relation_active,
        trust=trust,
        degraded=degraded,
        body_load=body_load,
        protection_bias=protection_bias,
        felt_energy=felt_energy,
        position_confidence=position_confidence,
        position_memory_weight=position_memory_weight,
        terrain_protect_bias=terrain_protect_bias,
        terrain_approach_bias=terrain_approach_bias,
        protection_mode_name=protection_mode_name,
        protection_mode_strength=protection_mode_strength,
    )
    mode_prior = _derive_memory_class_mode_prior(
        workspace_mode=workspace_mode,
        relation_active=relation_active,
        body_load=body_load,
        terrain_protect_bias=terrain_protect_bias,
        terrain_approach_bias=terrain_approach_bias,
        protection_mode_name=protection_mode_name,
        protection_mode_strength=protection_mode_strength,
    )

    memory_class = "episodic"
    reason = "default_episode"
    if relation_active and (
        protection_mode_name in {"contain", "stabilize", "shield"}
        or terrain_protect_bias >= 0.46
        or protection_bias >= 0.16
    ) and body_load < 0.18:
        memory_class = "bond_protection"
        reason = "bond_protection_pressure"
    elif (
        protection_mode_name in {"contain", "stabilize", "shield"}
        or terrain_protect_bias >= 0.42
        or body_load >= 0.12
        or degraded
    ):
        memory_class = "body_risk"
        reason = "protective_body_pressure"
    elif protection_mode_name == "repair":
        memory_class = "repair_trace"
        reason = "repair_mode"
    elif workspace_mode == "guarded_foreground" or deferred_object_labels or position_memory_weight >= 0.3:
        memory_class = "unresolved_tension"
        reason = "guarded_or_deferred"
    elif (
        relation_active
        and memory_write_priority in {"relation_episode", "prospective_trace", "foreground_trace"}
        and terrain_approach_bias >= 0.42
        and terrain_protect_bias <= 0.34
        and trust >= 0.54
        and not degraded
        and position_confidence >= 0.42
        and protection_mode_strength <= 0.58
    ):
        memory_class = "safe_repeat"
        reason = "safe_repeatable_contact"

    protective_lock = (
        memory_class == "body_risk"
        or degraded
        or body_load >= 0.12
        or protection_mode_name in {"contain", "stabilize", "shield"}
    )
    if not protective_lock:
        if (
            insight_prior["bond_protection"] >= 0.1
            and relation_active
            and memory_class in {"episodic", "safe_repeat", "unresolved_tension", "repair_trace"}
            and (terrain_protect_bias >= 0.18 or protection_bias >= 0.08 or protection_mode_name in {"repair", "monitor"})
        ):
            memory_class = "bond_protection"
            reason = "insight_reframed_relation_bias"
        elif (
            insight_prior["repair_trace"] >= 0.1
            and memory_class in {"episodic", "safe_repeat", "unresolved_tension"}
            and protection_mode_name in {"repair", "monitor"}
            and terrain_approach_bias >= 0.28
        ):
            memory_class = "repair_trace"
            reason = "insight_reframed_relation_bias"
        elif (
            insight_prior["insight_trace"] >= 0.1
            and memory_class in {"episodic", "safe_repeat", "unresolved_tension"}
            and trust >= 0.46
            and not degraded
            and body_load < 0.12
        ):
            memory_class = "insight_trace"
            reason = "insight_new_link_bias"

    combined_scores = _combine_memory_class_scores(
        same_turn_scores=same_turn_scores,
        mode_prior=mode_prior,
        insight_prior=insight_prior,
        protective_lock=protective_lock,
    )
    sorted_scores = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    runner_up_score = float(sorted_scores[1][1]) if len(sorted_scores) > 1 else 0.0
    selected_score = float(combined_scores.get(memory_class, 0.0) or 0.0)
    winner_margin = _clamp01(selected_score - runner_up_score)
    dominant_inputs = _derive_memory_class_dominant_inputs(
        memory_class=memory_class,
        reason=reason,
        same_turn_scores=same_turn_scores,
        mode_prior=mode_prior,
        insight_prior=insight_prior,
        protective_lock=protective_lock,
    )

    return {
        "memory_class": memory_class,
        "reason": reason,
        "trust": round(trust, 4),
        "degraded": degraded,
        "body_load": round(body_load, 4),
        "protection_bias": round(protection_bias, 4),
        "felt_energy": round(felt_energy, 4),
        "position_confidence": round(position_confidence, 4),
        "position_memory_weight": round(position_memory_weight, 4),
        "terrain_protect_bias": round(terrain_protect_bias, 4),
        "terrain_approach_bias": round(terrain_approach_bias, 4),
        "protection_mode": protection_mode_name,
        "protection_mode_strength": round(protection_mode_strength, 4),
        "relation_active": relation_active,
        "insight_reframing_bias": round(insight_reframing_bias, 4),
        "insight_class_focus": insight_class_focus,
        "same_turn_scores": {key: round(value, 4) for key, value in same_turn_scores.items()},
        "mode_prior": {key: round(value, 4) for key, value in mode_prior.items()},
        "insight_prior": insight_prior,
        "combined_scores": {key: round(value, 4) for key, value in combined_scores.items()},
        "winner_margin": round(winner_margin, 4),
        "dominant_inputs": dominant_inputs,
        "protective_lock": protective_lock,
    }


def _derive_insight_memory_prior(
    *,
    insight_reframing_bias: float,
    insight_class_focus: str,
    relation_active: bool,
    protection_mode_name: str,
    terrain_approach_bias: float,
    terrain_protect_bias: float,
    protection_bias: float,
    trust: float,
    degraded: bool,
    body_load: float,
) -> dict[str, float]:
    focus = str(insight_class_focus or "").strip()
    base = _clamp01(insight_reframing_bias)
    repair_prior = 0.0
    bond_prior = 0.0
    insight_trace_prior = 0.0

    if focus == "reframed_relation":
        repair_prior = _clamp01(
            base * 0.2
            * (1.0 if protection_mode_name in {"repair", "monitor"} else 0.65)
            * (1.0 if terrain_approach_bias >= 0.28 else 0.6)
        )
        bond_prior = _clamp01(
            base * 0.18
            * (1.0 if relation_active else 0.0)
            * (1.0 if terrain_protect_bias >= 0.18 or protection_bias >= 0.08 else 0.65)
        )
    elif focus == "new_link_hypothesis":
        insight_trace_prior = _clamp01(
            base * 0.18
            * (1.0 if trust >= 0.46 else 0.6)
            * (1.0 if not degraded and body_load < 0.12 else 0.45)
        )
    elif focus == "insight_trace":
        insight_trace_prior = _clamp01(
            base * 0.12
            * (1.0 if not degraded and body_load < 0.12 else 0.5)
        )

    return {
        "repair_trace": repair_prior,
        "bond_protection": bond_prior,
        "insight_trace": insight_trace_prior,
    }


def _derive_memory_class_same_turn_scores(
    *,
    workspace_mode: str,
    deferred_object_labels: Sequence[str],
    relation_active: bool,
    trust: float,
    degraded: bool,
    body_load: float,
    protection_bias: float,
    felt_energy: float,
    position_confidence: float,
    position_memory_weight: float,
    terrain_protect_bias: float,
    terrain_approach_bias: float,
    protection_mode_name: str,
    protection_mode_strength: float,
) -> dict[str, float]:
    guarded_value = 1.0 if workspace_mode == "guarded_foreground" else 0.0
    deferred_value = 1.0 if deferred_object_labels else 0.0
    degraded_value = 1.0 if degraded else 0.0
    protective_mode_value = 1.0 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0
    repair_mode_value = 1.0 if protection_mode_name == "repair" else 0.0
    monitor_value = 1.0 if protection_mode_name == "monitor" else 0.0
    same_turn_scores = {
        "episodic": _clamp01(
            0.14
            + 0.12 * trust
            + 0.08 * felt_energy
            + 0.06 * position_confidence
            - 0.12 * body_load
            - 0.12 * terrain_protect_bias
            - 0.1 * degraded_value
        ),
        "body_risk": _clamp01(
            0.34 * terrain_protect_bias
            + 0.28 * body_load
            + 0.16 * protection_bias
            + 0.16 * protective_mode_value
            + 0.12 * degraded_value
            + 0.06 * (1.0 - trust)
        ),
        "bond_protection": _clamp01(
            (0.24 if relation_active else 0.0)
            + 0.18 * terrain_protect_bias
            + 0.12 * protection_bias
            + 0.08 * protective_mode_value
            + 0.06 * position_confidence
            - 0.08 * body_load
        ),
        "repair_trace": _clamp01(
            0.34 * repair_mode_value
            + 0.2 * terrain_approach_bias
            + 0.08 * trust
            + 0.06 * monitor_value
            - 0.08 * degraded_value
            - 0.08 * body_load
        ),
        "unresolved_tension": _clamp01(
            0.26 * guarded_value
            + 0.22 * deferred_value
            + 0.18 * position_memory_weight
            + 0.08 * (1.0 - position_confidence)
            + 0.08 * degraded_value
        ),
        "safe_repeat": _clamp01(
            (0.2 if relation_active else 0.0)
            + 0.2 * terrain_approach_bias
            + 0.16 * trust
            + 0.12 * position_confidence
            + 0.08 * felt_energy
            - 0.14 * terrain_protect_bias
            - 0.12 * body_load
            - 0.12 * degraded_value
            - 0.08 * protective_mode_value
        ),
        "insight_trace": _clamp01(
            0.12 * felt_energy
            + 0.1 * trust
            + 0.08 * terrain_approach_bias
            - 0.08 * body_load
            - 0.08 * degraded_value
        ),
    }
    return same_turn_scores


def _derive_memory_class_mode_prior(
    *,
    workspace_mode: str,
    relation_active: bool,
    body_load: float,
    terrain_protect_bias: float,
    terrain_approach_bias: float,
    protection_mode_name: str,
    protection_mode_strength: float,
) -> dict[str, float]:
    protected_mode = protection_mode_name in {"contain", "stabilize", "shield"}
    repair_mode = protection_mode_name == "repair"
    guarded_value = 1.0 if workspace_mode == "guarded_foreground" else 0.0
    return {
        "episodic": _clamp01(0.04 * (1.0 if protection_mode_name == "monitor" else 0.0)),
        "body_risk": _clamp01(
            (0.32 if protected_mode else 0.0) * protection_mode_strength
            + 0.06 * body_load
        ),
        "bond_protection": _clamp01(
            (0.24 if relation_active and protected_mode and body_load < 0.18 else 0.0) * max(protection_mode_strength, terrain_protect_bias)
        ),
        "repair_trace": _clamp01(
            (0.34 if repair_mode else 0.0) * max(protection_mode_strength, terrain_approach_bias)
        ),
        "unresolved_tension": _clamp01(
            0.18 * guarded_value
            + (0.08 if protection_mode_name == "contain" else 0.0)
        ),
        "safe_repeat": _clamp01(
            (0.12 if relation_active and protection_mode_name == "monitor" else 0.0) * max(terrain_approach_bias, 0.5)
        ),
        "insight_trace": 0.0,
    }


def _combine_memory_class_scores(
    *,
    same_turn_scores: Mapping[str, float],
    mode_prior: Mapping[str, float],
    insight_prior: Mapping[str, float],
    protective_lock: bool,
) -> dict[str, float]:
    combined: dict[str, float] = {}
    keys = {
        *(str(key) for key in same_turn_scores.keys()),
        *(str(key) for key in mode_prior.keys()),
        *(str(key) for key in insight_prior.keys()),
    }
    for key in keys:
        same_turn = _clamp01(float(same_turn_scores.get(key, 0.0) or 0.0))
        mode = _clamp01(float(mode_prior.get(key, 0.0) or 0.0))
        insight = 0.0 if protective_lock else _clamp01(float(insight_prior.get(key, 0.0) or 0.0))
        combined[key] = _clamp01(same_turn + 0.24 * mode + 0.14 * insight)
    return combined


def _derive_memory_class_dominant_inputs(
    *,
    memory_class: str,
    reason: str,
    same_turn_scores: Mapping[str, float],
    mode_prior: Mapping[str, float],
    insight_prior: Mapping[str, float],
    protective_lock: bool,
) -> list[str]:
    dominant_inputs: list[str] = [reason]
    if float(same_turn_scores.get(memory_class, 0.0) or 0.0) >= 0.12:
        dominant_inputs.append("same_turn_evidence")
    if float(mode_prior.get(memory_class, 0.0) or 0.0) >= 0.08:
        dominant_inputs.append("mode_prior")
    if not protective_lock and float(insight_prior.get(memory_class, 0.0) or 0.0) >= 0.08:
        dominant_inputs.append("insight_prior")
    if protective_lock:
        dominant_inputs.append("protective_lock")
    deduped: list[str] = []
    for item in dominant_inputs:
        text = str(item).strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _derive_body_recovery_guard(
    *,
    self_state: Mapping[str, Any],
    temperament_estimate: Mapping[str, Any],
    qualia_planner_view: Mapping[str, Any],
    terrain_readout: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    contact_readiness: float,
) -> dict[str, object]:
    qualia = dict(qualia_planner_view or {})
    terrain = dict(terrain_readout or {})
    protection = dict(protection_mode or {})
    temperament = dict(temperament_estimate or {})

    stress = _clamp01(float(self_state.get("stress", 0.0) or 0.0))
    recovery_need = _clamp01(float(self_state.get("recovery_need", 0.0) or 0.0))
    safety_bias = _clamp01(float(self_state.get("safety_bias", 0.0) or 0.0))
    stabilization_drive = _clamp01(float(self_state.get("stabilization_drive", 0.0) or 0.0))
    qualia_body_load = _clamp01(float(qualia.get("body_load", 0.0) or 0.0))
    qualia_degraded = 1.0 if bool(qualia.get("degraded", False)) else 0.0
    qualia_protection_bias = _clamp01(float(qualia.get("protection_bias", 0.0) or 0.0))
    terrain_protect_bias = _clamp01(float(terrain.get("protect_bias", 0.0) or 0.0))
    terrain_approach_bias = _clamp01(float(terrain.get("approach_bias", 0.0) or 0.0))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_mode_strength = _clamp01(float(protection.get("strength", 0.0) or 0.0))
    protective_mode_value = 1.0 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0
    risk_tolerance = _clamp01(float(temperament.get("risk_tolerance", 0.0) or 0.0))
    recovery_discipline = _clamp01(float(temperament.get("recovery_discipline", 0.0) or 0.0))
    protect_floor = _clamp01(float(temperament.get("protect_floor", 0.0) or 0.0))

    scores = {
        "recovery_first": _clamp01(
            0.34 * recovery_need
            + 0.2 * stress
            + 0.16 * qualia_body_load
            + 0.12 * qualia_degraded
            + 0.1 * protective_mode_value * protection_mode_strength
            + 0.08 * stabilization_drive
            + 0.08 * recovery_discipline * max(recovery_need, qualia_body_load)
        ),
        "guarded": _clamp01(
            0.24 * terrain_protect_bias
            + 0.18 * qualia_protection_bias
            + 0.16 * protective_mode_value * protection_mode_strength
            + 0.14 * safety_bias
            + 0.12 * stress
            + 0.08 * (1.0 - _clamp01(contact_readiness))
            + 0.06 * qualia_degraded
            + 0.08 * protect_floor
            - 0.05 * risk_tolerance
        ),
        "open": _clamp01(
            0.28 * _clamp01(contact_readiness)
            + 0.18 * terrain_approach_bias
            + 0.08 * (1.0 - qualia_body_load)
            + 0.06 * (1.0 - qualia_degraded)
            + 0.08 * risk_tolerance
            - 0.18 * stress
            - 0.22 * recovery_need
            - 0.12 * terrain_protect_bias
            - 0.08 * protect_floor
            - 0.06 * recovery_discipline
        ),
    }
    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "recovery_need" if recovery_need >= 0.22 else "",
            "stress" if stress >= 0.18 else "",
            "body_load" if qualia_body_load >= 0.1 else "",
            "degraded_estimate" if qualia_degraded >= 1.0 else "",
            "protective_mode" if protective_mode_value >= 1.0 and protection_mode_strength >= 0.42 else "",
            "terrain_protect_bias" if terrain_protect_bias >= 0.22 else "",
            "contact_readiness" if state == "open" and contact_readiness >= 0.56 else "",
            "terrain_approach_bias" if state == "open" and terrain_approach_bias >= 0.24 else "",
            "temperament_forward_trace" if state == "open" and risk_tolerance >= 0.62 else "",
            "temperament_guard_floor" if state in {"guarded", "recovery_first"} and protect_floor >= 0.56 else "",
        ]
    )
    return {
        "state": state,
        "score": round(float(scores.get(state, 0.0) or 0.0), 4),
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "winner_margin": round(winner_margin, 4),
        "dominant_inputs": dominant_inputs,
    }


def _derive_body_homeostasis_state(
    *,
    self_state: Mapping[str, Any],
    temperament_estimate: Mapping[str, Any],
    qualia_planner_view: Mapping[str, Any],
    terrain_readout: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    contact_readiness: float,
) -> dict[str, object]:
    qualia = dict(qualia_planner_view or {})
    terrain = dict(terrain_readout or {})
    protection = dict(protection_mode or {})
    body_guard = dict(body_recovery_guard or {})
    temperament = dict(temperament_estimate or {})

    stress = _clamp01(float(self_state.get("stress", 0.0) or 0.0))
    recovery_need = _clamp01(float(self_state.get("recovery_need", 0.0) or 0.0))
    recent_strain = _clamp01(float(self_state.get("recent_strain", 0.0) or 0.0))
    stabilization_drive = _clamp01(float(self_state.get("stabilization_drive", 0.0) or 0.0))
    safety_bias = _clamp01(float(self_state.get("safety_bias", 0.0) or 0.0))
    autonomic_balance = _clamp01(float(self_state.get("autonomic_balance", 0.5) or 0.5))
    recovery_reopening = _clamp01(float(self_state.get("recovery_reopening", 0.0) or 0.0))
    qualia_body_load = _clamp01(float(qualia.get("body_load", 0.0) or 0.0))
    qualia_degraded = 1.0 if bool(qualia.get("degraded", False)) else 0.0
    qualia_protection_bias = _clamp01(float(qualia.get("protection_bias", 0.0) or 0.0))
    terrain_protect_bias = _clamp01(float(terrain.get("protect_bias", 0.0) or 0.0))
    terrain_approach_bias = _clamp01(float(terrain.get("approach_bias", 0.0) or 0.0))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_mode_strength = _clamp01(float(protection.get("strength", 0.0) or 0.0))
    body_guard_state = str(body_guard.get("state") or "open").strip() or "open"
    body_guard_score = _clamp01(float(body_guard.get("score", 0.0) or 0.0))
    recovery_discipline = _clamp01(float(temperament.get("recovery_discipline", 0.0) or 0.0))
    carry_focus = str(self_state.get("body_homeostasis_focus") or "").strip()
    carry_bias = _clamp01(float(self_state.get("body_homeostasis_carry_bias", 0.0) or 0.0))
    budget_focus = str(self_state.get("homeostasis_budget_focus") or "").strip()
    budget_bias = _clamp01(float(self_state.get("homeostasis_budget_bias", 0.0) or 0.0))

    strong_protective_mode = 1.0 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0
    recovering_mode = 1.0 if protection_mode_name in {"repair", "stabilize"} else 0.0

    scores = {
        "steady": _clamp01(
            0.24 * autonomic_balance
            + 0.16 * _clamp01(contact_readiness)
            + 0.14 * (1.0 - stress)
            + 0.14 * (1.0 - recovery_need)
            + 0.12 * (1.0 - qualia_body_load)
            + 0.08 * terrain_approach_bias
            - 0.14 * qualia_degraded
            - 0.12 * terrain_protect_bias
            - 0.08 * body_guard_score
        ),
        "strained": _clamp01(
            0.28 * stress
            + 0.2 * recent_strain
            + 0.16 * terrain_protect_bias
            + 0.12 * qualia_body_load
            + 0.1 * safety_bias
            + 0.08 * body_guard_score
            + 0.08 * qualia_protection_bias
            + 0.06 * (1.0 - _clamp01(contact_readiness))
            - 0.08 * autonomic_balance
        ),
        "recovering": _clamp01(
            0.24 * recovery_need
            + 0.18 * stabilization_drive
            + 0.14 * recovery_reopening
            + 0.12 * body_guard_score
            + 0.1 * recovering_mode * protection_mode_strength
            + 0.08 * recovery_discipline
            + 0.08 * qualia_body_load
            + 0.06 * terrain_protect_bias
        ),
        "depleted": _clamp01(
            0.32 * recovery_need
            + 0.22 * qualia_body_load
            + 0.16 * qualia_degraded
            + 0.12 * stress
            + 0.1 * (1.0 if body_guard_state == "recovery_first" else 0.0)
            + 0.08 * strong_protective_mode * protection_mode_strength
            + 0.06 * (1.0 - autonomic_balance)
        ),
    }
    if carry_focus in scores and carry_bias > 0.0:
        carry_scale = 0.16
        if carry_focus == "recovering":
            carry_scale = 0.18
        elif carry_focus == "depleted":
            carry_scale = 0.2
        elif carry_focus == "steady":
            carry_scale = 0.14
        scores[carry_focus] = _clamp01(scores[carry_focus] + carry_bias * carry_scale)
    if budget_focus in scores and budget_bias > 0.0:
        budget_scale = 0.08
        if budget_focus == "recovering":
            budget_scale = 0.1
        elif budget_focus == "depleted":
            budget_scale = 0.11
        elif budget_focus == "steady":
            budget_scale = 0.07
        scores[budget_focus] = _clamp01(scores[budget_focus] + budget_bias * budget_scale)
    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "recovery_need" if recovery_need >= 0.24 else "",
            "body_load" if qualia_body_load >= 0.12 else "",
            "degraded_estimate" if qualia_degraded >= 1.0 else "",
            "stress" if stress >= 0.2 else "",
            "recent_strain" if recent_strain >= 0.22 else "",
            "stabilization_drive" if state == "recovering" and stabilization_drive >= 0.2 else "",
            "recovery_reopening" if state == "recovering" and recovery_reopening >= 0.16 else "",
            "terrain_protect_bias" if terrain_protect_bias >= 0.24 else "",
            "terrain_approach_bias" if state == "steady" and terrain_approach_bias >= 0.22 else "",
            "autonomic_balance" if state == "steady" and autonomic_balance >= 0.56 else "",
            "body_recovery_guard" if body_guard_state in {"guarded", "recovery_first"} else "",
            "protective_mode" if strong_protective_mode >= 1.0 and protection_mode_strength >= 0.42 else "",
            "overnight_body_homeostasis" if carry_focus in scores and carry_bias >= 0.08 else "",
            "overnight_homeostasis_budget" if budget_focus in scores and budget_bias >= 0.08 else "",
        ]
    )
    return {
        "state": state,
        "score": round(float(scores.get(state, 0.0) or 0.0), 4),
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "winner_margin": round(winner_margin, 4),
        "dominant_inputs": dominant_inputs,
    }


def _derive_homeostasis_budget_state(
    *,
    self_state: Mapping[str, Any],
    temperament_estimate: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    contact_readiness: float,
) -> dict[str, object]:
    temperament = dict(temperament_estimate or {})
    body_homeostasis = dict(body_homeostasis_state or {})
    body_guard = dict(body_recovery_guard or {})
    protection = dict(protection_mode or {})

    stress = _clamp01(float(self_state.get("stress", 0.0) or 0.0))
    recovery_need = _clamp01(float(self_state.get("recovery_need", 0.0) or 0.0))
    recent_strain = _clamp01(float(self_state.get("recent_strain", 0.0) or 0.0))
    stabilization_drive = _clamp01(float(self_state.get("stabilization_drive", 0.0) or 0.0))
    autonomic_balance = _clamp01(float(self_state.get("autonomic_balance", 0.5) or 0.5))
    recovery_reopening = _clamp01(float(self_state.get("recovery_reopening", 0.0) or 0.0))
    recovery_discipline = _clamp01(float(temperament.get("recovery_discipline", 0.0) or 0.0))
    guard_bias = _clamp01(float(temperament.get("guard_bias", 0.0) or 0.0))
    carry_focus = str(self_state.get("homeostasis_budget_focus") or "").strip()
    carry_bias = _clamp01(float(self_state.get("homeostasis_budget_bias", 0.0) or 0.0))

    body_state_name = str(body_homeostasis.get("state") or "steady").strip() or "steady"
    body_state_score = _clamp01(float(body_homeostasis.get("score", 0.0) or 0.0))
    body_guard_state = str(body_guard.get("state") or "open").strip() or "open"
    body_guard_score = _clamp01(float(body_guard.get("score", 0.0) or 0.0))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_mode_strength = _clamp01(float(protection.get("strength", 0.0) or 0.0))

    steady_body = 1.0 if body_state_name == "steady" else 0.0
    strained_body = 1.0 if body_state_name == "strained" else 0.0
    recovering_body = 1.0 if body_state_name == "recovering" else 0.0
    depleted_body = 1.0 if body_state_name == "depleted" else 0.0
    strong_protective_mode = 1.0 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0
    budget_reserve = _clamp01(
        (1.0 - recovery_need) * 0.34
        + (1.0 - recent_strain) * 0.22
        + autonomic_balance * 0.18
        + steady_body * body_state_score * 0.14
        + recovery_discipline * 0.12
        - stress * 0.18
    )
    budget_debt = _clamp01(
        recovery_need * 0.34
        + recent_strain * 0.24
        + stress * 0.16
        + depleted_body * body_state_score * 0.16
        + max(0.0, body_guard_score - 0.2) * 0.1
        + strong_protective_mode * protection_mode_strength * 0.08
    )
    restoration_bias = _clamp01(
        recovery_need * 0.24
        + stabilization_drive * 0.22
        + recovery_reopening * 0.16
        + recovering_body * body_state_score * 0.16
        + recovery_discipline * 0.12
        + max(0.0, 1.0 - _clamp01(contact_readiness)) * 0.1
    )

    scores = {
        "steady": _clamp01(
            budget_reserve * 0.56
            + steady_body * body_state_score * 0.18
            + _clamp01(contact_readiness) * 0.08
            - budget_debt * 0.18
            - restoration_bias * 0.12
        ),
        "strained": _clamp01(
            recent_strain * 0.28
            + stress * 0.2
            + strained_body * body_state_score * 0.18
            + guard_bias * 0.08
            + body_guard_score * 0.1
            - budget_reserve * 0.12
        ),
        "recovering": _clamp01(
            restoration_bias * 0.48
            + recovery_need * 0.14
            + recovering_body * body_state_score * 0.18
            + max(0.0, body_guard_score - 0.12) * 0.08
            - budget_debt * 0.08
        ),
        "depleted": _clamp01(
            budget_debt * 0.54
            + depleted_body * body_state_score * 0.18
            + (1.0 if body_guard_state == "recovery_first" else 0.0) * 0.12
            + max(0.0, 1.0 - autonomic_balance) * 0.1
            - budget_reserve * 0.12
        ),
    }
    if carry_focus in scores and carry_bias > 0.0:
        carry_scale = 0.12
        if carry_focus == "recovering":
            carry_scale = 0.14
        elif carry_focus == "depleted":
            carry_scale = 0.16
        elif carry_focus == "steady":
            carry_scale = 0.1
        scores[carry_focus] = _clamp01(scores[carry_focus] + carry_bias * carry_scale)

    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "budget_reserve" if budget_reserve >= 0.46 and state == "steady" else "",
            "budget_debt" if budget_debt >= 0.32 and state in {"strained", "depleted"} else "",
            "restoration_bias" if restoration_bias >= 0.28 and state == "recovering" else "",
            "recovery_need" if recovery_need >= 0.22 else "",
            "recent_strain" if recent_strain >= 0.22 else "",
            "body_homeostasis_state" if body_state_name in {"strained", "recovering", "depleted"} else "",
            "body_recovery_guard" if body_guard_state in {"guarded", "recovery_first"} else "",
            "protective_mode" if strong_protective_mode >= 1.0 and protection_mode_strength >= 0.42 else "",
            "overnight_homeostasis_budget" if carry_focus in scores and carry_bias >= 0.08 else "",
        ]
    )
    return {
        "state": state,
        "score": round(float(scores.get(state, 0.0) or 0.0), 4),
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "winner_margin": round(winner_margin, 4),
        "reserve_level": round(budget_reserve, 4),
        "debt_level": round(budget_debt, 4),
        "restoration_bias": round(restoration_bias, 4),
        "dominant_inputs": dominant_inputs,
    }


def _derive_initiative_readiness(
    *,
    self_state: Mapping[str, Any],
    temperament_estimate: Mapping[str, Any],
    qualia_planner_view: Mapping[str, Any],
    terrain_readout: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    contact_readiness: float,
    coherence_score: float,
    human_presence_signal: float,
) -> dict[str, object]:
    qualia = dict(qualia_planner_view or {})
    terrain = dict(terrain_readout or {})
    protection = dict(protection_mode or {})
    body_guard = dict(body_recovery_guard or {})
    temperament = dict(temperament_estimate or {})

    stress = _clamp01(float(self_state.get("stress", 0.0) or 0.0))
    recovery_need = _clamp01(float(self_state.get("recovery_need", 0.0) or 0.0))
    recent_strain = _clamp01(float(self_state.get("recent_strain", 0.0) or 0.0))
    qualia_degraded = 1.0 if bool(qualia.get("degraded", False)) else 0.0
    qualia_body_load = _clamp01(float(qualia.get("body_load", 0.0) or 0.0))
    terrain_approach_bias = _clamp01(float(terrain.get("approach_bias", 0.0) or 0.0))
    terrain_protect_bias = _clamp01(float(terrain.get("protect_bias", 0.0) or 0.0))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_mode_strength = _clamp01(float(protection.get("strength", 0.0) or 0.0))
    guard_state = str(body_guard.get("state") or "open").strip()
    guard_score = _clamp01(float(body_guard.get("score", 0.0) or 0.0))
    followup_state = str(self_state.get("initiative_followup_state") or "hold").strip() or "hold"
    followup_bias = _clamp01(float(self_state.get("initiative_followup_bias", 0.0) or 0.0))
    agenda_focus = str(self_state.get("agenda_focus") or "").strip()
    agenda_bias = _clamp01(float(self_state.get("agenda_bias", 0.0) or 0.0))
    commitment_carry_bias = _clamp01(float(self_state.get("commitment_carry_bias", 0.0) or 0.0))
    commitment_followup_focus = str(self_state.get("commitment_followup_focus") or "").strip()
    risk_tolerance = _clamp01(float(temperament.get("risk_tolerance", 0.0) or 0.0))
    ambiguity_tolerance = _clamp01(float(temperament.get("ambiguity_tolerance", 0.0) or 0.0))
    curiosity_drive = _clamp01(float(temperament.get("curiosity_drive", 0.0) or 0.0))
    bond_drive = _clamp01(float(temperament.get("bond_drive", 0.0) or 0.0))
    recovery_discipline = _clamp01(float(temperament.get("recovery_discipline", 0.0) or 0.0))
    protect_floor = _clamp01(float(temperament.get("protect_floor", 0.0) or 0.0))
    temporal_membrane = _extract_temporal_membrane_bias(self_state)
    timeline_coherence = _clamp01(float(temporal_membrane.get("timeline_coherence", 0.0) or 0.0))
    reentry_pull = _clamp01(float(temporal_membrane.get("reentry_pull", 0.0) or 0.0))
    supersession_pressure = _clamp01(float(temporal_membrane.get("supersession_pressure", 0.0) or 0.0))
    continuity_pressure = _clamp01(float(temporal_membrane.get("continuity_pressure", 0.0) or 0.0))
    relation_reentry_pull = _clamp01(float(temporal_membrane.get("relation_reentry_pull", 0.0) or 0.0))
    temporal_mode = str(temporal_membrane.get("mode") or "ambient").strip() or "ambient"

    hold_score = _clamp01(
        0.28 * guard_score
        + 0.18 * recovery_need
        + 0.16 * stress
        + 0.12 * qualia_body_load
        + 0.1 * qualia_degraded
        + 0.1 * terrain_protect_bias
        + 0.06 * protection_mode_strength * (1.0 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0)
        + 0.08 * protect_floor
        + 0.08 * recovery_discipline * max(recovery_need, qualia_body_load)
    )
    tentative_score = _clamp01(
        0.2 * _clamp01(contact_readiness)
        + 0.14 * terrain_approach_bias
        + 0.12 * coherence_score
        + 0.08 * human_presence_signal
        + 0.08 * (1.0 - recent_strain)
        + 0.06 * (1.0 - guard_score)
        + 0.1 * ambiguity_tolerance
        + 0.08 * bond_drive
        - 0.08 * qualia_degraded
        - 0.08 * protect_floor
    )
    ready_score = _clamp01(
        0.3 * _clamp01(contact_readiness)
        + 0.18 * terrain_approach_bias
        + 0.14 * coherence_score
        + 0.12 * human_presence_signal
        + 0.1 * (1.0 - stress)
        + 0.08 * (1.0 - recovery_need)
        + 0.08 * (1.0 - qualia_body_load)
        + 0.12 * risk_tolerance
        + 0.12 * curiosity_drive
        + 0.08 * bond_drive
        - 0.12 * terrain_protect_bias
        - 0.1 * qualia_degraded
        - 0.08 * protection_mode_strength * (1.0 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0)
        - 0.12 * recovery_discipline * max(recovery_need, stress)
        - 0.08 * protect_floor
    )
    hold_score = _clamp01(
        hold_score
        + supersession_pressure * 0.12
        - timeline_coherence * 0.03
        - reentry_pull * 0.02
    )
    tentative_score = _clamp01(
        tentative_score
        + reentry_pull * 0.08
        + continuity_pressure * 0.06
        + relation_reentry_pull * 0.05
        - supersession_pressure * 0.05
    )
    ready_score = _clamp01(
        ready_score
        + reentry_pull * 0.1
        + timeline_coherence * 0.06
        + continuity_pressure * 0.04
        + relation_reentry_pull * 0.05
        - supersession_pressure * 0.1
    )
    if temporal_mode == "supersede":
        hold_score = _clamp01(hold_score + 0.04)
        ready_score = _clamp01(ready_score - 0.04)
    elif temporal_mode in {"reentry", "cohere"}:
        tentative_score = _clamp01(tentative_score + 0.03)
    if followup_state == "offer_next_step":
        ready_score = _clamp01(ready_score + followup_bias * 0.18)
        tentative_score = _clamp01(tentative_score + followup_bias * 0.08)
    elif followup_state == "reopen_softly":
        tentative_score = _clamp01(tentative_score + followup_bias * 0.14)
        ready_score = _clamp01(ready_score + followup_bias * 0.05)
    if commitment_followup_focus == "offer_next_step":
        ready_score = _clamp01(ready_score + commitment_carry_bias * 0.12)
        tentative_score = _clamp01(tentative_score + commitment_carry_bias * 0.04)
    elif commitment_followup_focus == "reopen_softly":
        tentative_score = _clamp01(tentative_score + commitment_carry_bias * 0.1)
        ready_score = _clamp01(ready_score + commitment_carry_bias * 0.03)
    elif commitment_followup_focus == "hold":
        hold_score = _clamp01(hold_score + commitment_carry_bias * 0.12)
    if agenda_focus == "step_forward":
        ready_score = _clamp01(ready_score + agenda_bias * 0.1)
        tentative_score = _clamp01(tentative_score + agenda_bias * 0.03)
    elif agenda_focus == "repair":
        tentative_score = _clamp01(tentative_score + agenda_bias * 0.08)
        hold_score = _clamp01(hold_score + agenda_bias * 0.01)
    elif agenda_focus == "revisit":
        tentative_score = _clamp01(tentative_score + agenda_bias * 0.08)
        hold_score = _clamp01(hold_score + agenda_bias * 0.02)
    elif agenda_focus == "hold":
        hold_score = _clamp01(hold_score + agenda_bias * 0.08)
    if guard_state == "recovery_first":
        ready_score = _clamp01(ready_score * 0.35)
        tentative_score = _clamp01(tentative_score * 0.75)
    elif guard_state == "guarded":
        ready_score = _clamp01(ready_score * 0.6)
    scores = {
        "hold": hold_score,
        "tentative": tentative_score,
        "ready": ready_score,
    }
    state, winner_margin = _winner_and_margin(scores)
    score = scores[state]
    dominant_inputs = _compact(
        [
            "contact_readiness" if contact_readiness >= 0.56 and state in {"tentative", "ready"} else "",
            "terrain_approach_bias" if terrain_approach_bias >= 0.24 and state in {"tentative", "ready"} else "",
            "coherence_score" if coherence_score >= 0.5 and state == "ready" else "",
            "body_recovery_guard" if guard_state in {"guarded", "recovery_first"} else "",
            "initiative_followup_bias" if followup_bias >= 0.18 and followup_state in {"reopen_softly", "offer_next_step"} else "",
            "overnight_commitment_carry" if commitment_carry_bias >= 0.12 and commitment_followup_focus in {"hold", "reopen_softly", "offer_next_step"} else "",
            "overnight_agenda_carry" if agenda_bias >= 0.1 and agenda_focus in {"hold", "revisit", "repair", "step_forward"} else "",
            "temperament_forward_trace" if state == "ready" and risk_tolerance >= 0.5 else "",
            "temperament_bond_trace" if state == "tentative" and bond_drive >= 0.56 else "",
            "temperament_guard_floor" if state == "hold" and protect_floor >= 0.56 else "",
            "degraded_estimate" if qualia_degraded >= 1.0 else "",
            "recovery_need" if recovery_need >= 0.22 else "",
            "stress" if stress >= 0.18 else "",
            "terrain_protect_bias" if terrain_protect_bias >= 0.22 and state == "hold" else "",
            "temporal_reentry_pull" if reentry_pull >= 0.2 and state in {"tentative", "ready"} else "",
            "temporal_supersession_pressure" if supersession_pressure >= 0.2 and state == "hold" else "",
            "temporal_timeline_coherence" if timeline_coherence >= 0.22 and state == "ready" else "",
        ]
    )
    return {
        "state": state,
        "score": round(score, 4),
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "winner_margin": round(winner_margin, 4),
        "dominant_inputs": dominant_inputs,
    }


def _derive_relational_continuity_state(
    *,
    self_state: Mapping[str, Any],
    temperament_estimate: Mapping[str, Any],
    relation_bias_strength: float,
    related_person_ids: Sequence[str],
    relation_competition_state: Mapping[str, Any] | None,
    social_topology_state: Mapping[str, Any] | None,
    partner_timing_hint: str,
    partner_stance_hint: str,
    partner_social_interpretation: str,
    contact_readiness: float,
    coherence_score: float,
    human_presence_signal: float,
    conscious_workspace: Mapping[str, Any],
    resonance_evaluation: Mapping[str, Any],
    conversational_objects: Mapping[str, Any],
    object_operations: Mapping[str, Any],
    interaction_effects: Mapping[str, Any],
    terrain_readout: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
) -> dict[str, object]:
    workspace = dict(conscious_workspace or {})
    resonance = dict(resonance_evaluation or {})
    objects = dict(conversational_objects or {})
    operations = dict(object_operations or {})
    effects = dict(interaction_effects or {})
    terrain = dict(terrain_readout or {})
    protection = dict(protection_mode or {})
    body_homeostasis = dict(body_homeostasis_state or {})
    temperament = dict(temperament_estimate or {})
    relation_competition = dict(relation_competition_state or {})
    social_topology = dict(social_topology_state or {})

    continuity_score = _clamp01(float(self_state.get("continuity_score", 0.0) or 0.0))
    social_grounding = _clamp01(float(self_state.get("social_grounding", 0.0) or 0.0))
    trust_memory = _clamp01(float(self_state.get("trust_memory", 0.0) or 0.0))
    familiarity = _clamp01(float(self_state.get("familiarity", 0.0) or 0.0))
    attachment = _clamp01(float(self_state.get("attachment", 0.0) or 0.0))
    recent_strain = _clamp01(float(self_state.get("recent_strain", 0.0) or 0.0))
    relation_active = 1.0 if (relation_bias_strength >= 0.28 or bool(related_person_ids)) else 0.0
    terrain_approach_bias = _clamp01(float(terrain.get("approach_bias", 0.0) or 0.0))
    terrain_protect_bias = _clamp01(float(terrain.get("protect_bias", 0.0) or 0.0))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_mode_strength = _clamp01(float(protection.get("strength", 0.0) or 0.0))
    body_homeostasis_name = str(body_homeostasis.get("state") or "steady").strip() or "steady"
    body_homeostasis_score = _clamp01(float(body_homeostasis.get("score", 0.0) or 0.0))
    bond_drive = _clamp01(float(temperament.get("bond_drive", 0.0) or 0.0))
    carry_focus = str(self_state.get("relational_continuity_focus") or "").strip()
    carry_bias = _clamp01(float(self_state.get("relational_continuity_carry_bias", 0.0) or 0.0))
    competition_state_name = str(relation_competition.get("state") or "ambient").strip() or "ambient"
    competition_level = _clamp01(float(relation_competition.get("competition_level", 0.0) or 0.0))
    competition_margin = _clamp01(float(relation_competition.get("winner_margin", 0.0) or 0.0))
    competition_total_people = int(relation_competition.get("total_people") or 0)
    topology_state_name = str(social_topology.get("state") or "ambient").strip() or "ambient"
    topology_visibility = _clamp01(float(social_topology.get("visibility_pressure", 0.0) or 0.0))
    topology_threading = _clamp01(float(social_topology.get("threading_pressure", 0.0) or 0.0))
    topology_hierarchy = _clamp01(float(social_topology.get("hierarchy_pressure", 0.0) or 0.0))
    temporal_membrane = _extract_temporal_membrane_bias(self_state)
    timeline_coherence = _clamp01(float(temporal_membrane.get("timeline_coherence", 0.0) or 0.0))
    reentry_pull = _clamp01(float(temporal_membrane.get("reentry_pull", 0.0) or 0.0))
    supersession_pressure = _clamp01(float(temporal_membrane.get("supersession_pressure", 0.0) or 0.0))
    continuity_pressure = _clamp01(float(temporal_membrane.get("continuity_pressure", 0.0) or 0.0))
    relation_reentry_pull = _clamp01(float(temporal_membrane.get("relation_reentry_pull", 0.0) or 0.0))
    temporal_mode = str(temporal_membrane.get("mode") or "ambient").strip() or "ambient"

    other_person_state = dict(resonance.get("estimated_other_person_state") or {})
    detail_room = _room_level_score(other_person_state.get("detail_room_level"))
    next_step_room = _room_level_score(other_person_state.get("next_step_room_level"))
    pressure_sensitivity = _room_level_score(other_person_state.get("pressure_sensitivity_level"))
    workspace_reportable = 1.0 if (workspace.get("reportable_slice") or []) else 0.0
    workspace_withheld = 1.0 if (workspace.get("withheld_slice") or []) else 0.0
    workspace_actionable = 1.0 if (workspace.get("actionable_slice") or []) else 0.0
    object_kinds = {
        str(item.get("object_kind") or "").strip()
        for item in objects.get("objects") or []
        if isinstance(item, Mapping) and str(item.get("object_kind") or "").strip()
    }
    operation_kinds = {
        str(item.get("operation_kind") or "").strip()
        for item in operations.get("operations") or []
        if isinstance(item, Mapping) and str(item.get("operation_kind") or "").strip()
    }
    effect_kinds = {
        str(item.get("effect_kind") or "").strip()
        for item in effects.get("effects") or []
        if isinstance(item, Mapping) and str(item.get("effect_kind") or "").strip()
    }

    shared_thread_signal = 1.0 if (
        "shared_thread" in object_kinds
        or "anchor_shared_thread" in operation_kinds
        or "preserve_continuity" in effect_kinds
    ) else 0.0
    repair_window_signal = 1.0 if (
        "repair_window" in str(partner_social_interpretation or "")
        or protection_mode_name == "repair"
        or "name_overreach" in operation_kinds
        or "reopen_carefully" in operation_kinds
    ) else 0.0
    respectful_delay = 1.0 if (
        str(partner_timing_hint).strip() == "delayed"
        or str(partner_stance_hint).strip() == "respectful"
    ) else 0.0
    future_open_signal = 1.0 if (
        "future_open" in str(partner_social_interpretation or "")
        or "keep_next_step_connected" in effect_kinds
        or "offer_small_next_step" in operation_kinds
    ) else 0.0
    protective_hold = 1.0 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0

    relation_signal = _clamp01(
        0.36 * _clamp01(float(relation_bias_strength or 0.0))
        + 0.16 * relation_active
        + 0.14 * attachment
        + 0.12 * familiarity
        + 0.12 * trust_memory
        + 0.1 * bond_drive
    )
    scores = {
        "distant": _clamp01(
            0.24 * (1.0 - relation_signal)
            + 0.18 * recent_strain
            + 0.14 * pressure_sensitivity
            + 0.12 * protective_hold * protection_mode_strength
            + 0.12 * (1.0 if body_homeostasis_name in {"strained", "depleted"} else 0.0) * body_homeostasis_score
            + 0.08 * respectful_delay
            - 0.12 * continuity_score
            - 0.1 * social_grounding
        ),
        "holding_thread": _clamp01(
            0.24 * relation_signal
            + 0.18 * shared_thread_signal
            + 0.16 * respectful_delay
            + 0.14 * workspace_withheld
            + 0.1 * continuity_score
            + 0.08 * protective_hold * protection_mode_strength
            + 0.08 * bond_drive
        ),
        "reopening": _clamp01(
            0.24 * relation_signal
            + 0.2 * repair_window_signal
            + 0.12 * _clamp01(contact_readiness)
            + 0.1 * coherence_score
            + 0.08 * detail_room
            + 0.08 * workspace_reportable
            + 0.08 * trust_memory
            + 0.06 * (1.0 if body_homeostasis_name not in {"depleted"} else 0.0)
            - 0.08 * pressure_sensitivity
        ),
        "co_regulating": _clamp01(
            0.24 * relation_signal
            + 0.18 * social_grounding
            + 0.14 * continuity_score
            + 0.12 * _clamp01(contact_readiness)
            + 0.1 * coherence_score
            + 0.08 * human_presence_signal
            + 0.08 * detail_room
            + 0.08 * next_step_room
            + 0.06 * terrain_approach_bias
            + 0.06 * future_open_signal
            - 0.08 * protective_hold * protection_mode_strength
            - 0.08 * terrain_protect_bias
            - 0.1 * (1.0 if body_homeostasis_name in {"recovering", "depleted"} else 0.0) * body_homeostasis_score
        ),
    }
    scores["distant"] = _clamp01(
        scores["distant"]
        + supersession_pressure * 0.12
        - continuity_pressure * 0.04
        - reentry_pull * 0.02
    )
    scores["holding_thread"] = _clamp01(
        scores["holding_thread"]
        + continuity_pressure * 0.12
        + timeline_coherence * 0.06
    )
    scores["reopening"] = _clamp01(
        scores["reopening"]
        + reentry_pull * 0.14
        + relation_reentry_pull * 0.12
        + continuity_pressure * 0.04
        - supersession_pressure * 0.08
    )
    scores["co_regulating"] = _clamp01(
        scores["co_regulating"]
        + timeline_coherence * 0.08
        + continuity_pressure * 0.1
        + relation_reentry_pull * 0.08
        - supersession_pressure * 0.08
    )
    if temporal_mode == "supersede":
        scores["distant"] = _clamp01(scores["distant"] + 0.04)
        scores["co_regulating"] = _clamp01(scores["co_regulating"] - 0.04)
    elif temporal_mode == "reentry":
        scores["reopening"] = _clamp01(scores["reopening"] + 0.04)
    if carry_focus in scores and carry_bias > 0.0:
        carry_scale = 0.16
        if carry_focus == "holding_thread":
            carry_scale = 0.17
        elif carry_focus == "reopening":
            carry_scale = 0.18
        elif carry_focus == "co_regulating":
            carry_scale = 0.14 if body_homeostasis_name in {"recovering", "depleted"} else 0.18
        scores[carry_focus] = _clamp01(scores[carry_focus] + carry_bias * carry_scale)
    if competition_state_name == "competing_threads" and competition_level > 0.0:
        competition_pressure = _clamp01(
            competition_level
            + (0.08 if competition_margin <= 0.12 else 0.0)
            + (0.06 if competition_total_people >= 3 else 0.0)
        )
        shared_thread_pressure = _clamp01(
            competition_pressure
            + (0.08 if shared_thread_signal >= 1.0 else 0.0)
            + (0.06 if workspace_withheld >= 1.0 else 0.0)
        )
        scores["holding_thread"] = _clamp01(scores["holding_thread"] + shared_thread_pressure * 0.3)
        scores["reopening"] = _clamp01(scores["reopening"] - competition_pressure * 0.12)
        scores["co_regulating"] = _clamp01(scores["co_regulating"] - shared_thread_pressure * 0.34)
        if pressure_sensitivity >= 0.5 or respectful_delay >= 1.0:
            scores["distant"] = _clamp01(scores["distant"] + competition_pressure * 0.08)
    elif competition_state_name == "dominant_thread" and competition_total_people > 1:
        scores["reopening"] = _clamp01(scores["reopening"] + competition_margin * 0.08)
        scores["co_regulating"] = _clamp01(scores["co_regulating"] + competition_margin * 0.06)
    if topology_state_name == "threaded_group" and topology_threading > 0.0:
        scores["holding_thread"] = _clamp01(scores["holding_thread"] + topology_threading * 0.2)
        scores["reopening"] = _clamp01(scores["reopening"] - topology_threading * 0.06)
        scores["co_regulating"] = _clamp01(scores["co_regulating"] - topology_threading * 0.16)
    elif topology_state_name == "public_visible" and topology_visibility > 0.0:
        scores["distant"] = _clamp01(scores["distant"] + topology_visibility * 0.12)
        scores["holding_thread"] = _clamp01(scores["holding_thread"] + topology_visibility * 0.14)
        scores["co_regulating"] = _clamp01(scores["co_regulating"] - topology_visibility * 0.18)
    elif topology_state_name == "hierarchical" and topology_hierarchy > 0.0:
        scores["distant"] = _clamp01(scores["distant"] + topology_hierarchy * 0.14)
        scores["holding_thread"] = _clamp01(scores["holding_thread"] + topology_hierarchy * 0.12)
        scores["reopening"] = _clamp01(scores["reopening"] - topology_hierarchy * 0.06)
        scores["co_regulating"] = _clamp01(scores["co_regulating"] - topology_hierarchy * 0.2)
    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "relation_bias" if relation_signal >= 0.34 else "",
            "relation_competition" if competition_level >= 0.22 else "",
            "multiple_people" if competition_total_people > 1 else "",
            "shared_thread" if shared_thread_signal >= 1.0 else "",
            "repair_window" if repair_window_signal >= 1.0 and state == "reopening" else "",
            "respectful_delay" if respectful_delay >= 1.0 and state in {"holding_thread", "distant"} else "",
            "detail_room" if detail_room >= 0.5 and state in {"reopening", "co_regulating"} else "",
            "next_step_room" if next_step_room >= 0.5 and state == "co_regulating" else "",
            "continuity_score" if continuity_score >= 0.56 and state in {"holding_thread", "co_regulating"} else "",
            "social_grounding" if social_grounding >= 0.56 and state == "co_regulating" else "",
            "social_topology_threaded_group" if topology_state_name == "threaded_group" and topology_threading >= 0.24 else "",
            "social_topology_public_visible" if topology_state_name == "public_visible" and topology_visibility >= 0.24 else "",
            "social_topology_hierarchical" if topology_state_name == "hierarchical" and topology_hierarchy >= 0.24 else "",
            "workspace_withheld" if workspace_withheld >= 1.0 and state == "holding_thread" else "",
            "workspace_actionable" if workspace_actionable >= 1.0 and state == "co_regulating" else "",
            "body_homeostasis" if body_homeostasis_name in {"strained", "recovering", "depleted"} else "",
            "protective_mode" if protective_hold >= 1.0 and protection_mode_strength >= 0.42 else "",
            "overnight_relational_continuity" if carry_focus in scores and carry_bias >= 0.08 else "",
            "temporal_reentry_pull" if reentry_pull >= 0.22 and state in {"reopening", "co_regulating"} else "",
            "temporal_supersession_pressure" if supersession_pressure >= 0.22 and state == "distant" else "",
            "temporal_timeline_coherence" if timeline_coherence >= 0.22 and state in {"holding_thread", "co_regulating"} else "",
        ]
    )
    return {
        "state": state,
        "score": round(float(scores.get(state, 0.0) or 0.0), 4),
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "winner_margin": round(winner_margin, 4),
        "dominant_inputs": dominant_inputs,
    }


def _extract_temporal_membrane_bias(self_state: Mapping[str, Any]) -> dict[str, object]:
    payload = dict(self_state or {})
    timeline_coherence = _clamp01(float(payload.get("temporal_timeline_coherence", 0.0) or 0.0))
    reentry_pull = _clamp01(float(payload.get("temporal_reentry_pull", 0.0) or 0.0))
    supersession_pressure = _clamp01(float(payload.get("temporal_supersession_pressure", 0.0) or 0.0))
    continuity_pressure = _clamp01(float(payload.get("temporal_continuity_pressure", 0.0) or 0.0))
    relation_reentry_pull = _clamp01(float(payload.get("temporal_relation_reentry_pull", 0.0) or 0.0))
    if timeline_coherence <= 0.0:
        timeline_coherence = _clamp01(float(payload.get("temporal_timeline_bias", 0.0) or 0.0))
    if reentry_pull <= 0.0:
        reentry_pull = _clamp01(float(payload.get("temporal_reentry_bias", 0.0) or 0.0))
    if supersession_pressure <= 0.0:
        supersession_pressure = _clamp01(float(payload.get("temporal_supersession_bias", 0.0) or 0.0))
    if continuity_pressure <= 0.0:
        continuity_pressure = _clamp01(float(payload.get("temporal_continuity_bias", 0.0) or 0.0))
    if relation_reentry_pull <= 0.0:
        relation_reentry_pull = _clamp01(float(payload.get("temporal_relation_reentry_bias", 0.0) or 0.0))
    mode = str(payload.get("temporal_membrane_mode") or payload.get("temporal_membrane_focus") or "ambient").strip() or "ambient"
    return {
        "timeline_coherence": timeline_coherence,
        "reentry_pull": reentry_pull,
        "supersession_pressure": supersession_pressure,
        "continuity_pressure": continuity_pressure,
        "relation_reentry_pull": relation_reentry_pull,
        "mode": mode,
    }


def _derive_attention_regulation_state(
    *,
    current_focus: str,
    reportable_slice: Sequence[str],
    withheld_slice: Sequence[str],
    current_risks: Sequence[str],
    primary_conversational_object_label: str,
    body_recovery_guard: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
    terrain_readout: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    relation_competition_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
) -> dict[str, object]:
    reportable_focus = str(reportable_slice[0]).strip() if reportable_slice else ""
    selective_target = _normalize_focus_label(
        reportable_focus or primary_conversational_object_label or current_focus or "ambient"
    )
    guard_state = str(body_recovery_guard.get("state") or "open").strip() or "open"
    body_state = str(body_homeostasis_state.get("state") or "steady").strip() or "steady"
    terrain_protect = _clamp01(float(terrain_readout.get("protect_bias", 0.0) or 0.0))
    terrain_approach = _clamp01(float(terrain_readout.get("approach_bias", 0.0) or 0.0))
    protection_mode_name = str(protection_mode.get("mode") or "").strip()
    protection_strength = _clamp01(float(protection_mode.get("strength", 0.0) or 0.0))
    competition_state = str(relation_competition_state.get("state") or "ambient").strip() or "ambient"
    competition_level = _clamp01(float(relation_competition_state.get("competition_level", 0.0) or 0.0))
    dominant_person_id = str(relation_competition_state.get("dominant_person_id") or "").strip()
    topology_state = str(social_topology_state.get("state") or "ambient").strip() or "ambient"
    topology_threading = _clamp01(float(social_topology_state.get("threading_pressure", 0.0) or 0.0))
    topology_visibility = _clamp01(float(social_topology_state.get("visibility_pressure", 0.0) or 0.0))
    topology_hierarchy = _clamp01(float(social_topology_state.get("hierarchy_pressure", 0.0) or 0.0))
    risk_signal = 1.0 if current_risks else 0.0

    if risk_signal >= 1.0 or protection_mode_name in {"contain", "stabilize", "shield"}:
        reflex_target = "boundary"
    elif guard_state == "recovery_first" or body_state in {"recovering", "depleted"}:
        reflex_target = "body"
    elif competition_state == "competing_threads" and dominant_person_id:
        reflex_target = f"person:{dominant_person_id}"
    elif topology_state in {"public_visible", "hierarchical"}:
        reflex_target = "boundary"
    else:
        reflex_target = selective_target

    selective_hold = _clamp01(
        0.34 * (1.0 if reportable_focus else 0.0)
        + 0.22 * terrain_approach
        + 0.16 * (1.0 if selective_target and selective_target != "ambient" else 0.0)
        + 0.12 * (1.0 if withheld_slice else 0.0)
        - 0.1 * terrain_protect
    )
    reflex_guard = _clamp01(
        0.34 * risk_signal
        + 0.22 * terrain_protect
        + 0.16 * protection_strength * (1.0 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0)
        + 0.14 * (1.0 if guard_state == "recovery_first" else 0.0)
        + 0.1 * (1.0 if body_state in {"recovering", "depleted"} else 0.0)
        + 0.08 * competition_level
        + 0.08 * max(topology_visibility, topology_hierarchy)
    )
    split_guarded = _clamp01(
        min(selective_hold, reflex_guard) * 0.74
        + max(competition_level, topology_threading) * 0.16
        + (0.08 if withheld_slice else 0.0)
    )
    scores = {
        "selective_hold": selective_hold,
        "reflex_guard": reflex_guard,
        "split_guarded": split_guarded,
    }
    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "reportable_focus" if reportable_focus and state in {"selective_hold", "split_guarded"} else "",
            "withheld_slice" if withheld_slice and state in {"selective_hold", "split_guarded"} else "",
            "current_risks" if current_risks and state in {"reflex_guard", "split_guarded"} else "",
            "terrain_protect_bias" if terrain_protect >= 0.22 and state in {"reflex_guard", "split_guarded"} else "",
            "terrain_approach_bias" if terrain_approach >= 0.24 and state == "selective_hold" else "",
            "protective_mode" if protection_mode_name in {"contain", "stabilize", "shield"} else "",
            "body_guard" if guard_state == "recovery_first" else "",
            "body_homeostasis" if body_state in {"recovering", "depleted"} else "",
            "relation_competition" if competition_level >= 0.22 else "",
            "social_topology_threaded" if topology_state == "threaded_group" and topology_threading >= 0.22 else "",
            "social_topology_visibility" if topology_state == "public_visible" and topology_visibility >= 0.22 else "",
            "social_topology_hierarchy" if topology_state == "hierarchical" and topology_hierarchy >= 0.22 else "",
        ]
    )
    return {
        "state": state,
        "score": round(float(scores.get(state, 0.0) or 0.0), 4),
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "winner_margin": round(winner_margin, 4),
        "selective_target": selective_target,
        "reflex_target": reflex_target,
        "dominant_inputs": dominant_inputs,
    }


def _derive_grice_guard_state(
    *,
    self_state: Mapping[str, Any],
    current_focus: str,
    attention_target: str,
    primary_conversational_object_label: str,
    reportable_slice: Sequence[str],
    withheld_slice: Sequence[str],
    actionable_slice: Sequence[str],
    relation_bias_strength: float,
    response_strategy: str,
    question_budget: int,
    question_pressure: float,
    relational_continuity_state: Mapping[str, Any],
    relational_style_memory_state: Mapping[str, Any],
    cultural_conversation_state: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    attention_regulation_state: Mapping[str, Any],
) -> dict[str, object]:
    state_payload = dict(self_state or {})
    semantic_seed_focus = str(state_payload.get("semantic_seed_focus") or "").strip()
    semantic_seed_strength = _clamp01(float(state_payload.get("semantic_seed_strength", 0.0) or 0.0))
    semantic_seed_recurrence = _clamp01(float(state_payload.get("semantic_seed_recurrence", 0.0) or 0.0) / 2.0)
    long_term_theme_focus = str(state_payload.get("long_term_theme_focus") or "").strip()
    long_term_theme_summary = str(state_payload.get("long_term_theme_summary") or "").strip()
    long_term_theme_strength = _clamp01(float(state_payload.get("long_term_theme_strength", 0.0) or 0.0))
    relation_seed_summary = str(state_payload.get("relation_seed_summary") or "").strip()
    relation_seed_strength = _clamp01(float(state_payload.get("relation_seed_strength", 0.0) or 0.0))
    guard_state = str(body_recovery_guard.get("state") or "open").strip() or "open"
    protection_mode_name = str(protection_mode.get("mode") or "").strip()
    protection_strength = _clamp01(float(protection_mode.get("strength", 0.0) or 0.0))
    relational_state = str(relational_continuity_state.get("state") or "").strip()
    relational_score = _clamp01(float(relational_continuity_state.get("score", 0.0) or 0.0))
    relational_style = dict(relational_style_memory_state or {})
    cultural_state = dict(cultural_conversation_state or {})
    advice_tolerance = _clamp01(float(relational_style.get("advice_tolerance", 0.0) or 0.0))
    lexical_familiarity = _clamp01(float(relational_style.get("lexical_familiarity", 0.0) or 0.0))
    style_playful_ceiling = _clamp01(float(relational_style.get("playful_ceiling", 0.0) or 0.0))
    cultural_politeness_pressure = _clamp01(float(cultural_state.get("politeness_pressure", 0.0) or 0.0))
    attention_state = str(attention_regulation_state.get("state") or "").strip()

    active_labels = [
        _normalize_focus_label(current_focus),
        _normalize_focus_label(attention_target),
        _normalize_focus_label(primary_conversational_object_label),
    ] + [_normalize_focus_label(item) for item in reportable_slice[:2]]
    memory_labels = [
        _normalize_focus_label(semantic_seed_focus),
        _normalize_focus_label(long_term_theme_focus),
        _normalize_focus_label(long_term_theme_summary),
        _normalize_focus_label(relation_seed_summary),
    ]
    label_overlap = _label_overlap(active_labels, memory_labels)
    knownness_pressure = _clamp01(
        label_overlap * 0.34
        + semantic_seed_strength * 0.18
        + semantic_seed_recurrence * 0.18
        + long_term_theme_strength * 0.14
        + relation_seed_strength * 0.12
        + (0.08 if relation_seed_summary and relation_bias_strength >= 0.28 else 0.0)
    )
    advice_pressure = _clamp01(
        (0.24 if response_strategy in {"shared_world_next_step", "attune_then_extend"} else 0.0)
        + min(question_budget, 2) * 0.16
        + _clamp01(question_pressure) * 0.16
        + (0.1 if actionable_slice else 0.0)
        + cultural_politeness_pressure * 0.08
    )
    caution_pressure = _clamp01(
        (0.18 if guard_state in {"guarded", "recovery_first"} else 0.0)
        + protection_strength * (0.16 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.08)
        + (0.12 if relational_state == "holding_thread" else 0.0)
        + (0.1 if attention_state in {"reflex_guard", "split_guarded"} else 0.0)
        + (0.08 if withheld_slice else 0.0)
        - advice_tolerance * 0.08
    )
    scores = {
        "advise_openly": _clamp01(
            advice_pressure * 0.7
            + (1.0 - knownness_pressure) * 0.22
            + advice_tolerance * 0.14
            + lexical_familiarity * 0.06
            - caution_pressure * 0.18
        ),
        "acknowledge_then_extend": _clamp01(
            knownness_pressure * 0.22
            + advice_pressure * 0.24
            + (0.16 if relational_state in {"reopening", "co_regulating"} else 0.0) * max(relational_score, 0.5)
            + advice_tolerance * 0.08
            - caution_pressure * 0.08
        ),
        "attune_without_repeating": _clamp01(
            knownness_pressure * 0.28
            + caution_pressure * 0.18
            + (0.14 if relational_state == "holding_thread" else 0.0) * max(relational_score, 0.5)
            + (0.08 if withheld_slice else 0.0)
            + (1.0 - advice_tolerance) * 0.08
        ),
        "hold_obvious_advice": _clamp01(
            knownness_pressure * 0.44
            + semantic_seed_recurrence * 0.18
            + caution_pressure * 0.16
            + (0.08 if question_budget == 0 else 0.0)
            + (1.0 - advice_tolerance) * 0.12
            + (1.0 - lexical_familiarity) * 0.06
        ),
    }
    state, winner_margin = _winner_and_margin(scores)
    if state == "hold_obvious_advice":
        question_budget_cap = 0
    elif state == "attune_without_repeating":
        question_budget_cap = min(question_budget, 0)
    elif state == "acknowledge_then_extend":
        question_budget_cap = min(question_budget, 1)
    else:
        question_budget_cap = int(question_budget)
    dominant_inputs = _compact(
        [
            "known_thread_overlap" if label_overlap >= 0.34 else "",
            "semantic_seed" if semantic_seed_strength >= 0.28 else "",
            "semantic_recurrence" if semantic_seed_recurrence >= 0.24 else "",
            "long_term_theme" if long_term_theme_strength >= 0.24 else "",
            "relation_seed" if relation_seed_strength >= 0.24 else "",
            "relational_style_memory" if advice_tolerance >= 0.24 or lexical_familiarity >= 0.24 or style_playful_ceiling >= 0.24 else "",
            "withheld_slice" if withheld_slice and state in {"attune_without_repeating", "hold_obvious_advice"} else "",
            "actionable_slice" if actionable_slice and state in {"advise_openly", "acknowledge_then_extend"} else "",
            "holding_thread" if relational_state == "holding_thread" else "",
            "body_guard" if guard_state in {"guarded", "recovery_first"} else "",
            "protective_mode" if protection_mode_name in {"contain", "stabilize", "shield"} else "",
            "reflex_attention" if attention_state in {"reflex_guard", "split_guarded"} else "",
        ]
    )
    return {
        "state": state,
        "score": round(float(scores.get(state, 0.0) or 0.0), 4),
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "winner_margin": round(winner_margin, 4),
        "knownness_pressure": round(knownness_pressure, 4),
        "question_budget_cap": int(question_budget_cap),
        "suppress_obvious_advice": bool(state in {"attune_without_repeating", "hold_obvious_advice"}),
        "prefer_acknowledge_only": bool(state == "hold_obvious_advice"),
        "dominant_inputs": dominant_inputs,
    }


def _label_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_tokens = {token for text in left for token in _focus_tokens(text)}
    right_tokens = {token for text in right for token in _focus_tokens(text)}
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    base = max(1, min(len(left_tokens), len(right_tokens)))
    return _clamp01(overlap / float(base))


def _focus_tokens(value: str) -> list[str]:
    raw = str(value or "").strip().lower().replace(":", " ").replace("/", " ").replace("_", " ")
    if not raw:
        return []
    return [token for token in raw.split() if token]


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    items = sorted(
        ((str(key), _clamp01(float(value or 0.0))) for key, value in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not items:
        return "none", 0.0
    winner_key, winner_score = items[0]
    runner_up_score = items[1][1] if len(items) > 1 else 0.0
    return winner_key, _clamp01(winner_score - runner_up_score)


def _room_level_score(value: Any) -> float:
    text = str(value or "").strip().lower()
    if text == "high":
        return 1.0
    if text == "medium":
        return 0.6
    if text == "low":
        return 0.2
    return 0.0
