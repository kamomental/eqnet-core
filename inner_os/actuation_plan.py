from __future__ import annotations

from typing import Any, Mapping


def derive_actuation_plan(
    interaction_policy: Mapping[str, Any] | None,
    action_posture: Mapping[str, Any] | None,
) -> dict[str, object]:
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

    return {
        "execution_mode": execution_mode,
        "primary_action": primary_action,
        "action_queue": _compact(action_queue),
        "reply_permission": reply_permission,
        "wait_before_action": wait_before_action,
        "repair_window_commitment": repair_window_commitment,
        "outcome_goal": outcome_goal,
        "boundary_mode": boundary_mode,
        "attention_target": attention_target,
        "memory_write_priority": memory_write_priority,
        "memory_write_class": memory_write_class,
        "memory_write_class_reason": memory_write_class_reason,
        "affordance_priority": affordance_priority,
        "do_not_cross": do_not_cross,
        "other_person_state": other_person_state,
        "resonance_prioritize_actions": resonance_prioritize_actions,
        "resonance_avoid_actions": resonance_avoid_actions,
        "conversational_objects": conversational_objects,
        "object_operations": object_operations,
        "interaction_effects": interaction_effects,
        "interaction_judgement_view": interaction_judgement_view,
        "contact_readiness": round(contact_readiness, 4),
        "repair_bias": repair_bias,
        "disclosure_depth": disclosure_depth,
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
        "commitment_state": commitment_state,
        "commitment_mode": commitment_mode,
        "commitment_target": commitment_target,
        "commitment_score": commitment_score,
        "relational_continuity_state": relational_continuity_state,
        "relational_continuity_name": relational_continuity_name,
        "relational_continuity_score": relational_continuity_score,
        "relation_competition_state": relation_competition_state,
        "relation_competition_name": relation_competition_name,
        "relation_competition_level": relation_competition_level,
        "social_topology_state": social_topology_state,
        "social_topology_name": social_topology_name,
        "social_topology_score": social_topology_score,
        "insight_event": insight_event,
    }


def _compact(values: list[str]) -> list[str]:
    return [value for value in values if value]


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
