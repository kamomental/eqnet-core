from __future__ import annotations

from typing import Any, Mapping


def derive_action_posture(
    interaction_policy: Mapping[str, Any] | None,
) -> dict[str, object]:
    packet = dict(interaction_policy or {})
    contract = dict(packet.get("conversation_contract") or {})
    response_action = dict(contract.get("response_action_now") or {})
    strategy = str(packet.get("response_strategy") or "").strip()
    opening_move = str(packet.get("opening_move") or "").strip()
    disclosure_depth = str(packet.get("disclosure_depth") or "").strip()
    attention_target = str(packet.get("attention_target") or "").strip()
    memory_write_priority = str(packet.get("memory_write_priority") or "").strip()
    memory_write_class = str(packet.get("memory_write_class") or "").strip()
    memory_write_class_reason = str(packet.get("memory_write_class_reason") or "").strip()
    affordance_priority = [str(item) for item in packet.get("affordance_priority") or [] if str(item).strip()]
    do_not_cross = [str(item) for item in packet.get("do_not_cross") or [] if str(item).strip()]
    actionable_slice = [str(item) for item in packet.get("actionable_slice") or [] if str(item).strip()]
    constraint_field = dict(packet.get("constraint_field") or {})
    conscious_workspace = dict(packet.get("conscious_workspace") or {})
    workspace_mode = str(conscious_workspace.get("workspace_mode") or "").strip()
    reportability_limit = str(constraint_field.get("reportability_limit") or "").strip()
    reportability_gate_mode = str(packet.get("reportability_gate_mode") or "").strip()
    other_person_state = dict(packet.get("other_person_state") or {})
    resonance_prioritize_actions = [
        str(item) for item in packet.get("resonance_prioritize_actions") or [] if str(item).strip()
    ]
    resonance_avoid_actions = [
        str(item) for item in packet.get("resonance_avoid_actions") or [] if str(item).strip()
    ]
    conversational_objects = dict(packet.get("conversational_objects") or {})
    object_operations = dict(packet.get("object_operations") or {})
    interaction_effects = dict(packet.get("interaction_effects") or {})
    interaction_judgement_view = dict(packet.get("interaction_judgement_view") or {})
    qualia_planner_view = dict(packet.get("qualia_planner_view") or {})
    terrain_readout = dict(packet.get("terrain_readout") or {})
    protection_mode = dict(packet.get("protection_mode") or {})
    body_recovery_guard = dict(packet.get("body_recovery_guard") or {})
    body_homeostasis_state = dict(packet.get("body_homeostasis_state") or {})
    homeostasis_budget_state = dict(packet.get("homeostasis_budget_state") or {})
    initiative_readiness = dict(packet.get("initiative_readiness") or {})
    commitment_state = dict(packet.get("commitment_state") or {})
    learning_mode_state = dict(packet.get("learning_mode_state") or {})
    social_experiment_loop_state = dict(packet.get("social_experiment_loop_state") or {})
    live_engagement_state = dict(packet.get("live_engagement_state") or {})
    situation_risk_state = dict(packet.get("situation_risk_state") or {})
    emergency_posture = dict(packet.get("emergency_posture") or {})
    relational_continuity_state = dict(packet.get("relational_continuity_state") or {})
    relation_competition_state = dict(packet.get("relation_competition_state") or {})
    social_topology_state = dict(packet.get("social_topology_state") or {})
    insight_event = dict(packet.get("insight_event") or {})
    primary_object_operation = dict(packet.get("primary_object_operation") or {})
    primary_operation_kind = str(primary_object_operation.get("operation_kind") or "").strip()
    if not primary_operation_kind:
        primary_operation_kind = str(response_action.get("primary_operation") or "").strip()
    ordered_operation_kinds = [
        str(item) for item in packet.get("ordered_operation_kinds") or [] if str(item).strip()
    ]
    if not ordered_operation_kinds:
        ordered_operation_kinds = [
            str(item).strip()
            for item in response_action.get("ordered_operations") or []
            if str(item).strip()
        ]
    ordered_effect_kinds = [
        str(item) for item in packet.get("ordered_effect_kinds") or [] if str(item).strip()
    ]
    if not ordered_effect_kinds:
        ordered_effect_kinds = [
            str(item).strip()
            for item in contract.get("ordered_effects") or []
            if str(item).strip()
        ]
    if not ordered_effect_kinds:
        ordered_effect_kinds = [
            str(item.get("effect") or "").strip()
            for item in contract.get("wanted_effect_on_other") or contract.get("intended_effects") or []
            if isinstance(item, Mapping) and str(item.get("effect") or "").strip()
        ]
    question_budget = int(packet.get("question_budget") or 0)
    if not question_budget:
        question_budget = int(response_action.get("question_budget") or 0)
    question_pressure = _float01(packet.get("question_pressure"))
    if question_pressure <= 0.0:
        question_pressure = _float01(response_action.get("question_pressure"))
    defer_dominance = _float01(packet.get("defer_dominance"))
    if defer_dominance <= 0.0:
        defer_dominance = _float01(response_action.get("defer_dominance"))
    boundary_pressure = _float01((constraint_field or {}).get("boundary_pressure"))
    qualia_trust = _float01(qualia_planner_view.get("trust"))
    qualia_degraded = bool(qualia_planner_view.get("degraded", False))
    qualia_body_load = _float01(qualia_planner_view.get("body_load"))
    qualia_protection_bias = _float01(qualia_planner_view.get("protection_bias"))
    qualia_felt_energy = _float01(qualia_planner_view.get("felt_energy"))
    terrain_protect_bias = _float01(terrain_readout.get("protect_bias"))
    terrain_approach_bias = _float01(terrain_readout.get("approach_bias"))
    terrain_avoid_bias = _float01(terrain_readout.get("avoid_bias"))
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
    situation_risk_name = str(situation_risk_state.get("state") or "").strip()
    situation_risk_immediacy = _float01(situation_risk_state.get("immediacy"))
    situation_risk_dialogue_room = _float01(situation_risk_state.get("dialogue_room"))
    situation_risk_relation_break = _float01(situation_risk_state.get("relation_break"))
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
    if not attention_target:
        attention_target = str(
            contract.get("focus_now")
            or contract.get("primary_object")
            or packet.get("primary_conversational_object_label")
            or ""
        ).strip()
    if not memory_write_priority:
        condition_summary = [str(item).strip() for item in contract.get("condition_summary") or [] if str(item).strip()]
        if any("relation" in item.lower() for item in condition_summary):
            memory_write_priority = "relation_episode"
        elif attention_target:
            memory_write_priority = "foreground_trace"
    if not memory_write_class:
        memory_write_class = "episodic"
    if not disclosure_depth:
        if defer_dominance >= 0.62 or question_pressure >= 0.62:
            disclosure_depth = "minimal"
        elif ordered_effect_kinds and any(
            item in {"enable_small_next_step", "keep_next_step_connected"} for item in ordered_effect_kinds
        ):
            disclosure_depth = "medium"
        else:
            disclosure_depth = "light"

    engagement_mode = "attune"
    outcome_goal = "increase_safe_contact"
    boundary_mode = "permeable"
    next_action_candidates = ["stay_visible", "invite_state", "gentle_approach"]

    if (
        primary_operation_kind == "offer_small_next_step"
        or "offer_small_next_step" in ordered_operation_kinds
        or "enable_small_next_step" in ordered_effect_kinds
        or "keep_next_step_connected" in ordered_effect_kinds
    ):
        engagement_mode = "co_move"
        outcome_goal = "shared_progress"
        boundary_mode = "forward_open"
        next_action_candidates = ["synchronize", "map_next_step", "pace_match", "keep_step_connected"]
    elif (
        opening_move == "name_overreach_and_reduce_force"
        or strategy == "repair_then_attune"
    ):
        engagement_mode = "repair"
        outcome_goal = "restore_contact_without_pressure"
        boundary_mode = "softened"
        next_action_candidates = ["name_overreach", "reduce_force", "reopen_carefully"]
    elif (
        opening_move == "reduce_force_and_secure_boundary"
        or strategy == "contain_then_stabilize"
        or (
            primary_operation_kind == "hold_without_probe"
            and "protect_boundary" in ordered_effect_kinds
            and boundary_pressure >= 0.74
        )
    ):
        engagement_mode = "contain"
        outcome_goal = "reduce_instability"
        boundary_mode = "protective"
        next_action_candidates = ["secure_boundary", "stabilize", "reduce_force"]
    elif (
        strategy == "respectful_wait"
        or primary_operation_kind in {"protect_unfinished_part", "defer_detail"}
        or (
            primary_operation_kind == "hold_without_probe"
            and (
                defer_dominance >= 0.58
                or question_pressure >= 0.62
                or "keep_return_point" in ordered_operation_kinds
                or "protect_unfinished_part" in ordered_operation_kinds
                or "defer_detail" in ordered_operation_kinds
            )
        )
        or (
            defer_dominance >= 0.72
            and ("protect_unfinished_part" in ordered_operation_kinds or "defer_detail" in ordered_operation_kinds)
        )
    ):
        engagement_mode = "wait"
        outcome_goal = "preserve_distance_with_connection"
        boundary_mode = "respectful"
        next_action_candidates = ["defer", "hold_presence", "leave_return_point"]
    elif primary_operation_kind == "narrow_clarify":
        engagement_mode = "attune"
        outcome_goal = "clarify_visible_part"
        boundary_mode = "bounded"
        next_action_candidates = ["check_visible_part", "bound_scope", "confirm_before_extend"]
    elif strategy == "reflect_without_settling":
        engagement_mode = "reflect"
        outcome_goal = "preserve_ambiguity"
        boundary_mode = "open_ambiguity"
        next_action_candidates = ["hold_meaning_open", "observe_more", "defer_closure"]
    elif strategy == "contain_then_stabilize":
        engagement_mode = "contain"
        outcome_goal = "reduce_instability"
        boundary_mode = "protective"
        next_action_candidates = ["secure_boundary", "stabilize", "reduce_force"]
    elif strategy == "shared_world_next_step":
        engagement_mode = "co_move"
        outcome_goal = "shared_progress"
        boundary_mode = "forward_open"
        next_action_candidates = ["synchronize", "map_next_step", "pace_match"]

    if workspace_mode == "guarded_foreground":
        boundary_mode = "guarded"
        if engagement_mode == "attune":
            engagement_mode = "contain"
            outcome_goal = "preserve_contact_without_forcing_report"
        if "withhold_detail" not in next_action_candidates:
            next_action_candidates.append("withhold_detail")

    if reportability_limit == "withhold":
        if "force_disclosure" not in do_not_cross:
            do_not_cross.append("force_disclosure")
    if actionable_slice and engagement_mode in {"attune", "contain", "repair"}:
        next_action_candidates = list(dict.fromkeys(next_action_candidates + ["hold_actionable_contact"]))
    if "anchor_shared_thread" in ordered_operation_kinds or "preserve_continuity" in ordered_effect_kinds:
        next_action_candidates = list(dict.fromkeys(next_action_candidates + ["keep_shared_thread"]))
    if "keep_connection_open" in ordered_effect_kinds:
        next_action_candidates = list(dict.fromkeys(next_action_candidates + ["leave_return_point"]))

    qualia_protective_hold = (
        qualia_degraded
        or qualia_trust <= 0.45
        or qualia_protection_bias >= 0.14
        or qualia_body_load >= 0.1
    )
    if qualia_protective_hold:
        if engagement_mode == "attune":
            engagement_mode = "contain"
            outcome_goal = "preserve_contact_without_forcing_report"
        boundary_mode = "protective" if boundary_mode in {"permeable", "forward_open"} else boundary_mode
        next_action_candidates = list(
            dict.fromkeys(
                ["stabilize_before_extend", "check_load_softly", *next_action_candidates]
            )
        )
    elif qualia_felt_energy >= 0.18 and engagement_mode == "attune":
        next_action_candidates = list(dict.fromkeys(["hold_felt_thread", *next_action_candidates]))

    if body_homeostasis_name == "depleted":
        if engagement_mode in {"attune", "co_move", "reflect"}:
            engagement_mode = "contain"
            outcome_goal = "preserve_body_margin"
        boundary_mode = "protective"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "honor_body_limit",
                    "reduce_load_first",
                    "restore_body_margin",
                    *next_action_candidates,
                ]
            )
        )
    elif body_homeostasis_name == "recovering" and body_homeostasis_score >= 0.32:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
            outcome_goal = "stabilize_while_reengaging"
        if boundary_mode == "permeable":
            boundary_mode = "softened"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "keep_step_small",
                    "stabilize_while_reengaging",
                    *next_action_candidates,
                ]
            )
        )
    elif body_homeostasis_name == "strained" and engagement_mode == "co_move":
        engagement_mode = "attune"
        outcome_goal = "preserve_contact_without_spike"
        next_action_candidates = list(dict.fromkeys(["keep_step_small", *next_action_candidates]))

    if homeostasis_budget_name == "depleted" and homeostasis_budget_score >= 0.3:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
            outcome_goal = "preserve_energy_budget"
        if boundary_mode == "forward_open":
            boundary_mode = "bounded"
        next_action_candidates = list(dict.fromkeys(["preserve_energy_budget", "keep_step_small", *next_action_candidates]))
    elif homeostasis_budget_name == "recovering" and homeostasis_budget_score >= 0.32:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
            outcome_goal = "reengage_with_budget_awareness"
        next_action_candidates = list(dict.fromkeys(["protect_recovery_budget", *next_action_candidates]))

    if body_recovery_guard_state == "recovery_first":
        if engagement_mode in {"attune", "co_move", "reflect"}:
            engagement_mode = "contain"
            outcome_goal = "restore_body_margin_before_extension"
        boundary_mode = "protective"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "restore_body_margin",
                    "wait_for_recovery",
                    "reduce_force",
                    *next_action_candidates,
                ]
            )
        )
    elif body_recovery_guard_state == "guarded" and body_recovery_guard_score >= 0.42:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
            outcome_goal = "protect_contact_while_holding_next_step"
        if boundary_mode == "forward_open":
            boundary_mode = "bounded"
        next_action_candidates = list(dict.fromkeys(["keep_step_small", *next_action_candidates]))

    if protection_mode_name in {"contain", "stabilize", "shield"}:
        if engagement_mode in {"attune", "reflect"}:
            engagement_mode = "contain"
            outcome_goal = "preserve_contact_without_forcing_report"
        boundary_mode = "protective"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "follow_protection_mode",
                    "stabilize_before_extend",
                    *next_action_candidates,
                ]
            )
        )
    elif protection_mode_name == "repair" and engagement_mode == "attune":
        next_action_candidates = list(dict.fromkeys(["follow_repair_opening", *next_action_candidates]))

    if terrain_protect_bias >= 0.22 and "terrain_protective_hold" not in next_action_candidates:
        next_action_candidates.append("terrain_protective_hold")
    if terrain_avoid_bias > terrain_approach_bias and boundary_mode == "permeable":
        boundary_mode = "bounded"
    if relational_continuity_name == "holding_thread":
        if engagement_mode == "attune":
            outcome_goal = "preserve_relational_thread"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "keep_shared_thread",
                    "leave_return_point",
                    *next_action_candidates,
                ]
            )
        )
    elif relational_continuity_name == "reopening" and body_homeostasis_name != "depleted":
        if engagement_mode == "wait":
            engagement_mode = "repair"
            outcome_goal = "reopen_contact_without_breaking_safety"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "reopen_shared_thread_gently",
                    "repair_without_rushing",
                    *next_action_candidates,
                ]
            )
        )
    elif (
        relational_continuity_name == "co_regulating"
        and relational_continuity_score >= 0.38
        and body_homeostasis_name not in {"recovering", "depleted"}
        and protection_mode_name not in {"contain", "stabilize", "shield"}
    ):
        if engagement_mode == "attune":
            outcome_goal = "co_regulate_then_progress"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "co_regulate_with_partner",
                    "keep_shared_thread",
                    *next_action_candidates,
                ]
            )
        )
    if relation_competition_name == "competing_threads" and relation_competition_level >= 0.34:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
            outcome_goal = "preserve_multiple_threads"
        if boundary_mode == "forward_open":
            boundary_mode = "bounded"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "hold_multiple_threads",
                    "do_not_collapse_to_single_thread",
                    "leave_return_point",
                    *next_action_candidates,
                ]
            )
        )
    if social_topology_name == "public_visible" and social_topology_score >= 0.34:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
            outcome_goal = "protect_visible_contact"
        if boundary_mode in {"permeable", "forward_open"}:
            boundary_mode = "bounded"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "respect_visible_context",
                    "keep_disclosure_small",
                    *next_action_candidates,
                ]
            )
        )
    elif social_topology_name == "hierarchical" and social_topology_score >= 0.34:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
            outcome_goal = "respect_role_gradient_while_contact"
        if boundary_mode == "permeable":
            boundary_mode = "bounded"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "respect_role_gradient",
                    "avoid_public_overreach",
                    *next_action_candidates,
                ]
            )
        )
    elif social_topology_name == "threaded_group" and social_topology_score >= 0.34:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
            outcome_goal = "preserve_group_threads"
        if boundary_mode == "forward_open":
            boundary_mode = "bounded"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "keep_group_threads_visible",
                    "avoid_collapsing_group_context",
                    *next_action_candidates,
                ]
            )
        )
    learning_mode_active = learning_mode_score >= 0.44
    social_experiment_active = social_experiment_score >= 0.44 or social_experiment_probe_intensity >= 0.36

    if learning_mode_name == "observe_only" and learning_mode_active:
        if engagement_mode == "co_move":
            engagement_mode = "reflect"
            outcome_goal = "read_before_extension"
        if boundary_mode == "forward_open":
            boundary_mode = "bounded"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "observe_more",
                    "read_reaction_first",
                    *next_action_candidates,
                ]
            )
        )
    elif learning_mode_name == "hold_and_wait" and learning_mode_active:
        if engagement_mode in {"attune", "co_move", "reflect"}:
            engagement_mode = "wait"
            outcome_goal = "hold_probe_without_losing_thread"
        if boundary_mode in {"permeable", "forward_open"}:
            boundary_mode = "bounded"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "hold_probe",
                    "leave_return_point",
                    *next_action_candidates,
                ]
            )
        )
    elif learning_mode_name == "repair_probe" and learning_mode_active:
        if engagement_mode == "wait":
            engagement_mode = "repair"
            outcome_goal = "test_small_repair_contact"
        next_action_candidates = list(dict.fromkeys(["test_repair_small", *next_action_candidates]))
    elif (
        learning_mode_name == "integrate_and_commit"
        and learning_mode_active
        and protection_mode_name not in {"contain", "stabilize", "shield"}
        and body_recovery_guard_state == "open"
    ):
        if engagement_mode == "attune":
            engagement_mode = "co_move"
            outcome_goal = "carry_decided_direction"
        next_action_candidates = list(dict.fromkeys(["confirm_shared_direction", *next_action_candidates]))
    elif learning_mode_name == "test_small" and learning_mode_active and body_recovery_guard_state != "recovery_first":
        next_action_candidates = list(dict.fromkeys(["test_small_step", *next_action_candidates]))

    if social_experiment_name == "watch_and_read" and social_experiment_active:
        next_action_candidates = list(dict.fromkeys(["read_reaction_first", *next_action_candidates]))
    elif social_experiment_name == "hold_probe" and social_experiment_active:
        next_action_candidates = list(dict.fromkeys(["hold_probe", "leave_return_point", *next_action_candidates]))
    elif social_experiment_name == "repair_signal_probe" and social_experiment_active:
        next_action_candidates = list(dict.fromkeys(["test_repair_small", *next_action_candidates]))
    elif social_experiment_name == "test_small_step" and social_experiment_active:
        next_action_candidates = list(dict.fromkeys(["test_small_step", *next_action_candidates]))
    elif social_experiment_name == "confirm_shared_direction" and social_experiment_active:
        next_action_candidates = list(dict.fromkeys(["confirm_shared_direction", *next_action_candidates]))

    if (
        initiative_readiness_state == "ready"
        and initiative_readiness_score >= 0.44
        and body_recovery_guard_state == "open"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
    ):
        if engagement_mode == "attune":
            engagement_mode = "co_move"
            outcome_goal = "shared_progress"
        if boundary_mode in {"permeable", "bounded"}:
            boundary_mode = "forward_open"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "offer_next_step_if_welcomed",
                    "map_next_step",
                    *next_action_candidates,
                ]
            )
        )
    elif initiative_readiness_state == "tentative" and body_recovery_guard_state != "recovery_first":
        next_action_candidates = list(dict.fromkeys(["test_small_next_step", *next_action_candidates]))
    if insight_triggered:
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "pause_for_orientation",
                    "name_new_connection_softly",
                    *next_action_candidates,
                ]
            )
        )
        if engagement_mode in {"attune", "co_move"} and protection_mode_name not in {"contain", "stabilize", "shield"}:
            outcome_goal = "stabilize_new_connection"
        if boundary_mode == "permeable":
            boundary_mode = "bounded"
        if insight_orient_bias >= 0.42 and engagement_mode == "co_move":
            engagement_mode = "attune"

    if commitment_mode == "commit":
        if commitment_target == "repair" and body_recovery_guard_state != "recovery_first":
            engagement_mode = "repair"
            outcome_goal = "follow_repair_commitment"
            next_action_candidates = list(
                dict.fromkeys(
                    [
                        "stay_with_decided_repair",
                        "repair_without_rushing",
                        *next_action_candidates,
                    ]
                )
            )
        elif commitment_target == "bond_protect":
            if engagement_mode == "attune":
                engagement_mode = "contain"
            boundary_mode = "protective" if boundary_mode in {"permeable", "bounded"} else boundary_mode
            outcome_goal = "protect_bond_without_breaking_contact"
            next_action_candidates = list(
                dict.fromkeys(
                    [
                        "hold_bond_guard_line",
                        "protect_connection_while_moving",
                        *next_action_candidates,
                    ]
                )
            )
        elif (
            commitment_target == "step_forward"
            and body_recovery_guard_state == "open"
            and protection_mode_name not in {"contain", "stabilize", "shield"}
        ):
            engagement_mode = "co_move"
            boundary_mode = "forward_open" if boundary_mode != "protective" else boundary_mode
            outcome_goal = "advance_on_decided_line"
            next_action_candidates = list(
                dict.fromkeys(
                    [
                        "advance_with_commitment",
                        "offer_next_step_if_welcomed",
                        *next_action_candidates,
                    ]
                )
            )
        elif commitment_target in {"stabilize", "hold"}:
            if engagement_mode in {"attune", "reflect"}:
                engagement_mode = "contain"
            boundary_mode = "protective" if boundary_mode in {"permeable", "bounded"} else boundary_mode
            next_action_candidates = list(
                dict.fromkeys(
                    [
                        "hold_decided_line",
                        "stay_with_protection_choice",
                        *next_action_candidates,
                    ]
                )
            )
    elif commitment_mode == "waver":
        if boundary_mode == "permeable":
            boundary_mode = "bounded"
        next_action_candidates = list(dict.fromkeys(["do_not_rush_commitment", *next_action_candidates]))

    if social_topology_name == "public_visible" and social_topology_score >= 0.34:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
        if boundary_mode in {"permeable", "forward_open"}:
            boundary_mode = "bounded"
    elif social_topology_name == "hierarchical" and social_topology_score >= 0.34:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
        if boundary_mode == "permeable":
            boundary_mode = "bounded"
    elif social_topology_name == "threaded_group" and social_topology_score >= 0.34:
        if engagement_mode == "co_move":
            engagement_mode = "attune"
        if boundary_mode == "forward_open":
            boundary_mode = "bounded"

    if live_engagement_name == "pickup_comment" and live_engagement_score >= 0.38:
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "pick_up_comment",
                    "answer_visible_comment",
                    "return_to_chat",
                    *next_action_candidates,
                ]
            )
        )
    elif live_engagement_name == "riff_with_comment" and live_engagement_score >= 0.38:
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "riff_current_comment",
                    "weave_light_callback",
                    "keep_chat_loop_open",
                    *next_action_candidates,
                ]
            )
        )
    elif live_engagement_name == "seed_topic" and live_engagement_score >= 0.38:
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "seed_small_topic",
                    "offer_chat_hook",
                    "check_audience_reaction",
                    *next_action_candidates,
                ]
            )
        )

    emergency_active = emergency_posture_score >= 0.38 and emergency_posture_name not in {"", "observe"}
    if emergency_active and emergency_posture_name == "de_escalate":
        if engagement_mode in {"attune", "co_move", "reflect"}:
            engagement_mode = "contain"
        outcome_goal = "lower_risk_without_escalation"
        boundary_mode = "guarded" if boundary_mode == "permeable" else "protective"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "set_clear_boundary",
                    "lower_heat",
                    "keep_words_short",
                    "watch_reaction",
                    *next_action_candidates,
                ]
            )
        )
    elif emergency_active and emergency_posture_name == "create_distance":
        engagement_mode = "contain"
        outcome_goal = "increase_distance_safely"
        boundary_mode = "protective"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "create_distance",
                    "orient_to_exit",
                    "keep_words_short",
                    "avoid_negotiation",
                    *next_action_candidates,
                ]
            )
        )
    elif emergency_active and emergency_posture_name == "exit":
        engagement_mode = "contain"
        outcome_goal = "leave_risk_field"
        boundary_mode = "protective"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "exit_space",
                    "move_to_safety",
                    "terminate_contact",
                    "seek_help_if_needed",
                    *next_action_candidates,
                ]
            )
        )
    elif emergency_active and emergency_posture_name == "seek_help":
        engagement_mode = "contain"
        outcome_goal = "bring_in_support"
        boundary_mode = "protective"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "seek_help",
                    "make_risk_visible",
                    "move_to_support",
                    "terminate_contact",
                    *next_action_candidates,
                ]
            )
        )
    elif emergency_active and emergency_posture_name == "emergency_protect":
        engagement_mode = "contain"
        outcome_goal = "protect_immediately"
        boundary_mode = "protective"
        next_action_candidates = list(
            dict.fromkeys(
                [
                    "emergency_protect",
                    "protect_others_if_present",
                    "reduce_exposure",
                    "terminate_contact",
                    *next_action_candidates,
                ]
            )
        )
    elif situation_risk_name in {"guarded_context", "unstable_contact", "acute_threat"}:
        next_action_candidates = list(dict.fromkeys(["assess_context_shift", *next_action_candidates]))

    return {
        "engagement_mode": engagement_mode,
        "outcome_goal": outcome_goal,
        "boundary_mode": boundary_mode,
        "next_action_candidates": next_action_candidates,
        "attention_target": attention_target,
        "memory_write_priority": memory_write_priority,
        "memory_write_class": memory_write_class,
        "memory_write_class_reason": memory_write_class_reason,
        "affordance_priority": affordance_priority,
        "do_not_cross": do_not_cross,
        "disclosure_depth": disclosure_depth,
        "workspace_mode": workspace_mode,
        "reportability_limit": reportability_limit,
        "reportability_gate_mode": reportability_gate_mode,
        "actionable_slice": actionable_slice,
        "other_person_state": other_person_state,
        "resonance_prioritize_actions": resonance_prioritize_actions,
        "resonance_avoid_actions": resonance_avoid_actions,
        "conversational_objects": conversational_objects,
        "object_operations": object_operations,
        "interaction_effects": interaction_effects,
        "interaction_judgement_view": interaction_judgement_view,
        "primary_operation_kind": primary_operation_kind,
        "ordered_operation_kinds": ordered_operation_kinds,
        "ordered_effect_kinds": ordered_effect_kinds,
        "question_budget": question_budget,
        "question_pressure": question_pressure,
        "defer_dominance": defer_dominance,
        "qualia_planner_view": qualia_planner_view,
        "terrain_readout": terrain_readout,
        "protection_mode": protection_mode,
        "protection_mode_name": protection_mode_name,
        "protection_mode_strength": protection_mode_strength,
        "body_recovery_guard": body_recovery_guard,
        "body_recovery_guard_state": body_recovery_guard_state,
        "body_recovery_guard_score": body_recovery_guard_score,
        "body_homeostasis_state": body_homeostasis_state,
        "body_homeostasis_name": body_homeostasis_name,
        "body_homeostasis_score": body_homeostasis_score,
        "homeostasis_budget_state": homeostasis_budget_state,
        "homeostasis_budget_name": homeostasis_budget_name,
        "homeostasis_budget_score": homeostasis_budget_score,
        "initiative_readiness": initiative_readiness,
        "initiative_readiness_state": initiative_readiness_state,
        "initiative_readiness_score": initiative_readiness_score,
        "commitment_state": commitment_state,
        "commitment_mode": commitment_mode,
        "commitment_target": commitment_target,
        "commitment_score": commitment_score,
        "learning_mode_state": learning_mode_state,
        "learning_mode_name": learning_mode_name,
        "learning_mode_probe_room": learning_mode_probe_room,
        "social_experiment_loop_state": social_experiment_loop_state,
        "social_experiment_name": social_experiment_name,
        "social_experiment_probe_intensity": social_experiment_probe_intensity,
        "live_engagement_state": live_engagement_state,
        "live_engagement_name": live_engagement_name,
        "live_engagement_score": live_engagement_score,
        "live_primary_move": live_primary_move,
        "situation_risk_state": situation_risk_state,
        "situation_risk_name": situation_risk_name,
        "situation_risk_immediacy": situation_risk_immediacy,
        "situation_risk_dialogue_room": situation_risk_dialogue_room,
        "situation_risk_relation_break": situation_risk_relation_break,
        "emergency_posture": emergency_posture,
        "emergency_posture_name": emergency_posture_name,
        "emergency_posture_score": emergency_posture_score,
        "emergency_dialogue_permission": emergency_dialogue_permission,
        "emergency_primary_action": emergency_primary_action,
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


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
