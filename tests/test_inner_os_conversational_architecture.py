from types import SimpleNamespace

from inner_os.conversational_objects import derive_conversational_objects
from inner_os.interaction_audit_bundle import build_interaction_audit_bundle
from inner_os.interaction_audit_comparison import compare_interaction_audit_bundles
from inner_os.interaction_audit_report import build_interaction_audit_report
from inner_os.interaction_condition_report import build_interaction_condition_report
from inner_os.interaction_effects import derive_interaction_effects
from inner_os.interaction_inspection_report import build_interaction_inspection_report
from inner_os.interaction_judgement_summary import derive_interaction_judgement_summary
from inner_os.interaction_judgement_view import derive_interaction_judgement_view
from inner_os.object_operations import derive_object_operations
from inner_os.policy_packet import derive_interaction_policy_packet
from inner_os.action_posture import derive_action_posture
from inner_os.actuation_plan import derive_actuation_plan
from inner_os.expression.content_policy import derive_content_sequence
from inner_os.expression.response_planner import _apply_interaction_policy_surface_bias


def test_conversational_architecture_derives_objects_operations_and_effects() -> None:
    conversational_objects = derive_conversational_objects(
        current_text="recently feeling worn out",
        current_focus="person:user",
        reportable_facts=["recent distress"],
        scene_state={
            "scene_family": "attuned_presence",
            "privacy_level": 0.76,
            "norm_pressure": 0.22,
            "safety_margin": 0.82,
            "environmental_load": 0.14,
        },
        relation_context={
            "relation_bias_strength": 0.74,
            "recent_strain": 0.62,
            "trust_memory": 0.68,
            "familiarity": 0.64,
            "attachment": 0.7,
            "partner_timing_hint": "delayed",
            "partner_stance_hint": "respectful",
            "partner_social_interpretation": "repair_window",
        },
        memory_context={
            "memory_anchor": "harbor slope",
            "relation_seed_summary": "shared harbor thread",
            "long_term_theme_summary": "quiet harbor routine",
            "conscious_residue_summary": "unfinished strain from the previous turn",
        },
        affect_blend_state={"dominant_mode": "care"},
        constraint_field={
            "reportability_limit": "withhold",
            "disclosure_limit": "minimal",
            "do_not_cross": ["force_reportability"],
        },
        conscious_workspace={
            "reportable_slice": ["recent distress"],
            "withheld_slice": ["cause detail"],
            "actionable_slice": ["rest tonight"],
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "medium",
            },
            "expected_effects": ["lower_pressure", "preserve_talk_choice"],
            "avoid_actions": ["probe_detail"],
        },
    )
    assert conversational_objects.primary_object_id
    assert conversational_objects.objects
    assert "recent distress" in conversational_objects.active_labels
    assert conversational_objects.deferred_object_ids
    assert conversational_objects.pressure_balance > 0.0
    assert any(item.object_kind == "shared_thread" for item in conversational_objects.objects)
    assert any(item.object_kind == "unfinished_part" for item in conversational_objects.objects)
    primary_object = next(
        item
        for item in conversational_objects.objects
        if item.object_id == conversational_objects.primary_object_id
    )
    assert 0.0 <= primary_object.touchability_score <= 1.0
    assert 0.0 <= primary_object.depth_room <= 1.0
    assert 0.0 <= primary_object.defer_pressure <= 1.0

    object_operations = derive_object_operations(
        conversational_objects=conversational_objects.to_dict(),
        scene_state={
            "scene_family": "guarded_boundary",
            "privacy_level": 0.2,
            "norm_pressure": 0.72,
            "safety_margin": 0.28,
        },
        relation_context={
            "relation_bias_strength": 0.74,
            "recent_strain": 0.62,
            "trust_memory": 0.68,
            "familiarity": 0.64,
            "attachment": 0.7,
            "partner_timing_hint": "delayed",
            "partner_stance_hint": "respectful",
        },
        memory_context={
            "memory_anchor": "harbor slope",
            "relation_seed_summary": "shared harbor thread",
            "long_term_theme_summary": "quiet harbor routine",
            "conscious_residue_summary": "unfinished strain from the previous turn",
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "medium",
            },
            "recommended_family_id": "wait",
        },
        constraint_field={"reportability_limit": "withhold"},
        conscious_workspace={"reportability_gate": {"gate_mode": "withhold"}},
        interaction_option_candidates=[{"family_id": "wait", "option_id": "wait:0"}],
    )
    assert object_operations.primary_operation_id
    assert object_operations.question_budget == 0
    assert object_operations.question_pressure > 0.0
    assert object_operations.defer_dominance > 0.0
    assert any(item.operation_kind == "hold_without_probe" for item in object_operations.operations)
    assert any(item.operation_kind == "defer_detail" for item in object_operations.operations)
    assert any(item.operation_kind == "anchor_shared_thread" for item in object_operations.operations)
    assert any(item.operation_kind == "protect_unfinished_part" for item in object_operations.operations)
    primary_operation = next(
        item
        for item in object_operations.operations
        if item.operation_id == object_operations.primary_operation_id
    )
    assert 0.0 <= primary_operation.operation_strength <= 1.0
    assert 0.0 <= primary_operation.burden_risk <= 1.0
    assert 0.0 <= primary_operation.connection_support <= 1.0

    interaction_effects = derive_interaction_effects(
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        resonance_evaluation={
            "estimated_other_person_state": {
                "acknowledgement_need_level": "high",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "medium",
            },
            "expected_effects": ["lower_pressure", "preserve_talk_choice"],
        },
        constraint_field={"boundary_pressure": 0.72},
    )
    assert interaction_effects.effects
    effect_kinds = {item.effect_kind for item in interaction_effects.effects}
    assert "feel_received" in effect_kinds
    assert "preserve_self_pacing" in effect_kinds
    assert "protect_boundary" in effect_kinds
    assert "preserve_continuity" in effect_kinds
    assert "avoid_forced_reopening" in effect_kinds
    assert all(0.0 <= item.intensity <= 1.0 for item in interaction_effects.effects)

    policy_packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=["danger"],
        reportable_facts=["recent distress"],
        relation_bias_strength=0.74,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="delayed",
        partner_stance_hint="respectful",
        partner_social_interpretation="repair_window",
        recent_strain=0.62,
        orchestration={
            "orchestration_mode": "repair",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.46,
            "coherence_score": 0.52,
            "human_presence_signal": 0.58,
            "distance_strategy": "holding_space",
            "repair_bias": True,
        },
        surface_profile={
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "careful_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=True,
            strained_pause=0.58,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.12,
        ),
        scene_state={
            "scene_family": "repair_window",
            "scene_tags": ["high_norm"],
        },
        interaction_option_candidates=[{"family_id": "repair", "option_id": "repair:0"}],
        affect_blend_state={"dominant_mode": "care", "conflict_level": 0.24, "residual_tension": 0.42},
        constraint_field={
            "body_cost": 0.32,
            "boundary_pressure": 0.72,
            "repair_pressure": 0.68,
            "shared_world_pressure": 0.22,
            "disclosure_limit": "minimal",
            "reportability_limit": "withhold",
            "do_not_cross": ["force_reportability"],
        },
        conscious_workspace={
            "workspace_mode": "guarded_foreground",
            "workspace_stability": 0.66,
            "reportable_slice": ["recent distress"],
            "withheld_slice": ["cause detail"],
            "actionable_slice": ["rest tonight"],
            "reportability_gate": {"gate_mode": "withhold"},
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "acknowledgement_need_level": "high",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "low",
            },
            "prioritize_actions": ["acknowledge_current_state"],
            "avoid_actions": ["press_for_detail"],
            "expected_effects": ["lower_pressure", "preserve_talk_choice"],
        },
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        interaction_effects=interaction_effects.to_dict(),
        interaction_judgement_view={},
        association_reweighting_focus="repeated_links",
        association_reweighting_reason="repeated_insight_trace",
        insight_terrain_shape_target="soft_relation",
    )
    assert policy_packet["dialogue_order"][0].startswith("open:")
    assert any(item.startswith("focus:") for item in policy_packet["dialogue_order"])
    assert policy_packet["primary_conversational_object_label"]
    assert any(item.startswith("operate:hold_without_probe") for item in policy_packet["dialogue_order"])
    assert any(item.startswith("effect:preserve_self_pacing") for item in policy_packet["dialogue_order"])
    assert any(item.startswith("defer:unfinished_part") for item in policy_packet["dialogue_order"])
    assert policy_packet["ordered_operation_kinds"]
    assert policy_packet["ordered_effect_kinds"]
    assert policy_packet["content_plan"]
    assert policy_packet["focus_now"] == policy_packet["primary_conversational_object_label"]
    assert policy_packet["response_action_now"]["primary_operation"]
    assert policy_packet["wanted_effect_on_other"]
    assert policy_packet["ordered_effects"]
    assert policy_packet["association_reweighting_focus"] == "repeated_links"
    assert policy_packet["association_reweighting_reason"] == "repeated_insight_trace"
    assert policy_packet["insight_terrain_shape_target"] == "soft_relation"
    assert policy_packet["overnight_bias_roles"]["association_reweighting_focus"] == "repeated_links"
    assert policy_packet["reaction_vs_overnight_bias"]["same_turn"]["memory_write_class"] == policy_packet["memory_write_class"]
    assert policy_packet["reaction_vs_overnight_bias"]["overnight"]["insight_terrain_shape_target"] == "soft_relation"
    assert policy_packet["protection_mode_decision"]["mode"] == ""
    assert policy_packet["memory_write_class_bias"]["same_turn_scores"]
    assert policy_packet["memory_write_class_bias"]["mode_prior"]
    assert policy_packet["memory_write_class_bias"]["combined_scores"]
    assert policy_packet["memory_write_class_bias"]["winner_margin"] >= 0.0
    assert policy_packet["memory_write_class_bias"]["dominant_inputs"]
    assert policy_packet["cultural_conversation_state"]["state"]
    assert policy_packet["agenda_window_state"]["state"] in {
        "now",
        "next_private_window",
        "next_same_group_window",
        "next_same_culture_window",
        "opportunistic_reentry",
        "long_hold",
    }
    assert policy_packet["agenda_window_state"]["deferral_budget"] >= 0.0
    assert "cultural_conversation_state" in policy_packet["reaction_vs_overnight_bias"]["same_turn"]

    interaction_judgement_view = derive_interaction_judgement_view(
        current_text="recently feeling worn out",
        reportable_facts=["recent distress"],
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        interaction_effects=interaction_effects.to_dict(),
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "acknowledgement_need_level": "high",
                "pressure_sensitivity_level": "high",
            }
        },
    )
    judgement_payload = interaction_judgement_view.to_dict()
    assert judgement_payload["observed_signals"]
    assert judgement_payload["inferred_signals"]
    assert judgement_payload["selected_object_labels"]
    assert judgement_payload["deferred_object_labels"]
    assert judgement_payload["active_operation_labels"]
    assert judgement_payload["intended_effect_labels"]
    assert judgement_payload["cues"]

    interaction_judgement_summary = derive_interaction_judgement_summary(
        interaction_judgement_view=interaction_judgement_view.to_dict(),
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        interaction_effects=interaction_effects.to_dict(),
    )
    summary_payload = interaction_judgement_summary.to_dict()
    assert summary_payload["observed_lines"]
    assert summary_payload["inferred_lines"]
    assert summary_payload["selected_object_lines"]
    assert summary_payload["deferred_object_lines"]
    assert summary_payload["operation_lines"]
    assert summary_payload["intended_effect_lines"]

    inspection_payload = build_interaction_inspection_report(
        {"current_case": interaction_judgement_summary.to_dict()}
    ).to_dict()
    assert inspection_payload["case_reports"]
    assert inspection_payload["report_lines"]
    assert any("current_case" in line for line in inspection_payload["report_lines"])

    condition_payload = build_interaction_condition_report(
        scene_state={
            "scene_family": "reverent_distance",
            "privacy_level": 0.2,
            "norm_pressure": 0.7,
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "acknowledgement_need_level": "high",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "low",
            }
        },
        relation_context={
            "recent_strain": 0.58,
            "relation_bias_strength": 0.62,
            "partner_timing_hint": "delayed",
            "partner_stance_hint": "respectful",
            "partner_social_interpretation": "repair_window",
        },
        memory_context={
            "relation_seed_summary": "gentle harbor companion thread",
            "long_term_theme_summary": "quiet harbor slope memory",
            "conscious_residue_summary": "previous unfinished strain",
            "memory_anchor": "harbor slope",
        },
    ).to_dict()
    assert condition_payload["scene_lines"]
    assert condition_payload["relation_lines"]
    assert condition_payload["memory_lines"]
    assert condition_payload["integration_lines"]
    assert condition_payload["report_lines"]

    audit_payload = build_interaction_audit_bundle(
        interaction_judgement_summary=summary_payload,
        interaction_condition_report=condition_payload,
        interaction_inspection_report=inspection_payload,
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        interaction_effects=interaction_effects.to_dict(),
        resonance_evaluation={
            "resonance_score": 0.72,
            "recommended_family_id": "wait",
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "acknowledgement_need_level": "high",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "low",
            },
        },
    ).to_dict()
    assert audit_payload["observed_lines"]
    assert audit_payload["scene_lines"]
    assert audit_payload["inspection_lines"]
    assert audit_payload["key_metrics"]["question_budget"] == 0
    assert audit_payload["report_lines"]

    open_audit = dict(audit_payload)
    guarded_audit = build_interaction_audit_bundle(
        interaction_judgement_summary={
            **summary_payload,
            "inferred_lines": [
                "system infers that the other person has less room to explain details right now."
            ],
            "operation_lines": [
                "system now keeps the unfinished part closed and reduces pressure."
            ],
        },
        interaction_condition_report={
            **condition_payload,
            "scene_lines": [
                "system sees a more guarded scene with less privacy and stronger norm pressure."
            ],
            "integration_lines": [
                "system therefore keeps the exchange narrower and more protective."
            ],
        },
        interaction_inspection_report=inspection_payload,
        conversational_objects=conversational_objects.to_dict(),
        object_operations={
            **object_operations.to_dict(),
            "question_budget": 0,
            "question_pressure": 0.91,
            "defer_dominance": 0.88,
        },
        interaction_effects=interaction_effects.to_dict(),
        resonance_evaluation={
            "resonance_score": 0.51,
            "recommended_family_id": "wait",
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "acknowledgement_need_level": "high",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "low",
            },
        },
    ).to_dict()
    comparison_payload = compare_interaction_audit_bundles(
        {
            "open_case": open_audit,
            "guarded_case": guarded_audit,
        }
    ).to_dict()
    assert comparison_payload["cases"]
    assert "inferred_lines" in comparison_payload["changed_sections"]
    assert "operation_lines" in comparison_payload["changed_sections"]
    assert comparison_payload["metric_differences"]
    assert any(
        item["metric_name"] == "question_pressure"
        for item in comparison_payload["metric_differences"]
    )
    assert comparison_payload["report_lines"]

    audit_report_payload = build_interaction_audit_report(
        {
            "open_case": open_audit,
            "guarded_case": guarded_audit,
        }
    ).to_dict()
    assert "inferred_lines" in audit_report_payload["changed_sections"]
    assert any(
        item["metric_name"] == "question_pressure"
        for item in audit_report_payload["metric_differences"]
    )
    assert any("question_pressure" in line for line in audit_report_payload["report_lines"])

    single_case_audit_report = build_interaction_audit_report(
        {"current_case": open_audit}
    ).to_dict()
    assert single_case_audit_report["report_lines"]


def test_same_input_changes_objects_and_operations_by_relation_and_memory() -> None:
    base_kwargs = {
        "current_text": "I want to stay with what feels difficult here.",
        "current_focus": "person:user",
        "reportable_facts": ["what feels difficult here"],
        "affect_blend_state": {"dominant_mode": "care"},
        "constraint_field": {
            "reportability_limit": "",
            "disclosure_limit": "light",
            "do_not_cross": [],
        },
        "conscious_workspace": {
            "reportable_slice": ["what feels difficult here"],
            "withheld_slice": [],
            "actionable_slice": ["one small next step"],
        },
    }

    open_objects = derive_conversational_objects(
        scene_state={
            "scene_family": "attuned_presence",
            "privacy_level": 0.82,
            "norm_pressure": 0.18,
            "safety_margin": 0.84,
            "environmental_load": 0.12,
        },
        relation_context={
            "relation_bias_strength": 0.76,
            "recent_strain": 0.12,
            "trust_memory": 0.74,
            "familiarity": 0.72,
            "attachment": 0.78,
            "partner_timing_hint": "open",
            "partner_stance_hint": "familiar",
            "partner_social_interpretation": "future_open",
        },
        memory_context={
            "memory_anchor": "harbor slope",
            "relation_seed_summary": "shared harbor thread",
            "long_term_theme_summary": "quiet harbor routine",
            "conscious_residue_summary": "",
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "high",
                "pressure_sensitivity_level": "low",
                "next_step_room_level": "high",
            }
        },
        **base_kwargs,
    )
    guarded_objects = derive_conversational_objects(
        scene_state={
            "scene_family": "guarded_boundary",
            "privacy_level": 0.22,
            "norm_pressure": 0.74,
            "safety_margin": 0.24,
            "environmental_load": 0.62,
        },
        relation_context={
            "relation_bias_strength": 0.42,
            "recent_strain": 0.68,
            "trust_memory": 0.44,
            "familiarity": 0.4,
            "attachment": 0.46,
            "partner_timing_hint": "delayed",
            "partner_stance_hint": "respectful",
            "partner_social_interpretation": "repair_window",
        },
        memory_context={
            "memory_anchor": "harbor slope",
            "relation_seed_summary": "shared harbor thread",
            "long_term_theme_summary": "quiet harbor routine",
            "conscious_residue_summary": "unfinished strain from the previous turn",
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "low",
            }
        },
        **base_kwargs,
    )

    open_primary = next(item for item in open_objects.objects if item.object_id == open_objects.primary_object_id)
    guarded_primary = next(item for item in guarded_objects.objects if item.object_id == guarded_objects.primary_object_id)
    assert open_primary.object_kind in {"topic", "shared_thread", "theme_anchor", "next_step"}
    assert guarded_primary.object_kind in {"unfinished_part", "topic", "shared_thread"}
    assert open_primary.object_kind != guarded_primary.object_kind or open_primary.label != guarded_primary.label

    open_operations = derive_object_operations(
        conversational_objects=open_objects.to_dict(),
        scene_state={
            "scene_family": "attuned_presence",
            "privacy_level": 0.82,
            "norm_pressure": 0.18,
            "safety_margin": 0.84,
        },
        relation_context={
            "relation_bias_strength": 0.76,
            "recent_strain": 0.12,
            "trust_memory": 0.74,
            "familiarity": 0.72,
            "attachment": 0.78,
            "partner_timing_hint": "open",
            "partner_stance_hint": "familiar",
        },
        memory_context={
            "memory_anchor": "harbor slope",
            "relation_seed_summary": "shared harbor thread",
            "long_term_theme_summary": "quiet harbor routine",
            "conscious_residue_summary": "",
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "high",
                "pressure_sensitivity_level": "low",
                "next_step_room_level": "high",
            },
            "recommended_family_id": "co_move",
        },
        interaction_option_candidates=[{"family_id": "co_move", "option_id": "co_move:0"}],
    )
    guarded_operations = derive_object_operations(
        conversational_objects=guarded_objects.to_dict(),
        scene_state={
            "scene_family": "guarded_boundary",
            "privacy_level": 0.22,
            "norm_pressure": 0.74,
            "safety_margin": 0.24,
        },
        relation_context={
            "relation_bias_strength": 0.42,
            "recent_strain": 0.68,
            "trust_memory": 0.44,
            "familiarity": 0.4,
            "attachment": 0.46,
            "partner_timing_hint": "delayed",
            "partner_stance_hint": "respectful",
        },
        memory_context={
            "memory_anchor": "harbor slope",
            "relation_seed_summary": "shared harbor thread",
            "long_term_theme_summary": "quiet harbor routine",
            "conscious_residue_summary": "unfinished strain from the previous turn",
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "low",
            },
            "recommended_family_id": "wait",
        },
        interaction_option_candidates=[{"family_id": "wait", "option_id": "wait:0"}],
    )

    open_operation_kinds = {item.operation_kind for item in open_operations.operations}
    guarded_operation_kinds = {item.operation_kind for item in guarded_operations.operations}
    assert open_operations.question_budget > guarded_operations.question_budget
    assert open_operations.question_pressure < guarded_operations.question_pressure
    assert "offer_small_next_step" in open_operation_kinds
    assert "protect_unfinished_part" in guarded_operation_kinds
    assert "anchor_shared_thread" in open_operation_kinds


def test_action_posture_and_actuation_can_follow_operation_and_effects_before_strategy() -> None:
    packet = {
        "response_strategy": "attune_then_extend",
        "opening_move": "anchor_visible_part",
        "disclosure_depth": "light",
        "attention_target": "what is visible first",
        "memory_write_priority": "foreground_trace",
        "affordance_priority": ["co_move", "pace_match"],
        "do_not_cross": [],
        "constraint_field": {"reportability_limit": "", "boundary_pressure": 0.18},
        "conscious_workspace": {"workspace_mode": "foreground"},
        "reportability_gate_mode": "",
        "actionable_slice": ["one next step"],
        "other_person_state": {},
        "resonance_prioritize_actions": [],
        "resonance_avoid_actions": [],
        "conversational_objects": {},
        "object_operations": {},
        "interaction_effects": {},
        "interaction_judgement_view": {},
        "primary_object_operation": {"operation_kind": "offer_small_next_step"},
        "ordered_operation_kinds": ["offer_small_next_step", "anchor_next_step_in_theme"],
        "ordered_effect_kinds": ["enable_small_next_step", "keep_next_step_connected"],
        "question_budget": 0,
        "question_pressure": 0.24,
        "defer_dominance": 0.12,
        "contact_readiness": 0.72,
        "repair_bias": False,
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert posture["engagement_mode"] == "co_move"
    assert posture["outcome_goal"] == "shared_progress"
    assert "map_next_step" in posture["next_action_candidates"]
    assert actuation["execution_mode"] == "shared_progression"
    assert actuation["primary_action"] == "co_move"
    assert "keep_step_connected" in actuation["action_queue"]


def test_action_posture_and_actuation_can_follow_conversation_contract_aliases_only() -> None:
    packet = {
        "conversation_contract": {
            "focus_now": "what is visible first",
            "response_action_now": {
                "primary_operation": "hold_without_probe",
                "operation_target": "what is visible first",
                "ordered_operations": ["hold_without_probe", "protect_unfinished_part", "keep_return_point"],
                "question_budget": 0,
                "question_pressure": 0.74,
                "defer_dominance": 0.82,
            },
            "ordered_effects": ["preserve_self_pacing", "keep_connection_open"],
            "wanted_effect_on_other": [
                {"effect": "preserve_self_pacing", "target": "what is visible first", "intensity": 0.73},
                {"effect": "keep_connection_open", "target": "return point", "intensity": 0.62},
            ],
            "condition_summary": [
                "System sees a relation where pace should stay gentle.",
                "System is keeping the shared thread visible.",
            ],
        },
        "constraint_field": {"boundary_pressure": 0.24},
        "conscious_workspace": {"workspace_mode": "foreground"},
        "contact_readiness": 0.46,
        "repair_bias": False,
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert posture["engagement_mode"] == "wait"
    assert posture["outcome_goal"] == "preserve_distance_with_connection"
    assert posture["attention_target"] == "what is visible first"
    assert posture["memory_write_priority"] == "relation_episode"
    assert actuation["execution_mode"] == "defer_with_presence"
    assert actuation["primary_action"] == "hold_presence"
    assert "leave_return_point" in actuation["action_queue"]


def test_action_posture_and_actuation_can_follow_same_turn_qualia_protection_bias() -> None:
    packet = {
        "conversation_contract": {
            "focus_now": "what is visible first",
            "response_action_now": {
                "primary_operation": "acknowledge",
                "operation_target": "what is visible first",
                "ordered_operations": ["acknowledge", "anchor_shared_thread"],
                "question_budget": 1,
                "question_pressure": 0.18,
                "defer_dominance": 0.16,
            },
            "ordered_effects": ["keep_connection_open"],
            "wanted_effect_on_other": [
                {"effect": "keep_connection_open", "target": "what is visible first", "intensity": 0.62},
            ],
        },
        "constraint_field": {"boundary_pressure": 0.18},
        "conscious_workspace": {"workspace_mode": "foreground"},
        "contact_readiness": 0.62,
        "repair_bias": False,
        "qualia_planner_view": {
            "trust": 0.42,
            "degraded": False,
            "dominant_axis": "stress_level",
            "dominant_value": 0.46,
            "body_load": 0.16,
            "protection_bias": 0.24,
            "felt_energy": 0.22,
        },
    }

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert posture["engagement_mode"] == "contain"
    assert posture["boundary_mode"] == "protective"
    assert "stabilize_before_extend" in posture["next_action_candidates"]
    assert actuation["execution_mode"] == "stabilize_boundary"
    assert "stabilize_before_extend" in actuation["action_queue"]
    assert actuation["repair_window_commitment"] == "protective"


def test_content_sequence_can_follow_moves_and_contract_without_strategy() -> None:
    sequence = derive_content_sequence(
        current_text="I want to stay with what feels difficult here.",
        interaction_policy={
            "dialogue_act": "check_in",
            "opening_move": "acknowledge_without_probe",
            "followup_move": "protect_talking_room",
            "closing_move": "leave_return_point",
            "primary_conversational_object_label": "what feels difficult here",
            "primary_object_operation": {
                "operation_kind": "hold_without_probe",
                "target_label": "what feels difficult here",
            },
            "ordered_operation_kinds": [
                "hold_without_probe",
                "protect_unfinished_part",
                "keep_return_point",
            ],
            "ordered_effect_kinds": [
                "keep_connection_open",
                "preserve_self_pacing",
            ],
            "deferred_object_labels": ["unfinished part"],
            "dialogue_order": [
                "open:acknowledge_without_probe",
                "focus:what feels difficult here",
                "operate:hold_without_probe",
                "operate:protect_unfinished_part",
                "follow:protect_talking_room",
                "effect:keep_connection_open",
                "effect:preserve_self_pacing",
                "defer:unfinished_part",
                "close:leave_return_point",
            ],
            "question_budget": 0,
        },
        conscious_access={"intent": "check_in"},
    )

    assert sequence
    assert sequence[0]["act"] == "respect_boundary"
    assert any("press this right now" in item["text"] for item in sequence)
    assert any("unfinished part" in item["text"] or "come back" in item["text"] for item in sequence)


def test_content_sequence_can_follow_conversation_contract_aliases_only() -> None:
    sequence = derive_content_sequence(
        current_text="I want to stay with what feels difficult here.",
        interaction_policy={
            "dialogue_act": "check_in",
            "conversation_contract": {
                "focus_now": "what feels difficult here",
                "leave_closed_for_now": ["unfinished part"],
                "response_action_now": {
                    "primary_operation": "hold_without_probe",
                    "operation_target": "what feels difficult here",
                    "ordered_operations": [
                        "hold_without_probe",
                        "protect_unfinished_part",
                        "keep_return_point",
                    ],
                    "question_budget": 0,
                },
                "ordered_effects": [
                    "keep_connection_open",
                    "preserve_self_pacing",
                ],
                "wanted_effect_on_other": [
                    {"effect": "keep_connection_open", "target": "what feels difficult here", "intensity": 0.62},
                    {"effect": "preserve_self_pacing", "target": "what feels difficult here", "intensity": 0.74},
                ],
            },
        },
        conscious_access={"intent": "check_in"},
    )

    assert sequence
    assert sequence[0]["act"] == "acknowledge_without_probe"
    assert any("what feels difficult here" in item["text"] for item in sequence)
    assert any("unfinished part" in item["text"] for item in sequence)
    assert any("unfinished part" in item["text"] or "come back" in item["text"] for item in sequence)


def test_policy_packet_memory_priority_can_follow_same_turn_qualia_bias() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="report",
        current_focus="ambient",
        current_risks=[],
        reportable_facts=[],
        relation_bias_strength=0.0,
        related_person_ids=[],
        partner_address_hint="",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.42,
            "coherence_score": 0.44,
            "human_presence_signal": 0.4,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.12,
            future_loop_pull=0.08,
            fantasy_loop_pull=0.0,
        ),
        scene_state={"scene_family": "attuned_presence", "scene_tags": []},
        interaction_option_candidates=[{"family_id": "attune", "option_id": "attune:0"}],
        affect_blend_state={"dominant_mode": "care"},
        constraint_field={
            "body_cost": 0.12,
            "boundary_pressure": 0.14,
            "repair_pressure": 0.08,
            "shared_world_pressure": 0.16,
            "disclosure_limit": "light",
            "reportability_limit": "",
            "do_not_cross": [],
        },
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.34,
            "reportable_slice": [],
            "withheld_slice": [],
            "actionable_slice": [],
            "reportability_gate": {"gate_mode": ""},
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "medium",
                "acknowledgement_need_level": "medium",
                "pressure_sensitivity_level": "medium",
                "next_step_room_level": "medium",
            }
        },
        conversational_objects={"objects": [], "deferred_object_ids": []},
        object_operations={"operations": [], "question_budget": 0, "question_pressure": 0.0, "defer_dominance": 0.0},
        interaction_effects={"effects": [], "primary_effect_ids": []},
        interaction_judgement_view={},
        qualia_planner_view={
            "trust": 0.82,
            "degraded": False,
            "dominant_axis": "stress_level",
            "dominant_value": 0.44,
            "body_load": 0.18,
            "protection_bias": 0.24,
            "felt_energy": 0.22,
        },
    )

    assert packet["memory_write_priority"] == "stability_trace"
    assert packet["memory_write_class"] == "body_risk"
    assert packet["memory_write_class_reason"] == "protective_body_pressure"
    assert packet["qualia_memory_bias"]["priority"] == "stability_trace"
    assert packet["qualia_memory_bias"]["reason"] == "protective_felt_bias"
    assert packet["qualia_memory_bias"]["applied"] is True


def test_policy_packet_derives_safe_repeat_memory_write_class_when_contact_is_open() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["quiet shared routine"],
        relation_bias_strength=0.52,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "advance",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.72,
            "coherence_score": 0.7,
            "human_presence_signal": 0.74,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.08,
            future_loop_pull=0.54,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.78,
            "reportable_slice": ["quiet shared routine"],
            "withheld_slice": [],
            "actionable_slice": ["repeat tomorrow"],
            "reportability_gate": {"gate_mode": ""},
        },
        qualia_planner_view={
            "trust": 0.82,
            "degraded": False,
            "dominant_axis": "continuity",
            "dominant_value": 0.34,
            "body_load": 0.04,
            "protection_bias": 0.06,
            "felt_energy": 0.18,
        },
        affective_position={
            "z_aff": [0.1, 0.0, 0.0, 0.0],
            "cov": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "confidence": 0.66,
            "source_weights": {"state": 0.48, "qualia": 0.28, "memory": 0.14, "carryover": 0.1},
        },
        terrain_readout={
            "value": 0.22,
            "grad": [0.04, 0.0, 0.0, 0.0],
            "curvature": [0.01, 0.01, 0.0, 0.0],
            "approach_bias": 0.62,
            "avoid_bias": 0.14,
            "protect_bias": 0.2,
            "active_patch_index": 0,
            "active_patch_label": "safe_repeat_hollow",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.22,
            "reasons": ["neutral_monitoring"],
        },
    )

    assert packet["memory_write_class"] == "safe_repeat"
    assert packet["memory_write_class_reason"] == "safe_repeatable_contact"


def test_policy_packet_uses_reframed_relation_as_weak_memory_class_prior() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["shared thread changed shape"],
        relation_bias_strength=0.56,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.62,
            "coherence_score": 0.64,
            "human_presence_signal": 0.68,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.08,
            future_loop_pull=0.24,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.74,
            "reportable_slice": ["shared thread changed shape"],
            "withheld_slice": [],
            "actionable_slice": [],
            "reportability_gate": {"gate_mode": ""},
        },
        qualia_planner_view={
            "trust": 0.8,
            "degraded": False,
            "dominant_axis": "continuity",
            "dominant_value": 0.28,
            "body_load": 0.04,
            "protection_bias": 0.08,
            "felt_energy": 0.18,
        },
        affective_position={
            "z_aff": [0.06, 0.0, 0.0, 0.0],
            "cov": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "confidence": 0.64,
            "source_weights": {"state": 0.48, "qualia": 0.26, "memory": 0.16, "carryover": 0.1},
        },
        terrain_readout={
            "value": 0.16,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.01, 0.01, 0.0, 0.0],
            "approach_bias": 0.46,
            "avoid_bias": 0.18,
            "protect_bias": 0.22,
            "active_patch_index": 0,
            "active_patch_label": "reframe_opening",
        },
        protection_mode={
            "mode": "repair",
            "strength": 0.34,
            "reasons": ["approach_opening"],
        },
        insight_reframing_bias=0.72,
        insight_class_focus="reframed_relation",
    )

    assert packet["memory_write_class"] in {"repair_trace", "bond_protection"}
    assert packet["memory_write_class_reason"] == "insight_reframed_relation_bias"
    assert packet["memory_write_class_bias"]["insight_prior"]["repair_trace"] > 0.0


def test_policy_packet_uses_new_link_hypothesis_as_weak_insight_trace_prior() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="clarify",
        current_focus="ambient",
        current_risks=[],
        reportable_facts=["new connection might be here"],
        relation_bias_strength=0.12,
        related_person_ids=[],
        partner_address_hint="",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "reflect",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.54,
            "coherence_score": 0.58,
            "human_presence_signal": 0.6,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.06,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.72,
            "reportable_slice": ["new connection might be here"],
            "withheld_slice": [],
            "actionable_slice": [],
            "reportability_gate": {"gate_mode": ""},
        },
        qualia_planner_view={
            "trust": 0.82,
            "degraded": False,
            "dominant_axis": "meaning_shift",
            "dominant_value": 0.24,
            "body_load": 0.02,
            "protection_bias": 0.04,
            "felt_energy": 0.22,
        },
        affective_position={
            "z_aff": [0.02, 0.0, 0.0, 0.0],
            "cov": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "confidence": 0.68,
            "source_weights": {"state": 0.44, "qualia": 0.3, "memory": 0.16, "carryover": 0.1},
        },
        terrain_readout={
            "value": 0.12,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.0, 0.0, 0.0, 0.0],
            "approach_bias": 0.34,
            "avoid_bias": 0.18,
            "protect_bias": 0.12,
            "active_patch_index": 0,
            "active_patch_label": "soft_opening",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.22,
            "reasons": ["neutral_monitoring"],
        },
        insight_reframing_bias=0.76,
        insight_class_focus="new_link_hypothesis",
    )

    assert packet["memory_write_class"] == "insight_trace"
    assert packet["memory_write_class_reason"] == "insight_new_link_bias"
    assert packet["memory_write_class_bias"]["insight_prior"]["insight_trace"] > 0.0


def test_policy_packet_keeps_body_risk_above_insight_memory_prior() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="clarify",
        current_focus="ambient",
        current_risks=[],
        reportable_facts=["new connection might be here"],
        relation_bias_strength=0.18,
        related_person_ids=[],
        partner_address_hint="",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "reflect",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.42,
            "coherence_score": 0.48,
            "human_presence_signal": 0.52,
        },
        surface_profile={
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "careful_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.18,
            future_loop_pull=0.14,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "guarded_foreground",
            "workspace_stability": 0.34,
            "reportable_slice": [],
            "withheld_slice": ["new connection might be here"],
            "actionable_slice": [],
            "reportability_gate": {"gate_mode": "withhold"},
        },
        qualia_planner_view={
            "trust": 0.34,
            "degraded": True,
            "dominant_axis": "body_load",
            "dominant_value": 0.28,
            "body_load": 0.22,
            "protection_bias": 0.18,
            "felt_energy": 0.18,
        },
        affective_position={
            "z_aff": [0.04, 0.0, 0.0, 0.0],
            "cov": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "confidence": 0.42,
            "source_weights": {"state": 0.42, "qualia": 0.24, "memory": 0.2, "carryover": 0.14},
        },
        terrain_readout={
            "value": -0.12,
            "grad": [0.04, 0.0, 0.0, 0.0],
            "curvature": [0.02, 0.02, 0.0, 0.0],
            "approach_bias": 0.32,
            "avoid_bias": 0.24,
            "protect_bias": 0.28,
            "active_patch_index": 0,
            "active_patch_label": "guarded_opening",
        },
        protection_mode={
            "mode": "contain",
            "strength": 0.62,
            "reasons": ["body_load", "degraded_estimate"],
        },
        insight_reframing_bias=0.9,
        insight_class_focus="new_link_hypothesis",
    )

    assert packet["memory_write_class"] == "body_risk"
    assert packet["memory_write_class_reason"] == "protective_body_pressure"


def test_protection_mode_biases_posture_and_plan_from_shared_inputs() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["recent distress"],
        relation_bias_strength=0.4,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="delayed",
        partner_stance_hint="respectful",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.44,
            "coherence_score": 0.56,
            "human_presence_signal": 0.6,
        },
        surface_profile={
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "careful_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.18,
            future_loop_pull=0.12,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "guarded_foreground",
            "workspace_stability": 0.32,
            "reportability_gate": {"gate_mode": "withhold"},
        },
        qualia_planner_view={
            "trust": 0.36,
            "degraded": True,
            "dominant_axis": "body_load",
            "dominant_value": 0.24,
            "body_load": 0.22,
            "protection_bias": 0.28,
            "felt_energy": 0.16,
        },
        affective_position={
            "z_aff": [0.2, 0.1, -0.1, 0.05],
            "cov": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "confidence": 0.42,
            "source_weights": {"state": 0.4, "qualia": 0.4, "memory": 0.2},
        },
        terrain_readout={
            "value": 0.18,
            "grad": [0.1, -0.04, 0.02, 0.0],
            "curvature": [0.02, 0.01, -0.01, 0.0],
            "approach_bias": 0.12,
            "avoid_bias": 0.26,
            "protect_bias": 0.64,
            "active_patch_index": 1,
            "active_patch_label": "repair_slope",
        },
        protection_mode={
            "mode": "stabilize",
            "strength": 0.74,
            "reasons": ["terrain_protect_bias", "degraded_estimate"],
        },
    )

    action_posture = derive_action_posture(packet)
    actuation_plan = derive_actuation_plan(packet, action_posture)

    assert packet["terrain_readout"]["protect_bias"] == 0.64
    assert packet["protection_mode"]["mode"] == "stabilize"
    assert action_posture["protection_mode_name"] == "stabilize"
    assert action_posture["boundary_mode"] == "protective"
    assert "follow_protection_mode" in action_posture["next_action_candidates"]
    assert actuation_plan["execution_mode"] == "stabilize_boundary"
    assert actuation_plan["primary_action"] == "secure_boundary"
    assert actuation_plan["protection_mode"]["mode"] == "stabilize"


def test_body_recovery_guard_slows_forward_motion() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["recent strain"],
        relation_bias_strength=0.38,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.62,
            "coherence_score": 0.58,
            "human_presence_signal": 0.66,
        },
        surface_profile={
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "soft_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.24,
            future_loop_pull=0.08,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.48,
            "reportability_gate": {"gate_mode": "soft_open"},
        },
        qualia_planner_view={
            "trust": 0.58,
            "degraded": False,
            "dominant_axis": "body_load",
            "dominant_value": 0.22,
            "body_load": 0.18,
            "protection_bias": 0.16,
            "felt_energy": 0.12,
        },
        terrain_readout={
            "value": 0.04,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.02, 0.01, 0.0, 0.0],
            "approach_bias": 0.28,
            "avoid_bias": 0.18,
            "protect_bias": 0.26,
            "active_patch_index": 1,
            "active_patch_label": "soft_guard",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.42,
            "reasons": ["check_load"],
        },
        self_state={
            "stress": 0.48,
            "recovery_need": 0.62,
            "safety_bias": 0.18,
            "stabilization_drive": 0.54,
        },
    )

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert packet["body_recovery_guard"]["state"] == "recovery_first"
    assert "recovery_need" in packet["body_recovery_guard"]["dominant_inputs"]
    assert packet["body_homeostasis_state"]["state"] in {"recovering", "depleted"}
    assert packet["commitment_state"]["target"] in {"hold", "stabilize"}
    assert packet["commitment_state"]["target"] != "step_forward"
    assert posture["engagement_mode"] == "contain"
    assert "restore_body_margin" in posture["next_action_candidates"]
    assert actuation["execution_mode"] == "stabilize_before_contact"
    assert actuation["primary_action"] == "restore_body_margin"
    assert actuation["wait_before_action"] == "extended"


def test_initiative_readiness_opens_small_next_step_when_guard_is_low() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["shared opening"],
        relation_bias_strength=0.44,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.76,
            "coherence_score": 0.72,
            "human_presence_signal": 0.78,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.08,
            future_loop_pull=0.26,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.66,
            "reportability_gate": {"gate_mode": "open"},
        },
        qualia_planner_view={
            "trust": 0.84,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.18,
            "body_load": 0.02,
            "protection_bias": 0.04,
            "felt_energy": 0.2,
        },
        terrain_readout={
            "value": 0.22,
            "grad": [0.04, 0.02, 0.0, 0.0],
            "curvature": [0.01, 0.0, 0.0, 0.0],
            "approach_bias": 0.62,
            "avoid_bias": 0.1,
            "protect_bias": 0.12,
            "active_patch_index": 2,
            "active_patch_label": "open_step",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.28,
            "reasons": ["stable_contact"],
        },
        self_state={
            "stress": 0.12,
            "recovery_need": 0.1,
            "recent_strain": 0.08,
            "safety_bias": 0.06,
            "stabilization_drive": 0.12,
        },
    )

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert packet["body_recovery_guard"]["state"] == "open"
    assert packet["body_homeostasis_state"]["state"] == "steady"
    assert packet["initiative_readiness"]["state"] == "ready"
    assert "contact_readiness" in packet["initiative_readiness"]["dominant_inputs"]
    assert packet["commitment_state"]["state"] == "commit"
    assert packet["commitment_state"]["target"] == "step_forward"
    assert packet["relational_continuity_state"]["state"] in {"reopening", "co_regulating"}
    assert posture["engagement_mode"] == "co_move"
    assert "offer_next_step_if_welcomed" in posture["next_action_candidates"]
    assert actuation["execution_mode"] == "shared_progression"
    assert actuation["primary_action"] == "co_move"


def test_relational_continuity_state_can_reach_co_regulating_without_external_profile() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["shared opening"],
        relation_bias_strength=0.68,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        partner_social_interpretation="future_open",
        orchestration={
            "orchestration_mode": "advance",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.74,
            "coherence_score": 0.72,
            "human_presence_signal": 0.8,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.08,
            future_loop_pull=0.34,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.72,
            "reportable_slice": ["shared opening"],
            "actionable_slice": ["next small step"],
            "reportability_gate": {"gate_mode": "open"},
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "high",
                "acknowledgement_need_level": "medium",
                "pressure_sensitivity_level": "low",
                "next_step_room_level": "high",
            },
            "prioritize_actions": ["acknowledge_current_state"],
            "expected_effects": ["keep_next_step_connected"],
        },
        conversational_objects={
            "objects": [
                {"object_id": "shared-thread", "object_kind": "shared_thread", "label": "shared opening"},
            ],
        },
        object_operations={
            "operations": [
                {"operation_id": "offer", "operation_kind": "offer_small_next_step", "target_label": "shared opening"},
                {"operation_id": "anchor", "operation_kind": "anchor_shared_thread", "target_label": "shared opening"},
            ],
        },
        interaction_effects={
            "effects": [
                {"effect_id": "e1", "effect_kind": "preserve_continuity", "target_label": "shared opening"},
                {"effect_id": "e2", "effect_kind": "keep_connection_open", "target_label": "shared opening"},
                {"effect_id": "e3", "effect_kind": "keep_next_step_connected", "target_label": "shared opening"},
            ],
        },
        qualia_planner_view={
            "trust": 0.84,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.18,
            "body_load": 0.02,
            "protection_bias": 0.04,
            "felt_energy": 0.22,
        },
        terrain_readout={
            "value": 0.22,
            "grad": [0.04, 0.02, 0.0, 0.0],
            "curvature": [0.01, 0.0, 0.0, 0.0],
            "approach_bias": 0.62,
            "avoid_bias": 0.08,
            "protect_bias": 0.1,
            "active_patch_index": 2,
            "active_patch_label": "open_step",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.22,
            "reasons": ["stable_contact"],
        },
        self_state={
            "stress": 0.12,
            "recovery_need": 0.1,
            "recent_strain": 0.08,
            "safety_bias": 0.06,
            "continuity_score": 0.74,
            "social_grounding": 0.72,
            "trust_memory": 0.76,
            "familiarity": 0.68,
            "attachment": 0.7,
        },
    )

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert packet["body_homeostasis_state"]["state"] == "steady"
    assert packet["relational_continuity_state"]["state"] == "co_regulating"
    assert "social_grounding" in packet["relational_continuity_state"]["dominant_inputs"]
    assert "co_regulate_with_partner" in posture["next_action_candidates"]
    assert "co_regulate_with_partner" in actuation["action_queue"]


def test_relation_competition_state_holds_multiple_threads_without_collapsing_to_one() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:harbor",
        current_risks=[],
        reportable_facts=["shared opening"],
        relation_bias_strength=0.66,
        related_person_ids=["person:harbor", "person:friend"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        partner_social_interpretation="future_open",
        orchestration={
            "orchestration_mode": "advance",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.72,
            "coherence_score": 0.7,
            "human_presence_signal": 0.82,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.08,
            future_loop_pull=0.3,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.74,
            "reportable_slice": ["shared opening"],
            "actionable_slice": ["next small step"],
            "reportability_gate": {"gate_mode": "open"},
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "high",
                "acknowledgement_need_level": "medium",
                "pressure_sensitivity_level": "medium",
                "next_step_room_level": "high",
            },
            "prioritize_actions": ["acknowledge_current_state"],
            "expected_effects": ["keep_next_step_connected"],
        },
        conversational_objects={
            "objects": [
                {"object_id": "shared-thread", "object_kind": "shared_thread", "label": "shared opening"},
            ],
        },
        object_operations={
            "operations": [
                {"operation_id": "offer", "operation_kind": "offer_small_next_step", "target_label": "shared opening"},
                {"operation_id": "anchor", "operation_kind": "anchor_shared_thread", "target_label": "shared opening"},
            ],
        },
        interaction_effects={
            "effects": [
                {"effect_id": "e1", "effect_kind": "preserve_continuity", "target_label": "shared opening"},
                {"effect_id": "e2", "effect_kind": "keep_connection_open", "target_label": "shared opening"},
            ],
        },
        qualia_planner_view={
            "trust": 0.84,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.16,
            "body_load": 0.02,
            "protection_bias": 0.05,
            "felt_energy": 0.22,
        },
        terrain_readout={
            "value": 0.2,
            "grad": [0.04, 0.02, 0.0, 0.0],
            "curvature": [0.01, 0.0, 0.0, 0.0],
            "approach_bias": 0.6,
            "avoid_bias": 0.08,
            "protect_bias": 0.12,
            "active_patch_index": 2,
            "active_patch_label": "open_step",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.24,
            "reasons": ["stable_contact"],
        },
        self_state={
            "related_person_id": "person:harbor",
            "person_registry_snapshot": {
                "persons": {
                    "person:harbor": {
                        "person_id": "person:harbor",
                        "adaptive_traits": {
                            "attachment": 0.72,
                            "familiarity": 0.68,
                            "trust_memory": 0.74,
                            "continuity_score": 0.7,
                            "social_grounding": 0.68,
                        },
                        "confidence": 0.8,
                    },
                    "person:friend": {
                        "person_id": "person:friend",
                        "adaptive_traits": {
                            "attachment": 0.68,
                            "familiarity": 0.66,
                            "trust_memory": 0.71,
                            "continuity_score": 0.67,
                            "social_grounding": 0.64,
                        },
                        "confidence": 0.76,
                    },
                },
                "top_person_ids": ["person:harbor", "person:friend"],
                "dominant_person_id": "person:harbor",
                "total_people": 2,
                "uncertainty": 0.18,
            },
            "stress": 0.12,
            "recovery_need": 0.1,
            "recent_strain": 0.1,
            "safety_bias": 0.08,
            "continuity_score": 0.72,
            "social_grounding": 0.69,
            "trust_memory": 0.75,
            "familiarity": 0.69,
            "attachment": 0.71,
        },
    )

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert packet["social_topology"] == "multi_person"
    assert packet["social_topology_state"]["state"] == "threaded_group"
    assert packet["social_topology_state"]["winner_margin"] >= 0.0
    assert packet["relation_competition_state"]["state"] == "competing_threads"
    assert packet["relation_competition_state"]["total_people"] == 2
    assert packet["relation_competition_state"]["winner_margin"] < 0.16
    assert packet["relational_continuity_state"]["scores"]["holding_thread"] >= packet["relational_continuity_state"]["scores"]["co_regulating"]
    assert "relation_competition" in packet["relational_continuity_state"]["dominant_inputs"]
    assert "hold_multiple_threads" in posture["next_action_candidates"]
    assert posture["relation_competition_name"] == "competing_threads"
    assert posture["social_topology_name"] == "threaded_group"
    assert "do_not_collapse_to_single_thread" in actuation["action_queue"]
    assert actuation["relation_competition_name"] == "competing_threads"
    assert actuation["social_topology_name"] == "threaded_group"


def test_social_topology_state_public_visible_keeps_contact_bounded() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["small visible opening"],
        relation_bias_strength=0.52,
        related_person_ids=["person:user"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        partner_social_interpretation="future_open",
        orchestration={
            "orchestration_mode": "advance",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.72,
            "coherence_score": 0.68,
            "human_presence_signal": 0.8,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.06,
            future_loop_pull=0.22,
            fantasy_loop_pull=0.0,
        ),
        scene_state={
            "social_topology": "public_visible",
            "privacy_level": 0.12,
            "norm_pressure": 0.66,
            "environmental_load": 0.24,
            "scene_family": "co_present",
            "scene_tags": ["socially_exposed", "high_norm"],
        },
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.72,
            "reportable_slice": ["small visible opening"],
            "actionable_slice": ["small next step"],
            "reportability_gate": {"gate_mode": "open"},
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "medium",
                "acknowledgement_need_level": "medium",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "medium",
            },
        },
        qualia_planner_view={
            "trust": 0.8,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.14,
            "body_load": 0.04,
            "protection_bias": 0.08,
            "felt_energy": 0.18,
        },
        terrain_readout={
            "value": 0.18,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.01, 0.0, 0.0, 0.0],
            "approach_bias": 0.58,
            "avoid_bias": 0.1,
            "protect_bias": 0.16,
            "active_patch_index": 1,
            "active_patch_label": "visible_threshold",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.2,
            "reasons": ["stable_contact"],
        },
        self_state={
            "related_person_id": "person:user",
            "stress": 0.12,
            "recovery_need": 0.08,
            "recent_strain": 0.06,
            "safety_bias": 0.1,
            "continuity_score": 0.64,
            "social_grounding": 0.62,
            "trust_memory": 0.68,
            "familiarity": 0.64,
            "attachment": 0.66,
        },
    )

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert packet["social_topology_state"]["state"] == "public_visible"
    assert "social_topology_public_visible" in packet["relational_continuity_state"]["dominant_inputs"]
    assert posture["social_topology_name"] == "public_visible"
    assert "respect_visible_context" in posture["next_action_candidates"]
    assert posture["boundary_mode"] in {"bounded", "guarded", "protective"}
    assert actuation["social_topology_name"] == "public_visible"
    assert "keep_visibility_safe" in actuation["action_queue"]


def test_initiative_followup_bias_softly_lifts_next_turn_readiness() -> None:
    base_kwargs = {
        "dialogue_act": "check_in",
        "current_focus": "person:user",
        "current_risks": [],
        "reportable_facts": ["shared opening"],
        "relation_bias_strength": 0.4,
        "related_person_ids": ["user"],
        "partner_address_hint": "companion",
        "partner_timing_hint": "open",
        "partner_stance_hint": "familiar",
        "orchestration": {
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.42,
            "coherence_score": 0.38,
            "human_presence_signal": 0.44,
        },
        "surface_profile": {
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "soft_return",
        },
        "live_regulation": SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.1,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        "conscious_workspace": {
            "workspace_mode": "foreground",
            "workspace_stability": 0.58,
            "reportability_gate": {"gate_mode": "open"},
        },
        "qualia_planner_view": {
            "trust": 0.72,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.14,
            "body_load": 0.06,
            "protection_bias": 0.08,
            "felt_energy": 0.16,
        },
        "terrain_readout": {
            "value": 0.14,
            "grad": [0.02, 0.01, 0.0, 0.0],
            "curvature": [0.02, 0.01, 0.0, 0.0],
            "approach_bias": 0.28,
            "avoid_bias": 0.16,
            "protect_bias": 0.16,
            "active_patch_index": 2,
            "active_patch_label": "open_step",
        },
        "protection_mode": {
            "mode": "monitor",
            "strength": 0.22,
            "reasons": ["stable_contact"],
        },
    }
    packet_without = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.16,
            "recovery_need": 0.14,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
        },
    )
    packet_with = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.16,
            "recovery_need": 0.14,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
            "initiative_followup_state": "offer_next_step",
            "initiative_followup_bias": 0.42,
        },
    )

    assert packet_with["initiative_followup_bias"]["state"] == "offer_next_step"
    assert packet_with["initiative_followup_bias"]["score"] == 0.42
    assert (
        packet_with["initiative_readiness"]["scores"]["ready"]
        > packet_without["initiative_readiness"]["scores"]["ready"]
    )
    assert "initiative_followup_bias" in packet_with["initiative_readiness"]["dominant_inputs"]
    assert (
        packet_with["reaction_vs_overnight_bias"]["overnight"]["initiative_followup_state"]
        == "offer_next_step"
    )


def test_temperament_estimate_lifts_ready_without_external_profile() -> None:
    base_kwargs = {
        "dialogue_act": "check_in",
        "current_focus": "person:user",
        "current_risks": [],
        "reportable_facts": ["shared opening"],
        "relation_bias_strength": 0.4,
        "related_person_ids": ["user"],
        "partner_address_hint": "companion",
        "partner_timing_hint": "open",
        "partner_stance_hint": "familiar",
        "orchestration": {
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.48,
            "coherence_score": 0.44,
            "human_presence_signal": 0.5,
        },
        "surface_profile": {
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "soft_return",
        },
        "live_regulation": SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.1,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        "conscious_workspace": {
            "workspace_mode": "foreground",
            "workspace_stability": 0.58,
            "reportability_gate": {"gate_mode": "open"},
        },
        "qualia_planner_view": {
            "trust": 0.74,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.14,
            "body_load": 0.06,
            "protection_bias": 0.08,
            "felt_energy": 0.16,
        },
        "terrain_readout": {
            "value": 0.14,
            "grad": [0.02, 0.01, 0.0, 0.0],
            "curvature": [0.02, 0.01, 0.0, 0.0],
            "approach_bias": 0.32,
            "avoid_bias": 0.16,
            "protect_bias": 0.18,
            "active_patch_index": 2,
            "active_patch_label": "open_step",
        },
        "protection_mode": {
            "mode": "monitor",
            "strength": 0.22,
            "reasons": ["stable_contact"],
        },
    }
    lower_trace = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.16,
            "recovery_need": 0.14,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
            "caution_bias": 0.46,
            "affiliation_bias": 0.52,
            "exploration_bias": 0.42,
            "reflective_bias": 0.48,
            "temperament_forward_trace": 0.08,
            "temperament_guard_trace": 0.26,
            "temperament_bond_trace": 0.1,
            "temperament_recovery_trace": 0.18,
        },
    )
    higher_trace = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.16,
            "recovery_need": 0.14,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
            "caution_bias": 0.32,
            "affiliation_bias": 0.58,
            "exploration_bias": 0.56,
            "reflective_bias": 0.42,
            "temperament_forward_trace": 0.62,
            "temperament_guard_trace": 0.08,
            "temperament_bond_trace": 0.34,
            "temperament_recovery_trace": 0.06,
        },
    )

    assert (
        higher_trace["temperament_estimate"]["risk_tolerance"]
        > lower_trace["temperament_estimate"]["risk_tolerance"]
    )
    assert (
        higher_trace["initiative_readiness"]["scores"]["ready"]
        > lower_trace["initiative_readiness"]["scores"]["ready"]
    )
    assert (
        higher_trace["temperament_estimate"]["hero_tendency"]
        > lower_trace["temperament_estimate"]["hero_tendency"]
    )


def test_overnight_temperament_bias_weakly_lifts_ready() -> None:
    base_kwargs = {
        "dialogue_act": "check_in",
        "current_focus": "person:user",
        "current_risks": [],
        "reportable_facts": ["shared opening"],
        "relation_bias_strength": 0.4,
        "related_person_ids": ["user"],
        "partner_address_hint": "companion",
        "partner_timing_hint": "open",
        "partner_stance_hint": "familiar",
        "orchestration": {
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.48,
            "coherence_score": 0.44,
            "human_presence_signal": 0.5,
        },
        "surface_profile": {
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "soft_return",
        },
        "live_regulation": SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.1,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        "conscious_workspace": {
            "workspace_mode": "foreground",
            "workspace_stability": 0.58,
            "reportability_gate": {"gate_mode": "open"},
        },
        "qualia_planner_view": {
            "trust": 0.74,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.14,
            "body_load": 0.06,
            "protection_bias": 0.08,
            "felt_energy": 0.16,
        },
        "terrain_readout": {
            "value": 0.14,
            "grad": [0.02, 0.01, 0.0, 0.0],
            "curvature": [0.02, 0.01, 0.0, 0.0],
            "approach_bias": 0.32,
            "avoid_bias": 0.16,
            "protect_bias": 0.18,
            "active_patch_index": 2,
            "active_patch_label": "open_step",
        },
        "protection_mode": {
            "mode": "monitor",
            "strength": 0.22,
            "reasons": ["stable_contact"],
        },
    }
    without_bias = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.16,
            "recovery_need": 0.14,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
            "caution_bias": 0.4,
            "affiliation_bias": 0.52,
            "exploration_bias": 0.48,
            "reflective_bias": 0.44,
            "temperament_forward_trace": 0.18,
            "temperament_guard_trace": 0.12,
            "temperament_bond_trace": 0.14,
            "temperament_recovery_trace": 0.1,
        },
    )
    with_bias = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.16,
            "recovery_need": 0.14,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
            "caution_bias": 0.4,
            "affiliation_bias": 0.52,
            "exploration_bias": 0.48,
            "reflective_bias": 0.44,
            "temperament_forward_trace": 0.18,
            "temperament_guard_trace": 0.12,
            "temperament_bond_trace": 0.14,
            "temperament_recovery_trace": 0.1,
            "temperament_focus": "forward",
            "temperament_forward_bias": 0.12,
            "temperament_guard_bias": 0.02,
            "temperament_bond_bias": 0.03,
            "temperament_recovery_bias": 0.01,
        },
    )

    assert (
        with_bias["temperament_estimate"]["risk_tolerance"]
        > without_bias["temperament_estimate"]["risk_tolerance"]
    )
    assert (
        with_bias["initiative_readiness"]["scores"]["ready"]
        > without_bias["initiative_readiness"]["scores"]["ready"]
    )


def test_commitment_carry_bias_weakly_lifts_initiative_readiness() -> None:
    base_kwargs = {
        "dialogue_act": "check_in",
        "current_focus": "person:user",
        "current_risks": [],
        "reportable_facts": ["shared opening"],
        "relation_bias_strength": 0.42,
        "related_person_ids": ["user"],
        "partner_address_hint": "companion",
        "partner_timing_hint": "open",
        "partner_stance_hint": "familiar",
        "orchestration": {
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.52,
            "coherence_score": 0.48,
            "human_presence_signal": 0.56,
        },
        "surface_profile": {
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "soft_return",
        },
        "live_regulation": SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.08,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        "conscious_workspace": {
            "workspace_mode": "foreground",
            "workspace_stability": 0.62,
            "reportability_gate": {"gate_mode": "open"},
        },
        "qualia_planner_view": {
            "trust": 0.76,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.12,
            "body_load": 0.04,
            "protection_bias": 0.08,
            "felt_energy": 0.14,
        },
        "terrain_readout": {
            "value": 0.12,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.02, 0.0, 0.0, 0.0],
            "approach_bias": 0.34,
            "avoid_bias": 0.16,
            "protect_bias": 0.12,
            "active_patch_index": 1,
            "active_patch_label": "open_step",
        },
        "protection_mode": {
            "mode": "monitor",
            "strength": 0.24,
            "reasons": ["stable_contact"],
        },
    }
    packet_without = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.16,
            "recovery_need": 0.14,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
        },
    )
    packet_with = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.16,
            "recovery_need": 0.14,
            "recent_strain": 0.12,
            "safety_bias": 0.08,
            "commitment_target_focus": "step_forward",
            "commitment_state_focus": "commit",
            "commitment_carry_bias": 0.42,
            "commitment_followup_focus": "offer_next_step",
            "commitment_mode_focus": "monitor",
            "commitment_carry_reason": "commit:step_forward",
        },
    )

    assert packet_with["commitment_carry"]["followup_focus"] == "offer_next_step"
    assert (
        packet_with["initiative_readiness"]["scores"]["ready"]
        > packet_without["initiative_readiness"]["scores"]["ready"]
    )
    assert "overnight_commitment_carry" in packet_with["initiative_readiness"]["dominant_inputs"]


def test_body_and_relational_overnight_carry_bias_weakly_lift_same_turn_states() -> None:
    base_kwargs = {
        "dialogue_act": "check_in",
        "current_focus": "person:user",
        "current_risks": [],
        "reportable_facts": ["shared opening"],
        "relation_bias_strength": 0.44,
        "related_person_ids": ["user"],
        "partner_address_hint": "companion",
        "partner_timing_hint": "open",
        "partner_stance_hint": "familiar",
        "orchestration": {
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.56,
            "coherence_score": 0.51,
            "human_presence_signal": 0.58,
        },
        "surface_profile": {
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "soft_return",
        },
        "live_regulation": SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=True,
            strained_pause=0.06,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        "conscious_workspace": {
            "workspace_mode": "foreground",
            "workspace_stability": 0.64,
            "reportability_gate": {"gate_mode": "open"},
        },
        "qualia_planner_view": {
            "trust": 0.74,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.11,
            "body_load": 0.06,
            "protection_bias": 0.1,
            "felt_energy": 0.16,
        },
        "terrain_readout": {
            "value": 0.14,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.03, 0.0, 0.0, 0.0],
            "approach_bias": 0.32,
            "avoid_bias": 0.15,
            "protect_bias": 0.11,
            "active_patch_index": 1,
            "active_patch_label": "open_step",
        },
        "protection_mode": {
            "mode": "repair",
            "strength": 0.28,
            "reasons": ["repair_window_open"],
        },
    }
    packet_without = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.14,
            "recovery_need": 0.12,
            "recent_strain": 0.1,
            "safety_bias": 0.08,
        },
    )
    packet_with = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.14,
            "recovery_need": 0.12,
            "recent_strain": 0.1,
            "safety_bias": 0.08,
            "body_homeostasis_focus": "recovering",
            "body_homeostasis_carry_bias": 0.16,
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.09,
            "relational_continuity_focus": "reopening",
            "relational_continuity_carry_bias": 0.14,
        },
    )

    assert packet_with["body_homeostasis_carry"]["focus"] == "recovering"
    assert packet_with["homeostasis_budget_carry"]["focus"] == "recovering"
    assert packet_with["relational_continuity_carry"]["focus"] == "reopening"
    assert (
        packet_with["body_homeostasis_state"]["scores"]["recovering"]
        > packet_without["body_homeostasis_state"]["scores"]["recovering"]
    )
    assert (
        packet_with["homeostasis_budget_state"]["scores"]["recovering"]
        > packet_without["homeostasis_budget_state"]["scores"]["recovering"]
    )
    assert (
        packet_with["relational_continuity_state"]["scores"]["reopening"]
        > packet_without["relational_continuity_state"]["scores"]["reopening"]
    )
    assert "overnight_body_homeostasis" in packet_with["body_homeostasis_state"]["dominant_inputs"]
    assert "overnight_homeostasis_budget" in packet_with["homeostasis_budget_state"]["dominant_inputs"]
    assert "overnight_relational_continuity" in packet_with["relational_continuity_state"]["dominant_inputs"]


def test_agenda_carry_bias_weakly_lifts_same_turn_readiness() -> None:
    base_kwargs = {
        "dialogue_act": "check_in",
        "current_focus": "person:user",
        "current_risks": [],
        "reportable_facts": ["shared opening"],
        "relation_bias_strength": 0.46,
        "related_person_ids": ["user"],
        "partner_address_hint": "companion",
        "partner_timing_hint": "open",
        "partner_stance_hint": "familiar",
        "orchestration": {
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.66,
            "coherence_score": 0.61,
            "human_presence_signal": 0.64,
        },
        "surface_profile": {
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        "live_regulation": SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.06,
            future_loop_pull=0.28,
            fantasy_loop_pull=0.0,
        ),
        "conscious_workspace": {
            "workspace_mode": "foreground",
            "workspace_stability": 0.68,
            "reportability_gate": {"gate_mode": "open"},
        },
        "qualia_planner_view": {
            "trust": 0.76,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.14,
            "body_load": 0.05,
            "protection_bias": 0.08,
            "felt_energy": 0.18,
        },
        "terrain_readout": {
            "value": 0.16,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.02, 0.0, 0.0, 0.0],
            "approach_bias": 0.36,
            "avoid_bias": 0.14,
            "protect_bias": 0.1,
            "active_patch_index": 1,
            "active_patch_label": "open_step",
        },
        "protection_mode": {
            "mode": "monitor",
            "strength": 0.24,
            "reasons": ["stable_contact"],
        },
    }
    packet_without = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.12,
            "recovery_need": 0.08,
            "recent_strain": 0.09,
            "safety_bias": 0.06,
        },
    )
    packet_with = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.12,
            "recovery_need": 0.08,
            "recent_strain": 0.09,
            "safety_bias": 0.06,
            "agenda_focus": "step_forward",
            "agenda_bias": 0.24,
            "agenda_reason": "offer_next_step",
        },
    )

    assert packet_with["agenda_carry"]["focus"] == "step_forward"
    assert packet_with["agenda_carry"]["reason"] == "offer_next_step"
    assert packet_with["reaction_vs_overnight_bias"]["overnight"]["agenda_focus"] == "step_forward"
    assert (
        packet_with["initiative_readiness"]["scores"]["ready"]
        > packet_without["initiative_readiness"]["scores"]["ready"]
    )
    assert "overnight_agenda_carry" in packet_with["initiative_readiness"]["dominant_inputs"]


def test_expressive_style_overnight_carry_bias_weakly_lifts_same_turn_style() -> None:
    base_kwargs = {
        "dialogue_act": "check_in",
        "current_focus": "person:user",
        "current_risks": [],
        "reportable_facts": ["shared opening"],
        "relation_bias_strength": 0.42,
        "related_person_ids": ["user"],
        "partner_address_hint": "companion",
        "partner_timing_hint": "open",
        "partner_stance_hint": "familiar",
        "orchestration": {
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.72,
            "coherence_score": 0.64,
            "human_presence_signal": 0.68,
        },
        "surface_profile": {
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        "live_regulation": SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.08,
            future_loop_pull=0.26,
            fantasy_loop_pull=0.0,
        ),
        "conscious_workspace": {
            "workspace_mode": "foreground",
            "workspace_stability": 0.66,
            "reportability_gate": {"gate_mode": "open"},
        },
        "qualia_planner_view": {
            "trust": 0.82,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "dominant_value": 0.14,
            "body_load": 0.03,
            "protection_bias": 0.06,
            "felt_energy": 0.2,
        },
        "terrain_readout": {
            "value": 0.16,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.02, 0.0, 0.0, 0.0],
            "approach_bias": 0.42,
            "avoid_bias": 0.1,
            "protect_bias": 0.1,
            "active_patch_index": 1,
            "active_patch_label": "open_step",
        },
        "protection_mode": {
            "mode": "monitor",
            "strength": 0.24,
            "reasons": ["stable_contact"],
        },
    }
    packet_without = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.12,
            "recovery_need": 0.1,
            "recent_strain": 0.08,
            "safety_bias": 0.06,
            "social_grounding": 0.62,
            "continuity_score": 0.58,
        },
    )
    packet_with = derive_interaction_policy_packet(
        **base_kwargs,
        self_state={
            "stress": 0.12,
            "recovery_need": 0.1,
            "recent_strain": 0.08,
            "safety_bias": 0.06,
            "social_grounding": 0.62,
            "continuity_score": 0.58,
            "expressive_style_focus": "warm_companion",
            "expressive_style_carry_bias": 0.12,
        },
    )

    assert packet_with["reaction_vs_overnight_bias"]["overnight"]["expressive_style_focus"] == "warm_companion"
    assert packet_with["reaction_vs_overnight_bias"]["overnight"]["expressive_style_carry_bias"] == 0.12
    assert (
        packet_with["expressive_style_state"]["scores"]["warm_companion"]
        > packet_without["expressive_style_state"]["scores"]["warm_companion"]
    )
    assert "overnight_expressive_style" in packet_with["expressive_style_state"]["dominant_inputs"]


def test_policy_packet_derives_relational_style_memory_and_lightness_budget_from_person_registry() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["shared opening"],
        relation_bias_strength=0.58,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.74,
            "coherence_score": 0.68,
            "human_presence_signal": 0.72,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.06,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.68,
            "reportability_gate": {"gate_mode": "open"},
        },
        qualia_planner_view={
            "trust": 0.82,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "body_load": 0.04,
            "protection_bias": 0.08,
            "felt_energy": 0.18,
        },
        terrain_readout={
            "approach_bias": 0.44,
            "avoid_bias": 0.1,
            "protect_bias": 0.1,
            "active_patch_label": "open_step",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.22,
            "winner_margin": 0.16,
            "reasons": ["stable_contact"],
        },
        self_state={
            "related_person_id": "user",
            "stress": 0.12,
            "recovery_need": 0.1,
            "recent_strain": 0.08,
            "safety_bias": 0.06,
            "attachment": 0.72,
            "familiarity": 0.76,
            "trust_memory": 0.7,
            "continuity_score": 0.64,
            "social_grounding": 0.62,
            "exploration_bias": 0.72,
            "caution_bias": 0.18,
            "person_registry_snapshot": {
                "persons": {
                    "user": {
                        "adaptive_traits": {
                            "style_warmth_memory": 0.68,
                            "playful_ceiling": 0.74,
                            "advice_tolerance": 0.58,
                            "lexical_familiarity": 0.62,
                        }
                    }
                }
            },
        },
    )

    assert packet["relational_style_memory_state"]["state"] in {"light_playful", "warm_companion"}
    assert packet["relational_style_memory_state"]["playful_ceiling"] >= 0.68
    assert packet["lightness_budget_state"]["state"] in {"open_play", "warm_only"}
    assert packet["reaction_vs_overnight_bias"]["same_turn"]["lightness_budget_state"]


def test_temperament_estimate_does_not_override_recovery_first() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["shared opening"],
        relation_bias_strength=0.4,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.72,
            "coherence_score": 0.64,
            "human_presence_signal": 0.66,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "steady_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="shared_world_orientation",
            repair_window_open=False,
            strained_pause=0.08,
            future_loop_pull=0.26,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.66,
            "reportability_gate": {"gate_mode": "open"},
        },
        qualia_planner_view={
            "trust": 0.72,
            "degraded": False,
            "dominant_axis": "body_load",
            "dominant_value": 0.2,
            "body_load": 0.2,
            "protection_bias": 0.16,
            "felt_energy": 0.18,
        },
        terrain_readout={
            "value": 0.08,
            "grad": [0.02, 0.0, 0.0, 0.0],
            "curvature": [0.02, 0.01, 0.0, 0.0],
            "approach_bias": 0.36,
            "avoid_bias": 0.18,
            "protect_bias": 0.24,
            "active_patch_index": 1,
            "active_patch_label": "soft_guard",
        },
        protection_mode={
            "mode": "stabilize",
            "strength": 0.58,
            "reasons": ["check_load"],
        },
        self_state={
            "stress": 0.48,
            "recovery_need": 0.74,
            "recent_strain": 0.28,
            "safety_bias": 0.12,
            "caution_bias": 0.22,
            "affiliation_bias": 0.6,
            "exploration_bias": 0.76,
            "reflective_bias": 0.34,
            "temperament_forward_trace": 0.74,
            "temperament_guard_trace": 0.04,
            "temperament_bond_trace": 0.28,
            "temperament_recovery_trace": 0.04,
        },
    )

    assert packet["temperament_estimate"]["hero_tendency"] > 0.0
    assert packet["body_recovery_guard"]["state"] == "recovery_first"
    assert packet["initiative_readiness"]["state"] != "ready"
    assert packet["commitment_state"]["target"] in {"hold", "stabilize"}


def test_commitment_state_commits_to_repair_when_history_and_mode_align() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=[],
        reportable_facts=["we can try again carefully"],
        relation_bias_strength=0.64,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="open",
        partner_stance_hint="familiar",
        orchestration={
            "orchestration_mode": "repair",
            "dominant_driver": "repair_window",
            "contact_readiness": 0.62,
            "coherence_score": 0.68,
            "human_presence_signal": 0.74,
        },
        surface_profile={
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "soft_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=True,
            strained_pause=0.1,
            future_loop_pull=0.12,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.72,
            "reportability_gate": {"gate_mode": "open"},
        },
        qualia_planner_view={
            "trust": 0.82,
            "degraded": False,
            "dominant_axis": "bond_repair",
            "dominant_value": 0.22,
            "body_load": 0.04,
            "protection_bias": 0.12,
            "felt_energy": 0.18,
        },
        terrain_readout={
            "value": 0.18,
            "grad": [0.02, 0.02, 0.0, 0.0],
            "curvature": [0.02, 0.01, 0.0, 0.0],
            "approach_bias": 0.54,
            "avoid_bias": 0.12,
            "protect_bias": 0.22,
            "active_patch_index": 3,
            "active_patch_label": "repair_basin",
        },
        protection_mode={
            "mode": "repair",
            "strength": 0.66,
            "winner_margin": 0.24,
            "reasons": ["repair_opening"],
        },
        self_state={
            "stress": 0.18,
            "recovery_need": 0.12,
            "recent_strain": 0.18,
            "safety_bias": 0.1,
            "initiative_followup_state": "reopen_softly",
            "initiative_followup_bias": 0.38,
            "caution_bias": 0.34,
            "affiliation_bias": 0.72,
            "exploration_bias": 0.42,
            "reflective_bias": 0.58,
            "temperament_forward_trace": 0.28,
            "temperament_guard_trace": 0.18,
            "temperament_bond_trace": 0.58,
            "temperament_recovery_trace": 0.42,
        },
    )

    assert packet["commitment_state"]["state"] == "commit"
    assert packet["commitment_state"]["target"] in {"repair", "bond_protect"}
    assert packet["commitment_state"]["accepted_cost"] > 0.0


def test_policy_packet_derives_attention_and_grice_guard_for_known_thread() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="shared opening",
        current_risks=[],
        reportable_facts=["shared opening"],
        relation_bias_strength=0.58,
        related_person_ids=["user"],
        partner_address_hint="companion",
        partner_timing_hint="delayed",
        partner_stance_hint="respectful",
        partner_social_interpretation="future_open",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.54,
            "coherence_score": 0.62,
            "human_presence_signal": 0.72,
        },
        surface_profile={
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "soft_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="holding_space",
            repair_window_open=False,
            strained_pause=0.18,
            future_loop_pull=0.18,
            fantasy_loop_pull=0.0,
        ),
        conscious_workspace={
            "workspace_mode": "foreground",
            "workspace_stability": 0.68,
            "reportable_slice": ["shared opening"],
            "withheld_slice": ["deeper meaning"],
            "actionable_slice": [],
            "reportability_gate": {"gate_mode": "guarded"},
        },
        qualia_planner_view={
            "trust": 0.8,
            "degraded": False,
            "dominant_axis": "shared_attention",
            "body_load": 0.06,
            "protection_bias": 0.14,
            "felt_energy": 0.16,
        },
        terrain_readout={
            "approach_bias": 0.32,
            "avoid_bias": 0.16,
            "protect_bias": 0.28,
            "active_patch_label": "holding_thread",
        },
        protection_mode={
            "mode": "contain",
            "strength": 0.58,
            "winner_margin": 0.2,
            "reasons": ["protect_boundary"],
        },
        self_state={
            "stress": 0.22,
            "recovery_need": 0.18,
            "recent_strain": 0.24,
            "semantic_seed_focus": "shared opening",
            "semantic_seed_strength": 0.78,
            "semantic_seed_recurrence": 1.34,
            "long_term_theme_focus": "shared opening",
            "long_term_theme_summary": "shared opening by the harbor",
            "long_term_theme_strength": 0.72,
            "relation_seed_summary": "shared opening thread",
            "relation_seed_strength": 0.64,
        },
    )

    assert packet["attention_regulation_state"]["state"] in {"reflex_guard", "split_guarded", "selective_hold"}
    assert packet["attention_regulation_state"]["selective_target"] == "shared opening"
    assert packet["grice_guard_state"]["state"] in {"attune_without_repeating", "hold_obvious_advice"}
    assert packet["grice_guard_state"]["question_budget_cap"] == 0
    assert packet["effective_question_budget"] == 0


def test_content_sequence_reduces_obvious_advice_when_grice_guard_holds() -> None:
    sequence = derive_content_sequence(
        current_text="Here is the next step I can see from what is already clear.",
        interaction_policy={
            "response_strategy": "shared_world_next_step",
            "attention_target": "shared opening",
            "primary_conversational_object_label": "shared opening",
            "question_budget": 1,
            "effective_question_budget": 0,
            "grice_guard_state": {
                "state": "hold_obvious_advice",
                "question_budget_cap": 0,
            },
        },
        conscious_access={"intent": "check_in"},
    )
    text = " ".join(step["text"] for step in sequence)
    assert "already clear" in text.lower()
    assert "next step" not in text.lower()


def test_surface_profile_softens_when_grice_guard_and_reflex_attention_are_active() -> None:
    updated = _apply_interaction_policy_surface_bias(
        {
            "opening_delay": "brief",
            "response_length": "forward_leaning",
            "sentence_temperature": "warm",
            "pause_insertion": "none",
            "certainty_style": "direct",
            "cues": [],
        },
        {
            "body_recovery_guard": {"state": "guarded", "score": 0.42},
            "body_homeostasis_state": {"state": "steady", "score": 0.24},
            "homeostasis_budget_state": {"state": "steady", "score": 0.24},
            "initiative_readiness": {"state": "ready", "score": 0.44},
            "commitment_state": {"state": "settle", "target": "hold", "score": 0.32},
            "attention_regulation_state": {"state": "reflex_guard", "score": 0.58},
            "grice_guard_state": {"state": "hold_obvious_advice", "score": 0.74},
            "relational_continuity_state": {"state": "holding_thread", "score": 0.62},
            "social_topology_state": {"state": "one_to_one", "score": 0.2},
            "protection_mode": {"mode": "contain", "strength": 0.58},
        },
    )

    assert updated["response_length"] == "short"
    assert updated["certainty_style"] == "careful"
    assert "surface_grice_hold_obvious_advice" in updated["cues"]
    assert "surface_reflex_attention_guard" in updated["cues"]
