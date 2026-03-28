from inner_os.access import AttentionState, ForegroundState, select_foreground
from inner_os.evaluation import evaluate_run, smoke_trace
from inner_os.expression import (
    DialogueContext,
    QUALIA_HINT_VERSION,
    build_expression_hints_from_gate_result,
    render_response,
)
from inner_os.qualia_kernel_adapter import QUALIA_AXIS_LABELS
from inner_os.grounding import ground_symbols, infer_affordances, observe, summarize_partner_grounding
from inner_os.memory import build_memory_appends, build_memory_context
from inner_os.memory_core import MemoryCore
from inner_os.self_model import PersonRegistry, SelfState, person_registry_from_snapshot, update_person_registry, update_self_state
from inner_os.value_system import compute_value_state
from inner_os.world_model import WorldState, update_world_state


def test_inner_os_bootstrap_smoke_flow(tmp_path) -> None:
    observation = observe({"entity_labels": ["harbor", "lamp", "user"]})
    world = update_world_state(WorldState(), observation, {})
    self_state = update_self_state(SelfState(curiosity=0.2), world, {"arousal_delta": 0.1})
    registry = update_person_registry(PersonRegistry(), {"summary": "user seen"}, {"person_id": "obs:2"})
    affordances = infer_affordances(observation, world, self_state)
    value = compute_value_state(world, self_state, registry)
    grounded = ground_symbols(["harbor"], observation, affordances, value)
    observation.symbol_groundings = grounded
    foreground = select_foreground(world, self_state, value, AttentionState(), registry)
    memory_context = build_memory_context(foreground, uncertainty=world.uncertainty)
    memory_core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    appended = memory_core.append_records(build_memory_appends(memory_context))
    plan = render_response(
        foreground,
        DialogueContext(user_text="これはどうですか"),
        memory_context,
    )
    report = evaluate_run(
        smoke_trace(
            foreground=foreground.reportable_facts,
            raw_observation_passed_to_llm=False,
        )
    )
    assert observation.entities
    assert foreground.reportable_facts
    assert foreground.candidates
    assert memory_context.episodic_candidates
    assert memory_context.semantic_hints
    assert len(appended) == 2
    assert memory_core.search("obs:2", limit=5)
    assert "foreground" in plan.llm_payload
    assert plan.llm_payload["qualia_hint_source"] == "none"
    assert plan.llm_payload["qualia_hint_fallback_reason"] == "neutral_no_qualia_material"
    assert plan.llm_payload["qualia_hint_expected_mismatch"] is False
    assert "memory_context" in plan.llm_payload
    assert "interaction_policy" in plan.llm_payload
    assert "scene_state" in plan.llm_payload
    assert "interaction_option_candidates" in plan.llm_payload
    assert "contact_field" in plan.llm_payload
    assert "contact_dynamics" in plan.llm_payload
    assert "contact_reflection_state" in plan.llm_payload
    assert "green_kernel_composition" in plan.llm_payload
    assert "access_projection" in plan.llm_payload
    assert "access_dynamics" in plan.llm_payload
    assert "affect_blend_state" in plan.llm_payload
    assert "constraint_field" in plan.llm_payload
    assert "conscious_workspace" in plan.llm_payload
    assert "conversational_objects" in plan.llm_payload
    assert "object_operations" in plan.llm_payload
    assert "interaction_effects" in plan.llm_payload
    assert "interaction_judgement_view" in plan.llm_payload
    assert "interaction_judgement_summary" in plan.llm_payload
    assert "interaction_condition_report" in plan.llm_payload
    assert "interaction_inspection_report" in plan.llm_payload
    assert "interaction_audit_bundle" in plan.llm_payload
    assert "interaction_audit_report" in plan.llm_payload
    assert "resonance_evaluation" in plan.llm_payload
    assert "content_skeleton" in plan.llm_payload
    assert "content_sequence" in plan.llm_payload
    assert "interaction_constraints" in plan.llm_payload
    assert "repetition_guard" in plan.llm_payload
    assert "recent_dialogue_state" in plan.llm_payload
    assert "discussion_thread_state" in plan.llm_payload
    assert "issue_state" in plan.llm_payload
    assert "turn_delta" in plan.llm_payload
    assert "surface_context_packet" in plan.llm_payload
    assert plan.interaction_policy["dialogue_act"] in {"report", "check_in"}
    assert plan.interaction_policy["dialogue_order"]
    assert plan.interaction_policy["scene_family"]
    assert plan.interaction_policy["interaction_option_candidates"]
    assert plan.llm_payload["conscious_workspace"]["workspace_mode"] in {"preconscious", "latent_foreground", "foreground", "guarded_foreground"}
    assert "actionable_slice" in plan.llm_payload["conscious_workspace"]
    assert plan.llm_payload["contact_field"]["points"]
    assert plan.llm_payload["contact_dynamics"]["stabilized_points"]
    assert plan.llm_payload["contact_reflection_state"]["reflection_style"] in {"reflect_then_question", "reflect_only", "boundary_only"}
    assert "field" in plan.llm_payload["green_kernel_composition"]
    assert plan.llm_payload["access_projection"]["regions"]
    assert plan.llm_payload["access_dynamics"]["stabilized_regions"]
    assert plan.llm_payload["resonance_evaluation"]["estimated_other_person_state"]["detail_room_level"] in {"low", "medium", "high"}
    assert plan.llm_payload["conversational_objects"]["objects"]
    assert plan.llm_payload["conversational_objects"]["pressure_balance"] >= 0.0
    assert plan.llm_payload["object_operations"]["operations"]
    assert plan.llm_payload["object_operations"]["question_pressure"] >= 0.0
    assert plan.llm_payload["object_operations"]["defer_dominance"] >= 0.0
    assert plan.llm_payload["interaction_effects"]["effects"]
    assert plan.llm_payload["interaction_judgement_view"]["observed_signals"]
    assert plan.llm_payload["interaction_judgement_view"]["inferred_signals"]
    assert isinstance(plan.llm_payload["interaction_judgement_view"]["selected_object_labels"], list)
    assert plan.llm_payload["interaction_judgement_summary"]["observed_lines"]
    assert plan.llm_payload["interaction_judgement_summary"]["inferred_lines"]
    assert any("相手は" in line for line in plan.llm_payload["interaction_judgement_summary"]["observed_lines"])
    assert plan.llm_payload["interaction_condition_report"]["scene_lines"]
    assert plan.llm_payload["interaction_condition_report"]["report_lines"]
    assert plan.llm_payload["interaction_inspection_report"]["case_reports"]
    assert any("current_case" in line for line in plan.llm_payload["interaction_inspection_report"]["report_lines"])
    assert plan.llm_payload["interaction_audit_bundle"]["report_lines"]
    assert plan.llm_payload["interaction_audit_bundle"]["key_metrics"]["question_budget"] >= 0
    assert plan.llm_payload["interaction_audit_report"]["report_lines"]
    assert all("act" in step and "text" in step for step in plan.llm_payload["content_sequence"])
    assert "keep_thread_visible" in plan.llm_payload["interaction_constraints"] or "prefer_return_point" in plan.llm_payload["interaction_constraints"]
    assert "recent_text_count" in plan.llm_payload["repetition_guard"]
    assert "preferred_act" in plan.llm_payload["turn_delta"]
    assert "conversation_phase" in plan.llm_payload["surface_context_packet"]
    assert "shared_core" in plan.llm_payload["surface_context_packet"]
    assert "response_role" in plan.llm_payload["surface_context_packet"]
    assert "constraints" in plan.llm_payload["surface_context_packet"]
    assert plan.llm_payload["interaction_policy"]["focus_now"]
    assert "response_action_now" in plan.llm_payload["interaction_policy"]
    assert plan.llm_payload["interaction_policy"]["wanted_effect_on_other"]
    assert "learning_mode_state" in plan.llm_payload["interaction_policy"]
    assert "social_experiment_loop_state" in plan.llm_payload["interaction_policy"]
    assert "persona_memory_selection" in plan.llm_payload["interaction_policy"]
    assert "persona_memory_fragments" in plan.llm_payload["interaction_policy"]
    assert "recent_dialogue_state" in plan.llm_payload["interaction_policy"]
    assert "discussion_thread_state" in plan.llm_payload["interaction_policy"]
    assert "issue_state" in plan.llm_payload["interaction_policy"]
    assert "contact_reflection_state" in plan.llm_payload["interaction_policy"]
    assert "green_kernel_composition" in plan.llm_payload["interaction_policy"]
    assert "reportability_scores" in plan.llm_payload["foreground"]
    assert "memory_candidates" in plan.llm_payload["foreground"]
    assert plan.llm_payload["memory_context"]["semantic_hints"]
    assert "related_person_ids" in plan.llm_payload["memory_context"]
    assert "harbor" in grounded
    assert report.score == 1.0
    assert "focus_current_object" in plan.multimodal_cues


def test_expression_bridge_never_exposes_raw_observation() -> None:
    foreground = select_foreground(
        WorldState(object_states={"obs:1": "harbor"}),
        SelfState(),
        compute_value_state(WorldState(), SelfState(), PersonRegistry()),
        AttentionState(),
    )
    plan = render_response(foreground, DialogueContext(user_text="話して"))
    assert "raw_observation" not in plan.llm_payload
    assert "foreground" in plan.llm_payload


def test_response_planner_consumes_shared_qualia_hints() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user looks strained"],
        memory_candidates=["user"],
        reportability_scores={"user": 0.8},
    )
    qualia = [0.0] * len(QUALIA_AXIS_LABELS)
    gate = [0.0] * len(QUALIA_AXIS_LABELS)
    value_grad = [0.0] * len(QUALIA_AXIS_LABELS)
    stress_index = QUALIA_AXIS_LABELS.index("stress_level")
    qualia[stress_index] = 0.72
    gate[stress_index] = 0.91
    value_grad[stress_index] = 0.64
    plan = render_response(
        foreground,
        DialogueContext(
            user_text="最近ちょっとしんどい",
            expression_hints={
                "qualia_state": {
                    "qualia": qualia,
                    "gate": gate,
                    "precision": [0.5] * len(QUALIA_AXIS_LABELS),
                    "observability": [0.5] * len(QUALIA_AXIS_LABELS),
                    "body_coupling": [0.0] * len(QUALIA_AXIS_LABELS),
                    "value_grad": value_grad,
                    "habituation": [0.0] * len(QUALIA_AXIS_LABELS),
                    "trust_applied": 0.88,
                    "degraded": False,
                    "reason": None,
                    "axis_labels": list(QUALIA_AXIS_LABELS),
                },
                "qualia_estimator_health": {
                    "trust": 0.88,
                    "degraded": False,
                    "reason": None,
                },
                "qualia_protection_grad_x": value_grad,
                "qualia_axis_labels": list(QUALIA_AXIS_LABELS),
            },
        ),
    )
    assert "access_qualia_input" in plan.llm_payload["access_projection"]["cues"]
    assert plan.llm_payload["qualia_planner_view"]["dominant_axis"] == "stress_level"
    assert "qualia_shared_input" in plan.multimodal_cues
    assert "qualia_axis_stress_level" in plan.multimodal_cues
    assert plan.llm_payload["qualia_state"]["qualia"][stress_index] == 0.72
    assert plan.llm_payload["qualia_hint_source"] == "fallback"
    assert plan.llm_payload["qualia_hint_version"] == QUALIA_HINT_VERSION
    assert plan.llm_payload["qualia_hint_fallback_reason"] == "bridge_reconstructed_from_raw_materials"
    assert plan.llm_payload["qualia_hint_expected_mismatch"] is False


def test_response_planner_prefers_prebuilt_qualia_planner_view() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user is present"],
        memory_candidates=["user"],
        reportability_scores={"user": 0.7},
    )
    quiet = [0.0] * len(QUALIA_AXIS_LABELS)
    plan = render_response(
        foreground,
        DialogueContext(
            user_text="ここにいるよ",
            expression_hints={
                "qualia_state": {
                    "qualia": quiet,
                    "gate": quiet,
                    "precision": [0.1] * len(QUALIA_AXIS_LABELS),
                    "observability": [0.1] * len(QUALIA_AXIS_LABELS),
                    "body_coupling": [0.0] * len(QUALIA_AXIS_LABELS),
                    "value_grad": quiet,
                    "habituation": quiet,
                    "trust_applied": 0.1,
                    "degraded": False,
                    "reason": None,
                    "axis_labels": list(QUALIA_AXIS_LABELS),
                },
                "qualia_estimator_health": {
                    "trust": 0.1,
                    "degraded": False,
                    "reason": None,
                },
                "qualia_protection_grad_x": quiet,
                "qualia_axis_labels": list(QUALIA_AXIS_LABELS),
                "qualia_planner_view": {
                    "trust": 0.93,
                    "degraded": False,
                    "dominant_axis": "recovery_need",
                    "dominant_value": 0.61,
                    "body_load": 0.18,
                    "protection_bias": 0.0,
                    "felt_energy": 0.24,
                },
            },
        ),
    )
    assert plan.llm_payload["qualia_planner_view"]["dominant_axis"] == "recovery_need"
    assert plan.llm_payload["qualia_planner_view"]["trust"] == 0.93
    assert plan.llm_payload["qualia_hint_source"] == "shared"
    assert plan.llm_payload["qualia_hint_version"] == QUALIA_HINT_VERSION
    assert plan.llm_payload["qualia_hint_fallback_reason"] == "prebuilt_shared_view"
    assert plan.llm_payload["qualia_hint_expected_mismatch"] is False
    assert "qualia_axis_recovery_need" in plan.multimodal_cues


def test_response_planner_qualia_degraded_biases_surface_conservatively() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user is here"],
        memory_candidates=["user"],
        reportability_scores={"user": 0.72},
    )
    quiet = [0.0] * len(QUALIA_AXIS_LABELS)
    plan = render_response(
        foreground,
        DialogueContext(
            user_text="どうしよう",
            expression_hints={
                "qualia_state": {
                    "qualia": quiet,
                    "gate": quiet,
                    "precision": [0.1] * len(QUALIA_AXIS_LABELS),
                    "observability": [0.1] * len(QUALIA_AXIS_LABELS),
                    "body_coupling": [0.0] * len(QUALIA_AXIS_LABELS),
                    "value_grad": quiet,
                    "habituation": quiet,
                    "trust_applied": 0.22,
                    "degraded": True,
                    "reason": "high_nis",
                    "axis_labels": list(QUALIA_AXIS_LABELS),
                },
                "qualia_estimator_health": {
                    "trust": 0.22,
                    "degraded": True,
                    "reason": "high_nis",
                },
                "qualia_protection_grad_x": quiet,
                "qualia_axis_labels": list(QUALIA_AXIS_LABELS),
            },
        ),
    )
    assert plan.surface_profile["certainty_style"] == "careful"
    assert plan.surface_profile["opening_delay"] in {"measured", "long"}
    assert "qualia_degraded" in plan.multimodal_cues


def test_response_planner_recovery_first_biases_surface_to_wait_and_restore() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user is here"],
        memory_candidates=["user"],
        reportability_scores={"user": 0.72},
        affective_summary={
            "stress": 0.48,
            "recovery_need": 0.62,
            "safety_bias": 0.18,
            "stabilization_drive": 0.54,
        },
    )
    plan = render_response(
        foreground,
        DialogueContext(
            user_text="いまどう？",
            expression_hints={
                "qualia_planner_view": {
                    "trust": 0.58,
                    "degraded": False,
                    "dominant_axis": "body_load",
                    "dominant_value": 0.22,
                    "body_load": 0.18,
                    "protection_bias": 0.16,
                    "felt_energy": 0.12,
                },
                "terrain_readout": {
                    "approach_bias": 0.28,
                    "avoid_bias": 0.18,
                    "protect_bias": 0.26,
                    "active_patch_label": "soft_guard",
                },
                "protection_mode": {
                    "mode": "monitor",
                    "strength": 0.42,
                    "reasons": ["check_load"],
                },
            },
        ),
    )
    assert plan.interaction_policy["body_recovery_guard"]["state"] == "recovery_first"
    assert plan.surface_profile["opening_delay"] == "long"
    assert plan.surface_profile["response_length"] == "short"
    assert plan.surface_profile["certainty_style"] == "careful"
    assert "surface_recovery_first" in plan.surface_profile["cues"]


def test_response_planner_ready_initiative_biases_surface_forward_gently() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user is here"],
        memory_candidates=["user"],
        reportability_scores={"user": 0.72},
        affective_summary={
            "stress": 0.1,
            "recovery_need": 0.08,
            "recent_strain": 0.06,
            "safety_bias": 0.04,
            "stabilization_drive": 0.1,
        },
    )
    plan = render_response(
        foreground,
        DialogueContext(
            user_text="つぎに進める？",
            expression_hints={
                "qualia_planner_view": {
                    "trust": 0.86,
                    "degraded": False,
                    "dominant_axis": "shared_attention",
                    "dominant_value": 0.18,
                    "body_load": 0.02,
                    "protection_bias": 0.04,
                    "felt_energy": 0.2,
                },
                "terrain_readout": {
                    "approach_bias": 0.62,
                    "avoid_bias": 0.1,
                    "protect_bias": 0.12,
                    "active_patch_label": "open_step",
                },
                "protection_mode": {
                    "mode": "monitor",
                    "strength": 0.28,
                    "reasons": ["stable_contact"],
                },
            },
        ),
    )
    assert plan.interaction_policy["initiative_readiness"]["state"] == "ready"
    assert plan.surface_profile["response_length"] == "forward_leaning"
    assert plan.surface_profile["sentence_temperature"] == "warm"
    assert "surface_initiative_ready" in plan.surface_profile["cues"]


def test_response_planner_preserves_shared_qualia_view_when_present() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user is present"],
        memory_candidates=["user"],
        reportability_scores={"user": 0.7},
    )
    quiet = [0.0] * len(QUALIA_AXIS_LABELS)

    expression_hints = build_expression_hints_from_gate_result(
        {
            "qualia_state": {
                "qualia": quiet,
                "gate": quiet,
                "precision": [0.1] * len(QUALIA_AXIS_LABELS),
                "observability": [0.1] * len(QUALIA_AXIS_LABELS),
                "body_coupling": [0.0] * len(QUALIA_AXIS_LABELS),
                "value_grad": quiet,
                "habituation": quiet,
                "trust_applied": 0.1,
                "degraded": False,
                "reason": None,
                "axis_labels": list(QUALIA_AXIS_LABELS),
            },
            "qualia_estimator_health": {
                "trust": 0.1,
                "degraded": False,
                "reason": None,
            },
            "qualia_protection_grad_x": quiet,
            "qualia_axis_labels": list(QUALIA_AXIS_LABELS),
            "qualia_planner_view": {
                "trust": 0.93,
                "degraded": False,
                "dominant_axis": "recovery_need",
                "dominant_value": 0.61,
                "body_load": 0.18,
                "protection_bias": 0.0,
                "felt_energy": 0.24,
            },
        }
    )
    plan = render_response(
        foreground,
        DialogueContext(
            user_text="stay here",
            expression_hints=expression_hints,
        ),
    )

    assert plan.llm_payload["qualia_planner_view"]["dominant_axis"] == "recovery_need"
    assert plan.llm_payload["qualia_hint_source"] == "shared"
    assert plan.llm_payload["qualia_hint_fallback_reason"] == "prebuilt_shared_view"
    assert "qualia_axis_recovery_need" in plan.multimodal_cues


def test_expression_hint_bridge_is_idempotent_for_shared_qualia_view() -> None:
    quiet = [0.0] * len(QUALIA_AXIS_LABELS)
    initial = build_expression_hints_from_gate_result(
        {
            "qualia_state": {
                "qualia": quiet,
                "gate": quiet,
                "precision": [0.1] * len(QUALIA_AXIS_LABELS),
                "observability": [0.1] * len(QUALIA_AXIS_LABELS),
                "body_coupling": [0.0] * len(QUALIA_AXIS_LABELS),
                "value_grad": quiet,
                "habituation": quiet,
                "trust_applied": 0.1,
                "degraded": False,
                "reason": None,
                "axis_labels": list(QUALIA_AXIS_LABELS),
            },
            "qualia_estimator_health": {
                "trust": 0.1,
                "degraded": False,
                "reason": None,
            },
            "qualia_protection_grad_x": quiet,
            "qualia_axis_labels": list(QUALIA_AXIS_LABELS),
            "qualia_planner_view": {
                "trust": 0.93,
                "degraded": False,
                "dominant_axis": "recovery_need",
                "dominant_value": 0.61,
                "body_load": 0.18,
                "protection_bias": 0.0,
                "felt_energy": 0.24,
            },
        }
    )
    rebuilt = build_expression_hints_from_gate_result(initial, existing_hints=initial)

    assert rebuilt == initial
    assert rebuilt["qualia_hint_source"] == "shared"
    assert rebuilt["qualia_hint_version"] == QUALIA_HINT_VERSION
    assert rebuilt["qualia_hint_fallback_reason"] == "prebuilt_shared_view"


def test_expression_bridge_can_shift_to_check_in_for_person_specific_memory() -> None:
    foreground = ForegroundState(
        salient_entities=["user", "harbor slope"],
        reportable_facts=["user", "harbor slope"],
        memory_candidates=["user"],
        memory_reasons={"user": ["social", "affiliation", "continuity"]},
        continuity_focus=["person:user"],
        reportability_scores={"user": 0.76},
    )
    memory_context = build_memory_context(foreground, uncertainty=0.18, episode_prefix="turn")
    plan = render_response(foreground, DialogueContext(user_text="一緒にいて"), memory_context)
    assert plan.speech_act == "check_in"
    assert "gentle_turn_toward_partner" in plan.multimodal_cues
    assert plan.llm_payload["memory_context"]["related_person_ids"] == ["user"]
    assert plan.llm_payload["memory_context"]["relation_bias_strength"] > 0.0
    assert "partner_memory_soft_start" in plan.multimodal_cues
    assert plan.llm_payload["memory_context"]["partner_semantic_summary"].startswith("relation:user:")
    assert plan.llm_payload["utterance_stance"] == "gentle_check_in"
    assert plan.interaction_policy["response_strategy"] in {"attune_then_extend", "repair_then_attune", "shared_world_next_step", "respectful_wait", "reflect_without_settling"}
    assert plan.interaction_policy["dialogue_order"][0].startswith("open:")
    assert plan.llm_payload["scene_state"]["scene_family"] in {"attuned_presence", "repair_window", "shared_world", "reverent_distance", "co_present"}
    assert plan.llm_payload["interaction_option_candidates"]
    assert any(
        phrase in plan.llm_payload["content_skeleton"].lower()
        for phrase in (
            "one small part of this first",
            "part that feels easiest to name",
            "open the whole thing at once",
            "next step",
            "without settling the meaning",
            "without asking you to unpack it",
            "i'm here with you",
            "thread that is already here between us",
        )
    )
    assert plan.llm_payload["content_sequence"]
    assert plan.surface_profile["sentence_temperature"] in {"gentle", "warm", "measured", "neutral"}
    assert "surface_temperature_gentle" in plan.multimodal_cues or "surface_temperature_warm" in plan.multimodal_cues


def test_partner_recognition_can_bias_foreground_at_observation_time() -> None:
    observation = observe(
        {
            "entity_labels": ["visitor", "lamp"],
            "entity_attributes": [
                {"person_id_hint": "user"},
                {},
            ],
        }
    )
    world = update_world_state(WorldState(), observation, {})
    registry = person_registry_from_snapshot(
        {
            "persons": {
                "user": {
                    "person_id": "user",
                    "stable_traits": {"community_marker": 1.0},
                    "adaptive_traits": {
                        "attachment": 0.82,
                        "familiarity": 0.78,
                        "trust_memory": 0.8,
                        "continuity_score": 0.73,
                        "social_grounding": 0.68,
                    },
                    "continuity_history": [{"observation": "user:social", "ambiguity": "0.18"}],
                    "confidence": 0.88,
                    "ambiguity_flag": False,
                }
            },
            "uncertainty": 0.18,
        }
    )
    value = compute_value_state(world, SelfState(fatigue=0.2), registry)
    foreground = select_foreground(
        world,
        SelfState(fatigue=0.2),
        value,
        AttentionState(continuity_bias=0.9),
        registry,
    )
    assert foreground.candidates[0].entity_id == "obs:0"
    assert "partner-trace" in foreground.candidates[0].reasons
    assert foreground.continuity_focus == ["person:user"]
    assert "obs:0" in foreground.memory_candidates


def test_grounding_can_shift_affordance_and_symbol_stance_by_partner_context() -> None:
    observation = observe(
        {
            "entity_labels": ["visitor"],
            "entity_attributes": [{"person_id_hint": "user"}],
        }
    )
    world = update_world_state(WorldState(), observation, {})
    open_context = {
        "user": {
            "stable_traits": {
                "community_marker": 1.0,
                "culture_marker": 0.9,
                "role_marker": 0.8,
            },
            "affiliation_bias": 0.76,
            "caution_bias": 0.18,
            "community_bias": 0.82,
            "culture_bias": 0.78,
        }
    }
    guarded_context = {
        "user": {
            "stable_traits": {
                "community_marker": 1.0,
                "culture_marker": 0.9,
                "role_marker": 0.8,
            },
            "affiliation_bias": 0.24,
            "caution_bias": 0.84,
            "community_bias": 0.2,
            "culture_bias": 0.18,
        }
    }

    open_affordances = infer_affordances(observation, world, SelfState(social_tension=0.1), open_context)
    guarded_affordances = infer_affordances(observation, world, SelfState(social_tension=0.66), guarded_context)
    open_grounded = ground_symbols(["visitor"], observation, open_affordances, compute_value_state(world, SelfState(), PersonRegistry()), open_context)
    guarded_grounded = ground_symbols(["visitor"], observation, guarded_affordances, compute_value_state(world, SelfState(), PersonRegistry()), guarded_context)
    open_grounding = summarize_partner_grounding(observation, open_affordances, open_grounded)

    open_engage = next(item for item in open_affordances["obs:0"] if item.action == "engage")
    guarded_engage = next(item for item in guarded_affordances["obs:0"] if item.action == "engage")
    assert open_engage.context_tags["stance_hint"] == "familiar"
    assert open_engage.context_tags["timing_hint"] == "open"
    assert guarded_engage.context_tags["timing_hint"] == "delayed"
    assert "wait_for_social_timing" in guarded_engage.constraints
    assert open_grounded["visitor"].context_tags["address_hint"] == "companion"
    assert guarded_grounded["visitor"].context_tags["address_hint"] == "respectful"
    foreground = ForegroundState(
        salient_entities=["obs:0"],
        reportable_facts=["obs:0"],
        memory_candidates=["obs:0"],
        memory_reasons={"obs:0": ["social", "continuity"]},
        continuity_focus=["person:user"],
        reportability_scores={"obs:0": 0.76},
    )
    memory_context = build_memory_context(
        foreground,
        uncertainty=0.18,
        episode_prefix="turn",
        grounding_context=open_grounding,
    )
    plan = render_response(foreground, DialogueContext(user_text="縺薙ｓ縺ｫ縺｡縺ｯ"), memory_context)
    assert plan.llm_payload["memory_context"]["partner_address_hint"] == "companion"
    assert plan.llm_payload["memory_context"]["partner_timing_hint"] == "open"
    assert plan.llm_payload["memory_context"]["partner_stance_hint"] == "familiar"
    assert plan.llm_payload["utterance_stance"] == "warm_check_in"
    assert plan.interaction_policy["attention_target"] == "person:user"
    assert plan.interaction_policy["memory_write_priority"] == "relation_episode"
    assert plan.action_posture["engagement_mode"] in {"attune", "co_move"}
    assert plan.action_posture["outcome_goal"] in {"increase_safe_contact", "shared_progress"}
    assert plan.actuation_plan["execution_mode"] in {"attuned_contact", "shared_progression"}
    assert plan.actuation_plan["primary_action"] in {"hold_presence", "co_move"}
    assert plan.llm_payload["action_posture"]["attention_target"] == "person:user"
    assert plan.llm_payload["action_posture"]["memory_write_priority"] == "relation_episode"
    assert plan.llm_payload["actuation_plan"]["attention_target"] == "person:user"
    assert plan.llm_payload["actuation_plan"]["memory_write_priority"] == "relation_episode"
    assert plan.llm_payload["nonverbal_profile"]["gaze_mode"] == "shared_attention_hold"
    assert plan.llm_payload["nonverbal_profile"]["pause_mode"] in {"short_warm", "patient_care", "confident_brief"}
    assert plan.llm_payload["nonverbal_profile"]["relational_mood"]["future_pull"] > 0.0
    assert plan.llm_payload["nonverbal_profile"]["relational_mood"]["shared_world_pull"] > 0.0
    assert plan.llm_payload["nonverbal_profile"]["live_regulation"]["future_loop_pull"] > 0.0
    assert plan.llm_payload["nonverbal_profile"]["live_regulation"]["distance_expectation"] in {"future_opening", "gentle_near", "holding_space", "respectful_distance"}
    assert plan.llm_payload["nonverbal_profile"]["interaction_orchestration"]["orchestration_mode"] in {"attune", "advance", "repair", "reflect", "contain"}
    assert plan.llm_payload["nonverbal_profile"]["interaction_orchestration"]["human_presence_signal"] >= 0.0
    assert plan.surface_profile["opening_delay"] in {"brief", "measured", "long"}
    assert plan.surface_profile["response_length"] in {"short", "balanced", "reflective", "forward_leaning"}
    assert plan.llm_payload["surface_profile"]["certainty_style"] in {"direct", "tentative", "careful"}
    assert "partner_address_companion" in plan.multimodal_cues
    assert "partner_timing_open" in plan.multimodal_cues
    assert "partner_stance_familiar" in plan.multimodal_cues
    assert "gaze_shared_attention_hold" in plan.multimodal_cues
    assert any(cue.startswith("pause_") for cue in plan.multimodal_cues)
    assert "shared_world_orientation" in plan.multimodal_cues
    assert any(cue in plan.multimodal_cues for cue in {"future_loop_active", "fantasy_loop_active", "past_loop_active"})
    assert any(cue.startswith("orchestration_") for cue in plan.multimodal_cues)
    assert any(cue.startswith("surface_") for cue in plan.multimodal_cues)
    assert any(cue.startswith("action_") for cue in plan.multimodal_cues)
    assert any(cue.startswith("actuation_") for cue in plan.multimodal_cues)
