from __future__ import annotations

from inner_os.schemas import INNER_OS_TRANSFER_PACKAGE_SCHEMA, transfer_package_contract
from inner_os.transfer_package import InnerOSTransferPackageBuilder


def test_transfer_package_builder_keeps_portable_state_and_runtime_seed() -> None:
    builder = InnerOSTransferPackageBuilder()

    package = builder.build(
        session_id="session-1",
        turn_id="turn-9",
        timestamp_ms=1234,
        current_state={
            "terrain_reweighting_bias": 0.21,
            "association_reweighting_bias": 0.17,
            "association_reweighting_focus": "repeated_links",
            "association_reweighting_reason": "repeated_insight_trace",
            "insight_reframing_bias": 0.19,
            "insight_class_focus": "reframed_relation",
            "insight_terrain_shape_target": "repair_basin",
            "initiative_followup_bias": 0.28,
            "initiative_followup_state": "reopen_softly",
            "commitment_target_focus": "repair",
            "commitment_state_focus": "commit",
            "commitment_carry_bias": 0.37,
            "commitment_followup_focus": "reopen_softly",
            "commitment_mode_focus": "repair",
            "commitment_carry_reason": "commit:repair",
            "agenda_focus": "repair",
            "agenda_bias": 0.22,
            "agenda_reason": "repair_window",
            "agenda_window_focus": "next_private_window",
            "agenda_window_bias": 0.18,
            "agenda_window_reason": "wait_for_private_window",
            "learning_mode_focus": "repair_probe",
            "learning_mode_carry_bias": 0.15,
            "social_experiment_focus": "repair_signal_probe",
            "social_experiment_carry_bias": 0.13,
            "temporal_membrane_focus": "coherent_reentry",
            "temporal_timeline_bias": 0.27,
            "temporal_reentry_bias": 0.21,
            "temporal_supersession_bias": 0.06,
            "temporal_continuity_bias": 0.19,
            "temporal_relation_reentry_bias": 0.17,
            "autobiographical_thread_mode": "unfinished_thread",
            "autobiographical_thread_anchor": "harbor promise",
            "autobiographical_thread_focus": "unfinished promise",
            "autobiographical_thread_strength": 0.43,
            "body_homeostasis_focus": "recovering",
            "body_homeostasis_carry_bias": 0.16,
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.09,
            "relational_continuity_focus": "reopening",
            "relational_continuity_carry_bias": 0.14,
            "expressive_style_focus": "quiet_repair",
            "expressive_style_carry_bias": 0.09,
            "expressive_style_history_focus": "warm_companion",
            "expressive_style_history_bias": 0.08,
            "banter_style_focus": "gentle_tease",
            "lexical_variation_carry_bias": 0.11,
            "group_thread_focus": "threaded_group",
            "group_thread_carry_bias": 0.12,
            "temperament_forward_trace": 0.24,
            "temperament_guard_trace": 0.18,
            "temperament_bond_trace": 0.31,
            "temperament_recovery_trace": 0.15,
            "temperament_forward_bias": 0.11,
            "temperament_guard_bias": 0.03,
            "temperament_bond_bias": 0.05,
            "temperament_recovery_bias": 0.02,
        },
        last_gate_context={
            "inner_os_prev_qualia": [0.2, 0.6],
            "inner_os_prev_qualia_habituation": [0.1, 0.4],
            "inner_os_prev_protection_grad_x": [0.3, 0.2],
            "inner_os_prev_affective_position": {"z_aff": [0.4, 0.2], "confidence": 0.66},
            "inner_os_affective_terrain_state": {"patches": [{"patch_id": "p0", "value": 0.31}]},
            "inner_os_association_graph_state": {"links": [{"link_key": "a->b", "weight": 0.44}]},
        },
        persona_meta_inner_os={
            "llm_model": "qwen-3.5-instruct",
            "llm_model_source": "live_list",
            "memory_anchor": "harbor slope",
            "semantic_seed_focus": "harbor slope",
            "semantic_seed_anchor": "harbor slope",
            "semantic_seed_strength": 0.73,
            "semantic_seed_recurrence": 1.22,
            "long_term_theme_focus": "quiet harbor routine",
            "long_term_theme_anchor": "harbor slope",
            "long_term_theme_kind": "place",
            "long_term_theme_summary": "quiet harbor routine",
            "long_term_theme_strength": 0.61,
            "identity_arc_kind": "repairing_bond",
            "identity_arc_phase": "shifting",
            "identity_arc_summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
            "identity_arc_open_tension": "timing_sensitive_reentry",
            "identity_arc_stability": 0.58,
            "relation_seed_summary": "harbor trust line",
            "relation_seed_strength": 0.58,
            "monument_salience": 0.47,
            "monument_kind": "shared_ritual",
            "workspace_decision": {"workspace_mode": "meaning", "winner_margin": 0.42},
            "protection_mode_decision": {"mode": "repair", "winner_margin": 0.36, "dominant_inputs": ["bond_drive"]},
            "memory_write_class_bias": {"selected_class": "repair_trace", "winner_margin": 0.29},
            "body_recovery_guard": {"state": "guarded", "score": 0.41},
            "body_homeostasis_state": {"state": "recovering", "score": 0.46, "winner_margin": 0.11},
            "social_topology_state": {"state": "threaded_group", "score": 0.52, "winner_margin": 0.1},
            "initiative_readiness": {"state": "tentative", "score": 0.38},
            "commitment_state": {"state": "commit", "target": "repair", "accepted_cost": 0.33},
            "expressive_style_state": {"state": "quiet_repair", "lightness_room": 0.24, "continuity_weight": 0.61},
            "relational_style_memory_state": {
                "state": "warm_companion",
                "playful_ceiling": 0.46,
                "advice_tolerance": 0.54,
                "lexical_variation_bias": 0.36,
                "banter_room": 0.32,
                "banter_style": "gentle_tease",
            },
            "relational_continuity_state": {"state": "reopening", "score": 0.54, "winner_margin": 0.13},
            "initiative_followup_bias": {"state": "reopen_softly", "score": 0.31},
            "overnight_bias_roles": {"association": "repeated_links"},
            "reaction_vs_overnight_bias": {"same_turn": {"protection_mode": "repair"}},
            "identity_arc": {
                "arc_kind": "repairing_bond",
                "phase": "shifting",
                "summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
                "open_tension": "timing_sensitive_reentry",
                "stability": 0.58,
                "memory_anchor": "harbor slope",
                "related_person_id": "person:harbor",
                "group_thread_focus": "threaded_group:person:friend|person:harbor",
                "long_term_theme_kind": "place",
                "long_term_theme_focus": "quiet harbor routine",
                "learning_mode_focus": "repair_probe",
                "social_experiment_focus": "repair_signal_probe",
            },
            "identity_arc_registry_summary": {
                "dominant_arc_id": "repairing_bond::person:harbor::place::quiet_harbor_routine",
                "dominant_arc_kind": "repairing_bond",
                "dominant_arc_phase": "shifting",
                "dominant_arc_summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
                "active_arc_count": 1,
                "total_arcs": 1,
                "top_arc_ids": ["repairing_bond::person:harbor::place::quiet_harbor_routine"],
                "status_counts": {"active": 1},
            },
            "relation_arc_kind": "repairing_relation",
            "relation_arc_phase": "shifting",
            "relation_arc_summary": "repair is gathering around a companion thread",
            "relation_arc_open_tension": "timing_sensitive_reentry",
            "relation_arc_stability": 0.56,
            "group_relation_arc_kind": "repairing_relation",
            "group_relation_arc_phase": "shifting",
            "group_relation_arc_summary": "repair is moving through a shared group thread in small steps",
            "group_relation_arc_boundary_mode": "same_group_reentry",
            "group_relation_arc_reentry_window_focus": "next_same_group_window",
            "group_relation_arc_group_thread_id": "threaded_group:person:friend|person:harbor",
            "group_relation_arc_topology_focus": "threaded_group",
            "group_relation_arc_dominant_person_id": "person:harbor",
            "group_relation_arc_stability": 0.61,
            "relation_arc_registry_summary": {
                "dominant_arc_id": "repairing_relation::person:harbor::companion",
                "dominant_arc_kind": "repairing_relation",
                "dominant_arc_phase": "shifting",
                "dominant_arc_summary": "repair is gathering around a companion thread",
                "dominant_person_id": "person:harbor",
                "dominant_group_thread_id": "threaded_group:person:friend|person:harbor",
                "active_arc_count": 1,
                "total_arcs": 1,
                "top_arc_ids": ["repairing_relation::person:harbor::companion"],
                "status_counts": {"active": 1},
            },
            "related_person_id": "person:harbor",
            "attachment": 0.61,
            "familiarity": 0.57,
            "trust_memory": 0.49,
            "relation_seed_summary": "harbor trust line",
            "partner_address_hint": "gentle",
            "partner_timing_hint": "slow",
            "partner_stance_hint": "careful",
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.09,
            "person_registry_snapshot": {
                "dominant_person_id": "person:harbor",
                "top_person_ids": ["person:harbor", "person:friend"],
                "total_people": 2,
            },
            "group_thread_registry_snapshot": {
                "dominant_thread_id": "threaded_group:person:friend|person:harbor",
                "top_thread_ids": ["threaded_group:person:friend|person:harbor"],
                "total_threads": 1,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "top_thread_ids": ["repair_anchor"],
                "total_threads": 1,
            },
        },
        response_meta={
            "model": "qwen-3.5-instruct",
            "model_source": "live_list",
            "trace_id": "turn-9",
        },
        nightly_summary={
            "inner_os_daily_carry_summary": {
                "same_turn_focus": {"memory_class": "repair_trace"},
                "overnight_focus": {"commitment_target_focus": "repair"},
                "carry_strengths": {"commitment_carry": 0.37},
            }
        },
    ).to_dict()

    assert package["schema"] == INNER_OS_TRANSFER_PACKAGE_SCHEMA
    assert package["source_model"]["source"] == "live_list"
    assert package["portable_state"]["same_turn"]["protection_mode"]["mode"] == "repair"
    assert package["portable_state"]["same_turn"]["body_homeostasis_state"]["state"] == "recovering"
    assert package["portable_state"]["same_turn"]["homeostasis_budget_state"]["winner_margin"] >= 0.0
    assert package["portable_state"]["same_turn"]["social_topology_state"]["state"] == "threaded_group"
    assert package["portable_state"]["same_turn"]["expressive_style_state"]["state"] == "quiet_repair"
    assert package["portable_state"]["same_turn"]["relational_style_memory_state"]["banter_style"] == "gentle_tease"
    assert package["portable_state"]["same_turn"]["relational_continuity_state"]["state"] == "reopening"
    assert package["portable_state"]["carry"]["daily_carry_summary"]["overnight_focus"]["commitment_target_focus"] == "repair"
    assert package["portable_state"]["carry"]["learning_mode_carry"]["focus"] == "repair_probe"
    assert package["portable_state"]["carry"]["social_experiment_carry"]["focus"] == "repair_signal_probe"
    assert package["portable_state"]["carry"]["temporal_membrane"]["focus"] == "coherent_reentry"
    assert package["portable_state"]["carry"]["temporal_membrane"]["reentry_bias"] == 0.21
    assert package["portable_state"]["carry"]["autobiographical_thread"]["mode"] == "unfinished_thread"
    assert package["portable_state"]["carry"]["autobiographical_thread"]["anchor"] == "harbor promise"
    assert package["portable_state"]["carry"]["autobiographical_thread"]["strength"] == 0.43
    assert package["portable_state"]["carry"]["style_history"]["expressive_style_history_focus"] == "warm_companion"
    assert package["portable_state"]["carry"]["style_history"]["banter_style_focus"] == "gentle_tease"
    assert package["portable_state"]["carry"]["identity_arc"]["arc_kind"] == "repairing_bond"
    assert package["portable_state"]["carry"]["identity_arc"]["learning_mode_focus"] == "repair_probe"
    assert package["portable_state"]["carry"]["identity_arc_registry_summary"]["dominant_arc_kind"] == "repairing_bond"
    assert package["portable_state"]["carry"]["relation_arc"]["arc_kind"] == "repairing_relation"
    assert package["portable_state"]["carry"]["relation_arc"]["summary"] == "repair is gathering around a companion thread"
    assert package["portable_state"]["carry"]["relation_arc_registry_summary"]["dominant_arc_kind"] == "repairing_relation"
    assert package["portable_state"]["carry"]["group_relation_arc"]["boundary_mode"] == "same_group_reentry"
    assert package["portable_state"]["carry"]["monument_carry"]["memory_anchor"] == "harbor slope"
    assert package["portable_state"]["carry"]["monument_carry"]["monument_kind"] == "shared_ritual"
    assert package["runtime_seed"]["commitment_state_focus"] == "commit"
    assert package["runtime_seed"]["agenda_focus"] == "repair"
    assert package["runtime_seed"]["agenda_bias"] == 0.22
    assert package["runtime_seed"]["agenda_window_focus"] == "next_private_window"
    assert package["runtime_seed"]["agenda_window_bias"] == 0.18
    assert package["runtime_seed"]["learning_mode_focus"] == "repair_probe"
    assert package["runtime_seed"]["learning_mode_carry_bias"] == 0.15
    assert package["runtime_seed"]["social_experiment_focus"] == "repair_signal_probe"
    assert package["runtime_seed"]["social_experiment_carry_bias"] == 0.13
    assert package["runtime_seed"]["temporal_membrane_focus"] == "coherent_reentry"
    assert package["runtime_seed"]["temporal_timeline_bias"] == 0.27
    assert package["runtime_seed"]["temporal_reentry_bias"] == 0.21
    assert package["runtime_seed"]["identity_arc_kind"] == "repairing_bond"
    assert package["runtime_seed"]["identity_arc_phase"] == "shifting"
    assert package["runtime_seed"]["identity_arc_registry_summary"]["total_arcs"] == 1
    assert package["runtime_seed"]["relation_arc_kind"] == "repairing_relation"
    assert package["runtime_seed"]["relation_arc_phase"] == "shifting"
    assert package["runtime_seed"]["relation_arc_registry_summary"]["total_arcs"] == 1
    assert package["runtime_seed"]["group_relation_arc_kind"] == "repairing_relation"
    assert package["runtime_seed"]["group_relation_arc_boundary_mode"] == "same_group_reentry"
    assert package["runtime_seed"]["group_relation_arc_group_thread_id"] == "threaded_group:person:friend|person:harbor"
    assert package["runtime_seed"]["body_homeostasis_focus"] == "recovering"
    assert package["runtime_seed"]["homeostasis_budget_focus"] == "recovering"
    assert package["runtime_seed"]["relational_continuity_focus"] == "reopening"
    assert package["runtime_seed"]["expressive_style_focus"] == "quiet_repair"
    assert package["runtime_seed"]["expressive_style_history_focus"] == "warm_companion"
    assert package["runtime_seed"]["banter_style_focus"] == "gentle_tease"
    assert package["portable_state"]["carry"]["relationship_registry_summary"]["total_people"] == 2
    assert package["portable_state"]["carry"]["group_thread_registry_summary"]["total_threads"] == 1
    assert package["portable_state"]["carry"]["discussion_thread_registry_summary"]["dominant_anchor"] == "repair anchor"
    assert package["runtime_seed"]["group_thread_focus"] == "threaded_group"
    assert package["runtime_seed"]["group_thread_carry_bias"] == 0.12
    assert package["runtime_seed"]["discussion_thread_registry_snapshot"]["dominant_issue_state"] == "pausing_issue"
    assert package["runtime_seed"]["autobiographical_thread_mode"] == "unfinished_thread"
    assert package["runtime_seed"]["autobiographical_thread_anchor"] == "harbor promise"
    assert package["runtime_seed"]["autobiographical_thread_strength"] == 0.43
    assert package["runtime_seed"]["prev_qualia"] == [0.2, 0.6]


def test_transfer_package_builder_restores_runtime_seed_shape() -> None:
    builder = InnerOSTransferPackageBuilder()

    seed = builder.to_runtime_seed(
        {
            "runtime_seed": {
                "prev_qualia": [0.4],
                "prev_qualia_habituation": [0.1],
                "prev_protection_grad_x": [0.2],
                "prev_affective_position": {"z_aff": [0.3, 0.7]},
                "affective_terrain_state": {"patches": []},
                "association_graph_state": {"links": []},
                "terrain_reweighting_bias": 0.18,
                "association_reweighting_bias": 0.22,
                "association_reweighting_focus": "repeated_links",
                "association_reweighting_reason": "nightly_repeat",
                "insight_reframing_bias": 0.15,
                "insight_class_focus": "reframed_relation",
                "insight_terrain_shape_target": "repair_basin",
                "initiative_followup_bias": 0.27,
                "initiative_followup_state": "offer_next_step",
                "commitment_target_focus": "step_forward",
                "commitment_state_focus": "settle",
                "commitment_carry_bias": 0.25,
                "commitment_followup_focus": "offer_next_step",
                "commitment_mode_focus": "monitor",
                "commitment_carry_reason": "settle:step_forward",
                "agenda_focus": "step_forward",
                "agenda_bias": 0.24,
                "agenda_reason": "offer_next_step",
                "agenda_window_focus": "now",
                "agenda_window_bias": 0.16,
                "agenda_window_reason": "step_forward_now",
                "temporal_membrane_focus": "coherent_reentry",
                "temporal_timeline_bias": 0.23,
                "temporal_reentry_bias": 0.18,
                "temporal_supersession_bias": 0.05,
                "temporal_continuity_bias": 0.16,
                "temporal_relation_reentry_bias": 0.14,
                "autobiographical_thread_mode": "unfinished_thread",
                "autobiographical_thread_anchor": "harbor promise",
                "autobiographical_thread_focus": "unfinished promise",
                "autobiographical_thread_strength": 0.39,
                "identity_arc_kind": "growing_edge",
                "identity_arc_phase": "holding",
                "identity_arc_summary": "a next step is starting to hold shape / phase=holding",
                "identity_arc_open_tension": "timing_sensitive_reentry",
                "identity_arc_stability": 0.41,
                "identity_arc_registry_summary": {
                    "dominant_arc_kind": "growing_edge",
                    "active_arc_count": 1,
                    "total_arcs": 1,
                },
                "relation_arc_kind": "group_thread_continuity",
                "relation_arc_phase": "holding",
                "relation_arc_summary": "the same group thread is being kept warm for a later return",
                "relation_arc_open_tension": "timing_sensitive_reentry",
                "relation_arc_stability": 0.39,
                "relation_arc_registry_summary": {
                    "dominant_arc_kind": "group_thread_continuity",
                    "active_arc_count": 1,
                    "total_arcs": 1,
                },
                "group_relation_arc_kind": "group_thread_continuity",
                "group_relation_arc_phase": "holding",
                "group_relation_arc_summary": "the same group thread is being kept available for a later return",
                "group_relation_arc_boundary_mode": "same_group_reentry",
                "group_relation_arc_reentry_window_focus": "next_same_group_window",
                "group_relation_arc_group_thread_id": "threaded_group:person:friend|person:harbor",
                "group_relation_arc_topology_focus": "threaded_group",
                "group_relation_arc_dominant_person_id": "person:harbor",
                "group_relation_arc_stability": 0.43,
                "body_homeostasis_focus": "recovering",
                "body_homeostasis_carry_bias": 0.13,
                "homeostasis_budget_focus": "recovering",
                "homeostasis_budget_bias": 0.08,
                "relational_continuity_focus": "holding_thread",
                "relational_continuity_carry_bias": 0.09,
                "expressive_style_focus": "warm_companion",
                "expressive_style_carry_bias": 0.08,
                "expressive_style_history_focus": "warm_companion",
                "expressive_style_history_bias": 0.07,
                "banter_style_focus": "gentle_tease",
                "lexical_variation_carry_bias": 0.1,
                "person_registry_snapshot": {
                    "dominant_person_id": "person:harbor",
                    "top_person_ids": ["person:harbor", "person:friend"],
                    "total_people": 2,
                },
                "group_thread_registry_snapshot": {
                    "dominant_thread_id": "threaded_group:person:friend|person:harbor",
                    "top_thread_ids": ["threaded_group:person:friend|person:harbor"],
                    "total_threads": 1,
                },
                "group_thread_focus": "threaded_group",
                "group_thread_carry_bias": 0.12,
                "temperament_forward_trace": 0.32,
                "temperament_guard_trace": 0.11,
                "temperament_bond_trace": 0.08,
                "temperament_recovery_trace": 0.05,
                "temperament_forward_bias": 0.12,
                "temperament_guard_bias": 0.03,
                "temperament_bond_bias": 0.02,
                "temperament_recovery_bias": 0.01,
            }
        }
    )

    assert seed["initiative_followup_state"] == "offer_next_step"
    assert seed["commitment_target_focus"] == "step_forward"
    assert seed["agenda_focus"] == "step_forward"
    assert seed["agenda_bias"] == 0.24
    assert seed["agenda_window_focus"] == "now"
    assert seed["agenda_window_bias"] == 0.16
    assert seed["temporal_membrane_focus"] == "coherent_reentry"
    assert seed["temporal_timeline_bias"] == 0.23
    assert seed["temporal_reentry_bias"] == 0.18
    assert seed["autobiographical_thread_mode"] == "unfinished_thread"
    assert seed["autobiographical_thread_anchor"] == "harbor promise"
    assert seed["autobiographical_thread_strength"] == 0.39
    assert seed["identity_arc_kind"] == "growing_edge"
    assert seed["identity_arc_phase"] == "holding"
    assert seed["identity_arc_registry_summary"]["dominant_arc_kind"] == "growing_edge"
    assert seed["relation_arc_kind"] == "group_thread_continuity"
    assert seed["relation_arc_phase"] == "holding"
    assert seed["relation_arc_registry_summary"]["dominant_arc_kind"] == "group_thread_continuity"
    assert seed["group_relation_arc_kind"] == "group_thread_continuity"
    assert seed["group_relation_arc_boundary_mode"] == "same_group_reentry"
    assert seed["group_relation_arc_group_thread_id"] == "threaded_group:person:friend|person:harbor"
    assert seed["body_homeostasis_focus"] == "recovering"
    assert seed["homeostasis_budget_focus"] == "recovering"
    assert seed["relational_continuity_focus"] == "holding_thread"
    assert seed["expressive_style_focus"] == "warm_companion"
    assert seed["expressive_style_history_focus"] == "warm_companion"
    assert seed["banter_style_focus"] == "gentle_tease"
    assert seed["person_registry_snapshot"]["total_people"] == 2
    assert seed["group_thread_registry_snapshot"]["total_threads"] == 1
    assert seed["group_thread_focus"] == "threaded_group"
    assert seed["group_thread_carry_bias"] == 0.12
    assert seed["temperament_forward_trace"] == 0.32
    assert seed["association_reweighting_focus"] == "repeated_links"


def test_transfer_package_contract_exposes_required_fields() -> None:
    contract = transfer_package_contract()

    assert contract["schema"] == INNER_OS_TRANSFER_PACKAGE_SCHEMA
    assert "portable_state" in contract["required_fields"]
    assert "runtime_seed" in contract["required_fields"]


def test_transfer_package_builder_normalizes_legacy_payload_into_v1_shape() -> None:
    builder = InnerOSTransferPackageBuilder()

    normalized = builder.normalize(
        {
            "source_model": "qwen-legacy",
            "working_memory_seed": {
                "semantic_seed_focus": "harbor slope",
                "semantic_seed_anchor": "harbor slope",
                "semantic_seed_strength": 0.71,
                "long_term_theme_focus": "quiet harbor routine",
                "long_term_theme_summary": "quiet harbor routine",
                "relation_seed_summary": "shared harbor thread",
                "related_person_id": "person:harbor",
                "attachment": 0.61,
            },
            "daily_carry_summary": {
                "overnight_focus": {"commitment_target_focus": "repair"},
            },
            "prev_qualia": [0.4, 0.2],
            "commitment_state_focus": "commit",
            "agenda_focus": "repair",
            "agenda_bias": 0.19,
            "agenda_reason": "repair_window",
            "agenda_window_focus": "next_same_group_window",
            "agenda_window_bias": 0.17,
            "agenda_window_reason": "wait_for_group_thread",
            "temporal_membrane_focus": "same_group_reentry",
            "temporal_timeline_bias": 0.22,
            "temporal_reentry_bias": 0.19,
            "temporal_supersession_bias": 0.04,
            "temporal_continuity_bias": 0.18,
            "temporal_relation_reentry_bias": 0.15,
            "identity_arc_summary": {
                "arc_kind": "holding_thread",
                "phase": "holding",
                "summary": "a relational thread is being kept warm without forcing reentry / phase=holding",
                "memory_anchor": "harbor slope",
                "stability": 0.46,
            },
            "identity_arc_registry_summary": {
                "dominant_arc_kind": "holding_thread",
                "active_arc_count": 1,
                "total_arcs": 1,
            },
            "homeostasis_budget_focus": "recovering",
            "homeostasis_budget_bias": 0.07,
            "person_registry_snapshot": {
                "dominant_person_id": "person:harbor",
                "top_person_ids": ["person:harbor"],
                "total_people": 1,
            },
            "group_thread_registry_snapshot": {
                "dominant_thread_id": "threaded_group:person:harbor",
                "top_thread_ids": ["threaded_group:person:harbor"],
                "total_threads": 1,
            },
            "group_thread_focus": "threaded_group",
            "group_thread_carry_bias": 0.1,
        }
    )

    assert normalized["schema"] == INNER_OS_TRANSFER_PACKAGE_SCHEMA
    assert normalized["package_version"] == "v1"
    assert normalized["migration"]["applied"] is True
    assert normalized["source_model"]["name"] == "qwen-legacy"
    assert normalized["portable_state"]["carry"]["monument_carry"]["semantic_seed_anchor"] == "harbor slope"
    assert normalized["portable_state"]["carry"]["relationship_summary"]["related_person_id"] == "person:harbor"
    assert normalized["portable_state"]["carry"]["daily_carry_summary"]["overnight_focus"]["commitment_target_focus"] == "repair"
    assert normalized["runtime_seed"]["prev_qualia"] == [0.4, 0.2]
    assert normalized["runtime_seed"]["commitment_state_focus"] == "commit"
    assert normalized["runtime_seed"]["agenda_focus"] == "repair"
    assert normalized["runtime_seed"]["agenda_bias"] == 0.19
    assert normalized["runtime_seed"]["agenda_window_focus"] == "next_same_group_window"
    assert normalized["runtime_seed"]["agenda_window_bias"] == 0.17
    assert normalized["portable_state"]["carry"]["temporal_membrane"]["focus"] == "same_group_reentry"
    assert normalized["runtime_seed"]["temporal_membrane_focus"] == "same_group_reentry"
    assert normalized["runtime_seed"]["temporal_reentry_bias"] == 0.19
    assert normalized["portable_state"]["carry"]["identity_arc"]["arc_kind"] == "holding_thread"
    assert normalized["portable_state"]["carry"]["identity_arc_registry_summary"]["dominant_arc_kind"] == "holding_thread"
    assert normalized["runtime_seed"]["identity_arc_kind"] == "holding_thread"
    assert normalized["runtime_seed"]["identity_arc_registry_summary"]["dominant_arc_kind"] == "holding_thread"
    assert normalized["runtime_seed"]["homeostasis_budget_focus"] == "recovering"
    assert normalized["portable_state"]["carry"]["relationship_registry_summary"]["dominant_person_id"] == "person:harbor"
    assert normalized["portable_state"]["carry"]["group_thread_registry_summary"]["dominant_thread_id"] == "threaded_group:person:harbor"
    assert normalized["runtime_seed"]["group_thread_focus"] == "threaded_group"
