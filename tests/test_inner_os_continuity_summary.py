from inner_os.continuity_summary import ContinuitySummaryBuilder


def test_continuity_summary_includes_temporal_membrane_same_turn_and_overnight_fields() -> None:
    summary = ContinuitySummaryBuilder().build(
        interaction_policy_packet={
            "protection_mode": {"mode": "repair"},
            "memory_write_class": "repair_trace",
            "agenda_state": {"state": "repair", "reason": "repair_window"},
            "agenda_window_state": {
                "state": "next_private_window",
                "reason": "wait_for_private_window",
                "carry_target": "same_person_private_window",
                "deferral_budget": 0.34,
            },
            "learning_mode_state": {"state": "repair_probe", "probe_room": 0.42},
            "social_experiment_loop_state": {"state": "repair_signal_probe", "probe_intensity": 0.38},
            "live_engagement_state": {"state": "pickup_comment", "score": 0.54, "primary_move": "pick_up_comment"},
            "commitment_state": {"target": "repair"},
            "relation_competition_state": {"state": "single_relation", "total_people": 1},
            "active_relation_table": {"total_people": 1},
            "reaction_vs_overnight_bias": {
                "same_turn": {},
                "overnight": {
                    "temporal_membrane_focus": "reentry",
                    "temporal_reentry_bias": 0.16,
                    "temporal_timeline_bias": 0.12,
                },
            },
        },
        current_state={
            "temporal_membrane_mode": "reentry",
            "temporal_timeline_coherence": 0.44,
            "temporal_reentry_pull": 0.58,
            "temporal_supersession_pressure": 0.08,
            "temporal_continuity_pressure": 0.36,
            "temporal_relation_reentry_pull": 0.41,
            "temporal_membrane_focus": "reentry",
            "temporal_timeline_bias": 0.12,
            "temporal_reentry_bias": 0.16,
            "identity_arc_kind": "repairing_bond",
            "growth_state": {
                "relational_trust": 0.63,
                "epistemic_maturity": 0.48,
                "expressive_range": 0.57,
                "residue_integration": 0.51,
                "playfulness_range": 0.36,
                "self_coherence": 0.59,
                "dominant_transition": "relational_trust",
            },
            "growth_replay_axes": {
                "bond": {"value": 0.62, "delta": 0.04},
                "stability": {"value": 0.55, "delta": 0.03},
                "curiosity": {"value": 0.41, "delta": 0.02},
            },
            "epistemic_state": {
                "freshness": 0.74,
                "source_confidence": 0.69,
                "verification_pressure": 0.28,
                "change_likelihood": 0.34,
                "stale_risk": 0.29,
                "epistemic_caution": 0.33,
                "dominant_posture": "carry_forward",
            },
            "epistemic_packet_axes": {
                "grounding": {"value": 0.72, "delta": 0.05},
                "volatility": {"value": 0.31, "delta": 0.02},
                "verification": {"value": 0.3, "delta": -0.01},
            },
            "memory_dynamics_state": {
                "palace_topology": 0.56,
                "dominant_link_key": "harbor->promise",
                "monument_kind": "shared_ritual",
                "monument_salience": 0.48,
                "ignition_readiness": 0.52,
                "consolidation_pull": 0.43,
                "memory_tension": 0.29,
                "dominant_mode": "ignite",
            },
            "memory_dynamics_axes": {
                "topology": {"value": 0.56, "delta": 0.05},
                "salience": {"value": 0.48, "delta": 0.04},
                "ignition": {"value": 0.52, "delta": 0.03},
                "consolidation": {"value": 0.43, "delta": 0.02},
                "tension": {"value": 0.29, "delta": -0.01},
            },
            "organism_state": {
                "attunement": 0.61,
                "coherence": 0.58,
                "grounding": 0.63,
                "protective_tension": 0.24,
                "expressive_readiness": 0.57,
                "play_window": 0.39,
                "relation_pull": 0.56,
                "social_exposure": 0.18,
                "dominant_posture": "attune",
                "relation_focus": "user",
                "social_mode": "one_to_one",
                "trace": [{"step": 1, "dominant_posture": "attune"}],
            },
            "organism_axes": {
                "attunement": {"value": 0.61, "delta": 0.05},
                "coherence": {"value": 0.58, "delta": 0.03},
                "grounding": {"value": 0.63, "delta": 0.04},
                "protection": {"value": 0.24, "delta": -0.01},
                "expression": {"value": 0.49, "delta": 0.02},
                "relation": {"value": 0.56, "delta": 0.04},
            },
            "external_field_state": {
                "dominant_field": "continuity_field",
                "social_mode": "one_to_one",
                "thread_mode": "reopening_thread",
                "environmental_load": 0.24,
                "social_pressure": 0.18,
                "continuity_pull": 0.63,
                "ambiguity_load": 0.22,
                "safety_envelope": 0.73,
                "novelty": 0.16,
                "trace": [{"step": 1, "dominant_field": "continuity_field"}],
            },
            "external_field_axes": {
                "environment": {"value": 0.24, "delta": 0.02},
                "social": {"value": 0.18, "delta": 0.01},
                "continuity": {"value": 0.63, "delta": 0.05},
                "ambiguity": {"value": 0.22, "delta": 0.01},
                "safety": {"value": 0.73, "delta": 0.04},
                "novelty": {"value": 0.16, "delta": 0.0},
            },
            "terrain_dynamics_state": {
                "dominant_basin": "continuity_basin",
                "dominant_flow": "reenter",
                "terrain_energy": 0.41,
                "entropy": 0.19,
                "ignition_pressure": 0.36,
                "barrier_height": 0.22,
                "recovery_gradient": 0.49,
                "basin_pull": 0.59,
                "trace": [{"step": 1, "dominant_basin": "continuity_basin"}],
            },
            "terrain_dynamics_axes": {
                "energy": {"value": 0.41, "delta": 0.03},
                "entropy": {"value": 0.19, "delta": -0.01},
                "ignition": {"value": 0.36, "delta": 0.02},
                "barrier": {"value": 0.22, "delta": -0.01},
                "recovery": {"value": 0.49, "delta": 0.04},
                "basin": {"value": 0.59, "delta": 0.05},
            },
        },
        transfer_summary={
            "growth_state": {
                "relational_trust": 0.58,
                "epistemic_maturity": 0.52,
                "expressive_range": 0.54,
                "residue_integration": 0.49,
                "playfulness_range": 0.33,
                "self_coherence": 0.57,
                "dominant_transition": "self_coherence",
            },
            "growth_replay_axes": {
                "bond": {"value": 0.57, "delta": 0.02},
                "stability": {"value": 0.53, "delta": 0.04},
                "curiosity": {"value": 0.39, "delta": 0.01},
            },
            "epistemic_state": {
                "freshness": 0.68,
                "source_confidence": 0.63,
                "verification_pressure": 0.34,
                "change_likelihood": 0.41,
                "stale_risk": 0.36,
                "epistemic_caution": 0.39,
                "dominant_posture": "carry_forward",
            },
            "epistemic_packet_axes": {
                "grounding": {"value": 0.66, "delta": 0.03},
                "volatility": {"value": 0.38, "delta": 0.05},
                "verification": {"value": 0.37, "delta": 0.02},
            },
            "memory_dynamics_state": {
                "palace_topology": 0.47,
                "dominant_link_key": "harbor->promise",
                "monument_kind": "shared_ritual",
                "monument_salience": 0.44,
                "ignition_readiness": 0.39,
                "consolidation_pull": 0.51,
                "memory_tension": 0.35,
                "dominant_mode": "reconsolidate",
            },
            "memory_dynamics_axes": {
                "topology": {"value": 0.47, "delta": 0.03},
                "salience": {"value": 0.44, "delta": 0.02},
                "ignition": {"value": 0.39, "delta": 0.01},
                "consolidation": {"value": 0.51, "delta": 0.05},
                "tension": {"value": 0.35, "delta": 0.04},
            },
            "organism_state": {
                "attunement": 0.54,
                "coherence": 0.55,
                "grounding": 0.59,
                "protective_tension": 0.31,
                "expressive_readiness": 0.49,
                "play_window": 0.28,
                "relation_pull": 0.51,
                "social_exposure": 0.24,
                "dominant_posture": "steady",
                "relation_focus": "user",
                "social_mode": "one_to_one",
                "trace": [{"step": 1, "dominant_posture": "steady"}],
            },
            "organism_axes": {
                "attunement": {"value": 0.54, "delta": 0.02},
                "coherence": {"value": 0.55, "delta": 0.03},
                "grounding": {"value": 0.59, "delta": 0.02},
                "protection": {"value": 0.31, "delta": 0.03},
                "expression": {"value": 0.4, "delta": 0.01},
                "relation": {"value": 0.51, "delta": 0.02},
            },
            "external_field_state": {
                "dominant_field": "social_pressure_field",
                "social_mode": "threaded_group",
                "thread_mode": "continuing_thread",
                "environmental_load": 0.34,
                "social_pressure": 0.46,
                "continuity_pull": 0.41,
                "ambiguity_load": 0.27,
                "safety_envelope": 0.55,
                "novelty": 0.21,
                "trace": [{"step": 1, "dominant_field": "social_pressure_field"}],
            },
            "external_field_axes": {
                "environment": {"value": 0.34, "delta": 0.03},
                "social": {"value": 0.46, "delta": 0.04},
                "continuity": {"value": 0.41, "delta": 0.02},
                "ambiguity": {"value": 0.27, "delta": 0.03},
                "safety": {"value": 0.55, "delta": -0.02},
                "novelty": {"value": 0.21, "delta": 0.02},
            },
            "terrain_dynamics_state": {
                "dominant_basin": "recovery_basin",
                "dominant_flow": "recover",
                "terrain_energy": 0.36,
                "entropy": 0.28,
                "ignition_pressure": 0.24,
                "barrier_height": 0.33,
                "recovery_gradient": 0.57,
                "basin_pull": 0.48,
                "trace": [{"step": 1, "dominant_basin": "recovery_basin"}],
            },
            "terrain_dynamics_axes": {
                "energy": {"value": 0.36, "delta": 0.02},
                "entropy": {"value": 0.28, "delta": 0.03},
                "ignition": {"value": 0.24, "delta": 0.01},
                "barrier": {"value": 0.33, "delta": 0.02},
                "recovery": {"value": 0.57, "delta": 0.05},
                "basin": {"value": 0.48, "delta": 0.03},
            },
        },
    ).to_dict()

    assert summary["same_turn"]["temporal_membrane_mode"] == "reentry"
    assert summary["same_turn"]["temporal_reentry_pull"] == 0.58
    assert summary["same_turn"]["temporal_relation_reentry_pull"] == 0.41
    assert summary["same_turn"]["boundary_gate_mode"] == ""
    assert summary["same_turn"]["residual_reflection_mode"] == ""
    assert summary["same_turn"]["live_engagement_state"] == "pickup_comment"
    assert summary["same_turn"]["live_primary_move"] == "pick_up_comment"
    assert summary["same_turn"]["growth_relational_trust"] == 0.63
    assert summary["same_turn"]["growth_dominant_transition"] == "relational_trust"
    assert summary["same_turn"]["growth_bond_axis"] == 0.62
    assert summary["same_turn"]["epistemic_freshness"] == 0.74
    assert summary["same_turn"]["epistemic_posture"] == "carry_forward"
    assert summary["same_turn"]["epistemic_grounding_axis"] == 0.72
    assert summary["same_turn"]["memory_dynamics_mode"] == "ignite"
    assert summary["same_turn"]["memory_dominant_link"] == "harbor->promise"
    assert summary["same_turn"]["memory_monument_kind"] == "shared_ritual"
    assert summary["same_turn"]["memory_monument_salience"] == 0.48
    assert summary["same_turn"]["memory_ignition_axis"] == 0.52
    assert summary["same_turn"]["organism_posture"] == "attune"
    assert summary["same_turn"]["organism_relation_focus"] == "user"
    assert summary["same_turn"]["organism_attunement"] == 0.61
    assert summary["same_turn"]["organism_axis_grounding"] == 0.63


def test_continuity_summary_includes_joint_state_fields() -> None:
    summary = ContinuitySummaryBuilder().build(
        interaction_policy_packet={},
        current_state={
            "joint_state": {
                "dominant_mode": "shared_attention",
                "shared_delight": 0.48,
                "shared_tension": 0.21,
                "repair_readiness": 0.44,
                "common_ground": 0.63,
                "joint_attention": 0.67,
                "mutual_room": 0.41,
                "coupling_strength": 0.58,
                "trace": [{"step": 1, "dominant_mode": "shared_attention"}],
            },
            "joint_axes": {
                "delight": {"value": 0.48, "delta": 0.03},
                "tension": {"value": 0.21, "delta": -0.01},
                "repair": {"value": 0.44, "delta": 0.02},
                "ground": {"value": 0.63, "delta": 0.05},
                "attention": {"value": 0.67, "delta": 0.06},
                "coupling": {"value": 0.58, "delta": 0.04},
            },
        },
        transfer_summary={
            "joint_state": {
                "dominant_mode": "repair_attunement",
                "shared_delight": 0.31,
                "shared_tension": 0.29,
                "repair_readiness": 0.57,
                "common_ground": 0.49,
                "joint_attention": 0.46,
                "mutual_room": 0.38,
                "coupling_strength": 0.52,
                "trace": [{"step": 1, "dominant_mode": "repair_attunement"}],
            },
            "joint_axes": {
                "delight": {"value": 0.31, "delta": 0.01},
                "tension": {"value": 0.29, "delta": 0.02},
                "repair": {"value": 0.57, "delta": 0.04},
                "ground": {"value": 0.49, "delta": 0.03},
                "attention": {"value": 0.46, "delta": 0.02},
                "coupling": {"value": 0.52, "delta": 0.03},
            },
        },
    ).to_dict()

    assert summary["same_turn"]["joint_mode"] == "shared_attention"
    assert summary["same_turn"]["joint_attention"] == 0.67
    assert summary["same_turn"]["joint_axis_coupling"] == 0.58
    assert summary["same_turn"]["joint_common_ground"] == 0.63
    assert summary["same_turn"]["joint_trace_len"] == 1
    assert summary["overnight"]["joint_mode"] == "repair_attunement"
    assert summary["overnight"]["joint_repair_readiness"] == 0.57
    assert summary["overnight"]["joint_axis_ground"] == 0.49
    assert summary["overnight"]["joint_axis_attention"] == 0.46


def test_continuity_summary_surfaces_boundary_transform_and_residual_reflection() -> None:
    summary = ContinuitySummaryBuilder().build(
        interaction_policy_packet={
            "contact_reflection_state": {
                "state": "guarded_reflection",
                "reflection_style": "reflect_only",
                "transmit_share": 0.32,
                "reflect_share": 0.46,
                "absorb_share": 0.22,
                "block_share": 0.0,
                "dominant_inputs": ["withheld_candidate", "guarded_contact"],
            }
        },
        current_state={
            "boundary_gate_mode": "narrow",
            "boundary_transform_mode": "soften",
            "boundary_authority_scope": "user_guarded",
            "boundary_softened_acts": ["offer_small_opening_line"],
            "boundary_withheld_acts": ["clarify_question"],
            "boundary_deferred_topics": ["unfinished part"],
            "boundary_residual_pressure": 0.48,
            "residual_reflection_mode": "withheld",
            "residual_reflection_focus": "unfinished part",
            "residual_reflection_strength": 0.62,
            "residual_reflection_reasons": ["withheld_candidate", "deferred_topic"],
        },
    ).to_dict()

    assert summary["same_turn"]["boundary_gate_mode"] == "narrow"
    assert summary["same_turn"]["boundary_transform_mode"] == "soften"
    assert summary["same_turn"]["contact_reflection_state"] == "guarded_reflection"
    assert summary["same_turn"]["contact_reflection_style"] == "reflect_only"
    assert summary["same_turn"]["contact_reflect_share"] == 0.46
    assert "guarded_contact" in summary["same_turn"]["contact_reflection_inputs"]
    assert summary["same_turn"]["boundary_softened_acts"] == ["offer_small_opening_line"]
    assert summary["same_turn"]["boundary_withheld_acts"] == ["clarify_question"]
    assert summary["same_turn"]["residual_reflection_mode"] == "withheld"
    assert summary["same_turn"]["residual_reflection_focus"] == "unfinished part"
    assert summary["same_turn"]["residual_reflection_strength"] == 0.62


def test_continuity_summary_surfaces_autobiographical_thread() -> None:
    summary = ContinuitySummaryBuilder().build(
        interaction_policy_packet={},
        current_state={
            "autobiographical_thread_mode": "unfinished_thread",
            "autobiographical_thread_anchor": "harbor promise",
            "autobiographical_thread_focus": "unfinished promise",
            "autobiographical_thread_strength": 0.56,
            "autobiographical_thread_reasons": ["discussion_registry", "residual_reflection"],
        },
    ).to_dict()

    assert summary["same_turn"]["autobiographical_thread_mode"] == "unfinished_thread"
    assert summary["same_turn"]["autobiographical_thread_anchor"] == "harbor promise"
    assert summary["same_turn"]["autobiographical_thread_strength"] == 0.56
    assert "residual_reflection" in summary["same_turn"]["autobiographical_thread_reasons"]


def test_continuity_summary_surfaces_situation_risk_and_emergency_posture() -> None:
    summary = ContinuitySummaryBuilder().build(
        interaction_policy_packet={
            "situation_risk_state": {
                "state": "acute_threat",
                "immediacy": 0.72,
                "intent_clarity": 0.58,
                "escape_room": 0.34,
                "relation_break": 0.12,
            },
            "emergency_posture": {
                "state": "create_distance",
                "dialogue_permission": "boundary_only",
                "primary_action": "create_distance",
            },
            "reaction_vs_overnight_bias": {"same_turn": {}, "overnight": {}},
        },
        current_state={},
    ).to_dict()

    assert summary["same_turn"]["situation_risk_state"] == "acute_threat"
    assert summary["same_turn"]["situation_risk_immediacy"] == 0.72
    assert summary["same_turn"]["emergency_posture"] == "create_distance"
    assert summary["same_turn"]["emergency_dialogue_permission"] == "boundary_only"
    assert summary["same_turn"]["emergency_primary_action"] == "create_distance"


def test_continuity_summary_surfaces_recent_dialogue_state() -> None:
    summary = ContinuitySummaryBuilder().build(
        interaction_policy_packet={
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "overlap_score": 0.22,
                "reopen_pressure": 0.64,
                "thread_carry": 0.58,
                "recent_anchor": "前に話した引っかかり",
                "dominant_inputs": ["history_overlap", "reopen_marker"],
            }
        },
        current_state={},
    ).to_dict()

    assert summary["same_turn"]["recent_dialogue_state"] == "reopening_thread"
    assert summary["same_turn"]["recent_dialogue_overlap"] == 0.22
    assert summary["same_turn"]["recent_dialogue_reopen_pressure"] == 0.64
    assert summary["same_turn"]["recent_dialogue_thread_carry"] == 0.58
    assert summary["same_turn"]["recent_dialogue_anchor"] == "前に話した引っかかり"
    assert "reopen_marker" in summary["same_turn"]["recent_dialogue_inputs"]


def test_continuity_summary_surfaces_discussion_thread_state() -> None:
    summary = ContinuitySummaryBuilder().build(
        interaction_policy_packet={
            "discussion_thread_state": {
                "state": "revisit_issue",
                "topic_anchor": "前に話した引っかかり",
                "unresolved_pressure": 0.48,
                "revisit_readiness": 0.62,
                "thread_visibility": 0.54,
                "dominant_inputs": ["history_overlap", "revisit_marker"],
            }
        },
        current_state={},
    ).to_dict()

    assert summary["same_turn"]["discussion_thread_state"] == "revisit_issue"
    assert summary["same_turn"]["discussion_thread_anchor"] == "前に話した引っかかり"
    assert summary["same_turn"]["discussion_unresolved_pressure"] == 0.48
    assert summary["same_turn"]["discussion_revisit_readiness"] == 0.62
    assert summary["same_turn"]["discussion_thread_visibility"] == 0.54
    assert "revisit_marker" in summary["same_turn"]["discussion_thread_inputs"]


def test_continuity_summary_surfaces_issue_state() -> None:
    summary = ContinuitySummaryBuilder().build(
        interaction_policy_packet={
            "issue_state": {
                "state": "pausing_issue",
                "issue_anchor": "さっきの引っかかり",
                "question_pressure": 0.22,
                "pause_readiness": 0.58,
                "resolution_readiness": 0.18,
                "dominant_inputs": ["pause_marker", "discussion:revisit_issue"],
            }
        },
        current_state={},
    ).to_dict()

    assert summary["same_turn"]["issue_state"] == "pausing_issue"
    assert summary["same_turn"]["issue_anchor"] == "さっきの引っかかり"
    assert summary["same_turn"]["issue_question_pressure"] == 0.22
    assert summary["same_turn"]["issue_pause_readiness"] == 0.58
    assert summary["same_turn"]["issue_resolution_readiness"] == 0.18
    assert "pause_marker" in summary["same_turn"]["issue_inputs"]


def test_continuity_summary_surfaces_discussion_thread_registry_snapshot() -> None:
    summary = ContinuitySummaryBuilder().build(
        interaction_policy_packet={},
        current_state={
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "さっきの引っかかり",
                "dominant_anchor": "さっきの引っかかり",
                "dominant_issue_state": "pausing_issue",
                "total_threads": 1,
            }
        },
    ).to_dict()

    assert summary["same_turn"]["discussion_registry_dominant_thread"] == "さっきの引っかかり"
    assert summary["same_turn"]["discussion_registry_dominant_anchor"] == "さっきの引っかかり"
    assert summary["same_turn"]["discussion_registry_dominant_issue_state"] == "pausing_issue"
    assert summary["same_turn"]["discussion_registry_total_threads"] == 1
