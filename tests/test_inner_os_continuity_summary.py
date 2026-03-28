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
        },
    ).to_dict()

    assert summary["same_turn"]["temporal_membrane_mode"] == "reentry"
    assert summary["same_turn"]["temporal_reentry_pull"] == 0.58
    assert summary["same_turn"]["temporal_relation_reentry_pull"] == 0.41
    assert summary["same_turn"]["boundary_gate_mode"] == ""
    assert summary["same_turn"]["residual_reflection_mode"] == ""
    assert summary["same_turn"]["live_engagement_state"] == "pickup_comment"
    assert summary["same_turn"]["live_primary_move"] == "pick_up_comment"
    assert summary["overnight"]["temporal_membrane_focus"] == "reentry"
    assert summary["overnight"]["temporal_reentry_bias"] == 0.16
    assert summary["carry_strengths"]["temporal_timeline"] == 0.12
    assert summary["carry_strengths"]["temporal_reentry"] == 0.16


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
