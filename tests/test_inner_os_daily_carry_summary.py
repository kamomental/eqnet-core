from inner_os.daily_carry_summary import DailyCarrySummaryBuilder


def test_daily_carry_summary_builder_collects_same_turn_and_overnight_bias() -> None:
    report = {
        "inner_os_memory_class_summary": {
            "dominant_class": "bond_protection",
            "dominant_reason": "bond_protection_pressure",
        },
        "inner_os_commitment_summary": {
            "dominant_target": "repair",
            "dominant_state": "commit",
        },
        "inner_os_insight_summary": {
            "dominant_insight_class": "reframed_relation",
            "dominant_reframed_topic": "shared thread",
        },
        "inner_os_partner_relation_summary": {
            "person_id": "user",
            "social_role": "companion",
        },
        "inner_os_identity_arc_summary": {
            "arc_kind": "repairing_bond",
            "phase": "shifting",
            "summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
        },
        "inner_os_relation_arc_summary": {
            "arc_kind": "repairing_relation",
            "phase": "shifting",
            "summary": "repair is gathering around a companion thread",
        },
        "inner_os_relation_arc_registry_summary": {
            "dominant_arc_kind": "repairing_relation",
            "active_arc_count": 1,
            "total_arcs": 1,
        },
        "inner_os_discussion_thread_registry_summary": {
            "dominant_thread_id": "repair_anchor",
            "dominant_anchor": "repair anchor",
            "dominant_issue_state": "pausing_issue",
            "top_thread_ids": ["repair_anchor"],
            "total_threads": 1,
            "thread_scores": {"repair_anchor": 0.71},
            "uncertainty": 0.18,
        },
        "inner_os_same_turn_temporal_membrane_mode": "reentry",
        "inner_os_same_turn_temporal_timeline_coherence": 0.42,
        "inner_os_same_turn_temporal_reentry_pull": 0.56,
        "inner_os_same_turn_temporal_supersession_pressure": 0.08,
        "inner_os_same_turn_temporal_continuity_pressure": 0.34,
        "inner_os_same_turn_temporal_relation_reentry_pull": 0.38,
        "inner_os_same_turn_boundary_gate_mode": "narrow",
        "inner_os_same_turn_boundary_transform_mode": "soften",
        "inner_os_same_turn_boundary_softened_count": 1,
        "inner_os_same_turn_boundary_withheld_count": 2,
        "inner_os_same_turn_boundary_deferred_count": 1,
        "inner_os_same_turn_boundary_residual_pressure": 0.48,
        "inner_os_same_turn_residual_reflection_mode": "withheld",
        "inner_os_same_turn_residual_reflection_focus": "unfinished part",
        "inner_os_same_turn_residual_reflection_strength": 0.62,
        "inner_os_same_turn_contact_reflection_state": "guarded_reflection",
        "inner_os_same_turn_contact_reflection_style": "reflect_only",
        "inner_os_same_turn_contact_transmit_share": 0.31,
        "inner_os_same_turn_contact_reflect_share": 0.47,
        "inner_os_same_turn_contact_absorb_share": 0.22,
        "inner_os_same_turn_contact_block_share": 0.0,
        "inner_os_same_turn_autobiographical_thread_mode": "unfinished_thread",
        "inner_os_same_turn_autobiographical_thread_anchor": "harbor promise",
        "inner_os_same_turn_autobiographical_thread_focus": "unfinished promise",
        "inner_os_same_turn_autobiographical_thread_strength": 0.58,
        "inner_os_sleep_temporal_membrane_focus": "reentry",
        "inner_os_sleep_temporal_timeline_bias": 0.12,
        "inner_os_sleep_temporal_reentry_bias": 0.17,
        "inner_os_sleep_temporal_supersession_bias": 0.04,
        "inner_os_sleep_temporal_continuity_bias": 0.11,
        "inner_os_sleep_temporal_relation_reentry_bias": 0.09,
        "inner_os_sleep_autobiographical_thread_mode": "unfinished_thread",
        "inner_os_sleep_autobiographical_thread_anchor": "harbor promise",
        "inner_os_sleep_autobiographical_thread_focus": "unfinished promise",
        "inner_os_sleep_autobiographical_thread_strength": 0.41,
        "inner_os_sleep_memory_class_focus": "bond_protection",
        "inner_os_sleep_commitment_target_focus": "repair",
        "inner_os_sleep_commitment_state_focus": "commit",
        "inner_os_sleep_commitment_followup_focus": "reopen_softly",
        "inner_os_sleep_commitment_mode_focus": "repair",
        "inner_os_sleep_association_reweighting_focus": "repeated_links",
        "inner_os_sleep_association_reweighting_reason": "repeated_insight_trace",
        "inner_os_sleep_insight_class_focus": "reframed_relation",
        "inner_os_sleep_insight_terrain_shape_target": "soft_relation",
        "inner_os_sleep_insight_terrain_shape_reason": "reframed_relation",
        "inner_os_sleep_temperament_focus": "forward",
        "inner_os_sleep_terrain_reweighting_bias": 0.28,
        "inner_os_sleep_commitment_carry_bias": 0.37,
        "inner_os_sleep_association_reweighting_bias": 0.22,
        "inner_os_sleep_insight_reframing_bias": 0.19,
        "inner_os_sleep_insight_terrain_shape_bias": 0.16,
        "inner_os_sleep_temperament_forward_bias": 0.11,
        "inner_os_sleep_temperament_guard_bias": 0.03,
        "inner_os_sleep_temperament_bond_bias": 0.05,
        "inner_os_sleep_temperament_recovery_bias": 0.02,
    }

    summary = DailyCarrySummaryBuilder().build(report).to_dict()

    assert summary["same_turn_focus"]["memory_class"] == "bond_protection"
    assert summary["same_turn_focus"]["commitment_target"] == "repair"
    assert summary["same_turn_focus"]["insight_class"] == "reframed_relation"
    assert summary["same_turn_focus"]["identity_arc_kind"] == "repairing_bond"
    assert summary["same_turn_focus"]["relation_arc_kind"] == "repairing_relation"
    assert summary["same_turn_focus"]["discussion_registry_dominant_anchor"] == "repair anchor"
    assert summary["same_turn_focus"]["discussion_registry_dominant_issue_state"] == "pausing_issue"
    assert summary["same_turn_focus"]["discussion_registry_total_threads"] == 1
    assert summary["same_turn_focus"]["temporal_membrane_mode"] == "reentry"
    assert summary["same_turn_focus"]["temporal_reentry_pull"] == 0.56
    assert summary["same_turn_focus"]["boundary_gate_mode"] == "narrow"
    assert summary["same_turn_focus"]["boundary_withheld_count"] == 2
    assert summary["same_turn_focus"]["contact_reflection_state"] == "guarded_reflection"
    assert summary["same_turn_focus"]["contact_reflection_style"] == "reflect_only"
    assert summary["same_turn_focus"]["contact_reflect_share"] == 0.47
    assert summary["same_turn_focus"]["residual_reflection_mode"] == "withheld"
    assert summary["same_turn_focus"]["residual_reflection_strength"] == 0.62
    assert summary["same_turn_focus"]["autobiographical_thread_mode"] == "unfinished_thread"
    assert summary["same_turn_focus"]["autobiographical_thread_anchor"] == "harbor promise"
    assert summary["same_turn_focus"]["autobiographical_thread_strength"] == 0.58
    assert summary["overnight_focus"]["association_focus"] == "repeated_links"
    assert summary["overnight_focus"]["discussion_registry_dominant_anchor"] == "repair anchor"
    assert summary["overnight_focus"]["discussion_registry_dominant_issue_state"] == "pausing_issue"
    assert summary["overnight_focus"]["autobiographical_thread_mode"] == "unfinished_thread"
    assert summary["overnight_focus"]["autobiographical_thread_anchor"] == "harbor promise"
    assert summary["overnight_focus"]["autobiographical_thread_strength"] == 0.41
    assert summary["overnight_focus"]["temporal_membrane_focus"] == "reentry"
    assert summary["overnight_focus"]["temporal_reentry_bias"] == 0.17
    assert summary["overnight_focus"]["identity_arc_phase"] == "shifting"
    assert summary["overnight_focus"]["relation_arc_phase"] == "shifting"
    assert summary["carry_strengths"]["commitment_carry"] == 0.37
    assert summary["carry_strengths"]["temporal_reentry"] == 0.17
    assert summary["temporal_alignment"]["same_turn_mode"] == "reentry"
    assert summary["temporal_alignment"]["overnight_focus"] == "reentry"
    assert summary["temporal_alignment"]["focus_alignment"] is True
    assert summary["temporal_alignment"]["same_to_overnight_reentry_delta"] == -0.39
    assert summary["temporal_alignment"]["reentry_carry_visible"] is True
    assert summary["temporal_alignment"]["reentry_carry_strength"] == 0.17
    assert summary["boundary_alignment"]["gate_mode"] == "narrow"
    assert summary["boundary_alignment"]["withheld_count"] == 2
    assert summary["boundary_alignment"]["contact_state"] == "guarded_reflection"
    assert summary["boundary_alignment"]["contact_style"] == "reflect_only"
    assert summary["boundary_alignment"]["contact_reflect_share"] == 0.47
    assert summary["boundary_alignment"]["contact_reflection_visible"] is True
    assert summary["boundary_alignment"]["residual_mode"] == "withheld"
    assert summary["boundary_alignment"]["residual_strength"] == 0.62
    assert summary["boundary_alignment"]["unsaid_pressure_visible"] is True
    assert summary["dominant_carry_channel"] == "autobiographical_thread"
    assert "commitment_carry" in summary["active_carry_channels"]
    assert "autobiographical_thread" in summary["active_carry_channels"]
    assert summary["carry_alignment"]["memory_carry_visible"] is True
    assert summary["carry_alignment"]["commitment_carry_visible"] is True
    assert summary["carry_alignment"]["insight_carry_visible"] is True
    assert summary["carry_alignment"]["temporal_membrane_visible"] is True
    assert summary["carry_alignment"]["boundary_visible"] is True
    assert summary["carry_alignment"]["contact_reflection_visible"] is True
    assert summary["carry_alignment"]["residual_visible"] is True
    assert summary["carry_alignment"]["identity_arc_visible"] is True
    assert summary["carry_alignment"]["relation_arc_visible"] is True
    assert summary["carry_alignment"]["relation_arc_registry_visible"] is True
    assert summary["carry_alignment"]["discussion_registry_visible"] is True
    assert summary["carry_alignment"]["autobiographical_thread_visible"] is True
    assert summary["carry_alignment"]["temperament_carry_visible"] is True
