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
    assert summary["overnight_focus"]["association_focus"] == "repeated_links"
    assert summary["carry_strengths"]["commitment_carry"] == 0.37
    assert summary["dominant_carry_channel"] == "commitment_carry"
    assert "commitment_carry" in summary["active_carry_channels"]
    assert summary["carry_alignment"]["memory_carry_visible"] is True
    assert summary["carry_alignment"]["commitment_carry_visible"] is True
    assert summary["carry_alignment"]["insight_carry_visible"] is True
    assert summary["carry_alignment"]["temperament_carry_visible"] is True
