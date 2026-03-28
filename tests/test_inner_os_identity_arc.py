from inner_os.identity_arc import IdentityArcSummaryBuilder


def test_identity_arc_builder_derives_repairing_bond_arc() -> None:
    summary = IdentityArcSummaryBuilder().build(
        {
            "inner_os_long_term_theme_summary": {
                "focus": "harbor slope",
                "anchor": "harbor slope",
                "kind": "meaning",
                "summary": "fragile harbor promise",
                "strength": 0.63,
            },
            "inner_os_memory_class_summary": {
                "dominant_class": "bond_protection",
            },
            "inner_os_agenda_summary": {
                "dominant_agenda": "repair",
                "dominant_reason": "repair_window",
                "agenda_carry_bias": 0.31,
            },
            "inner_os_commitment_summary": {
                "dominant_target": "repair",
                "dominant_state": "commit",
                "dominant_reason": "repair_trace",
                "commitment_carry_bias": 0.37,
            },
            "inner_os_insight_summary": {
                "dominant_insight_class": "reframed_relation",
                "dominant_reframed_topic": "harbor thread",
                "insight_reframing_bias": 0.18,
            },
            "inner_os_partner_relation_summary": {
                "person_id": "user",
                "memory_anchor": "harbor slope",
                "strength": 0.71,
            },
            "inner_os_group_thread_registry_summary": {
                "dominant_thread_id": "threaded_group:user|friend",
                "total_threads": 1,
            },
            "inner_os_same_turn_learning_mode_state": "repair_probe",
            "inner_os_same_turn_social_experiment_state": "repair_signal_probe",
        }
    ).to_dict()

    assert summary["arc_kind"] == "repairing_bond"
    assert summary["phase"] in {"shifting", "integrating"}
    assert summary["memory_anchor"] == "harbor slope"
    assert summary["related_person_id"] == "user"
    assert summary["group_thread_focus"] == "threaded_group:user|friend"
    assert summary["stability"] > 0.4
    assert summary["learning_mode_focus"] == "repair_probe"
    assert summary["social_experiment_focus"] == "repair_signal_probe"
    assert "learning:repair_probe" in summary["supporting_drivers"]
    assert "probe:repair_signal_probe" in summary["supporting_drivers"]


def test_identity_arc_builder_uses_stabilizing_self_for_body_risk() -> None:
    summary = IdentityArcSummaryBuilder().build(
        {
            "inner_os_memory_class_summary": {"dominant_class": "body_risk"},
            "inner_os_agenda_summary": {"dominant_agenda": "hold"},
            "inner_os_commitment_summary": {
                "dominant_target": "stabilize",
                "dominant_state": "settle",
            },
        }
    ).to_dict()

    assert summary["arc_kind"] == "stabilizing_self"
    assert summary["phase"] in {"forming", "holding"}


def test_identity_arc_builder_carries_probe_drivers_into_growing_edge() -> None:
    summary = IdentityArcSummaryBuilder().build(
        {
            "inner_os_long_term_theme_summary": {
                "focus": "workshop threshold",
                "anchor": "workshop threshold",
                "kind": "identity",
                "summary": "careful return to a social edge",
                "strength": 0.41,
            },
            "inner_os_agenda_summary": {
                "dominant_agenda": "revisit",
                "agenda_carry_bias": 0.14,
            },
            "inner_os_same_turn_learning_mode_state": "test_small",
            "inner_os_same_turn_social_experiment_state": "test_small_step",
        }
    ).to_dict()

    assert summary["arc_kind"] == "growing_edge"
    assert summary["open_tension"] == "careful_probe"
    assert "learning=test_small" in summary["summary"]
    assert "probe=test_small_step" in summary["summary"]
