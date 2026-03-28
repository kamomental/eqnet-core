from inner_os.relation_arc import RelationArcSummaryBuilder


def test_relation_arc_builder_derives_repairing_relation_arc() -> None:
    summary = RelationArcSummaryBuilder().build(
        {
            "inner_os_partner_relation_summary": {
                "person_id": "person:harbor",
                "social_role": "companion",
                "social_interpretation": "familiar:companion:open",
                "community_id": "harbor_collective",
                "culture_id": "coastal",
                "attachment": 0.72,
                "familiarity": 0.68,
                "trust_memory": 0.7,
                "strength": 0.71,
            },
            "inner_os_partner_relation_registry_summary": {
                "dominant_person_id": "person:harbor",
                "total_people": 2,
            },
            "inner_os_group_thread_registry_summary": {
                "dominant_thread_id": "threaded_group:person:friend|person:harbor",
                "total_threads": 1,
            },
            "inner_os_agenda_summary": {
                "dominant_agenda": "repair",
                "dominant_reason": "repair_window",
            },
            "inner_os_commitment_summary": {
                "dominant_target": "repair",
                "dominant_state": "commit",
            },
            "inner_os_same_turn_learning_mode_state": "repair_probe",
            "inner_os_same_turn_social_experiment_state": "repair_signal_probe",
        }
    ).to_dict()

    assert summary["arc_kind"] == "repairing_relation"
    assert summary["phase"] in {"shifting", "integrating"}
    assert summary["related_person_id"] == "person:harbor"
    assert summary["group_thread_id"] == "threaded_group:person:friend|person:harbor"
    assert summary["social_role"] == "companion"
    assert summary["learning_mode_focus"] == "repair_probe"
    assert summary["social_experiment_focus"] == "repair_signal_probe"
    assert summary["open_tension"] in {"careful_probe", "timing_sensitive_reentry"}
    assert "person:person:harbor" not in summary["summary"]


def test_relation_arc_builder_prefers_group_thread_continuity() -> None:
    summary = RelationArcSummaryBuilder().build(
        {
            "inner_os_partner_relation_registry_summary": {
                "dominant_person_id": "person:harbor",
                "total_people": 3,
            },
            "inner_os_group_thread_registry_summary": {
                "dominant_thread_id": "threaded_group:person:friend|person:harbor|person:mentor",
                "total_threads": 2,
            },
            "inner_os_sleep_group_thread_focus": "threaded_group",
            "inner_os_sleep_agenda_window_focus": "next_same_group_window",
            "inner_os_sleep_agenda_window_reason": "wait_for_group_thread",
        }
    ).to_dict()

    assert summary["arc_kind"] == "group_thread_continuity"
    assert summary["open_tension"] == "timing_sensitive_reentry"
    assert summary["topology_focus"] == "threaded_group"
