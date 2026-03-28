from inner_os.relation_memory import RelationArcRegistry


def test_relation_arc_registry_merges_same_relation_across_days() -> None:
    registry = RelationArcRegistry()

    registry.update(
        day_key="2026-03-21",
        relation_arc_summary={
            "arc_kind": "repairing_relation",
            "phase": "shifting",
            "summary": "repair is gathering around a companion thread",
            "dominant_driver": "person:person:harbor",
            "supporting_drivers": ["person:person:harbor", "learning:repair_probe"],
            "open_tension": "careful_probe",
            "stability": 0.44,
            "related_person_id": "person:harbor",
            "group_thread_id": "threaded_group:person:friend|person:harbor",
            "social_role": "companion",
            "topology_focus": "threaded_group",
            "learning_mode_focus": "repair_probe",
            "social_experiment_focus": "repair_signal_probe",
        },
    )
    registry.update(
        day_key="2026-03-22",
        relation_arc_summary={
            "arc_kind": "repairing_relation",
            "phase": "integrating",
            "summary": "repair is holding more steadily around the same companion thread",
            "dominant_driver": "group:threaded_group:person:friend|person:harbor",
            "supporting_drivers": ["group:threaded_group:person:friend|person:harbor", "learning:integrate_and_commit"],
            "stability": 0.73,
            "related_person_id": "person:harbor",
            "group_thread_id": "threaded_group:person:friend|person:harbor",
            "social_role": "companion",
            "topology_focus": "threaded_group",
            "learning_mode_focus": "integrate_and_commit",
            "social_experiment_focus": "confirm_shared_direction",
        },
    )

    summary = registry.summary()

    assert summary["total_arcs"] == 1
    assert summary["dominant_arc_kind"] == "repairing_relation"
    assert summary["dominant_person_id"] == "person:harbor"
    assert summary["dominant_group_thread_id"] == "threaded_group:person:friend|person:harbor"
    record = summary["top_arcs"][0]
    assert record["days_seen"] == 2
    assert record["phase"] == "integrating"
    assert record["stability_peak"] == 0.73
    assert record["learning_mode_focus"] == "integrate_and_commit"
    assert record["social_experiment_focus"] == "confirm_shared_direction"


def test_relation_arc_registry_roundtrip_keeps_group_arc_summary() -> None:
    registry = RelationArcRegistry()
    registry.update(
        day_key="2026-03-22",
        relation_arc_summary={
            "arc_kind": "group_thread_continuity",
            "phase": "holding",
            "summary": "the same group thread is being kept warm for a later return",
            "group_thread_id": "threaded_group:person:friend|person:harbor",
            "topology_focus": "threaded_group",
            "open_tension": "timing_sensitive_reentry",
            "stability": 0.39,
            "learning_mode_focus": "hold_and_wait",
            "social_experiment_focus": "hold_probe",
        },
    )

    restored = RelationArcRegistry.from_dict(registry.to_dict())

    assert restored.summary()["dominant_arc_kind"] == "group_thread_continuity"
    assert restored.summary()["top_arcs"][0]["group_thread_id"] == "threaded_group:person:friend|person:harbor"
    assert restored.summary()["top_arcs"][0]["learning_mode_focus"] == "hold_and_wait"
