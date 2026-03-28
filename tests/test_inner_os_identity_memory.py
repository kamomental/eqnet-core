from inner_os.identity_memory import IdentityArcRegistry


def test_identity_arc_registry_merges_same_arc_across_days() -> None:
    registry = IdentityArcRegistry()

    registry.update(
        day_key="2026-03-21",
        identity_arc_summary={
            "arc_kind": "repairing_bond",
            "phase": "shifting",
            "summary": "repair is gathering around a relationship thread / phase=shifting",
            "dominant_driver": "commitment:repair",
            "supporting_drivers": ["commitment:repair", "memory:bond_protection"],
            "open_tension": "timing_sensitive_reentry",
            "stability": 0.42,
            "memory_anchor": "harbor slope",
            "related_person_id": "person:harbor",
            "group_thread_focus": "threaded_group:person:friend|person:harbor",
            "long_term_theme_kind": "meaning",
            "long_term_theme_focus": "quiet harbor promise",
            "learning_mode_focus": "repair_probe",
            "social_experiment_focus": "repair_signal_probe",
        },
    )
    registry.update(
        day_key="2026-03-22",
        identity_arc_summary={
            "arc_kind": "repairing_bond",
            "phase": "integrating",
            "summary": "repair is holding more steadily around the same relationship thread",
            "dominant_driver": "theme:meaning",
            "supporting_drivers": ["theme:meaning", "commitment:repair"],
            "stability": 0.73,
            "memory_anchor": "harbor slope",
            "related_person_id": "person:harbor",
            "group_thread_focus": "threaded_group:person:friend|person:harbor",
            "long_term_theme_kind": "meaning",
            "long_term_theme_focus": "quiet harbor promise",
            "learning_mode_focus": "integrate_and_commit",
            "social_experiment_focus": "confirm_shared_direction",
        },
    )

    summary = registry.summary()

    assert summary["total_arcs"] == 1
    assert summary["dominant_arc_kind"] == "repairing_bond"
    record = summary["top_arcs"][0]
    assert record["days_seen"] == 2
    assert record["phase"] == "integrating"
    assert record["stability_peak"] == 0.73
    assert record["summary"] == "repair is holding more steadily around the same relationship thread"
    assert record["learning_mode_focus"] == "integrate_and_commit"
    assert record["social_experiment_focus"] == "confirm_shared_direction"


def test_identity_arc_registry_roundtrip_keeps_top_arc_summary() -> None:
    registry = IdentityArcRegistry()
    registry.update(
        day_key="2026-03-22",
        identity_arc_summary={
            "arc_kind": "stabilizing_self",
            "phase": "holding",
            "summary": "stabilization is protecting continuity before further movement",
            "memory_anchor": "room edge",
            "stability": 0.36,
            "learning_mode_focus": "hold_and_wait",
            "social_experiment_focus": "hold_probe",
        },
    )

    restored = IdentityArcRegistry.from_dict(registry.to_dict())

    assert restored.summary()["dominant_arc_kind"] == "stabilizing_self"
    assert restored.summary()["top_arcs"][0]["memory_anchor"] == "room edge"
    assert restored.summary()["top_arcs"][0]["learning_mode_focus"] == "hold_and_wait"
