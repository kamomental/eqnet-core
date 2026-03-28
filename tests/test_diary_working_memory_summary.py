from datetime import date

from emot_terrain_lab.terrain.diary import DiaryEntry, DiaryManager


def test_diary_entry_roundtrip_keeps_working_memory_summary() -> None:
    entry = DiaryEntry(
        day="2026-03-15",
        text="memo",
        metrics={"entropy": 0.4},
        tags=["rest-mode"],
        highlights=["anchor"],
        working_memory_summary={
            "available": True,
            "current_focus": "meaning",
            "focus_anchor": "harbor slope",
            "promotion_readiness": 0.63,
        },
        working_memory_signature_summary={
            "dominant_focus": "meaning",
            "dominant_anchor": "harbor slope",
            "promotion_readiness_mean": 0.64,
            "recurrence_weight": 2.4,
            "semantic_seed_strength": 0.41,
        },
        working_memory_replay_summary={
            "focus": "meaning",
            "anchor": "harbor slope",
            "matched_events": 2,
            "strength": 0.57,
            "top_matches": [{"id": "trace-1", "alignment": 0.8, "boost": 0.05}],
        },
        long_term_theme_summary={
            "focus": "meaning",
            "anchor": "harbor slope",
            "kind": "meaning",
            "strength": 0.63,
            "seed_strength": 0.41,
            "summary": "fragile harbor promise",
        },
        identity_arc_summary={
            "arc_kind": "repairing_bond",
            "phase": "shifting",
            "summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
            "memory_anchor": "harbor slope",
            "stability": 0.58,
        },
        relation_arc_summary={
            "arc_kind": "repairing_relation",
            "phase": "shifting",
            "summary": "repair is gathering around a companion thread / phase=shifting",
            "related_person_id": "person:harbor",
            "group_thread_id": "threaded_group:person:friend|person:harbor",
            "social_role": "companion",
            "open_tension": "timing_sensitive_reentry",
            "stability": 0.56,
            "learning_mode_focus": "repair_probe",
            "social_experiment_focus": "repair_signal_probe",
        },
    )

    payload = entry.to_json()
    restored = DiaryEntry.from_json(payload)

    assert restored.working_memory_summary is not None
    assert restored.working_memory_summary["current_focus"] == "meaning"
    assert restored.working_memory_signature_summary is not None
    assert restored.working_memory_signature_summary["dominant_focus"] == "meaning"
    assert restored.working_memory_replay_summary is not None
    assert restored.working_memory_replay_summary["matched_events"] == 2
    assert restored.long_term_theme_summary is not None
    assert restored.long_term_theme_summary["kind"] == "meaning"
    assert restored.identity_arc_summary is not None
    assert restored.identity_arc_summary["arc_kind"] == "repairing_bond"
    assert restored.relation_arc_summary is not None
    assert restored.relation_arc_summary["arc_kind"] == "repairing_relation"


def test_diary_manager_records_working_memory_summary() -> None:
    manager = DiaryManager()

    entry = manager.record_daily_entry(
        day=date(2026, 3, 15),
        metrics={"entropy": 0.42, "enthalpy_mean": 0.31, "dissipation": 0.08},
        top_axes=["valence"],
        catalyst_highlights=[],
        gentle_quotes=[],
        rest_snapshot={"active": False},
        loop_alert=False,
        fatigue_flag=False,
        use_llm=False,
        working_memory_summary={
            "available": True,
            "current_focus": "meaning",
            "focus_anchor": "harbor slope",
            "unresolved_count": 2,
            "pending_meaning": 0.57,
            "promotion_readiness": 0.62,
        },
        working_memory_signature_summary={
            "dominant_focus": "meaning",
            "dominant_anchor": "harbor slope",
            "promotion_readiness_mean": 0.68,
            "autobiographical_pressure_mean": 0.51,
            "recurrence_weight": 2.6,
            "semantic_seed_strength": 0.44,
            "long_term_theme": {
                "focus": "meaning",
                "anchor": "harbor slope",
                "kind": "meaning",
                "strength": 0.63,
            },
        },
        working_memory_replay_summary={
            "focus": "meaning",
            "anchor": "harbor slope",
            "matched_events": 2,
            "strength": 0.58,
            "top_matches": [{"id": "trace-1", "alignment": 0.82, "boost": 0.06}],
        },
        long_term_theme_summary={
            "focus": "meaning",
            "anchor": "harbor slope",
            "kind": "meaning",
            "strength": 0.63,
            "seed_strength": 0.44,
            "recurrence_weight": 2.6,
            "summary": "fragile harbor promise",
        },
        identity_arc_summary={
            "arc_kind": "repairing_bond",
            "phase": "shifting",
            "summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
            "memory_anchor": "harbor slope",
            "stability": 0.58,
        },
        relation_arc_summary={
            "arc_kind": "repairing_relation",
            "phase": "shifting",
            "summary": "repair is gathering around a companion thread / phase=shifting",
            "related_person_id": "person:harbor",
            "group_thread_id": "threaded_group:person:friend|person:harbor",
            "social_role": "companion",
            "open_tension": "timing_sensitive_reentry",
            "stability": 0.56,
            "learning_mode_focus": "repair_probe",
            "social_experiment_focus": "repair_signal_probe",
        },
    )

    assert entry.working_memory_summary is not None
    assert entry.working_memory_summary["focus_anchor"] == "harbor slope"
    assert "Working memory:" in entry.text
    assert entry.working_memory_signature_summary is not None
    assert "Working memory signature:" in entry.text
    assert "recurrence=2.60" in entry.text
    assert "seed=0.44" in entry.text
    assert "theme=meaning" in entry.text
    assert "theme_strength=0.63" in entry.text
    assert entry.working_memory_replay_summary is not None
    assert "Working memory replay:" in entry.text
    assert entry.long_term_theme_summary is not None
    assert "Long-term theme:" in entry.text
    assert "summary=fragile harbor promise" in entry.text
    assert entry.identity_arc_summary is not None
    assert "Identity arc:" in entry.text
    assert "kind=repairing_bond" in entry.text
    assert entry.relation_arc_summary is not None
    assert "Relation arc:" in entry.text
    assert "kind=repairing_relation" in entry.text
    registry_summary = manager.identity_arc_registry_summary()
    assert registry_summary["dominant_arc_kind"] == "repairing_bond"
    assert registry_summary["total_arcs"] == 1
    relation_registry_summary = manager.relation_arc_registry_summary()
    assert relation_registry_summary["dominant_arc_kind"] == "repairing_relation"
    assert relation_registry_summary["total_arcs"] == 1


def test_diary_manager_updates_identity_arc_registry_across_days() -> None:
    manager = DiaryManager()

    manager.record_daily_entry(
        day=date(2026, 3, 15),
        metrics={"entropy": 0.42, "enthalpy_mean": 0.31, "dissipation": 0.08},
        top_axes=["valence"],
        catalyst_highlights=[],
        gentle_quotes=[],
        rest_snapshot={"active": False},
        loop_alert=False,
        fatigue_flag=False,
        use_llm=False,
        identity_arc_summary={
            "arc_kind": "repairing_bond",
            "phase": "shifting",
            "summary": "repair is gathering around a relationship thread / phase=shifting / anchor=harbor slope",
            "memory_anchor": "harbor slope",
            "related_person_id": "person:harbor",
            "long_term_theme_kind": "meaning",
            "long_term_theme_focus": "quiet harbor promise",
            "stability": 0.58,
        },
    )
    manager.record_daily_entry(
        day=date(2026, 3, 16),
        metrics={"entropy": 0.38, "enthalpy_mean": 0.29, "dissipation": 0.06},
        top_axes=["valence"],
        catalyst_highlights=[],
        gentle_quotes=[],
        rest_snapshot={"active": False},
        loop_alert=False,
        fatigue_flag=False,
        use_llm=False,
        identity_arc_summary={
            "arc_kind": "repairing_bond",
            "phase": "holding",
            "summary": "repair is holding more steadily around the same relationship thread",
            "memory_anchor": "harbor slope",
            "related_person_id": "person:harbor",
            "long_term_theme_kind": "meaning",
            "long_term_theme_focus": "quiet harbor promise",
            "stability": 0.71,
        },
    )

    payload = manager.to_json()
    restored = DiaryManager.from_json(payload)
    registry_summary = restored.identity_arc_registry_summary()

    assert registry_summary["total_arcs"] == 1
    assert registry_summary["top_arcs"][0]["days_seen"] == 2
    assert registry_summary["top_arcs"][0]["phase"] == "holding"
    assert registry_summary["top_arcs"][0]["stability_peak"] == 0.71


def test_diary_manager_updates_relation_arc_registry_across_days() -> None:
    manager = DiaryManager()

    manager.record_daily_entry(
        day=date(2026, 3, 15),
        metrics={"entropy": 0.42, "enthalpy_mean": 0.31, "dissipation": 0.08},
        top_axes=["valence"],
        catalyst_highlights=[],
        gentle_quotes=[],
        rest_snapshot={"active": False},
        loop_alert=False,
        fatigue_flag=False,
        use_llm=False,
        relation_arc_summary={
            "arc_kind": "repairing_relation",
            "phase": "shifting",
            "summary": "repair is gathering around a companion thread",
            "related_person_id": "person:harbor",
            "group_thread_id": "threaded_group:person:friend|person:harbor",
            "social_role": "companion",
            "topology_focus": "threaded_group",
            "stability": 0.56,
            "learning_mode_focus": "repair_probe",
            "social_experiment_focus": "repair_signal_probe",
        },
    )
    manager.record_daily_entry(
        day=date(2026, 3, 16),
        metrics={"entropy": 0.38, "enthalpy_mean": 0.29, "dissipation": 0.06},
        top_axes=["valence"],
        catalyst_highlights=[],
        gentle_quotes=[],
        rest_snapshot={"active": False},
        loop_alert=False,
        fatigue_flag=False,
        use_llm=False,
        relation_arc_summary={
            "arc_kind": "repairing_relation",
            "phase": "integrating",
            "summary": "repair is holding more steadily around the same companion thread",
            "related_person_id": "person:harbor",
            "group_thread_id": "threaded_group:person:friend|person:harbor",
            "social_role": "companion",
            "topology_focus": "threaded_group",
            "stability": 0.74,
            "learning_mode_focus": "integrate_and_commit",
            "social_experiment_focus": "confirm_shared_direction",
        },
    )

    restored = DiaryManager.from_json(manager.to_json())
    registry_summary = restored.relation_arc_registry_summary()

    assert registry_summary["total_arcs"] == 1
    assert registry_summary["dominant_arc_kind"] == "repairing_relation"
    assert registry_summary["top_arcs"][0]["days_seen"] == 2
    assert registry_summary["top_arcs"][0]["phase"] == "integrating"
    assert registry_summary["top_arcs"][0]["stability_peak"] == 0.74
