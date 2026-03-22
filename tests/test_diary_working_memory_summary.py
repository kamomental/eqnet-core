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
