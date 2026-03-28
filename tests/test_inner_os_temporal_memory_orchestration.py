from inner_os.temporal_memory_orchestration import build_temporal_memory_evidence_bundle


def test_temporal_memory_orchestration_builds_reentry_and_constraints() -> None:
    bundle = build_temporal_memory_evidence_bundle(
        cue_text="harbor promise",
        current_state={
            "agenda_window_state": "next_same_group_window",
            "agenda_window_reason": "wait for the harbor group to reopen it",
            "agenda_window_bias": 0.68,
            "group_thread_id": "thread:harbor",
        },
        world_snapshot={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
        recall_payload={
            "summary": "prefers warm tea now",
            "memory_anchor": "harbor slope",
            "record_kind": "relationship_trace",
            "source_episode_id": "ep-2",
            "primed_weight": 0.82,
            "related_person_id": "person:harbor",
            "replay_signature_focus": "harbor",
            "replay_signature_strength": 0.74,
            "semantic_seed_focus": "warm tea",
            "semantic_seed_strength": 0.52,
            "long_term_theme_focus": "harbor routine",
            "long_term_theme_strength": 0.63,
            "relation_seed_summary": "reopen in the same harbor group",
        },
        retrieval_summary={"hits": [{"id": "vision-1"}]},
    )

    assert bundle["facts_current"][0]["summary"] == "prefers warm tea now"
    assert bundle["timeline_events"][0]["temporal_status"] == "timeline"
    assert {item["kind"] for item in bundle["temporal_constraints"]} >= {
        "replay_signature",
        "semantic_seed",
        "long_term_theme",
    }
    assert bundle["reentry_contexts"][0]["window"] == "next_same_group_window"
    assert bundle["reentry_contexts"][0]["group_thread_id"] == "thread:harbor"
    assert "vision-1" in bundle["source_refs"]
