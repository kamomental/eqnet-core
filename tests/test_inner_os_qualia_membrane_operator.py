from inner_os.qualia_membrane_operator import derive_qualia_membrane_temporal_bias


def test_qualia_membrane_temporal_bias_prefers_reentry_when_window_and_relation_align() -> None:
    bias = derive_qualia_membrane_temporal_bias(
        memory_evidence_bundle={
            "schema": "inner_os_memory_evidence_bundle/v1",
            "cue_text": "harbor promise",
            "facts_current": [
                {
                    "evidence_id": "ep-2",
                    "kind": "relationship_trace",
                    "summary": "prefers warm tea now",
                    "temporal_status": "current",
                    "weight": 0.82,
                    "related_person_id": "person:harbor",
                }
            ],
            "facts_superseded": [],
            "timeline_events": [
                {
                    "evidence_id": "timeline:ep-2",
                    "kind": "timeline_event",
                    "summary": "prefers warm tea now",
                    "temporal_status": "timeline",
                    "weight": 0.74,
                    "group_thread_id": "thread:harbor",
                }
            ],
            "temporal_constraints": [
                {
                    "kind": "long_term_theme",
                    "summary": "keep the harbor routine coherent",
                    "focus": "harbor routine",
                    "priority": 0.63,
                    "source": "long_term_theme",
                }
            ],
            "reentry_contexts": [
                {
                    "window": "next_same_group_window",
                    "summary": "reopen in the same harbor group",
                    "related_person_id": "person:harbor",
                    "group_thread_id": "thread:harbor",
                    "culture_id": "coastal",
                    "community_id": "harbor_collective",
                    "priority": 0.68,
                }
            ],
            "source_refs": ["ep-2"],
            "ambiguity_notes": [],
        },
        current_state={"agenda_window_bias": 0.68, "related_person_id": "person:harbor"},
        world_snapshot={"culture_id": "coastal", "community_id": "harbor_collective"},
    )

    assert bias.reentry_pull > 0.5
    assert bias.relation_reentry_pull > 0.4
    assert bias.continuity_pressure > 0.2
    assert bias.dominant_mode == "reentry"
    assert "temporal_membrane_reentry" in bias.cues


def test_qualia_membrane_temporal_bias_surfaces_supersession_pressure() -> None:
    bias = derive_qualia_membrane_temporal_bias(
        memory_evidence_bundle={
            "schema": "inner_os_memory_evidence_bundle/v1",
            "cue_text": "old preference",
            "facts_current": [],
            "facts_superseded": [
                {
                    "evidence_id": "old-1",
                    "kind": "memory",
                    "summary": "liked bitter coffee",
                    "temporal_status": "superseded",
                    "weight": 0.84,
                }
            ],
            "timeline_events": [],
            "temporal_constraints": [],
            "reentry_contexts": [],
            "source_refs": ["old-1"],
            "ambiguity_notes": ["no_current_fact_selected"],
        },
    )

    assert bias.supersession_pressure > 0.5
    assert bias.timeline_coherence < 0.2
    assert bias.dominant_mode == "supersede"
    assert "temporal_membrane_supersession" in bias.cues
