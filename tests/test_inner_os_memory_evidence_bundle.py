from inner_os.memory_evidence_bundle import (
    MemoryEvidenceBundle,
    MemoryEvidenceItem,
    ReentryContext,
    TemporalConstraint,
    build_memory_evidence_bundle,
)
from inner_os.schemas import INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA


def test_memory_evidence_bundle_roundtrip_preserves_temporal_fields() -> None:
    bundle = build_memory_evidence_bundle(
        cue_text="harbor promise",
        focus="latest preference and same-group reentry",
        facts_current=[
            MemoryEvidenceItem(
                evidence_id="pref:tea:new",
                kind="preference",
                summary="prefers warm tea now",
                temporal_status="current",
                source_session_id="s2",
                weight=0.84,
                related_person_id="person:harbor",
            )
        ],
        facts_superseded=[
            {
                "evidence_id": "pref:tea:old",
                "kind": "preference",
                "summary": "used to prefer coffee",
                "temporal_status": "superseded",
                "source_session_id": "s1",
                "weight": 0.42,
            }
        ],
        timeline_events=[
            {
                "evidence_id": "event:repair-window",
                "kind": "event",
                "summary": "repair thread reopened after the harbor meeting",
                "temporal_status": "timeline",
                "group_thread_id": "thread:harbor",
            }
        ],
        temporal_constraints=[
            TemporalConstraint(
                kind="latest_update",
                summary="prefer the newest corrected preference",
                focus="latest",
                priority=0.88,
            )
        ],
        reentry_contexts=[
            ReentryContext(
                window="next_same_group_window",
                summary="reopen during the same harbor group",
                related_person_id="person:harbor",
                group_thread_id="thread:harbor",
                priority=0.73,
            )
        ],
        source_refs=["session:s1", "session:s2"],
        ambiguity_notes=["time anchor is approximate"],
    )

    payload = bundle.to_dict()
    restored = MemoryEvidenceBundle.from_mapping(payload)

    assert restored.schema == INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA
    assert restored.focus == "latest preference and same-group reentry"
    assert restored.facts_current[0].summary == "prefers warm tea now"
    assert restored.facts_superseded[0].temporal_status == "superseded"
    assert restored.timeline_events[0].group_thread_id == "thread:harbor"
    assert restored.temporal_constraints[0].kind == "latest_update"
    assert restored.reentry_contexts[0].window == "next_same_group_window"
    assert restored.source_refs == ("session:s1", "session:s2")
    assert restored.ambiguity_notes == ("time anchor is approximate",)


def test_memory_evidence_bundle_drops_incomplete_entries() -> None:
    bundle = build_memory_evidence_bundle(
        facts_current=[
            {"evidence_id": "", "kind": "fact", "summary": "missing id"},
            {"evidence_id": "ok", "kind": "fact", "summary": "kept"},
        ],
        temporal_constraints=[
            {"kind": "", "summary": "missing kind"},
            {"kind": "latest", "summary": "prefer latest"},
        ],
        reentry_contexts=[
            {"window": "", "summary": "missing window"},
            {"window": "next_private_window", "summary": "reopen privately"},
        ],
    )

    assert len(bundle.facts_current) == 1
    assert bundle.facts_current[0].evidence_id == "ok"
    assert len(bundle.temporal_constraints) == 1
    assert bundle.temporal_constraints[0].kind == "latest"
    assert len(bundle.reentry_contexts) == 1
    assert bundle.reentry_contexts[0].window == "next_private_window"
