from inner_os.memory_bridge import (
    collect_runtime_memory_candidates,
    memory_reference_to_record,
    observed_vision_to_record,
)


def test_memory_reference_to_record_maps_source_quality() -> None:
    record = memory_reference_to_record(
        {
            "reply": "I remember the harbor slope.",
            "fidelity": 0.9,
            "meta": {"source_class": "self", "audit_event": "OK", "memory_kind": "experience"},
            "candidate": {"label": "harbor slope"},
        },
        relational_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "surface_policy_active": 1.0,
            "surface_policy_level": "layered",
            "surface_policy_intent": "clarify",
        },
    )
    assert record is not None
    assert record["kind"] == "verified"
    assert record["culture_id"] == "coastal"
    assert record["policy_hint"] == "experience"
    assert record["provenance"] == "eqnet_memory_reference"
    assert record["surface_policy_active"] == 1.0
    assert record["surface_policy_level"] == "layered"
    assert record["surface_policy_intent"] == "clarify"


def test_observed_vision_to_record_skips_suppressed_entries() -> None:
    assert observed_vision_to_record({"suppressed": True, "text": "scene"}, relational_context={}) is None


def test_collect_runtime_memory_candidates_dedupes_and_preserves_order() -> None:
    records = collect_runtime_memory_candidates(
        recall_payload={"kind": "observed_real", "memory_anchor": "harbor slope", "summary": "harbor slope", "provenance": "recall"},
        memory_reference={
            "reply": "I remember the harbor slope.",
            "fidelity": 0.9,
            "meta": {"source_class": "self", "audit_event": "OK"},
            "candidate": {"label": "harbor slope"},
        },
        vision_entry={"id": "vision-1", "text": "harbor slope and signboard", "memory_anchor": "harbor slope"},
        relational_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "surface_policy_active": 1.0,
            "surface_policy_level": "layered",
            "surface_policy_intent": "clarify",
        },
    )
    assert len(records) == 3
    assert records[0]["provenance"] == "recall"
    assert records[1]["provenance"] == "eqnet_memory_reference"
    assert records[2]["provenance"] == "observed_vision"
    assert records[1]["surface_policy_intent"] == "clarify"
    assert records[2]["surface_policy_active"] == 1.0
