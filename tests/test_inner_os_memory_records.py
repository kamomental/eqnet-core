from pathlib import Path

from inner_os.memory_records import normalize_memory_record


def test_normalize_memory_record_preserves_memory_classes() -> None:
    observed = normalize_memory_record({"kind": "observed_real", "summary": "harbor walk", "culture_id": "coastal"})
    reconstructed = normalize_memory_record({"kind": "reconstructed", "text": "maybe we turned here"})
    verified = normalize_memory_record({"kind": "verified", "summary": "building was rebuilt", "confidence": 0.9})
    sim = normalize_memory_record({"kind": "experienced_sim", "summary": "trial route"})
    transferred = normalize_memory_record({"kind": "transferred_learning", "summary": "pause before commitment"})
    identity = normalize_memory_record({"kind": "identity_trace", "summary": "slow trace", "continuity_score": 0.62})
    relation = normalize_memory_record({"kind": "relationship_trace", "summary": "slow relation", "attachment": 0.68, "profile_scope": "community_role_place", "access_count": 2, "primed_weight": 0.31})
    context_shift = normalize_memory_record({"kind": "context_shift_trace", "summary": "slow context shift", "transition_intensity": 0.52})
    community_profile = normalize_memory_record({
        "kind": "community_profile_trace",
        "summary": "slow communal pattern",
        "culture_resonance": 0.71,
        "community_resonance": 0.75,
        "ritual_memory": 0.62,
        "institutional_memory": 0.58,
        "community_profile_pressure": 0.68,
    })

    assert observed["kind"] == "observed_real"
    assert observed["culture_id"] == "coastal"
    assert reconstructed["provenance"] == "reconstruction"
    assert verified["provenance"] == "verification"
    assert sim["provenance"] == "simulation"
    assert transferred["provenance"] == "simulation_transfer"
    assert identity["provenance"] == "inner_state"
    assert identity["continuity_score"] == 0.62
    assert relation["provenance"] == "inner_relation"
    assert relation["attachment"] == 0.68
    assert context_shift["provenance"] == "inner_context"
    assert context_shift["transition_intensity"] == 0.52
    assert community_profile["provenance"] == "inner_community"
    assert community_profile["community_profile_pressure"] == 0.68
    assert relation["access_count"] == 2
    assert relation["primed_weight"] == 0.31


from inner_os.memory_core import MemoryCore


def test_memory_core_build_recall_payload_exposes_record_contract(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {
            "kind": "verified",
            "summary": "harbor slope memory",
            "text": "harbor slope memory from a trusted recall",
            "memory_anchor": "harbor slope",
            "source_episode_id": "episode-7",
            "policy_hint": "experience",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        }
    ])
    payload = core.build_recall_payload("harbor slope")
    assert payload["memory_anchor"] == "harbor slope"
    assert payload["summary"] == "harbor slope memory"
    assert payload["record_kind"] == "verified"
    assert payload["record_provenance"] == "verification"
    assert payload["source_episode_id"] == "episode-7"
    assert payload["policy_hint"] == "experience"
    assert payload["culture_id"] == "coastal"
    assert payload["community_id"] == "harbor_collective"
    assert payload["social_role"] == "companion"


def test_memory_core_load_latest_identity_trace(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "identity_trace", "summary": "trace one", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "continuity_score": 0.58},
        {"kind": "identity_trace", "summary": "trace two", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "continuity_score": 0.67},
    ])
    latest = core.load_latest_identity_trace(culture_id="coastal", community_id="harbor_collective", social_role="companion")
    assert latest["summary"] == "trace two"
    assert latest["continuity_score"] == 0.67


def test_memory_core_load_latest_profile_record(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "relationship_trace", "summary": "relation one", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "memory_anchor": "harbor slope", "attachment": 0.58},
        {"kind": "relationship_trace", "summary": "relation two", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "memory_anchor": "harbor slope", "attachment": 0.69, "related_person_id": "user"},
    ])
    latest = core.load_latest_profile_record(kind="relationship_trace", culture_id="coastal", community_id="harbor_collective", social_role="companion", memory_anchor="harbor slope")
    assert latest["summary"] == "relation two"
    assert latest["attachment"] == 0.69
    person_latest = core.load_latest_profile_record(
        kind="relationship_trace",
        culture_id="coastal",
        community_id="harbor_collective",
        social_role="companion",
        memory_anchor="harbor slope",
        related_person_id="user",
    )
    assert person_latest["summary"] == "relation two"
    assert person_latest["related_person_id"] == "user"


def test_memory_core_build_recall_payload_biases_toward_relational_match(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "verified", "summary": "generic harbor memory", "text": "generic harbor memory", "memory_anchor": "harbor", "culture_id": "default", "community_id": "other", "social_role": "visitor"},
        {"kind": "relationship_trace", "summary": "harbor slope relation", "text": "harbor slope relation", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "related_person_id": "user"},
    ])
    payload = core.build_recall_payload(
        "harbor slope",
        bias_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "related_person_id": "user",
            "memory_anchor": "harbor slope",
            "caution_bias": 0.42,
            "affiliation_bias": 0.68,
            "continuity_score": 0.63,
        },
    )
    assert payload["record_kind"] == "relationship_trace"
    assert payload["memory_anchor"] == "harbor slope"
    assert payload["related_person_id"] == "user"


def test_memory_core_build_recall_payload_biases_by_environment_pressure(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "verified", "summary": "verified shelter route", "text": "verified shelter route", "memory_anchor": "shelter", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        {"kind": "reconstructed", "summary": "ritual recollection", "text": "ritual recollection", "memory_anchor": "shelter", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
    ])
    payload = core.build_recall_payload(
        "shelter",
        bias_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "shelter",
            "caution_bias": 0.72,
            "affiliation_bias": 0.41,
            "continuity_score": 0.52,
            "hazard_pressure": 0.7,
            "institutional_pressure": 0.12,
            "ritual_pressure": 0.08,
            "resource_pressure": 0.2,
        },
    )
    assert payload["record_kind"] == "verified"


def test_memory_core_build_recall_payload_biases_by_development_profile(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "observed_real", "summary": "plain harbor fact", "text": "plain harbor fact", "memory_anchor": "harbor", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        {"kind": "relationship_trace", "summary": "shared harbor bond", "text": "shared harbor bond", "memory_anchor": "harbor", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
    ])
    payload = core.build_recall_payload(
        "harbor",
        bias_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor",
            "caution_bias": 0.35,
            "affiliation_bias": 0.62,
            "continuity_score": 0.58,
            "kind_biases": {
                "relationship_trace": 0.18,
                "observed_real": 0.01,
            },
        },
    )
    assert payload["record_kind"] == "relationship_trace"


def test_memory_core_build_recall_payload_biases_toward_stable_records_under_context_roughness(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "verified", "summary": "verified shelter route", "text": "verified shelter route", "memory_anchor": "shelter", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        {"kind": "reconstructed", "summary": "fragile reinterpretation", "text": "fragile reinterpretation", "memory_anchor": "shelter", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
    ])
    payload = core.build_recall_payload(
        "shelter",
        bias_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "shelter",
            "caution_bias": 0.52,
            "affiliation_bias": 0.48,
            "continuity_score": 0.44,
            "terrain_transition_roughness": 0.72,
        },
    )
    assert payload["record_kind"] == "verified"
    assert payload["anchored_allocation"] is not None
    assert payload["reconstructive_allocation"] is not None
    assert payload["anchored_allocation"] > payload["reconstructive_allocation"]

def test_memory_core_recall_penalizes_high_tentative_reconstructed_records(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "reconstructed", "summary": "steady harbor read", "text": "steady harbor read", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "tentative_bias": 0.08},
        {"kind": "reconstructed", "summary": "highly tentative harbor read", "text": "highly tentative harbor read", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "tentative_bias": 0.82},
    ])
    payload = core.build_recall_payload(
        "harbor slope",
        bias_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
            "caution_bias": 0.52,
            "affiliation_bias": 0.48,
            "continuity_score": 0.44,
            "terrain_transition_roughness": 0.72,
        },
    )
    assert payload["summary"] == "steady harbor read"
    assert payload["tentative_bias"] == 0.08


def test_memory_core_recall_uses_interaction_afterglow_to_prefer_grounded_or_relational_records(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "observed_real", "summary": "grounded harbor observation", "text": "grounded harbor observation", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        {"kind": "relationship_trace", "summary": "gentle harbor bond", "text": "gentle harbor bond", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        {"kind": "reconstructed", "summary": "speculative harbor read", "text": "speculative harbor read", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "tentative_bias": 0.41},
    ])
    payload = core.build_recall_payload(
        "harbor slope",
        bias_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
            "continuity_score": 0.56,
            "affiliation_bias": 0.54,
            "interaction_afterglow": 0.52,
            "interaction_afterglow_intent": "check_in",
        },
    )
    assert payload["record_kind"] in {"observed_real", "relationship_trace"}
    assert payload["record_kind"] != "reconstructed"


def test_memory_core_recall_reopens_low_tentative_reconstructed_records_under_recovery(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "verified", "summary": "verified harbor route", "text": "verified harbor route", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        {"kind": "reconstructed", "summary": "careful harbor reinterpretation", "text": "careful harbor reinterpretation", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "tentative_bias": 0.08, "recovery_reopening": 0.3},
    ])
    closed = core.build_recall_payload(
        "harbor slope",
        bias_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
            "caution_bias": 0.58,
            "affiliation_bias": 0.35,
            "continuity_score": 0.46,
            "relational_clarity": 0.62,
            "anticipation_tension": 0.12,
            "meaning_inertia": 0.18,
            "recovery_reopening": 0.0,
        },
    )
    reopened = core.build_recall_payload(
        "harbor slope",
        bias_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
            "caution_bias": 0.58,
            "affiliation_bias": 0.35,
            "continuity_score": 0.46,
            "relational_clarity": 0.62,
            "anticipation_tension": 0.12,
            "meaning_inertia": 0.18,
            "recovery_reopening": 0.72,
        },
    )
    assert closed["record_kind"] == "verified"
    assert reopened["record_kind"] == "reconstructed"
    assert reopened["summary"] == "careful harbor reinterpretation"
    assert reopened["reconstructive_allocation"] > reopened["anchored_allocation"]


def test_memory_core_touch_record_usage_updates_access_and_priming(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    appended = core.append_records([
        {"kind": "observed_real", "summary": "harbor slope walk", "memory_anchor": "harbor slope"}
    ])
    touched = core.touch_record_usage(str(appended[0]["id"]))
    assert touched["access_count"] == 1.0
    assert touched["primed_weight"] >= 0.28
    payload = core.build_recall_payload("harbor slope")
    assert payload["access_count"] == 1.0
    assert payload["primed_weight"] >= 0.28


def test_memory_core_recall_forgetting_pressure_penalizes_reconstructed_records(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {"kind": "verified", "summary": "verified harbor route", "text": "verified harbor route", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        {"kind": "reconstructed", "summary": "careful harbor reinterpretation", "text": "careful harbor reinterpretation", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "tentative_bias": 0.08, "primed_weight": 0.7, "last_accessed_at": 9999999999},
    ])
    payload = core.build_recall_payload(
        "harbor slope",
        bias_context={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
            "affiliation_bias": 0.44,
            "relational_clarity": 0.58,
            "recovery_reopening": 0.32,
            "forgetting_pressure": 0.82,
            "replay_horizon": 1,
        },
    )
    assert payload["record_kind"] == "verified"
    assert payload["forgetting_pressure"] == 0.82
    assert payload["replay_horizon"] == 1
