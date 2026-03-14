from inner_os.reinterpretation_core import ReinterpretationCore


def test_reinterpretation_core_reframes_memory_under_social_pressure() -> None:
    core = ReinterpretationCore()
    snapshot = core.snapshot(
        recall_payload={
            "record_kind": "observed_real",
            "summary": "harbor slope walk",
            "memory_anchor": "harbor slope",
        },
        current_state={
            "norm_pressure": 0.72,
            "trust_bias": 0.34,
            "belonging": 0.41,
            "role_commitment": 0.69,
            "temporal_pressure": 0.46,
            "continuity_score": 0.34,
            "social_grounding": 0.39,
            "recent_strain": 0.61,
        },
        relational_world={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
        environment_pressure={
            "resource_pressure": 0.48,
            "hazard_pressure": 0.36,
            "ritual_pressure": 0.42,
            "institutional_pressure": 0.58,
            "social_density": 0.54,
        },
    )
    assert snapshot.mode in {"social_reframing", "reflective_reconsolidation"}
    assert snapshot.social_self_pressure > 0.4
    assert snapshot.reflective_tension > 0.3
    assert snapshot.summary


def test_reinterpretation_core_builds_reconstructed_record() -> None:
    core = ReinterpretationCore()
    record = core.build_reconstructed_record(
        recall_payload={
            "record_kind": "observed_real",
            "summary": "harbor slope walk",
            "text": "we walked past the signboard on the slope",
            "memory_anchor": "harbor slope",
            "source_episode_id": "ep-1",
            "policy_hint": "gentle_clarification_before_commitment",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
        current_state={
            "norm_pressure": 0.74,
            "trust_bias": 0.31,
            "belonging": 0.4,
            "role_commitment": 0.7,
            "temporal_pressure": 0.5,
            "continuity_score": 0.31,
            "social_grounding": 0.36,
            "recent_strain": 0.58,
            "terrain_transition_roughness": 0.14,
        },
        relational_world={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
        environment_pressure={
            "resource_pressure": 0.5,
            "hazard_pressure": 0.34,
            "ritual_pressure": 0.41,
            "institutional_pressure": 0.62,
            "social_density": 0.57,
        },
        reply_text="I remember the slope differently now.",
    )
    assert record is not None
    assert record["kind"] == "reconstructed"
    assert record["reinterpretation_mode"]
    assert record["environment_summary"]
    assert record["meaning_shift"] >= 0.16
    assert record["terrain_transition_roughness"] == 0.14


def test_reinterpretation_core_marks_community_transition_reframing() -> None:
    core = ReinterpretationCore()
    snapshot = core.snapshot(
        recall_payload={
            "record_kind": "observed_real",
            "summary": "harbor slope walk",
            "memory_anchor": "harbor slope",
        },
        current_state={
            "norm_pressure": 0.58,
            "trust_bias": 0.36,
            "belonging": 0.42,
            "role_commitment": 0.61,
            "temporal_pressure": 0.4,
            "continuity_score": 0.38,
            "social_grounding": 0.35,
            "recent_strain": 0.62,
            "culture_resonance": 0.41,
            "community_resonance": 0.29,
        },
        relational_world={
            "culture_id": "coastal",
            "community_id": "new_collective",
            "social_role": "companion",
        },
        environment_pressure={
            "resource_pressure": 0.44,
            "hazard_pressure": 0.32,
            "ritual_pressure": 0.28,
            "institutional_pressure": 0.57,
            "social_density": 0.52,
        },
        transition_signal={"transition_intensity": 0.7},
    )
    assert snapshot.mode == "community_transition_reframing"
    assert snapshot.meaning_shift > 0.2


def test_reinterpretation_core_builds_transition_marked_record() -> None:
    core = ReinterpretationCore()
    record = core.build_reconstructed_record(
        recall_payload={
            "record_kind": "observed_real",
            "summary": "harbor slope walk",
            "text": "we walked past the signboard on the slope",
            "memory_anchor": "harbor slope",
            "source_episode_id": "ep-1",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
        current_state={
            "norm_pressure": 0.62,
            "trust_bias": 0.34,
            "belonging": 0.41,
            "role_commitment": 0.66,
            "temporal_pressure": 0.44,
            "continuity_score": 0.36,
            "social_grounding": 0.34,
            "recent_strain": 0.61,
            "culture_resonance": 0.39,
            "community_resonance": 0.27,
            "terrain_transition_roughness": 0.48,
        },
        relational_world={
            "culture_id": "coastal",
            "community_id": "new_collective",
            "social_role": "companion",
        },
        environment_pressure={
            "resource_pressure": 0.46,
            "hazard_pressure": 0.31,
            "ritual_pressure": 0.25,
            "institutional_pressure": 0.55,
            "social_density": 0.5,
        },
        transition_signal={"transition_intensity": 0.7},
        reply_text="it feels different after moving here",
    )
    assert record is not None
    assert record["reinterpretation_mode"] == "community_transition_reframing"
    assert record["transition_intensity"] == 0.7
    assert record["terrain_transition_roughness"] == 0.48


def test_reinterpretation_core_slows_meaning_under_transition_roughness() -> None:
    core = ReinterpretationCore()
    base_state = {
        "norm_pressure": 0.61,
        "trust_bias": 0.35,
        "belonging": 0.43,
        "role_commitment": 0.64,
        "temporal_pressure": 0.43,
        "continuity_score": 0.37,
        "social_grounding": 0.36,
        "recent_strain": 0.59,
        "culture_resonance": 0.34,
        "community_resonance": 0.26,
    }
    low = core.snapshot(
        recall_payload={"record_kind": "observed_real", "summary": "harbor slope walk", "memory_anchor": "harbor slope"},
        current_state={**base_state, "terrain_transition_roughness": 0.0},
        relational_world={"culture_id": "coastal", "community_id": "new_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.44, "hazard_pressure": 0.3, "ritual_pressure": 0.26, "institutional_pressure": 0.54, "social_density": 0.49},
        transition_signal={"transition_intensity": 0.68},
    )
    high = core.snapshot(
        recall_payload={"record_kind": "observed_real", "summary": "harbor slope walk", "memory_anchor": "harbor slope"},
        current_state={**base_state, "terrain_transition_roughness": 0.72},
        relational_world={"culture_id": "coastal", "community_id": "new_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.44, "hazard_pressure": 0.3, "ritual_pressure": 0.26, "institutional_pressure": 0.54, "social_density": 0.49},
        transition_signal={"transition_intensity": 0.68},
    )
    assert high.terrain_transition_roughness == 0.72
    assert high.meaning_shift < low.meaning_shift
    assert abs(high.meaning_push + high.meaning_hold - 1.0) < 1e-6
    assert high.narrative_pull < low.narrative_pull

def test_reinterpretation_core_lowers_confidence_when_transition_is_rough() -> None:
    core = ReinterpretationCore()
    common = {
        "recall_payload": {
            "record_kind": "observed_real",
            "summary": "harbor slope walk",
            "text": "we walked past the signboard on the slope",
            "memory_anchor": "harbor slope",
            "source_episode_id": "ep-1",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
        "relational_world": {"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        "environment_pressure": {"resource_pressure": 0.46, "hazard_pressure": 0.31, "ritual_pressure": 0.25, "institutional_pressure": 0.55, "social_density": 0.5},
        "transition_signal": {"transition_intensity": 0.7},
        "reply_text": "it feels different after moving here",
    }
    low = core.build_reconstructed_record(current_state={"norm_pressure": 0.62, "trust_bias": 0.34, "belonging": 0.41, "role_commitment": 0.66, "temporal_pressure": 0.44, "continuity_score": 0.36, "social_grounding": 0.34, "recent_strain": 0.61, "culture_resonance": 0.39, "community_resonance": 0.27, "terrain_transition_roughness": 0.08}, **common)
    high = core.build_reconstructed_record(current_state={"norm_pressure": 0.62, "trust_bias": 0.34, "belonging": 0.41, "role_commitment": 0.66, "temporal_pressure": 0.44, "continuity_score": 0.36, "social_grounding": 0.34, "recent_strain": 0.61, "culture_resonance": 0.39, "community_resonance": 0.27, "terrain_transition_roughness": 0.58}, **common)
    assert low is not None and high is not None
    assert high["tentative_bias"] > low["tentative_bias"]
    assert high["confidence"] < low["confidence"]


def test_reinterpretation_core_prefers_relational_check_in_over_meaning_push() -> None:
    core = ReinterpretationCore()
    base_state = {
        "norm_pressure": 0.61,
        "trust_bias": 0.35,
        "belonging": 0.43,
        "role_commitment": 0.64,
        "temporal_pressure": 0.43,
        "continuity_score": 0.37,
        "social_grounding": 0.36,
        "recent_strain": 0.59,
        "culture_resonance": 0.34,
        "community_resonance": 0.26,
        "terrain_transition_roughness": 0.18,
    }
    low = core.snapshot(
        recall_payload={"record_kind": "observed_real", "summary": "harbor slope walk", "memory_anchor": "harbor slope"},
        current_state={**base_state, "interaction_afterglow": 0.0},
        relational_world={"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.44, "hazard_pressure": 0.3, "ritual_pressure": 0.26, "institutional_pressure": 0.54, "social_density": 0.49},
        transition_signal={"transition_intensity": 0.18},
    )
    high = core.snapshot(
        recall_payload={"record_kind": "observed_real", "summary": "harbor slope walk", "memory_anchor": "harbor slope"},
        current_state={**base_state, "interaction_afterglow": 0.52, "interaction_afterglow_intent": "check_in"},
        relational_world={"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.44, "hazard_pressure": 0.3, "ritual_pressure": 0.26, "institutional_pressure": 0.54, "social_density": 0.49},
        transition_signal={"transition_intensity": 0.18},
    )
    assert high.mode == "relational_check_in_reframing"
    assert high.meaning_shift < low.meaning_shift
    assert "relational check-in" in high.summary


def test_reinterpretation_core_reopens_when_recovery_returns() -> None:
    core = ReinterpretationCore()
    snapshot = core.snapshot(
        recall_payload={"record_kind": "observed_real", "summary": "harbor slope walk", "memory_anchor": "harbor slope"},
        current_state={
            "norm_pressure": 0.52,
            "trust_bias": 0.58,
            "belonging": 0.49,
            "role_commitment": 0.43,
            "temporal_pressure": 0.24,
            "continuity_score": 0.72,
            "social_grounding": 0.68,
            "recent_strain": 0.18,
            "terrain_transition_roughness": 0.16,
            "relational_clarity": 0.54,
            "meaning_inertia": 0.22,
            "anticipation_tension": 0.18,
            "recovery_reopening": 0.34,
        },
        relational_world={"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.18, "hazard_pressure": 0.12, "ritual_pressure": 0.08, "institutional_pressure": 0.18, "social_density": 0.22},
        transition_signal={"transition_intensity": 0.12},
    )
    assert snapshot.mode == "reopening_reframing"
    assert snapshot.recovery_reopening == 0.34
    assert snapshot.meaning_push > snapshot.meaning_hold
    assert "reopening" in snapshot.summary


def test_reinterpretation_core_carries_check_in_afterglow_into_reconstructed_record() -> None:
    core = ReinterpretationCore()
    record = core.build_reconstructed_record(
        recall_payload={
            "record_kind": "observed_real",
            "summary": "harbor slope walk",
            "text": "we walked past the signboard on the slope",
            "memory_anchor": "harbor slope",
            "source_episode_id": "ep-1",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
        current_state={
            "norm_pressure": 0.62,
            "trust_bias": 0.34,
            "belonging": 0.41,
            "role_commitment": 0.66,
            "temporal_pressure": 0.44,
            "continuity_score": 0.36,
            "social_grounding": 0.34,
            "recent_strain": 0.61,
            "culture_resonance": 0.39,
            "community_resonance": 0.27,
            "terrain_transition_roughness": 0.18,
            "interaction_afterglow": 0.48,
            "interaction_afterglow_intent": "check_in",
        },
        relational_world={"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.46, "hazard_pressure": 0.31, "ritual_pressure": 0.25, "institutional_pressure": 0.55, "social_density": 0.5},
        transition_signal={"transition_intensity": 0.18},
        reply_text="I want to stay with this gently first.",
    )
    assert record is not None
    assert record["reinterpretation_mode"] == "relational_check_in_reframing"
    assert "relational check-in" in record["reinterpretation_summary"]


def test_reinterpretation_core_holds_meaning_when_object_is_fragile() -> None:
    core = ReinterpretationCore()
    base_state = {
        "norm_pressure": 0.52,
        "trust_bias": 0.42,
        "belonging": 0.48,
        "role_commitment": 0.54,
        "temporal_pressure": 0.32,
        "continuity_score": 0.46,
        "social_grounding": 0.44,
        "recent_strain": 0.28,
        "terrain_transition_roughness": 0.12,
    }
    low = core.snapshot(
        recall_payload={"record_kind": "observed_real", "summary": "glass lantern", "memory_anchor": "harbor lantern"},
        current_state={**base_state, "fragility_guard": 0.0, "object_attachment": 0.0},
        relational_world={"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.2, "hazard_pressure": 0.16, "ritual_pressure": 0.24, "institutional_pressure": 0.18, "social_density": 0.2},
        transition_signal={"transition_intensity": 0.18},
    )
    high = core.snapshot(
        recall_payload={"record_kind": "observed_real", "summary": "glass lantern", "memory_anchor": "harbor lantern"},
        current_state={**base_state, "fragility_guard": 0.52, "object_attachment": 0.34},
        relational_world={"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.2, "hazard_pressure": 0.16, "ritual_pressure": 0.24, "institutional_pressure": 0.18, "social_density": 0.2},
        transition_signal={"transition_intensity": 0.18},
    )
    assert high.fragility_guard == 0.52
    assert high.meaning_shift < low.meaning_shift
    assert high.meaning_hold > low.meaning_hold


def test_reinterpretation_core_slows_meaning_under_defensive_salience() -> None:
    core = ReinterpretationCore()
    base_state = {
        "norm_pressure": 0.52,
        "trust_bias": 0.42,
        "belonging": 0.48,
        "role_commitment": 0.54,
        "temporal_pressure": 0.32,
        "continuity_score": 0.46,
        "social_grounding": 0.44,
        "recent_strain": 0.28,
        "terrain_transition_roughness": 0.12,
    }
    low = core.snapshot(
        recall_payload={"record_kind": "observed_real", "summary": "lantern near hand", "memory_anchor": "harbor lantern"},
        current_state={**base_state, "defensive_salience": 0.0, "approach_confidence": 0.0},
        relational_world={"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.2, "hazard_pressure": 0.16, "ritual_pressure": 0.24, "institutional_pressure": 0.18, "social_density": 0.2},
        transition_signal={"transition_intensity": 0.18},
    )
    high = core.snapshot(
        recall_payload={"record_kind": "observed_real", "summary": "lantern near hand", "memory_anchor": "harbor lantern"},
        current_state={**base_state, "defensive_salience": 0.48, "near_body_risk": 0.42, "approach_confidence": 0.08},
        relational_world={"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        environment_pressure={"resource_pressure": 0.2, "hazard_pressure": 0.16, "ritual_pressure": 0.24, "institutional_pressure": 0.18, "social_density": 0.2},
        transition_signal={"transition_intensity": 0.18},
    )
    assert high.defensive_salience == 0.48
    assert high.meaning_shift < low.meaning_shift
