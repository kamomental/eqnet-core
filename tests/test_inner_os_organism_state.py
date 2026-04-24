from inner_os.organism_state import (
    OrganismState,
    coerce_organism_state,
    derive_organism_state,
)


def test_organism_state_integrates_growth_qualia_heartbeat_epistemic_and_relation() -> None:
    state = derive_organism_state(
        previous_state=None,
        growth_state={
            "relational_trust": 0.66,
            "epistemic_maturity": 0.52,
            "expressive_range": 0.58,
            "residue_integration": 0.49,
            "playfulness_range": 0.44,
            "self_coherence": 0.61,
        },
        epistemic_state={
            "freshness": 0.72,
            "source_confidence": 0.68,
            "verification_pressure": 0.29,
            "stale_risk": 0.24,
            "epistemic_caution": 0.31,
        },
        qualia_structure_state={
            "phase": "echoing",
            "emergence": 0.55,
            "stability": 0.62,
            "memory_resonance": 0.59,
            "temporal_coherence": 0.57,
            "drift": 0.18,
            "protection_bias": 0.16,
        },
        heartbeat_structure_state={
            "dominant_reaction": "attune",
            "attunement": 0.58,
            "containment_bias": 0.24,
            "recovery_pull": 0.16,
            "bounce_room": 0.41,
            "response_tempo": 0.47,
            "entrainment": 0.54,
            "activation_drive": 0.48,
        },
        relation_competition_state={
            "dominant_person_id": "user",
            "dominant_score": 0.71,
            "competition_level": 0.12,
        },
        social_topology_state={
            "state": "one_to_one",
            "visibility_pressure": 0.12,
            "threading_pressure": 0.08,
            "hierarchy_pressure": 0.04,
        },
    )

    assert state.relation_focus == "user"
    assert state.social_mode == "one_to_one"
    assert state.attunement >= 0.0
    assert state.coherence >= 0.0
    assert state.grounding >= 0.0
    assert state.relation_pull >= 0.0
    assert state.dominant_posture in {
        "steady",
        "attune",
        "open",
        "play",
        "protect",
        "recover",
        "verify",
    }
    assert len(state.trace) == 1
    axes = state.to_packet_axes()
    assert axes["attunement"]["value"] >= 0.0
    assert axes["expression"]["value"] >= 0.0
    assert axes["relation"]["value"] >= 0.0


def test_organism_state_preserves_temporal_trace_and_can_shift_to_protect() -> None:
    previous = coerce_organism_state(
        {
            "attunement": 0.61,
            "coherence": 0.57,
            "grounding": 0.54,
            "protective_tension": 0.22,
            "expressive_readiness": 0.49,
            "play_window": 0.38,
            "relation_pull": 0.56,
            "social_exposure": 0.14,
            "dominant_posture": "attune",
            "relation_focus": "user",
            "social_mode": "one_to_one",
            "trace": [{"step": 1, "dominant_posture": "attune"}],
        }
    )

    state = derive_organism_state(
        previous_state=previous,
        growth_state={
            "relational_trust": 0.44,
            "epistemic_maturity": 0.41,
            "expressive_range": 0.33,
            "residue_integration": 0.52,
            "playfulness_range": 0.19,
            "self_coherence": 0.48,
        },
        epistemic_state={
            "freshness": 0.42,
            "source_confidence": 0.44,
            "verification_pressure": 0.58,
            "stale_risk": 0.49,
            "epistemic_caution": 0.62,
        },
        qualia_structure_state={
            "phase": "fragmenting",
            "emergence": 0.28,
            "stability": 0.34,
            "memory_resonance": 0.24,
            "temporal_coherence": 0.31,
            "drift": 0.49,
            "protection_bias": 0.53,
        },
        heartbeat_structure_state={
            "dominant_reaction": "recover",
            "attunement": 0.24,
            "containment_bias": 0.71,
            "recovery_pull": 0.67,
            "bounce_room": 0.08,
            "response_tempo": 0.21,
            "entrainment": 0.26,
            "activation_drive": 0.19,
        },
        relation_competition_state={
            "dominant_person_id": "user",
            "dominant_score": 0.46,
            "competition_level": 0.41,
        },
        social_topology_state={
            "state": "public_visible",
            "visibility_pressure": 0.66,
            "threading_pressure": 0.24,
            "hierarchy_pressure": 0.18,
        },
    )

    assert state.dominant_posture in {"protect", "recover", "verify"}
    assert len(state.trace) == 2
    assert state.protective_tension >= previous.protective_tension
    assert state.social_exposure >= 0.0


def test_organism_state_coercion_handles_plain_mapping() -> None:
    state = coerce_organism_state(
        {
            "attunement": 0.4,
            "coherence": 0.51,
            "grounding": 0.55,
            "protective_tension": 0.22,
            "expressive_readiness": 0.48,
            "play_window": 0.29,
            "relation_pull": 0.46,
            "social_exposure": 0.14,
            "dominant_posture": "steady",
            "relation_focus": "user",
            "social_mode": "one_to_one",
        }
    )

    assert isinstance(state, OrganismState)
    assert state.relation_focus == "user"
    assert state.dominant_posture == "steady"
