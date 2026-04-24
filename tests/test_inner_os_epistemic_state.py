from inner_os.epistemic_state import EpistemicState, coerce_epistemic_state
from inner_os.epistemic_update_policy import derive_epistemic_state


def test_epistemic_state_exposes_packet_axes() -> None:
    previous = EpistemicState(
        freshness=0.42,
        source_confidence=0.48,
        verification_pressure=0.38,
        change_likelihood=0.31,
        stale_risk=0.36,
        epistemic_caution=0.41,
    )
    current = EpistemicState(
        freshness=0.68,
        source_confidence=0.73,
        verification_pressure=0.44,
        change_likelihood=0.52,
        stale_risk=0.47,
        epistemic_caution=0.46,
    )

    axes = current.to_packet_axes(previous)

    assert axes["grounding"]["value"] > 0.0
    assert axes["verification"]["value"] > 0.0
    assert axes["volatility"]["delta"] != 0.0


def test_derive_epistemic_state_prefers_carry_forward_for_fresh_verified_signal() -> None:
    state = derive_epistemic_state(
        current_state={
            "freshness_days": 1,
            "knowledge_scope": "autobiographical",
            "source_type": "verification",
        },
        memory_evidence={
            "record_kind": "verified",
            "source_confidence": 0.84,
            "freshness": 0.92,
        },
        observation_state={
            "freshness": 0.88,
            "source_confidence": 0.8,
        },
    )

    assert state.freshness > 0.6
    assert state.source_confidence > 0.7
    assert state.stale_risk < 0.45
    assert state.verification_pressure < 0.5
    assert state.dominant_posture == "carry_forward"


def test_derive_epistemic_state_prefers_reverify_for_stale_dynamic_signal() -> None:
    state = derive_epistemic_state(
        previous_epistemic={
            "freshness": 0.52,
            "source_confidence": 0.48,
            "verification_pressure": 0.42,
            "change_likelihood": 0.46,
            "stale_risk": 0.44,
            "epistemic_caution": 0.47,
        },
        current_state={
            "freshness_days": 24,
            "knowledge_scope": "live_trend",
            "source_type": "social",
            "epistemic_update_strength": 0.62,
        },
        memory_evidence={
            "record_kind": "reconstructed",
            "source_confidence": 0.28,
            "change_likelihood": 0.82,
        },
        observation_state={
            "freshness": 0.18,
            "change_likelihood": 0.76,
        },
    )

    assert state.freshness < 0.45
    assert state.source_confidence < 0.5
    assert state.stale_risk > 0.55
    assert state.verification_pressure > 0.6
    assert state.dominant_posture == "reverify"


def test_coerce_epistemic_state_uses_defaults_for_missing_payload() -> None:
    state = coerce_epistemic_state({})

    assert state.freshness == 0.5
    assert state.source_confidence == 0.5
    assert state.dominant_posture == "carry_forward"
