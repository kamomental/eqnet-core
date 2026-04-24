from inner_os.heartbeat_structure_state import derive_heartbeat_structure_state


def test_derive_heartbeat_structure_state_builds_reaction_profile() -> None:
    state = derive_heartbeat_structure_state(
        heart_snapshot={"rate": 1.42, "phase": 0.37},
        metrics={
            "life_indicator": 0.58,
            "tension_score": 0.22,
            "phi_norm": 0.46,
            "heart_rate_norm": 0.47,
        },
        qualia_structure_state={
            "phase": "echoing",
            "emergence": 0.51,
            "stability": 0.62,
            "memory_resonance": 0.57,
            "temporal_coherence": 0.61,
            "drift": 0.18,
        },
        qualia_planner_view={
            "trust": 0.66,
            "body_load": 0.14,
            "protection_bias": 0.11,
        },
        growth_state={
            "relational_trust": 0.63,
            "expressive_range": 0.56,
            "playfulness_range": 0.42,
        },
    )

    assert state.pulse_band in {"soft_pulse", "lifted_pulse", "racing_pulse"}
    assert state.phase_window in {"downbeat", "upswing", "crest", "release"}
    assert 0.0 <= state.activation_drive <= 1.0
    assert 0.0 <= state.response_tempo <= 1.0
    assert state.dominant_reaction in {"recover", "contain", "bounce", "attune", "steady"}
    axes = state.to_packet_axes()
    assert axes["activation"]["value"] >= 0.0
    assert axes["tempo"]["value"] >= 0.0


def test_derive_heartbeat_structure_state_accumulates_trace_and_deltas() -> None:
    first = derive_heartbeat_structure_state(
        heart_snapshot={"rate": 0.92, "phase": 0.12},
        metrics={
            "life_indicator": 0.44,
            "tension_score": 0.19,
            "phi_norm": 0.31,
            "heart_rate_norm": 0.31,
        },
        qualia_structure_state={
            "emergence": 0.34,
            "stability": 0.66,
            "memory_resonance": 0.29,
            "temporal_coherence": 0.47,
            "drift": 0.12,
        },
    )
    second = derive_heartbeat_structure_state(
        previous_state=first.to_dict(),
        heart_snapshot={"rate": 1.58, "phase": 0.66},
        metrics={
            "life_indicator": 0.6,
            "tension_score": 0.27,
            "phi_norm": 0.52,
            "heart_rate_norm": 0.53,
        },
        qualia_structure_state={
            "emergence": 0.56,
            "stability": 0.49,
            "memory_resonance": 0.54,
            "temporal_coherence": 0.43,
            "drift": 0.24,
        },
        growth_state={
            "relational_trust": 0.58,
            "expressive_range": 0.63,
            "playfulness_range": 0.51,
        },
    )

    assert len(second.trace) == 2
    assert second.trace[-1].phase_window in {"downbeat", "upswing", "crest", "release"}
    axes = second.to_packet_axes(first)
    assert axes["activation"]["delta"] != 0.0
    assert axes["tempo"]["delta"] != 0.0
