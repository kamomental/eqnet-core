from inner_os.qualia_structure_state import derive_qualia_structure_state


def test_derive_qualia_structure_state_builds_temporal_frame() -> None:
    state = derive_qualia_structure_state(
        previous_state=None,
        qualia_state={
            "qualia": [0.82, 0.14, 0.09],
            "gate": [0.91, 0.42, 0.28],
            "habituation": [0.2, 0.1, 0.05],
            "body_coupling": [0.12, 0.08, 0.04],
            "trust_applied": 0.78,
            "axis_labels": ["joy", "fear", "calm"],
            "degraded": False,
        },
        temporal_membrane_bias={
            "timeline_coherence": 0.68,
            "reentry_pull": 0.41,
            "continuity_pressure": 0.56,
            "relation_reentry_pull": 0.33,
            "supersession_pressure": 0.09,
        },
        qualia_planner_view={
            "trust": 0.79,
            "felt_energy": 0.72,
            "body_load": 0.11,
            "protection_bias": 0.14,
            "degraded": False,
        },
    )

    assert state.dominant_axis == "joy"
    assert state.phase in {"rising", "echoing", "settling", "holding", "shifting"}
    assert state.emergence > 0.0
    assert state.stability > 0.0
    assert state.memory_resonance > 0.0
    assert len(state.trace) == 1
    axes = state.to_packet_axes()
    assert axes["emergence"]["value"] > 0.0
    assert axes["resonance"]["value"] > 0.0


def test_derive_qualia_structure_state_accumulates_trace_and_momentum() -> None:
    first = derive_qualia_structure_state(
        previous_state=None,
        qualia_state={
            "qualia": [0.24, 0.66],
            "gate": [0.55, 0.82],
            "habituation": [0.1, 0.18],
            "body_coupling": [0.04, 0.19],
            "trust_applied": 0.63,
            "axis_labels": ["approval", "fear"],
        },
        temporal_membrane_bias={
            "timeline_coherence": 0.44,
            "reentry_pull": 0.26,
            "continuity_pressure": 0.31,
            "relation_reentry_pull": 0.18,
            "supersession_pressure": 0.08,
        },
        qualia_planner_view={
            "trust": 0.65,
            "felt_energy": 0.51,
            "body_load": 0.16,
            "protection_bias": 0.22,
        },
    )
    second = derive_qualia_structure_state(
        previous_state=first.to_dict(),
        qualia_state={
            "qualia": [0.51, 0.48],
            "gate": [0.72, 0.68],
            "habituation": [0.14, 0.24],
            "body_coupling": [0.09, 0.21],
            "trust_applied": 0.69,
            "axis_labels": ["approval", "fear"],
        },
        temporal_membrane_bias={
            "timeline_coherence": 0.57,
            "reentry_pull": 0.39,
            "continuity_pressure": 0.48,
            "relation_reentry_pull": 0.27,
            "supersession_pressure": 0.05,
        },
        qualia_planner_view={
            "trust": 0.71,
            "felt_energy": 0.63,
            "body_load": 0.18,
            "protection_bias": 0.16,
        },
    )

    assert len(second.trace) == 2
    assert second.trace[-1].step == 2
    assert any(abs(value) > 0.0 for value in second.momentum)
    assert second.drift > 0.0
    assert abs(second.to_packet_axes(first)["drift"]["delta"]) >= 0.0
