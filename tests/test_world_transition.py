from heartos.world_transition import TransitionParams, apply_transition, build_transition_record


def test_apply_transition_decay_and_uncertainty():
    params = TransitionParams(decay=0.5, uncertainty_factor=0.5, base_uncertainty=0.2)
    state = {"drive": 1.0, "uncertainty": 0.8}
    updated = apply_transition(state, params)
    assert updated["drive"] == 0.5
    assert updated["uncertainty"] == 0.4


def test_build_transition_record_shape():
    params = TransitionParams(decay=0.7, uncertainty_factor=0.6, base_uncertainty=0.2)
    record = build_transition_record(
        turn_id="turn-1",
        transition_turn_index=3,
        scenario_id="commute",
        from_world="infrastructure",
        to_world="community",
        params=params,
    )
    assert record["schema_version"] == "trace_v1"
    assert record["source_loop"] == "world_transition"
    assert record["transition"]["from_world_type"] == "infrastructure"
    assert record["event_type"] == "world_transition"
    assert record["transition"]["transition_turn_index"] == 3
