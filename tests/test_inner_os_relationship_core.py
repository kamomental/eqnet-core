from inner_os.relationship_core import RelationshipCore


def test_relationship_core_tracks_attachment_and_rupture() -> None:
    core = RelationshipCore()
    state = core.snapshot(
        relational_world={"community_id": "harbor_collective", "social_role": "companion"},
        sensor_input={"voice_level": 0.46, "person_count": 1, "body_stress_index": 0.22},
        current_state={},
    )
    assert state.attachment > 0.42
    assert state.familiarity > 0.35
    assert state.rupture_sensitivity >= 0.0
