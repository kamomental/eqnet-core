from inner_os.self_model import derive_self_other_attribution_state


def test_self_other_attribution_prefers_shared_when_shared_cues_are_strong() -> None:
    state = derive_self_other_attribution_state(
        camera_observation={
            "appearance_match": 0.42,
            "contingency_match": 0.66,
            "perspective_match": 0.72,
            "sensorimotor_consistency": 0.58,
            "shared_reference_score": 0.84,
            "other_reference_score": 0.24,
            "identity_uncertainty": 0.12,
        },
        subjective_scene_state={
            "self_related_salience": 0.46,
            "shared_scene_potential": 0.78,
            "workspace_proximity": 0.7,
            "frontal_alignment": 0.74,
            "comfort": 0.62,
            "familiarity": 0.54,
            "uncertainty": 0.18,
        },
        self_state={"uncertainty": 0.2},
        person_registry={"other_presence": 0.3},
    ).to_dict()

    assert state["dominant_attribution"] == "shared"
    assert state["shared_likelihood"] > state["self_likelihood"]
    assert state["shared_likelihood"] > state["other_likelihood"]
    assert state["attribution_confidence"] > 0.2


def test_self_other_attribution_preserves_unknown_when_signals_are_weak() -> None:
    state = derive_self_other_attribution_state(
        camera_observation={"identity_uncertainty": 0.82},
        subjective_scene_state={"uncertainty": 0.74, "tension": 0.42},
        self_state={"uncertainty": 0.7},
    ).to_dict()

    assert state["unknown_likelihood"] >= 0.35
    assert state["dominant_attribution"] in {"unknown", "shared"}
