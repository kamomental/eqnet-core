from inner_os.world_model import derive_subjective_scene_state


def test_subjective_scene_state_maps_camera_observation_to_lived_space() -> None:
    state = derive_subjective_scene_state(
        camera_observation={
            "distance_closeness": 0.82,
            "workspace_overlap": 0.74,
            "frontality": 0.7,
            "movement_score": 0.36,
            "self_reference_score": 0.68,
            "shared_reference_score": 0.58,
            "familiarity_hint": 0.62,
            "comfort_hint": 0.56,
            "scene_uncertainty": 0.14,
        },
        world_state={"task_relevance": 0.48, "scene_familiarity": 0.52},
        self_state={"curiosity": 0.34, "safety_margin": 0.7},
        external_field_state={"safety_envelope": 0.72, "continuity_pull": 0.46},
    ).to_dict()

    assert state["dominant_zone"] in {"near_working_field", "near_personal_space"}
    assert state["anchor_frame"] == "self_margin"
    assert state["workspace_proximity"] >= 0.6
    assert state["self_related_salience"] >= 0.55
    assert state["shared_scene_potential"] >= 0.45
    assert state["comfort"] >= 0.45
    assert state["uncertainty"] < 0.4


def test_subjective_scene_state_defaults_to_previous_when_empty() -> None:
    previous = derive_subjective_scene_state(
        camera_observation={"distance_closeness": 0.6, "workspace_overlap": 0.62},
        self_state={"curiosity": 0.2},
    )

    current = derive_subjective_scene_state(previous_state=previous)
    assert current == previous
