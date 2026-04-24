from inner_os import derive_joint_state
from inner_os.shared_presence_state import derive_shared_presence_state
from inner_os.self_model import derive_self_other_attribution_state
from inner_os.world_model import derive_subjective_scene_state


def test_shared_presence_state_builds_co_presence_from_subjective_scene() -> None:
    subjective_scene = derive_subjective_scene_state(
        camera_observation={
            "distance_closeness": 0.76,
            "workspace_overlap": 0.68,
            "frontality": 0.72,
            "movement_score": 0.34,
            "self_reference_score": 0.48,
            "shared_reference_score": 0.8,
            "familiarity_hint": 0.58,
            "comfort_hint": 0.6,
        },
        self_state={"curiosity": 0.28, "safety_margin": 0.72},
        external_field_state={"safety_envelope": 0.74, "continuity_pull": 0.54},
    )
    attribution = derive_self_other_attribution_state(
        camera_observation={
            "contingency_match": 0.62,
            "perspective_match": 0.72,
            "sensorimotor_consistency": 0.52,
            "shared_reference_score": 0.82,
        },
        subjective_scene_state=subjective_scene.to_dict(),
        self_state={"uncertainty": 0.18},
    )

    shared_presence = derive_shared_presence_state(
        subjective_scene_state=subjective_scene.to_dict(),
        self_other_attribution_state=attribution.to_dict(),
        joint_state={"joint_attention": 0.42, "common_ground": 0.38, "mutual_room": 0.34},
        organism_state={"attunement": 0.56, "grounding": 0.62},
        external_field_state={"safety_envelope": 0.76, "continuity_pull": 0.5},
    ).to_dict()

    assert shared_presence["dominant_mode"] in {"inhabited_shared_space", "soft_projection"}
    assert shared_presence["co_presence"] >= 0.45
    assert shared_presence["shared_attention"] >= 0.45
    assert shared_presence["boundary_stability"] >= 0.45


def test_joint_state_can_use_shared_presence_signals() -> None:
    baseline = derive_joint_state(
        shared_moment_state={"state": "shared_moment", "moment_kind": "laugh", "score": 0.42, "jointness": 0.38},
        organism_state={"attunement": 0.42},
        external_field_state={"continuity_pull": 0.22},
    )
    boosted = derive_joint_state(
        previous_state=baseline,
        shared_moment_state={"state": "shared_moment", "moment_kind": "laugh", "score": 0.42, "jointness": 0.38},
        organism_state={"attunement": 0.42, "grounding": 0.58},
        external_field_state={"continuity_pull": 0.22, "safety_envelope": 0.7},
        subjective_scene_state={
            "shared_scene_potential": 0.78,
            "frontal_alignment": 0.72,
            "workspace_proximity": 0.64,
            "motion_salience": 0.28,
            "comfort": 0.62,
            "tension": 0.12,
        },
        self_other_attribution_state={
            "dominant_attribution": "shared",
            "shared_likelihood": 0.76,
            "unknown_likelihood": 0.12,
        },
        shared_presence_state={
            "co_presence": 0.82,
            "shared_attention": 0.74,
            "shared_scene_salience": 0.66,
            "self_projection_strength": 0.58,
            "other_projection_receptivity": 0.62,
            "boundary_stability": 0.72,
        },
    )

    assert boosted.common_ground >= baseline.common_ground
    assert boosted.joint_attention >= baseline.joint_attention
    assert boosted.coupling_strength >= baseline.coupling_strength
