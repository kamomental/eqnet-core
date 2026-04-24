from inner_os.external_field_state import (
    ExternalFieldState,
    coerce_external_field_state,
    derive_external_field_state,
)


def test_external_field_state_exposes_packet_axes() -> None:
    state = ExternalFieldState(
        environmental_load=0.28,
        social_pressure=0.36,
        continuity_pull=0.64,
        ambiguity_load=0.22,
        safety_envelope=0.71,
        novelty=0.18,
    )

    axes = state.to_packet_axes(
        {
            "environmental_load": 0.21,
            "social_pressure": 0.31,
            "continuity_pull": 0.55,
            "ambiguity_load": 0.17,
            "safety_envelope": 0.66,
            "novelty": 0.12,
        }
    )

    assert axes["environment"]["value"] == 0.28
    assert axes["social"]["delta"] == 0.05
    assert axes["continuity"]["delta"] == 0.09
    assert axes["ambiguity"]["delta"] == 0.05
    assert axes["safety"]["delta"] == 0.05
    assert axes["novelty"]["delta"] == 0.06


def test_derive_external_field_state_integrates_environment_relation_and_thread_context() -> None:
    state = derive_external_field_state(
        previous_state=None,
        environment_pressure={
            "resource_pressure": 0.24,
            "hazard_pressure": 0.12,
            "ritual_pressure": 0.08,
            "institutional_pressure": 0.07,
            "social_density": 0.19,
        },
        social_topology_state={
            "state": "one_to_one",
            "visibility_pressure": 0.11,
            "threading_pressure": 0.14,
            "hierarchy_pressure": 0.05,
        },
        relation_competition_state={
            "state": "single_relation",
            "competition_level": 0.08,
            "dominant_score": 0.63,
        },
        recent_dialogue_state={
            "state": "reopening_thread",
            "overlap_score": 0.19,
            "reopen_pressure": 0.52,
            "thread_carry": 0.61,
        },
        discussion_thread_state={
            "state": "revisit_issue",
            "unresolved_pressure": 0.27,
            "revisit_readiness": 0.56,
            "thread_visibility": 0.62,
        },
        issue_state={
            "state": "pausing_issue",
            "question_pressure": 0.18,
            "pause_readiness": 0.47,
        },
        transition_signal={
            "transition_intensity": 0.16,
            "place_changed": False,
            "privacy_shift": False,
        },
        organism_state={
            "grounding": 0.59,
            "relation_pull": 0.66,
        },
    )

    assert state.dominant_field == "continuity_field"
    assert state.social_mode == "one_to_one"
    assert state.thread_mode == "revisit_issue"
    assert state.continuity_pull >= 0.5
    assert state.safety_envelope >= 0.5
    assert len(state.trace) == 1


def test_derive_external_field_state_prefers_social_pressure_under_public_visibility() -> None:
    state = derive_external_field_state(
        previous_state=coerce_external_field_state(
            {
                "environmental_load": 0.21,
                "social_pressure": 0.29,
                "continuity_pull": 0.24,
                "ambiguity_load": 0.18,
                "safety_envelope": 0.61,
                "novelty": 0.14,
            }
        ),
        environment_pressure={
            "resource_pressure": 0.2,
            "hazard_pressure": 0.11,
            "ritual_pressure": 0.12,
            "institutional_pressure": 0.14,
            "social_density": 0.54,
        },
        social_topology_state={
            "state": "public_visible",
            "visibility_pressure": 0.72,
            "threading_pressure": 0.49,
            "hierarchy_pressure": 0.31,
        },
        relation_competition_state={
            "state": "multi_relation",
            "competition_level": 0.37,
            "dominant_score": 0.33,
        },
        recent_dialogue_state={"state": "continuing_thread"},
        discussion_thread_state={"state": "continuing_thread"},
        issue_state={"state": "open_thread", "question_pressure": 0.33},
        transition_signal={"transition_intensity": 0.29, "privacy_shift": True},
        organism_state={"grounding": 0.42, "relation_pull": 0.38},
    )

    assert state.social_pressure >= 0.45
    assert state.dominant_field == "social_pressure_field"
    assert state.novelty > 0.0
