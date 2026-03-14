from inner_os.persistence_core import PersistenceCore


def test_persistence_snapshot_balances_grounding_and_strain() -> None:
    core = PersistenceCore()
    state = core.snapshot(
        current_state={"stress": 0.48, "temporal_pressure": 0.42},
        development={"belonging": 0.64, "trust_bias": 0.58, "norm_pressure": 0.36},
        relationship={"attachment": 0.66, "familiarity": 0.55},
        personality={"caution_bias": 0.44, "affiliation_bias": 0.61},
        environment_pressure={"hazard_pressure": 0.18, "resource_pressure": 0.14, "institutional_pressure": 0.22},
        transition_signal={"transition_intensity": 0.0},
    )
    assert state.social_grounding > 0.44
    assert state.recent_strain > 0.32
    assert 0.0 <= state.continuity_score <= 1.0
    assert state.culture_resonance >= 0.0
    assert state.community_resonance >= 0.0


def test_persistence_post_turn_rewards_reconstruction_and_reply() -> None:
    core = PersistenceCore()
    state = core.post_turn(
        previous={"continuity_score": 0.46, "social_grounding": 0.41, "recent_strain": 0.37, "stress": 0.42},
        development={"belonging": 0.62, "trust_bias": 0.57, "norm_pressure": 0.34},
        relationship={"attachment": 0.63, "familiarity": 0.52},
        personality={"caution_bias": 0.41, "affiliation_bias": 0.6},
        environment_pressure={"hazard_pressure": 0.12, "resource_pressure": 0.18, "institutional_pressure": 0.2},
        transition_signal={"transition_intensity": 0.0},
        reply_present=True,
        reconstructed_memory_appended=True,
        transferred_lessons_used=1,
    )
    assert state.continuity_score > 0.46
    assert state.social_grounding >= 0.41
    assert state.culture_resonance >= 0.0
    assert state.community_resonance >= 0.0


def test_persistence_post_turn_keeps_resonance_over_time() -> None:
    core = PersistenceCore()
    state = core.post_turn(
        previous={
            "continuity_score": 0.52,
            "social_grounding": 0.5,
            "recent_strain": 0.24,
            "stress": 0.22,
            "culture_resonance": 0.64,
            "community_resonance": 0.68,
        },
        development={"belonging": 0.66, "trust_bias": 0.61, "norm_pressure": 0.58},
        relationship={"attachment": 0.62, "familiarity": 0.57},
        personality={"caution_bias": 0.38, "affiliation_bias": 0.63},
        environment_pressure={"hazard_pressure": 0.14, "resource_pressure": 0.16, "institutional_pressure": 0.4},
        reply_present=True,
        reconstructed_memory_appended=True,
        transferred_lessons_used=1,
    )
    assert state.culture_resonance > 0.0
    assert state.community_resonance > 0.0


def test_persistence_transition_signal_rebalances_resonance() -> None:
    core = PersistenceCore()
    state = core.snapshot(
        current_state={
            "continuity_score": 0.54,
            "social_grounding": 0.58,
            "recent_strain": 0.26,
            "culture_resonance": 0.72,
            "community_resonance": 0.76,
            "stress": 0.2,
            "temporal_pressure": 0.18,
        },
        development={"belonging": 0.62, "trust_bias": 0.57, "norm_pressure": 0.48},
        relationship={"attachment": 0.61, "familiarity": 0.56},
        personality={"caution_bias": 0.38, "affiliation_bias": 0.63},
        environment_pressure={"hazard_pressure": 0.16, "resource_pressure": 0.2, "institutional_pressure": 0.22},
        transition_signal={"transition_intensity": 0.7},
    )
    assert state.community_resonance < 0.76
    assert state.recent_strain > 0.26
