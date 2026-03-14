from inner_os.personality_core import PersonalityIndexCore


def test_personality_core_derives_biases_from_state() -> None:
    core = PersonalityIndexCore()
    state = core.snapshot(
        current_state={"stress": 0.42, "temporal_pressure": 0.36, "continuity_score": 0.66, "social_grounding": 0.61, "recent_strain": 0.34},
        development={"belonging": 0.58, "trust_bias": 0.52, "norm_pressure": 0.61, "role_commitment": 0.57},
        relationship={"attachment": 0.63, "familiarity": 0.54, "rupture_sensitivity": 0.33, "trust_memory": 0.62, "role_alignment": 0.58},
        environment_pressure={"hazard_pressure": 0.24, "resource_pressure": 0.18, "institutional_pressure": 0.42, "ritual_pressure": 0.31},
    )
    assert state.caution_bias >= 0.0
    assert state.affiliation_bias > 0.45
    assert state.reflective_bias > 0.45
    assert state.exploration_bias > 0.4


def test_personality_core_responds_to_low_continuity_and_high_strain() -> None:
    core = PersonalityIndexCore()
    state = core.snapshot(
        current_state={"stress": 0.5, "temporal_pressure": 0.42, "continuity_score": 0.22, "social_grounding": 0.28, "recent_strain": 0.74},
        development={"belonging": 0.38, "trust_bias": 0.34, "norm_pressure": 0.58, "role_commitment": 0.51},
        relationship={"attachment": 0.36, "familiarity": 0.31, "rupture_sensitivity": 0.61, "trust_memory": 0.3, "role_alignment": 0.33},
        environment_pressure={"hazard_pressure": 0.48, "resource_pressure": 0.32, "institutional_pressure": 0.44, "ritual_pressure": 0.21},
    )
    assert state.caution_bias > 0.5
    assert state.affiliation_bias < 0.55


def test_personality_core_reads_community_resonance() -> None:
    from inner_os.personality_core import PersonalityIndexCore
    core = PersonalityIndexCore()
    low = core.snapshot(
        current_state={"continuity_score": 0.6, "social_grounding": 0.58, "culture_resonance": 0.0, "community_resonance": 0.0},
        development={"trust_bias": 0.52, "belonging": 0.5, "norm_pressure": 0.48, "role_commitment": 0.46},
        relationship={"attachment": 0.55, "familiarity": 0.52, "rupture_sensitivity": 0.35, "trust_memory": 0.54, "role_alignment": 0.5},
        environment_pressure={"hazard_pressure": 0.2, "resource_pressure": 0.2, "institutional_pressure": 0.35, "ritual_pressure": 0.3},
    )
    high = core.snapshot(
        current_state={"continuity_score": 0.6, "social_grounding": 0.58, "culture_resonance": 0.66, "community_resonance": 0.72},
        development={"trust_bias": 0.52, "belonging": 0.5, "norm_pressure": 0.48, "role_commitment": 0.46},
        relationship={"attachment": 0.55, "familiarity": 0.52, "rupture_sensitivity": 0.35, "trust_memory": 0.54, "role_alignment": 0.5},
        environment_pressure={"hazard_pressure": 0.2, "resource_pressure": 0.2, "institutional_pressure": 0.35, "ritual_pressure": 0.3},
    )
    assert high.affiliation_bias > low.affiliation_bias
    assert high.caution_bias < low.caution_bias
