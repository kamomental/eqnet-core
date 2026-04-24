from inner_os.development_transition_policy import derive_growth_state
from inner_os.growth_state import GrowthState, coerce_growth_state


def test_growth_state_exposes_replay_axes() -> None:
    previous = GrowthState(
        relational_trust=0.42,
        epistemic_maturity=0.36,
        expressive_range=0.38,
        residue_integration=0.34,
        playfulness_range=0.24,
        self_coherence=0.44,
    )
    current = GrowthState(
        relational_trust=0.64,
        epistemic_maturity=0.58,
        expressive_range=0.62,
        residue_integration=0.57,
        playfulness_range=0.52,
        self_coherence=0.66,
    )

    axes = current.to_replay_axes(previous)

    assert set(axes.keys()) == {"bond", "stability", "curiosity"}
    assert axes["bond"]["value"] > previous.to_replay_axes()["bond"]["value"]
    assert axes["stability"]["delta"] > 0.0
    assert axes["curiosity"]["delta"] > 0.0


def test_development_transition_policy_integrates_slow_state_fragments() -> None:
    growth = derive_growth_state(
        previous_growth=None,
        development_state={
            "belonging": 0.74,
            "trust_bias": 0.71,
            "norm_pressure": 0.44,
            "role_commitment": 0.68,
            "social_update_strength": 0.82,
            "identity_update_strength": 0.76,
        },
        forgetting_snapshot={
            "forgetting_pressure": 0.18,
        },
        sleep_consolidation={
            "replay_priority": 0.72,
            "reconsolidation_priority": 0.69,
            "autobiographical_pull": 0.66,
            "abstraction_readiness": 0.62,
            "identity_preservation_bias": 0.74,
            "expressive_style_carry_bias": 0.58,
        },
        transfer_package={
            "portable_state": {
                "same_turn": {
                    "expressive_style_state": {"lightness_room": 0.61},
                    "relational_style_memory_state": {"lexical_variation_bias": 0.57},
                },
                "carry": {
                    "memory_carry": {
                        "monument_carry": {"monument_salience": 0.64},
                        "autobiographical_thread": {"strength": 0.68},
                    }
                },
            },
            "runtime_seed": {
                "continuity_score": 0.72,
                "social_grounding": 0.63,
                "association_reweighting_bias": 0.44,
                "expressive_style_carry_bias": 0.55,
                "relational_continuity_carry_bias": 0.52,
                "initiative_followup_bias": 0.38,
            },
        },
    )

    assert growth.relational_trust > 0.45
    assert growth.epistemic_maturity > 0.35
    assert growth.residue_integration > 0.35
    assert growth.self_coherence > 0.45
    assert growth.dominant_transition in growth.to_dict()


def test_development_transition_policy_reduces_range_under_high_forgetting_pressure() -> None:
    low_pressure = derive_growth_state(
        development_state={
            "belonging": 0.55,
            "trust_bias": 0.56,
            "norm_pressure": 0.46,
            "role_commitment": 0.52,
            "social_update_strength": 0.78,
            "identity_update_strength": 0.74,
        },
        forgetting_snapshot={"forgetting_pressure": 0.14},
        sleep_consolidation={
            "replay_priority": 0.44,
            "reconsolidation_priority": 0.46,
            "autobiographical_pull": 0.42,
            "abstraction_readiness": 0.48,
            "identity_preservation_bias": 0.52,
            "expressive_style_carry_bias": 0.5,
        },
        transfer_package={
            "portable_state": {
                "same_turn": {
                    "expressive_style_state": {"lightness_room": 0.48},
                    "relational_style_memory_state": {"lexical_variation_bias": 0.46},
                }
            }
        },
    )
    high_pressure = derive_growth_state(
        development_state={
            "belonging": 0.55,
            "trust_bias": 0.56,
            "norm_pressure": 0.46,
            "role_commitment": 0.52,
            "social_update_strength": 0.78,
            "identity_update_strength": 0.74,
        },
        forgetting_snapshot={"forgetting_pressure": 0.82},
        sleep_consolidation={
            "replay_priority": 0.44,
            "reconsolidation_priority": 0.46,
            "autobiographical_pull": 0.42,
            "abstraction_readiness": 0.48,
            "identity_preservation_bias": 0.52,
            "expressive_style_carry_bias": 0.5,
        },
        transfer_package={
            "portable_state": {
                "same_turn": {
                    "expressive_style_state": {"lightness_room": 0.48},
                    "relational_style_memory_state": {"lexical_variation_bias": 0.46},
                }
            }
        },
    )

    assert low_pressure.epistemic_maturity > high_pressure.epistemic_maturity
    assert low_pressure.expressive_range > high_pressure.expressive_range
    assert low_pressure.self_coherence > high_pressure.self_coherence


def test_coerce_growth_state_uses_defaults_for_missing_payload() -> None:
    state = coerce_growth_state({})
    assert state.relational_trust == 0.45
    assert state.self_coherence == 0.45
