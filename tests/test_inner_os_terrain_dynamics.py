from inner_os.terrain_dynamics import (
    TerrainDynamicsState,
    coerce_terrain_dynamics_state,
    derive_terrain_dynamics_state,
)


def test_terrain_dynamics_state_exposes_packet_axes() -> None:
    state = TerrainDynamicsState(
        terrain_energy=0.42,
        entropy=0.24,
        ignition_pressure=0.36,
        barrier_height=0.21,
        recovery_gradient=0.49,
        basin_pull=0.58,
    )

    axes = state.to_packet_axes(
        {
            "terrain_energy": 0.35,
            "entropy": 0.28,
            "ignition_pressure": 0.31,
            "barrier_height": 0.26,
            "recovery_gradient": 0.44,
            "basin_pull": 0.52,
        }
    )

    assert axes["energy"]["delta"] == 0.07
    assert axes["entropy"]["delta"] == -0.04
    assert axes["ignition"]["delta"] == 0.05
    assert axes["barrier"]["delta"] == -0.05
    assert axes["recovery"]["delta"] == 0.05
    assert axes["basin"]["delta"] == 0.06


def test_derive_terrain_dynamics_state_prefers_continuity_basin_under_reentry_conditions() -> None:
    state = derive_terrain_dynamics_state(
        previous_state=None,
        organism_state={
            "expressive_readiness": 0.55,
            "grounding": 0.62,
            "relation_pull": 0.66,
            "play_window": 0.34,
        },
        external_field_state={
            "continuity_pull": 0.68,
            "novelty": 0.14,
            "ambiguity_load": 0.19,
            "social_pressure": 0.18,
            "environmental_load": 0.22,
            "safety_envelope": 0.73,
        },
        memory_dynamics_state={
            "ignition_readiness": 0.41,
            "monument_salience": 0.57,
            "consolidation_pull": 0.46,
            "memory_tension": 0.24,
        },
        qualia_structure_state={
            "emergence": 0.44,
            "stability": 0.67,
            "memory_resonance": 0.59,
            "temporal_coherence": 0.61,
            "drift": 0.15,
        },
        heartbeat_structure_state={
            "activation_drive": 0.39,
            "containment_bias": 0.19,
            "recovery_pull": 0.51,
            "entrainment": 0.58,
        },
        terrain_readout={
            "protect_bias": 0.18,
            "approach_bias": 0.47,
            "value": 0.42,
        },
    )

    assert state.dominant_basin == "recovery_basin"
    assert state.dominant_flow == "recover"
    assert state.basin_pull >= 0.5
    assert state.recovery_gradient >= 0.4
    assert len(state.trace) == 1


def test_derive_terrain_dynamics_state_prefers_protective_basin_under_barrier_load() -> None:
    state = derive_terrain_dynamics_state(
        previous_state=coerce_terrain_dynamics_state(
            {
                "terrain_energy": 0.31,
                "entropy": 0.26,
                "ignition_pressure": 0.21,
                "barrier_height": 0.41,
                "recovery_gradient": 0.29,
                "basin_pull": 0.37,
            }
        ),
        organism_state={
            "expressive_readiness": 0.27,
            "grounding": 0.43,
            "relation_pull": 0.34,
            "protective_tension": 0.63,
            "play_window": 0.12,
            "coherence": 0.46,
        },
        external_field_state={
            "continuity_pull": 0.28,
            "novelty": 0.22,
            "ambiguity_load": 0.33,
            "social_pressure": 0.56,
            "environmental_load": 0.44,
            "safety_envelope": 0.39,
        },
        memory_dynamics_state={
            "ignition_readiness": 0.18,
            "monument_salience": 0.31,
            "consolidation_pull": 0.28,
            "memory_tension": 0.51,
        },
        qualia_structure_state={
            "emergence": 0.29,
            "stability": 0.38,
            "memory_resonance": 0.33,
            "temporal_coherence": 0.34,
            "drift": 0.41,
        },
        heartbeat_structure_state={
            "activation_drive": 0.34,
            "containment_bias": 0.62,
            "recovery_pull": 0.27,
            "entrainment": 0.31,
        },
        terrain_readout={
            "protect_bias": 0.58,
            "approach_bias": 0.18,
            "value": 0.26,
        },
    )

    assert state.barrier_height >= 0.5
    assert state.dominant_basin == "protective_basin"
    assert state.dominant_flow in {"contain", "settle"}


def test_derive_terrain_dynamics_state_reflects_causal_memory_field() -> None:
    reopened = derive_terrain_dynamics_state(
        previous_state=None,
        organism_state={
            "expressive_readiness": 0.42,
            "grounding": 0.48,
            "relation_pull": 0.52,
            "protective_tension": 0.34,
            "play_window": 0.18,
            "coherence": 0.54,
        },
        external_field_state={
            "continuity_pull": 0.5,
            "novelty": 0.16,
            "ambiguity_load": 0.24,
            "social_pressure": 0.28,
            "environmental_load": 0.2,
            "safety_envelope": 0.64,
        },
        memory_dynamics_state={
            "dominant_relation_type": "unfinished_carry",
            "relation_generation_mode": "competitive",
            "dominant_causal_type": "reopened_by",
            "causal_generation_mode": "contested",
            "meta_relations": [{"meta_type": "competes_with"}],
            "ignition_readiness": 0.34,
            "monument_salience": 0.44,
            "consolidation_pull": 0.32,
            "memory_tension": 0.46,
        },
        qualia_structure_state={
            "emergence": 0.32,
            "stability": 0.52,
            "memory_resonance": 0.48,
            "temporal_coherence": 0.5,
            "drift": 0.28,
        },
        heartbeat_structure_state={
            "activation_drive": 0.31,
            "containment_bias": 0.36,
            "recovery_pull": 0.33,
            "entrainment": 0.42,
        },
        terrain_readout={
            "protect_bias": 0.36,
            "approach_bias": 0.22,
            "value": 0.3,
        },
    )

    reframed = derive_terrain_dynamics_state(
        previous_state=None,
        organism_state={
            "expressive_readiness": 0.42,
            "grounding": 0.48,
            "relation_pull": 0.52,
            "protective_tension": 0.34,
            "play_window": 0.18,
            "coherence": 0.54,
        },
        external_field_state={
            "continuity_pull": 0.5,
            "novelty": 0.16,
            "ambiguity_load": 0.24,
            "social_pressure": 0.28,
            "environmental_load": 0.2,
            "safety_envelope": 0.64,
        },
        memory_dynamics_state={
            "dominant_relation_type": "cross_context_bridge",
            "relation_generation_mode": "anchored",
            "dominant_causal_type": "reframed_by",
            "causal_generation_mode": "reinforced",
            "meta_relations": [{"meta_type": "reinforces"}],
            "ignition_readiness": 0.34,
            "monument_salience": 0.44,
            "consolidation_pull": 0.32,
            "memory_tension": 0.46,
        },
        qualia_structure_state={
            "emergence": 0.32,
            "stability": 0.52,
            "memory_resonance": 0.48,
            "temporal_coherence": 0.5,
            "drift": 0.28,
        },
        heartbeat_structure_state={
            "activation_drive": 0.31,
            "containment_bias": 0.36,
            "recovery_pull": 0.33,
            "entrainment": 0.42,
        },
        terrain_readout={
            "protect_bias": 0.36,
            "approach_bias": 0.22,
            "value": 0.3,
        },
    )

    assert reopened.barrier_height > reframed.barrier_height
    assert reopened.entropy > reframed.entropy
    assert reframed.recovery_gradient > reopened.recovery_gradient
    assert reframed.basin_pull > reopened.basin_pull
