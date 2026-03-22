from inner_os.development_core import DevelopmentCore
from inner_os.simulation_transfer import SimulationTransferCore


def test_development_core_snapshot_reflects_relation_and_privacy() -> None:
    core = DevelopmentCore()
    state = core.snapshot(
        relational_world={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
        sensor_input={
            "person_count": 1,
            "voice_level": 0.4,
            "privacy_tags": ["private"],
            "body_stress_index": 0.2,
        },
        current_state={"attachment": 0.68, "familiarity": 0.58, "trust_memory": 0.62, "role_alignment": 0.57},
        safety_bias=0.1,
        environment_pressure={
            "resource_pressure": 0.55,
            "hazard_pressure": 0.42,
            "ritual_pressure": 0.3,
            "institutional_pressure": 0.6,
            "social_density": 0.5,
        },
    )
    assert state.belonging > 0.45
    assert state.norm_pressure > 0.35
    assert state.role_commitment > 0.4
    assert state.trust_bias > 0.45


def test_development_core_post_turn_updates_belonging_and_trust() -> None:
    core = DevelopmentCore()
    state = core.post_turn(
        previous={"belonging": 0.5, "trust_bias": 0.45, "norm_pressure": 0.35, "role_commitment": 0.4, "attachment": 0.66, "familiarity": 0.61, "trust_memory": 0.64, "role_alignment": 0.58},
        relational_world={"community_id": "harbor_collective", "social_role": "companion"},
        reply_present=True,
        stress=0.2,
        recovery_need=0.1,
        environment_pressure={
            "resource_pressure": 0.2,
            "hazard_pressure": 0.3,
            "ritual_pressure": 0.45,
            "institutional_pressure": 0.5,
            "social_density": 0.35,
        },
    )
    assert state.belonging >= 0.45
    assert state.trust_bias >= 0.4
    assert state.social_update_strength == 1.0
    assert state.identity_update_strength == 1.0


def test_development_core_slows_identity_updates_under_tentative_recall() -> None:
    core = DevelopmentCore()
    previous = {
        "belonging": 0.5,
        "trust_bias": 0.45,
        "norm_pressure": 0.35,
        "role_commitment": 0.4,
        "attachment": 0.66,
        "familiarity": 0.61,
        "trust_memory": 0.64,
        "role_alignment": 0.58,
    }
    relational_world = {"community_id": "harbor_collective", "social_role": "companion"}
    pressure = {
        "resource_pressure": 0.2,
        "hazard_pressure": 0.3,
        "ritual_pressure": 0.45,
        "institutional_pressure": 0.5,
        "social_density": 0.35,
    }
    low = core.post_turn(
        previous=previous,
        relational_world=relational_world,
        reply_present=True,
        stress=0.2,
        recovery_need=0.1,
        environment_pressure=pressure,
        terrain_transition_roughness=0.05,
        recalled_tentative_bias=0.0,
    )
    high = core.post_turn(
        previous=previous,
        relational_world=relational_world,
        reply_present=True,
        stress=0.2,
        recovery_need=0.1,
        environment_pressure=pressure,
        terrain_transition_roughness=0.78,
        recalled_tentative_bias=0.72,
    )
    assert high.norm_pressure < low.norm_pressure
    assert high.role_commitment < low.role_commitment
    assert high.belonging < low.belonging
    assert high.identity_update_strength < low.identity_update_strength
    assert high.social_update_strength < low.social_update_strength


def test_development_core_reopens_updates_when_recovery_returns() -> None:
    core = DevelopmentCore()
    previous = {
        "belonging": 0.5,
        "trust_bias": 0.45,
        "norm_pressure": 0.35,
        "role_commitment": 0.4,
        "attachment": 0.66,
        "familiarity": 0.61,
        "trust_memory": 0.64,
        "role_alignment": 0.58,
    }
    relational_world = {"community_id": "harbor_collective", "social_role": "companion"}
    pressure = {
        "resource_pressure": 0.2,
        "hazard_pressure": 0.3,
        "ritual_pressure": 0.45,
        "institutional_pressure": 0.5,
        "social_density": 0.35,
    }
    held = core.post_turn(
        previous=previous,
        relational_world=relational_world,
        reply_present=True,
        stress=0.2,
        recovery_need=0.1,
        environment_pressure=pressure,
        terrain_transition_roughness=0.52,
        recalled_tentative_bias=0.38,
        recovery_reopening=0.0,
    )
    reopened = core.post_turn(
        previous=previous,
        relational_world=relational_world,
        reply_present=True,
        stress=0.2,
        recovery_need=0.1,
        environment_pressure=pressure,
        terrain_transition_roughness=0.52,
        recalled_tentative_bias=0.38,
        recovery_reopening=0.62,
    )
    assert reopened.social_update_strength > held.social_update_strength
    assert reopened.identity_update_strength > held.identity_update_strength
    assert reopened.norm_pressure >= held.norm_pressure
    assert reopened.role_commitment >= held.role_commitment


def test_development_core_can_absorb_transferred_learning() -> None:
    transfer = SimulationTransferCore()
    lessons = transfer.promote([
        {
            "episode_id": "sim-001",
            "summary": "pause, observe, then clarify before acting",
            "patterns": [
                "pause and observe when signals conflict",
                "clarify before commitment when the other person is uncertain",
            ],
            "benefit_score": 0.85,
            "risk_score": 0.15,
            "transfer_ready": True,
        }
    ])
    core = DevelopmentCore()
    state = core.apply_transferred_learning(
        previous={"belonging": 0.45, "trust_bias": 0.45, "norm_pressure": 0.35, "role_commitment": 0.4, "social_update_strength": 0.72, "identity_update_strength": 0.61},
        lessons=[lesson.to_memory_record() for lesson in lessons],
    )
    assert state.norm_pressure > 0.35
    assert state.trust_bias > 0.45
    assert state.social_update_strength == 0.72
    assert state.identity_update_strength == 0.61


def test_development_core_exposes_memory_kind_biases() -> None:
    core = DevelopmentCore()
    biases = core.memory_kind_biases(
        state={
            "belonging": 0.72,
            "trust_bias": 0.64,
            "norm_pressure": 0.7,
            "role_commitment": 0.68,
        }
    )
    assert biases["relationship_trace"] > biases["observed_real"]
    assert biases["verified"] > 0.0
    assert biases["transferred_learning"] > 0.0


def test_development_core_reads_field_residue() -> None:
    core = DevelopmentCore()
    calm = core.snapshot(
        relational_world={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
        sensor_input={
            "person_count": 1,
            "voice_level": 0.4,
            "privacy_tags": [],
            "body_stress_index": 0.2,
        },
        current_state={
            "attachment": 0.68,
            "familiarity": 0.58,
            "trust_memory": 0.62,
            "role_alignment": 0.57,
            "roughness_level": 0.04,
            "roughness_dwell": 0.0,
            "defensive_level": 0.02,
            "defensive_dwell": 0.0,
        },
        safety_bias=0.1,
        environment_pressure={
            "resource_pressure": 0.22,
            "hazard_pressure": 0.18,
            "ritual_pressure": 0.24,
            "institutional_pressure": 0.3,
            "social_density": 0.4,
        },
    )
    rough = core.snapshot(
        relational_world={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
        sensor_input={
            "person_count": 1,
            "voice_level": 0.4,
            "privacy_tags": [],
            "body_stress_index": 0.2,
        },
        current_state={
            "attachment": 0.68,
            "familiarity": 0.58,
            "trust_memory": 0.62,
            "role_alignment": 0.57,
            "roughness_level": 0.58,
            "roughness_dwell": 0.72,
            "defensive_level": 0.44,
            "defensive_dwell": 0.66,
        },
        safety_bias=0.1,
        environment_pressure={
            "resource_pressure": 0.22,
            "hazard_pressure": 0.18,
            "ritual_pressure": 0.24,
            "institutional_pressure": 0.3,
            "social_density": 0.4,
        },
    )
    assert rough.belonging < calm.belonging
    assert rough.trust_bias < calm.trust_bias
    assert rough.norm_pressure > calm.norm_pressure
