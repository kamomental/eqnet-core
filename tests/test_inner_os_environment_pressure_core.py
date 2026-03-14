from inner_os.environment_pressure_core import EnvironmentPressureCore


def test_environment_pressure_core_estimates_social_and_institutional_pressure() -> None:
    core = EnvironmentPressureCore()
    snapshot = core.snapshot(
        relational_world={
            "world_type": "institutional",
            "zone_id": "shrine",
            "time_phase": "night",
            "weather": "storm",
            "resource_scarcity": 0.62,
            "hazard_level": 0.55,
            "ritual_signal": 0.7,
            "institutional_pressure": 0.68,
            "mode": "simulation",
        },
        sensor_input={
            "person_count": 4,
            "motion_score": 0.42,
            "body_stress_index": 0.38,
        },
        current_state={"stress": 0.35},
    )
    assert snapshot.resource_pressure > 0.4
    assert snapshot.hazard_pressure > 0.4
    assert snapshot.ritual_pressure > 0.4
    assert snapshot.institutional_pressure > 0.4
    assert snapshot.summary
