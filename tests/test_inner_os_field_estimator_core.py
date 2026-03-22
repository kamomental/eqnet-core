from inner_os.field_estimator_core import FieldEstimatorCore


def test_field_estimator_core_tracks_velocity_momentum_and_dwell() -> None:
    core = FieldEstimatorCore()
    first = core.snapshot(observed_roughness=0.6, observed_defensive_salience=0.42)
    second = core.snapshot(current_state=first.to_dict(), observed_roughness=0.6, observed_defensive_salience=0.42)
    cooled = core.snapshot(current_state=second.to_dict(), observed_roughness=0.04, observed_defensive_salience=0.05)

    assert first.roughness_level > 0.0
    assert first.roughness_velocity > 0.0
    assert first.roughness_dwell > 0.0
    assert second.roughness_momentum > 0.0
    assert second.roughness_dwell > first.roughness_dwell
    assert cooled.roughness_velocity < 0.0
    assert cooled.roughness_dwell < second.roughness_dwell
    assert cooled.defensive_velocity < 0.0
