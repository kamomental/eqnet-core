from inner_os.value_system.emotional_dft import (
    AccessState,
    DynamicFieldConfig,
    DynamicFieldInput,
    DynamicFieldState,
    TerrainSnapshot,
    dynamic_step,
    subjective_intensity,
    surprise_score,
)


def test_dynamic_step_reduces_repeated_input_by_habituation() -> None:
    repeated = dynamic_step(
        DynamicFieldState(
            arousal=0.2,
            prediction_error_mass=0.1,
            relation_value=0.3,
            habituation=0.8,
        ),
        terrain=TerrainSnapshot(
            state_energy=0.4,
            gradient_norm=0.2,
            max_curvature=0.1,
        ),
        graph_coupling=(0.2, 0.2, 0.1),
        external_input=DynamicFieldInput(
            arousal_drive=0.5,
            prediction_drive=0.4,
            relation_drive=0.1,
            novelty=0.2,
            repeated_exposure=0.8,
        ),
        access=AccessState(attention=0.5, reportability=0.4),
        dt=0.1,
        config=DynamicFieldConfig(relation_gamma=0.2),
    )
    fresh = dynamic_step(
        DynamicFieldState(
            arousal=0.2,
            prediction_error_mass=0.1,
            relation_value=0.3,
            habituation=0.0,
        ),
        terrain=TerrainSnapshot(
            state_energy=0.4,
            gradient_norm=0.2,
            max_curvature=0.1,
        ),
        graph_coupling=(0.2, 0.2, 0.1),
        external_input=DynamicFieldInput(
            arousal_drive=0.5,
            prediction_drive=0.4,
            relation_drive=0.1,
            novelty=0.2,
            repeated_exposure=0.0,
        ),
        access=AccessState(attention=0.5, reportability=0.4),
        dt=0.1,
        config=DynamicFieldConfig(relation_gamma=0.2),
    )
    assert fresh.arousal > repeated.arousal
    assert repeated.habituation >= 0.8


def test_geometry_interaction_requires_gradient_and_curvature() -> None:
    access_low = AccessState(attention=0.5, interface_curvature=0.2)
    access_high = AccessState(attention=0.5, interface_curvature=0.8)
    low = subjective_intensity(
        terrain=TerrainSnapshot(
            state_energy=0.4,
            gradient_norm=0.6,
            max_curvature=0.2,
        ),
        access=access_low,
        hypothesis="geometry_interaction",
    )
    high = subjective_intensity(
        terrain=TerrainSnapshot(
            state_energy=0.4,
            gradient_norm=0.6,
            max_curvature=0.2,
        ),
        access=access_high,
        hypothesis="geometry_interaction",
    )
    flat = subjective_intensity(
        terrain=TerrainSnapshot(
            state_energy=0.4,
            gradient_norm=0.0,
            max_curvature=0.2,
        ),
        access=access_high,
        hypothesis="geometry_interaction",
    )
    assert high > low
    assert flat == 0.0


def test_access_mode_uses_reportability_and_uncertainty() -> None:
    terrain = TerrainSnapshot(
        state_energy=0.5,
        gradient_norm=0.4,
        max_curvature=0.2,
    )
    uncertain = subjective_intensity(
        terrain=terrain,
        access=AccessState(
            attention=0.7,
            reportability=0.8,
            access_uncertainty=0.9,
        ),
        hypothesis="access_mode",
    )
    certain = subjective_intensity(
        terrain=terrain,
        access=AccessState(
            attention=0.7,
            reportability=0.8,
            access_uncertainty=0.1,
        ),
        hypothesis="access_mode",
    )
    assert certain > uncertain


def test_surprise_score_separates_prediction_error_and_curvature() -> None:
    flat = surprise_score(0.4, 0.1, 0.0)
    steep = surprise_score(0.4, 0.1, 0.6)
    assert steep > flat
