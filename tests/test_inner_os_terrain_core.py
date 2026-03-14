from inner_os.terrain_core import AffectiveTerrainCore


def test_affective_terrain_core_guarded_edge() -> None:
    core = AffectiveTerrainCore()
    snap = core.snapshot(
        valence=-0.4,
        arousal=0.8,
        stress=0.85,
        temporal_pressure=0.7,
        memory_ignition=0.2,
    )
    assert snap.attractor == "guarded_edge"
    assert snap.danger_slope > snap.recovery_basin


def test_affective_terrain_core_warm_rest() -> None:
    core = AffectiveTerrainCore()
    snap = core.snapshot(
        valence=0.6,
        arousal=0.2,
        stress=0.15,
        temporal_pressure=0.2,
        memory_ignition=0.1,
    )
    assert snap.attractor == "warm_rest"
    assert snap.recovery_basin > 0.5


def test_affective_terrain_core_transition_forms_unfamiliar_slope() -> None:
    core = AffectiveTerrainCore()
    snap = core.snapshot(
        valence=0.08,
        arousal=0.34,
        stress=0.26,
        temporal_pressure=0.22,
        transition_intensity=0.82,
        social_grounding=0.21,
        community_resonance=0.18,
    )
    assert snap.transition_roughness > 0.4
    assert snap.attractor == "unfamiliar_slope"
