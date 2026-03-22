from __future__ import annotations

import numpy as np

from inner_os.affective_localizer import BasicAffectiveLocalizer
from inner_os.affective_position import make_neutral_affective_position
from inner_os.affective_terrain import (
    AffectiveTerrainState,
    BasicAffectiveTerrain,
)
from inner_os.association_graph import BasicAssociationGraph, apply_association_reinforcement
from inner_os.dot_seed import derive_dot_seeds
from inner_os.insight_event import BasicInsightDetector
from inner_os.protection_mode import derive_protection_mode
from inner_os.qualia_projector import QualiaState
from inner_os.self_estimator import Estimate, EstimatorHealth
from inner_os.terrain_plasticity import (
    apply_terrain_plasticity,
    derive_terrain_plasticity_update,
)


def test_affective_localizer_changes_position_when_memory_changes() -> None:
    localizer = BasicAffectiveLocalizer(position_dim=3)
    estimate = _estimate()
    health = _health(trust=0.86)
    qualia_state = _qualia_state()
    prev_position = make_neutral_affective_position(3)

    position_a = localizer.localize(
        estimate=estimate,
        health=health,
        qualia_state=qualia_state,
        memory=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        prev_position=prev_position,
        dt=1.0,
    )
    position_b = localizer.localize(
        estimate=estimate,
        health=health,
        qualia_state=qualia_state,
        memory=[0.0, 0.0, 0.0, 0.9, 0.9, 0.9],
        prev_position=prev_position,
        dt=1.0,
    )

    assert position_a.z_aff.shape == (3,)
    assert position_a.cov.shape == (3, 3)
    assert 0.0 <= position_a.confidence <= 1.0
    assert set(position_a.source_weights) == {"state", "qualia", "memory", "carryover"}
    assert abs(sum(position_a.source_weights.values()) - 1.0) <= 1.0e-6
    assert not np.allclose(position_a.z_aff, position_b.z_aff)


def test_affective_terrain_readout_changes_with_position() -> None:
    terrain = BasicAffectiveTerrain()
    terrain_state = AffectiveTerrainState(
        centers=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        widths=np.asarray([0.45, 0.45], dtype=np.float32),
        value_weights=np.asarray([0.8, -0.2], dtype=np.float32),
        approach_weights=np.asarray([0.75, 0.15], dtype=np.float32),
        avoid_weights=np.asarray([0.15, 0.75], dtype=np.float32),
        protect_weights=np.asarray([0.25, 0.85], dtype=np.float32),
        anchor_labels=("safe_hollow", "guarded_ridge"),
    )
    near_safe = make_neutral_affective_position(3)
    near_guarded = make_neutral_affective_position(3)
    near_guarded = near_guarded.__class__(
        z_aff=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        cov=near_guarded.cov,
        confidence=0.5,
        source_weights=near_guarded.source_weights,
    )

    safe_readout = terrain.read(terrain_state, near_safe)
    guarded_readout = terrain.read(terrain_state, near_guarded)

    assert safe_readout.active_patch_label == "safe_hollow"
    assert guarded_readout.active_patch_label == "guarded_ridge"
    assert safe_readout.approach_bias > guarded_readout.approach_bias
    assert guarded_readout.protect_bias > safe_readout.protect_bias
    assert safe_readout.grad.shape == (3,)
    assert safe_readout.curvature.shape == (3,)


def test_protection_mode_moves_toward_stabilize_when_protect_bias_rises() -> None:
    low_mode = derive_protection_mode(
        terrain_readout={
            "value": 0.1,
            "grad": [0.0, 0.0, 0.0],
            "curvature": [0.0, 0.0, 0.0],
            "approach_bias": 0.34,
            "avoid_bias": 0.1,
            "protect_bias": 0.12,
            "active_patch_index": 0,
            "active_patch_label": "plain",
        },
        workspace={"workspace_stability": 0.82, "workspace_mode": "open_foreground"},
        qualia_planner_view={"trust": 0.84, "degraded": False, "body_load": 0.04, "protection_bias": 0.08},
    )
    high_mode = derive_protection_mode(
        terrain_readout={
            "value": -0.4,
            "grad": [0.2, 0.1, 0.0],
            "curvature": [0.1, 0.1, 0.1],
            "approach_bias": 0.18,
            "avoid_bias": 0.66,
            "protect_bias": 0.82,
            "active_patch_index": 1,
            "active_patch_label": "guarded_ridge",
        },
        workspace={"workspace_stability": 0.32, "workspace_mode": "guarded_foreground"},
        qualia_planner_view={"trust": 0.42, "degraded": True, "body_load": 0.18, "protection_bias": 0.22},
    )

    assert low_mode.mode == "monitor"
    assert high_mode.mode in {"contain", "stabilize", "shield"}
    assert high_mode.strength >= low_mode.strength
    assert high_mode.reasons


def test_protection_mode_reads_affective_position_and_self_state() -> None:
    terrain_readout = {
        "value": -0.18,
        "grad": [0.08, 0.02, 0.0],
        "curvature": [0.02, 0.01, 0.01],
        "approach_bias": 0.28,
        "avoid_bias": 0.34,
        "protect_bias": 0.36,
        "active_patch_index": 0,
        "active_patch_label": "guarded_middle",
    }
    light_mode = derive_protection_mode(
        terrain_readout=terrain_readout,
        affective_position={
            "z_aff": [0.0, 0.0, 0.0],
            "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "confidence": 0.84,
            "source_weights": {"state": 0.52, "qualia": 0.3, "memory": 0.12, "carryover": 0.06},
        },
        self_state={
            "stress": 0.22,
            "recovery_need": 0.2,
            "continuity_score": 0.72,
            "recent_strain": 0.18,
        },
        workspace={"workspace_stability": 0.74, "workspace_mode": "foreground"},
        qualia_planner_view={"trust": 0.8, "degraded": False, "body_load": 0.04, "protection_bias": 0.08},
    )
    heavy_mode = derive_protection_mode(
        terrain_readout=terrain_readout,
        affective_position={
            "z_aff": [0.0, 0.0, 0.0],
            "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "confidence": 0.22,
            "source_weights": {"state": 0.22, "qualia": 0.18, "memory": 0.38, "carryover": 0.22},
        },
        self_state={
            "stress": 0.74,
            "recovery_need": 0.68,
            "continuity_score": 0.28,
            "recent_strain": 0.62,
        },
        workspace={"workspace_stability": 0.34, "workspace_mode": "guarded_foreground"},
        qualia_planner_view={"trust": 0.42, "degraded": True, "body_load": 0.18, "protection_bias": 0.16},
    )

    assert heavy_mode.mode in {"contain", "stabilize", "shield"}
    assert heavy_mode.strength >= light_mode.strength
    assert any(reason in heavy_mode.reasons for reason in {"self_stress", "recovery_need", "carryover_pull"})


def test_protection_mode_reads_low_frequency_reframing_bias_as_weak_repair_prior() -> None:
    base_kwargs = {
        "terrain_readout": {
            "value": 0.12,
            "grad": [0.01, 0.0, 0.0],
            "curvature": [0.0, 0.0, 0.0],
            "approach_bias": 0.41,
            "avoid_bias": 0.18,
            "protect_bias": 0.16,
            "active_patch_index": 0,
            "active_patch_label": "open_basin",
        },
        "workspace": {"workspace_stability": 0.82, "workspace_mode": "foreground"},
        "self_state": {
            "stress": 0.16,
            "recovery_need": 0.14,
            "continuity_score": 0.72,
            "recent_strain": 0.12,
        },
        "qualia_planner_view": {
            "trust": 0.82,
            "degraded": False,
            "body_load": 0.02,
            "protection_bias": 0.06,
        },
    }
    neutral_mode = derive_protection_mode(**base_kwargs)
    reframed_mode = derive_protection_mode(
        **base_kwargs,
        insight_reframing_bias=0.72,
        insight_class_focus="reframed_relation",
    )

    assert neutral_mode.mode == "monitor"
    assert reframed_mode.mode == "repair"
    assert "insight_reframing_prior" in reframed_mode.reasons
    assert reframed_mode.strength >= neutral_mode.strength


def test_protection_mode_keeps_body_protection_above_low_frequency_insight_prior() -> None:
    mode = derive_protection_mode(
        terrain_readout={
            "value": -0.24,
            "grad": [0.08, 0.0, 0.0],
            "curvature": [0.04, 0.02, 0.02],
            "approach_bias": 0.44,
            "avoid_bias": 0.22,
            "protect_bias": 0.32,
            "active_patch_index": 0,
            "active_patch_label": "strained_opening",
        },
        workspace={"workspace_stability": 0.28, "workspace_mode": "guarded_foreground"},
        self_state={
            "stress": 0.48,
            "recovery_need": 0.72,
            "continuity_score": 0.34,
            "recent_strain": 0.46,
        },
        qualia_planner_view={
            "trust": 0.34,
            "degraded": True,
            "body_load": 0.22,
            "protection_bias": 0.18,
        },
        insight_reframing_bias=0.82,
        insight_class_focus="reframed_relation",
    )

    assert mode.mode in {"contain", "stabilize", "shield"}
    assert "recovery_need" in mode.reasons or "body_load" in mode.reasons
    assert "insight_reframing_prior" in mode.reasons


def test_protection_mode_keeps_body_protection_above_overnight_commitment_prior() -> None:
    mode = derive_protection_mode(
        terrain_readout={
            "value": -0.18,
            "grad": [0.06, 0.01, 0.0],
            "curvature": [0.03, 0.02, 0.01],
            "approach_bias": 0.52,
            "avoid_bias": 0.18,
            "protect_bias": 0.28,
            "active_patch_index": 0,
            "active_patch_label": "strained_opening",
        },
        self_state={
            "stress": 0.46,
            "recovery_need": 0.76,
            "recent_strain": 0.42,
            "continuity_score": 0.36,
            "commitment_carry_bias": 0.84,
            "commitment_mode_focus": "repair",
        },
        workspace={"workspace_stability": 0.26, "workspace_mode": "guarded_foreground"},
        qualia_planner_view={
            "trust": 0.34,
            "degraded": True,
            "body_load": 0.22,
            "protection_bias": 0.18,
        },
    )

    assert mode.mode in {"contain", "stabilize", "shield"}
    assert "overnight_commitment_repair_prior" in mode.reasons
    assert "recovery_need" in mode.reasons or "body_load" in mode.reasons


def test_protection_mode_exposes_mode_scores_for_mixed_activation() -> None:
    mode = derive_protection_mode(
        terrain_readout={
            "value": -0.06,
            "grad": [0.04, 0.01, 0.0],
            "curvature": [0.02, 0.01, 0.01],
            "approach_bias": 0.48,
            "avoid_bias": 0.26,
            "protect_bias": 0.44,
            "active_patch_index": 0,
            "active_patch_label": "mixed_edge",
        },
        affective_position={
            "z_aff": [0.0, 0.0, 0.0],
            "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "confidence": 0.52,
            "source_weights": {"state": 0.38, "qualia": 0.32, "memory": 0.2, "carryover": 0.1},
        },
        self_state={
            "stress": 0.46,
            "recovery_need": 0.34,
            "continuity_score": 0.54,
            "recent_strain": 0.3,
        },
        workspace={"workspace_stability": 0.5, "workspace_mode": "guarded_foreground"},
        qualia_planner_view={
            "trust": 0.58,
            "degraded": False,
            "body_load": 0.1,
            "protection_bias": 0.16,
        },
    )

    assert set(mode.scores) == {"monitor", "contain", "stabilize", "repair", "shield"}
    assert mode.scores[mode.mode] == mode.strength
    assert mode.winner_margin >= 0.0
    assert mode.dominant_inputs


def test_protection_mode_prefers_contain_when_repair_and_guard_compete() -> None:
    mode = derive_protection_mode(
        terrain_readout={
            "value": -0.1,
            "grad": [0.06, 0.02, 0.0],
            "curvature": [0.03, 0.02, 0.01],
            "approach_bias": 0.54,
            "avoid_bias": 0.3,
            "protect_bias": 0.52,
            "active_patch_index": 0,
            "active_patch_label": "guarded_opening",
        },
        affective_position={
            "z_aff": [0.0, 0.0, 0.0],
            "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "confidence": 0.48,
            "source_weights": {"state": 0.34, "qualia": 0.3, "memory": 0.24, "carryover": 0.12},
        },
        self_state={
            "stress": 0.52,
            "recovery_need": 0.38,
            "continuity_score": 0.5,
            "recent_strain": 0.34,
        },
        workspace={"workspace_stability": 0.46, "workspace_mode": "guarded_foreground"},
        qualia_planner_view={
            "trust": 0.56,
            "degraded": False,
            "body_load": 0.12,
            "protection_bias": 0.16,
        },
    )

    assert mode.mode == "contain"
    assert mode.scores["contain"] > mode.scores["repair"]
    assert any(item in mode.dominant_inputs for item in {"protect_pressure", "terrain_protect_bias", "guarded_workspace"})


def test_temperament_estimate_shifts_mode_but_not_beyond_guard() -> None:
    base_kwargs = {
        "terrain_readout": {
            "value": -0.08,
            "grad": [0.05, 0.02, 0.0],
            "curvature": [0.03, 0.02, 0.01],
            "approach_bias": 0.5,
            "avoid_bias": 0.28,
            "protect_bias": 0.42,
            "active_patch_index": 0,
            "active_patch_label": "guarded_opening",
        },
        "affective_position": {
            "z_aff": [0.0, 0.0, 0.0],
            "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "confidence": 0.5,
            "source_weights": {"state": 0.34, "qualia": 0.3, "memory": 0.24, "carryover": 0.12},
        },
        "self_state": {
            "stress": 0.42,
            "recovery_need": 0.24,
            "continuity_score": 0.52,
            "recent_strain": 0.24,
        },
        "workspace": {"workspace_stability": 0.5, "workspace_mode": "foreground"},
        "qualia_planner_view": {
            "trust": 0.6,
            "degraded": False,
            "body_load": 0.08,
            "protection_bias": 0.14,
        },
    }
    cautious = derive_protection_mode(
        **base_kwargs,
        temperament_estimate={
            "risk_tolerance": 0.18,
            "ambiguity_tolerance": 0.22,
            "curiosity_drive": 0.28,
            "bond_drive": 0.34,
            "recovery_discipline": 0.72,
            "protect_floor": 0.74,
        },
    )
    forward = derive_protection_mode(
        **base_kwargs,
        temperament_estimate={
            "risk_tolerance": 0.82,
            "ambiguity_tolerance": 0.7,
            "curiosity_drive": 0.78,
            "bond_drive": 0.58,
            "recovery_discipline": 0.24,
            "protect_floor": 0.18,
        },
    )

    assert forward.scores["monitor"] > cautious.scores["monitor"]
    assert forward.scores["repair"] > cautious.scores["repair"]
    assert cautious.scores["contain"] >= forward.scores["contain"]


def test_temperament_estimate_does_not_override_severe_guard() -> None:
    mode = derive_protection_mode(
        terrain_readout={
            "value": -0.2,
            "grad": [0.08, 0.03, 0.0],
            "curvature": [0.04, 0.03, 0.02],
            "approach_bias": 0.38,
            "avoid_bias": 0.34,
            "protect_bias": 0.72,
            "active_patch_index": 0,
            "active_patch_label": "hard_guard",
        },
        affective_position={
            "z_aff": [0.0, 0.0, 0.0],
            "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "confidence": 0.42,
            "source_weights": {"state": 0.3, "qualia": 0.28, "memory": 0.26, "carryover": 0.16},
        },
        self_state={
            "stress": 0.78,
            "recovery_need": 0.68,
            "continuity_score": 0.48,
            "recent_strain": 0.52,
        },
        workspace={"workspace_stability": 0.42, "workspace_mode": "guarded_foreground"},
        qualia_planner_view={
            "trust": 0.48,
            "degraded": True,
            "body_load": 0.2,
            "protection_bias": 0.22,
        },
        temperament_estimate={
            "risk_tolerance": 0.88,
            "ambiguity_tolerance": 0.8,
            "curiosity_drive": 0.82,
            "bond_drive": 0.6,
            "recovery_discipline": 0.18,
            "protect_floor": 0.12,
        },
    )

    assert mode.mode in {"contain", "stabilize", "shield"}
    assert mode.scores["monitor"] < mode.scores[mode.mode]


def test_terrain_plasticity_changes_next_readout_after_update() -> None:
    terrain = BasicAffectiveTerrain()
    terrain_state = AffectiveTerrainState(
        centers=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        widths=np.asarray([0.55], dtype=np.float32),
        value_weights=np.asarray([0.0], dtype=np.float32),
        approach_weights=np.asarray([0.2], dtype=np.float32),
        avoid_weights=np.asarray([0.2], dtype=np.float32),
        protect_weights=np.asarray([0.2], dtype=np.float32),
        anchor_labels=("plain",),
    )
    position = make_neutral_affective_position(3)
    before = terrain.read(terrain_state, position)
    update = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=before,
        safety_gain=0.8,
        strain_load=0.1,
        bond_weight=0.6,
        unresolved_tension=0.0,
        dt=1.0,
    )
    updated_state = apply_terrain_plasticity(terrain_state, update)
    after = terrain.read(updated_state, position)

    assert after.approach_bias > before.approach_bias
    assert after.protect_bias > before.protect_bias
    assert after.value != before.value


def test_terrain_plasticity_reweighting_bias_changes_update_gain_by_memory_class() -> None:
    position = make_neutral_affective_position(3)
    terrain = BasicAffectiveTerrain()
    terrain_state = AffectiveTerrainState(
        centers=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        widths=np.asarray([0.55], dtype=np.float32),
        value_weights=np.asarray([0.0], dtype=np.float32),
        approach_weights=np.asarray([0.2], dtype=np.float32),
        avoid_weights=np.asarray([0.2], dtype=np.float32),
        protect_weights=np.asarray([0.2], dtype=np.float32),
        anchor_labels=("plain",),
    )
    readout = terrain.read(terrain_state, position)

    baseline = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.3,
        strain_load=0.52,
        bond_weight=0.44,
        unresolved_tension=0.18,
        terrain_reweighting_bias=0.0,
        memory_class_focus="bond_protection",
        dt=1.0,
    )
    reweighted = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.3,
        strain_load=0.52,
        bond_weight=0.44,
        unresolved_tension=0.18,
        terrain_reweighting_bias=0.6,
        memory_class_focus="bond_protection",
        dt=1.0,
    )
    safe_repeat = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.3,
        strain_load=0.52,
        bond_weight=0.44,
        unresolved_tension=0.18,
        terrain_reweighting_bias=0.6,
        memory_class_focus="safe_repeat",
        dt=1.0,
    )

    assert reweighted.reweighting_bias == 0.6
    assert reweighted.memory_class_focus == "bond_protection"
    assert reweighted.protect_delta > baseline.protect_delta
    assert reweighted.width <= baseline.width
    assert reweighted.protect_delta > safe_repeat.protect_delta
    assert reweighted.spread >= baseline.spread
    assert safe_repeat.width > reweighted.width
    assert safe_repeat.spread > reweighted.spread
    assert reweighted.driver_scores["overnight_reweighting"] > 0.0
    assert reweighted.winner_margin >= 0.0
    assert reweighted.dominant_inputs


def test_safe_repeat_plasticity_makes_local_shape_smoother_than_body_risk() -> None:
    terrain = BasicAffectiveTerrain()
    terrain_state = AffectiveTerrainState(
        centers=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        widths=np.asarray([0.42, 0.42], dtype=np.float32),
        value_weights=np.asarray([0.35, -0.18], dtype=np.float32),
        approach_weights=np.asarray([0.22, 0.18], dtype=np.float32),
        avoid_weights=np.asarray([0.18, 0.22], dtype=np.float32),
        protect_weights=np.asarray([0.18, 0.2], dtype=np.float32),
        anchor_labels=("near", "far"),
    )
    position = make_neutral_affective_position(3)
    before = terrain.read(terrain_state, position)

    safe_update = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=before,
        safety_gain=0.62,
        strain_load=0.16,
        bond_weight=0.28,
        unresolved_tension=0.08,
        terrain_reweighting_bias=0.65,
        memory_class_focus="safe_repeat",
        dt=1.0,
    )
    risk_update = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=before,
        safety_gain=0.22,
        strain_load=0.54,
        bond_weight=0.18,
        unresolved_tension=0.22,
        terrain_reweighting_bias=0.65,
        memory_class_focus="body_risk",
        dt=1.0,
    )

    safe_state = apply_terrain_plasticity(terrain_state, safe_update)
    risk_state = apply_terrain_plasticity(terrain_state, risk_update)
    safe_readout = terrain.read(safe_state, position)
    risk_readout = terrain.read(risk_state, position)

    assert float(np.mean(safe_state.widths)) > float(np.mean(risk_state.widths))
    assert float(np.linalg.norm(safe_readout.curvature)) < float(np.linalg.norm(risk_readout.curvature))
    assert safe_readout.approach_bias >= before.approach_bias
    assert risk_readout.protect_bias > safe_readout.protect_bias
    assert risk_readout.avoid_bias > safe_readout.avoid_bias


def test_reframed_relation_bias_weakly_shapes_local_terrain_on_next_turn() -> None:
    position = make_neutral_affective_position(3)
    terrain = BasicAffectiveTerrain()
    terrain_state = AffectiveTerrainState(
        centers=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        widths=np.asarray([0.48], dtype=np.float32),
        value_weights=np.asarray([0.1], dtype=np.float32),
        approach_weights=np.asarray([0.24], dtype=np.float32),
        avoid_weights=np.asarray([0.18], dtype=np.float32),
        protect_weights=np.asarray([0.18], dtype=np.float32),
        anchor_labels=("plain",),
    )
    readout = terrain.read(terrain_state, position)

    baseline = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.34,
        strain_load=0.18,
        bond_weight=0.26,
        unresolved_tension=0.08,
        dt=1.0,
    )
    reframed = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.34,
        strain_load=0.18,
        bond_weight=0.26,
        unresolved_tension=0.08,
        insight_terrain_shape_bias=0.22,
        insight_terrain_shape_reason="reframed_relation",
        insight_anchor_center=[0.24, -0.06, 0.1],
        insight_anchor_dispersion=0.28,
        qualia_body_load=0.02,
        qualia_degraded=False,
        protection_mode_name="monitor",
        dt=1.0,
    )

    assert reframed.insight_shape_bias > 0.0
    assert reframed.insight_shape_reason == "reframed_relation"
    assert reframed.insight_shape_target == "soft_relation"
    assert reframed.center[0] > baseline.center[0]
    assert reframed.width > baseline.width
    assert reframed.smoothing > baseline.smoothing
    assert reframed.approach_delta > baseline.approach_delta


def test_commitment_carry_bias_weakly_shapes_local_terrain_on_next_turn() -> None:
    position = make_neutral_affective_position(3)
    terrain = BasicAffectiveTerrain()
    terrain_state = AffectiveTerrainState(
        centers=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        widths=np.asarray([0.48], dtype=np.float32),
        value_weights=np.asarray([0.1], dtype=np.float32),
        approach_weights=np.asarray([0.24], dtype=np.float32),
        avoid_weights=np.asarray([0.18], dtype=np.float32),
        protect_weights=np.asarray([0.18], dtype=np.float32),
        anchor_labels=("plain",),
    )
    readout = terrain.read(terrain_state, position)

    baseline = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.34,
        strain_load=0.18,
        bond_weight=0.26,
        unresolved_tension=0.08,
        dt=1.0,
    )
    carried = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.34,
        strain_load=0.18,
        bond_weight=0.26,
        unresolved_tension=0.08,
        commitment_carry_bias=0.4,
        commitment_target_focus="repair",
        commitment_state_focus="commit",
        qualia_body_load=0.02,
        qualia_degraded=False,
        protection_mode_name="monitor",
        dt=1.0,
    )

    assert carried.commitment_shape_bias > 0.0
    assert carried.commitment_shape_target == "repair_commitment_basin"
    assert carried.approach_delta > baseline.approach_delta
    assert carried.driver_scores["commitment_carry"] > 0.0
    assert "overnight_commitment:repair" in carried.dominant_inputs


def test_reframed_relation_bias_targets_repair_or_stabilize_basin_by_mode() -> None:
    position = make_neutral_affective_position(3)
    terrain = BasicAffectiveTerrain()
    terrain_state = AffectiveTerrainState(
        centers=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        widths=np.asarray([0.48], dtype=np.float32),
        value_weights=np.asarray([0.1], dtype=np.float32),
        approach_weights=np.asarray([0.24], dtype=np.float32),
        avoid_weights=np.asarray([0.18], dtype=np.float32),
        protect_weights=np.asarray([0.18], dtype=np.float32),
        anchor_labels=("plain",),
    )
    readout = terrain.read(terrain_state, position)

    repair_basin = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.34,
        strain_load=0.18,
        bond_weight=0.26,
        unresolved_tension=0.08,
        insight_terrain_shape_bias=0.24,
        insight_terrain_shape_reason="reframed_relation",
        insight_anchor_center=[0.24, -0.06, 0.1],
        insight_anchor_dispersion=0.28,
        qualia_body_load=0.02,
        qualia_degraded=False,
        protection_mode_name="repair",
        dt=1.0,
    )
    stabilize_basin = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.34,
        strain_load=0.18,
        bond_weight=0.26,
        unresolved_tension=0.08,
        insight_terrain_shape_bias=0.24,
        insight_terrain_shape_reason="reframed_relation",
        insight_anchor_center=[0.24, -0.06, 0.1],
        insight_anchor_dispersion=0.28,
        qualia_body_load=0.02,
        qualia_degraded=False,
        protection_mode_name="stabilize",
        dt=1.0,
    )

    assert repair_basin.insight_shape_target == "repair_basin"
    assert stabilize_basin.insight_shape_target == "stabilize_basin"
    assert repair_basin.approach_delta > stabilize_basin.approach_delta
    assert stabilize_basin.protect_delta > repair_basin.protect_delta


def test_new_link_hypothesis_bias_keeps_terrain_shape_change_nearly_neutral() -> None:
    position = make_neutral_affective_position(3)
    terrain = BasicAffectiveTerrain()
    terrain_state = AffectiveTerrainState(
        centers=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        widths=np.asarray([0.48], dtype=np.float32),
        value_weights=np.asarray([0.1], dtype=np.float32),
        approach_weights=np.asarray([0.24], dtype=np.float32),
        avoid_weights=np.asarray([0.18], dtype=np.float32),
        protect_weights=np.asarray([0.18], dtype=np.float32),
        anchor_labels=("plain",),
    )
    readout = terrain.read(terrain_state, position)

    weak_hypothesis = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.34,
        strain_load=0.18,
        bond_weight=0.26,
        unresolved_tension=0.08,
        insight_terrain_shape_bias=0.32,
        insight_terrain_shape_reason="new_link_hypothesis",
        insight_anchor_center=[0.24, -0.06, 0.1],
        insight_anchor_dispersion=0.28,
        qualia_body_load=0.02,
        qualia_degraded=False,
        protection_mode_name="monitor",
        dt=1.0,
    )
    strong_reframed = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.34,
        strain_load=0.18,
        bond_weight=0.26,
        unresolved_tension=0.08,
        insight_terrain_shape_bias=0.32,
        insight_terrain_shape_reason="reframed_relation",
        insight_anchor_center=[0.24, -0.06, 0.1],
        insight_anchor_dispersion=0.28,
        qualia_body_load=0.02,
        qualia_degraded=False,
        protection_mode_name="monitor",
        dt=1.0,
    )

    assert weak_hypothesis.insight_shape_target == "hypothesis_hold"
    assert weak_hypothesis.insight_shape_bias < strong_reframed.insight_shape_bias
    assert abs(weak_hypothesis.approach_delta - strong_reframed.approach_delta) > 0.0
    assert weak_hypothesis.width <= strong_reframed.width
    assert weak_hypothesis.smoothing < strong_reframed.smoothing


def test_body_risk_conditions_suppress_insight_shape_bias() -> None:
    position = make_neutral_affective_position(3)
    terrain = BasicAffectiveTerrain()
    terrain_state = AffectiveTerrainState(
        centers=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        widths=np.asarray([0.48], dtype=np.float32),
        value_weights=np.asarray([0.1], dtype=np.float32),
        approach_weights=np.asarray([0.24], dtype=np.float32),
        avoid_weights=np.asarray([0.18], dtype=np.float32),
        protect_weights=np.asarray([0.18], dtype=np.float32),
        anchor_labels=("plain",),
    )
    readout = terrain.read(terrain_state, position)

    open_case = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.34,
        strain_load=0.18,
        bond_weight=0.26,
        unresolved_tension=0.08,
        insight_terrain_shape_bias=0.28,
        insight_terrain_shape_reason="reframed_relation",
        insight_anchor_center=[0.24, -0.06, 0.1],
        insight_anchor_dispersion=0.28,
        qualia_body_load=0.02,
        qualia_degraded=False,
        protection_mode_name="monitor",
        memory_class_focus="episodic",
        dt=1.0,
    )
    guarded_case = derive_terrain_plasticity_update(
        position_state=position,
        terrain_readout=readout,
        safety_gain=0.18,
        strain_load=0.48,
        bond_weight=0.12,
        unresolved_tension=0.24,
        insight_terrain_shape_bias=0.28,
        insight_terrain_shape_reason="reframed_relation",
        insight_anchor_center=[0.24, -0.06, 0.1],
        insight_anchor_dispersion=0.28,
        qualia_body_load=0.22,
        qualia_degraded=True,
        protection_mode_name="shield",
        memory_class_focus="body_risk",
        dt=1.0,
    )

    assert guarded_case.insight_shape_bias < open_case.insight_shape_bias
    assert guarded_case.smoothing < open_case.smoothing
    assert float(np.linalg.norm(guarded_case.center - position.z_aff)) < float(np.linalg.norm(open_case.center - position.z_aff))
    assert open_case.driver_scores["insight_shape"] > guarded_case.driver_scores["insight_shape"]
    assert open_case.dominant_inputs


def test_dot_seed_association_and_insight_event_link_weak_points() -> None:
    seeds = derive_dot_seeds(
        qualia_state={**_qualia_state().to_dict(), "axis_labels": ["care", "strain", "bond", "quiet"]},
        current_state={
            "memory_anchor": "shared_thread",
            "relation_seed_summary": "shared thread",
            "replay_intensity": 0.64,
            "meaning_inertia": 0.42,
            "pending_meaning": 0.38,
            "unresolved_count": 1,
            "related_person_id": "user",
            "attachment": 0.62,
            "trust_memory": 0.54,
        },
        current_text="shared thread feels different now",
        current_focus="shared_thread",
    )
    graph = BasicAssociationGraph().build(dot_seeds=seeds)
    insight = BasicInsightDetector().detect(
        dot_seeds=seeds,
        association_graph=graph,
        qualia_trust=0.82,
    )

    assert seeds.seeds
    assert {seed.source for seed in seeds.seeds} >= {"felt", "memory", "external_cue"}
    assert graph.edges
    assert graph.dominant_weight > 0.0
    assert graph.winner_margin >= 0.0
    assert graph.dominant_inputs
    assert insight.score.total > 0.0
    assert insight.connected_seed_ids
    assert insight.summary
    assert insight.triggered is True


def test_association_reinforcement_exposes_post_turn_dominant_inputs() -> None:
    previous_state = {
        "link_weights": {
            "bond:user|memory:shared_thread": 0.32,
            "felt:care|memory:shared_thread": 0.21,
        },
        "link_counts": {
            "bond:user|memory:shared_thread": 2,
            "felt:care|memory:shared_thread": 1,
        },
    }
    updated = apply_association_reinforcement(
        previous_state,
        {
            "triggered": True,
            "link_key": "bond:user|memory:shared_thread",
            "score": {"total": 0.74},
            "reasons": ["source_diversity", "coherent_link"],
        },
        association_reweighting_focus="repeated_links",
        association_reweighting_reason="repeated_insight_trace",
    )

    assert updated.dominant_link_key == "bond:user|memory:shared_thread"
    assert updated.dominant_weight > 0.0
    assert updated.winner_margin >= 0.0
    assert "source_diversity" in updated.dominant_inputs
    assert "overnight_focus:repeated_links" in updated.dominant_inputs


def test_association_reinforcement_reads_commitment_followup_bias() -> None:
    previous_state = {
        "link_weights": {
            "bond:user|memory:shared_thread": 0.32,
        },
        "link_counts": {
            "bond:user|memory:shared_thread": 2,
        },
    }
    reopen = apply_association_reinforcement(
        previous_state,
        {
            "triggered": True,
            "link_key": "bond:user|memory:shared_thread",
            "score": {"total": 0.64},
            "reasons": ["source_diversity"],
        },
        commitment_followup_focus="reopen_softly",
        commitment_carry_bias=0.42,
    )
    hold = apply_association_reinforcement(
        previous_state,
        {
            "triggered": True,
            "link_key": "bond:user|memory:shared_thread",
            "score": {"total": 0.64},
            "reasons": ["source_diversity"],
        },
        commitment_followup_focus="hold",
        commitment_carry_bias=0.42,
    )

    assert reopen.link_weights["bond:user|memory:shared_thread"] > hold.link_weights["bond:user|memory:shared_thread"]
    assert "overnight_commitment:reopen_softly" in reopen.dominant_inputs


def test_association_reinforcement_updates_graph_only_after_triggered_insight() -> None:
    seeds = derive_dot_seeds(
        qualia_state={**_qualia_state().to_dict(), "axis_labels": ["care", "strain", "bond", "quiet"]},
        current_state={
            "memory_anchor": "repair_window",
            "relation_seed_summary": "repair window",
            "replay_intensity": 0.72,
            "meaning_inertia": 0.44,
            "pending_meaning": 0.4,
            "unresolved_count": 1,
        },
        current_text="repair window opens here",
        current_focus="repair_window",
    )
    graph = BasicAssociationGraph().build(dot_seeds=seeds)
    insight = BasicInsightDetector().detect(
        dot_seeds=seeds,
        association_graph=graph,
        qualia_trust=0.78,
    )
    reinforced = apply_association_reinforcement({}, insight.to_dict())
    untouched = apply_association_reinforcement({}, {"triggered": False})

    assert insight.triggered is True
    assert insight.link_key
    assert reinforced.link_weights[insight.link_key] > 0.0
    assert reinforced.link_counts[insight.link_key] == 1
    assert untouched.link_weights == {}


def test_association_graph_reads_low_frequency_insight_bias_on_next_turn() -> None:
    seeds = derive_dot_seeds(
        qualia_state={**_qualia_state().to_dict(), "axis_labels": ["care", "strain", "bond", "quiet"]},
        current_state={
            "memory_anchor": "shared_thread",
            "relation_seed_summary": "shared thread",
            "replay_intensity": 0.7,
            "meaning_inertia": 0.46,
            "pending_meaning": 0.36,
            "unresolved_count": 1,
            "related_person_id": "user",
            "attachment": 0.66,
            "trust_memory": 0.58,
        },
        current_text="something new connects with the shared thread",
        current_focus="shared_thread",
    )
    baseline = BasicAssociationGraph().build(dot_seeds=seeds)
    biased = BasicAssociationGraph().build(
        dot_seeds=seeds,
        association_reweighting_bias=0.68,
        insight_reframing_bias=0.28,
        insight_class_focus="new_link_hypothesis",
    )

    assert baseline.edges
    assert biased.edges
    assert biased.dominant_weight > baseline.dominant_weight


def _estimate() -> Estimate:
    return Estimate(
        x_hat=np.asarray([0.25, -0.1, 0.42, 0.18], dtype=np.float32),
        cov=np.diag(np.asarray([0.2, 0.28, 0.24, 0.22], dtype=np.float32)).astype(np.float32),
        y_hat=np.zeros(4, dtype=np.float32),
        innovation=np.zeros(4, dtype=np.float32),
        innovation_cov=np.eye(4, dtype=np.float32),
        H=np.eye(4, dtype=np.float32),
        nis=0.8,
    )


def _health(*, trust: float) -> EstimatorHealth:
    return EstimatorHealth(
        innovation_norm=0.2,
        nis=0.8,
        observability_mean=0.9,
        observed_fraction=1.0,
        trust=trust,
        degraded=False,
        reason="healthy",
        overconfident_estimate=False,
        observation_contract_break=False,
        low_observability=False,
    )


def _qualia_state() -> QualiaState:
    vector = np.asarray([0.12, 0.45, 0.2, 0.08], dtype=np.float32)
    zeros = np.zeros_like(vector)
    return QualiaState(
        gate=np.asarray([0.3, 0.7, 0.45, 0.2], dtype=np.float32),
        qualia=vector,
        precision=np.asarray([0.8, 0.7, 0.6, 0.7], dtype=np.float32),
        observability=np.asarray([0.5, 0.9, 0.4, 0.2], dtype=np.float32),
        body_coupling=np.asarray([0.2, 0.6, 0.2, 0.1], dtype=np.float32),
        value_grad=np.asarray([0.1, 0.3, 0.2, 0.05], dtype=np.float32),
        habituation=zeros,
        trust_applied=0.82,
        degraded=False,
        reason="healthy",
    )
