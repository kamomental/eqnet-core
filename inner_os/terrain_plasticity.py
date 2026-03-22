from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from .affective_position import AffectivePositionState
from .affective_terrain import AffectiveTerrainState, TerrainReadout
from .observation_model import Vector


@dataclass(frozen=True)
class TerrainPlasticityUpdate:
    center: Vector
    width: float
    spread: float
    value_delta: float
    approach_delta: float
    avoid_delta: float
    protect_delta: float
    confidence: float
    reweighting_bias: float
    memory_class_focus: str
    smoothing: float
    commitment_shape_bias: float = 0.0
    commitment_shape_target: str = ""
    insight_shape_bias: float = 0.0
    insight_shape_reason: str = ""
    insight_shape_target: str = ""
    insight_anchor_center: Vector | None = None
    insight_anchor_dispersion: float = 0.0
    driver_scores: Dict[str, float] = field(default_factory=dict)
    winner_margin: float = 0.0
    dominant_inputs: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "width": float(self.width),
            "spread": float(self.spread),
            "value_delta": float(self.value_delta),
            "approach_delta": float(self.approach_delta),
            "avoid_delta": float(self.avoid_delta),
            "protect_delta": float(self.protect_delta),
            "confidence": float(self.confidence),
            "reweighting_bias": float(self.reweighting_bias),
            "memory_class_focus": str(self.memory_class_focus),
            "smoothing": float(self.smoothing),
            "commitment_shape_bias": float(self.commitment_shape_bias),
            "commitment_shape_target": str(self.commitment_shape_target),
            "insight_shape_bias": float(self.insight_shape_bias),
            "insight_shape_reason": str(self.insight_shape_reason),
            "insight_shape_target": str(self.insight_shape_target),
            "insight_anchor_center": self.insight_anchor_center.tolist() if self.insight_anchor_center is not None else [],
            "insight_anchor_dispersion": float(self.insight_anchor_dispersion),
            "driver_scores": {
                str(key): float(value)
                for key, value in self.driver_scores.items()
            },
            "winner_margin": float(self.winner_margin),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_terrain_plasticity_update(
    *,
    position_state: AffectivePositionState,
    terrain_readout: TerrainReadout,
    safety_gain: float = 0.0,
    strain_load: float = 0.0,
    bond_weight: float = 0.0,
    unresolved_tension: float = 0.0,
    terrain_reweighting_bias: float = 0.0,
    memory_class_focus: str = "",
    commitment_carry_bias: float = 0.0,
    commitment_target_focus: str = "",
    commitment_state_focus: str = "",
    insight_terrain_shape_bias: float = 0.0,
    insight_terrain_shape_reason: str = "",
    insight_anchor_center: Any = None,
    insight_anchor_dispersion: float = 0.0,
    qualia_body_load: float = 0.0,
    qualia_degraded: bool = False,
    protection_mode_name: str = "",
    dt: float = 1.0,
) -> TerrainPlasticityUpdate:
    if dt <= 0.0:
        raise ValueError("terrain plasticity dt must be positive")

    safety_gain = _clamp01(safety_gain)
    strain_load = _clamp01(strain_load)
    bond_weight = _clamp01(bond_weight)
    unresolved_tension = _clamp01(unresolved_tension)
    terrain_reweighting_bias = _clamp01(terrain_reweighting_bias)
    memory_class_focus = str(memory_class_focus or "").strip() or "episodic"
    commitment_carry_bias = _clamp01(commitment_carry_bias)
    commitment_target_focus = str(commitment_target_focus or "").strip()
    commitment_state_focus = str(commitment_state_focus or "").strip()
    insight_terrain_shape_bias = _clamp01(insight_terrain_shape_bias)
    insight_terrain_shape_reason = str(insight_terrain_shape_reason or "").strip()
    qualia_body_load = _clamp01(qualia_body_load)
    protection_mode_name = str(protection_mode_name or "").strip()
    reweighting = _reweighting_profile(memory_class_focus)
    low_frequency_scale = 1.0 + 0.25 * terrain_reweighting_bias
    center = position_state.z_aff.astype(np.float32).copy()

    base_value_delta = 0.08 * safety_gain - 0.08 * strain_load - 0.04 * unresolved_tension
    base_approach_delta = 0.06 * safety_gain + 0.05 * bond_weight - 0.03 * strain_load
    base_avoid_delta = 0.06 * strain_load + 0.03 * unresolved_tension - 0.03 * safety_gain
    base_protect_delta = 0.06 * strain_load + 0.04 * bond_weight + 0.03 * unresolved_tension
    value_delta = base_value_delta * low_frequency_scale * (1.0 + reweighting["value"] * terrain_reweighting_bias)
    approach_delta = base_approach_delta * low_frequency_scale * (1.0 + reweighting["approach"] * terrain_reweighting_bias)
    avoid_delta = base_avoid_delta * low_frequency_scale * (1.0 + reweighting["avoid"] * terrain_reweighting_bias)
    protect_delta = base_protect_delta * low_frequency_scale * (1.0 + reweighting["protect"] * terrain_reweighting_bias)
    confidence = _clamp01(position_state.confidence)
    base_width = max(
        0.25,
        0.75 - 0.3 * confidence + 0.15 * unresolved_tension,
    )
    width = max(
        0.25,
        base_width * (1.0 + 0.18 * terrain_reweighting_bias * reweighting["width_pull"]),
    )
    base_spread = max(0.18, base_width)
    spread = max(
        0.18,
        width * (1.0 + 0.24 * terrain_reweighting_bias * reweighting["spread_pull"]),
    )
    smoothing = float(np.clip(terrain_reweighting_bias * reweighting["smoothing"], -1.0, 1.0))
    reweighted_width = width
    reweighted_spread = spread
    reweighting_strength = _driver_strength(
        value_delta - base_value_delta,
        approach_delta - base_approach_delta,
        avoid_delta - base_avoid_delta,
        protect_delta - base_protect_delta,
        reweighted_width - base_width,
        reweighted_spread - base_spread,
        smoothing,
    )
    commitment_scale = _derive_commitment_shape_scale(
        commitment_carry_bias=commitment_carry_bias,
        commitment_target_focus=commitment_target_focus,
        commitment_state_focus=commitment_state_focus,
        qualia_body_load=qualia_body_load,
        qualia_degraded=qualia_degraded,
        memory_class_focus=memory_class_focus,
        protection_mode_name=protection_mode_name,
    )
    commitment_strength = 0.0
    commitment_shape_target = ""
    if commitment_scale > 0.0:
        commitment_shape_target, commitment_profile = _commitment_shape_profile(commitment_target_focus)
        commitment_width_before = width
        commitment_spread_before = spread
        width = max(
            0.22,
            width * (1.0 + commitment_profile["width_pull"] * commitment_scale),
        )
        spread = max(
            0.18,
            spread * (1.0 + commitment_profile["spread_pull"] * commitment_scale),
        )
        value_delta += commitment_profile["value"] * commitment_scale
        approach_delta += commitment_profile["approach"] * commitment_scale
        avoid_delta += commitment_profile["avoid"] * commitment_scale
        protect_delta += commitment_profile["protect"] * commitment_scale
        smoothing = float(np.clip(smoothing + commitment_profile["smoothing"] * commitment_scale, -1.0, 1.0))
        commitment_strength = _driver_strength(
            commitment_profile["value"] * commitment_scale,
            commitment_profile["approach"] * commitment_scale,
            commitment_profile["avoid"] * commitment_scale,
            commitment_profile["protect"] * commitment_scale,
            width - commitment_width_before,
            spread - commitment_spread_before,
            commitment_profile["smoothing"] * commitment_scale,
        )
    insight_anchor_vector = _coerce_anchor_center(insight_anchor_center, position_state.position_dim)
    insight_scale = _derive_insight_shape_scale(
        insight_terrain_shape_bias=insight_terrain_shape_bias,
        insight_terrain_shape_reason=insight_terrain_shape_reason,
        qualia_body_load=qualia_body_load,
        qualia_degraded=qualia_degraded,
        memory_class_focus=memory_class_focus,
        protection_mode_name=protection_mode_name,
    )
    insight_strength = 0.0
    insight_shape_target = ""
    if insight_scale > 0.0:
        insight_shape_target, insight_profile = _insight_shape_profile(
            insight_terrain_shape_reason,
            protection_mode_name=protection_mode_name,
        )
        insight_width_before = width
        insight_spread_before = spread
        if insight_anchor_vector is not None:
            center_blend = min(0.18, insight_scale * 0.24)
            center = ((1.0 - center_blend) * center + center_blend * insight_anchor_vector).astype(np.float32)
        width = max(
            0.22,
            width * (1.0 + insight_profile["width_pull"] * insight_scale),
        )
        spread = max(
            0.18,
            spread * (1.0 + insight_profile["spread_pull"] * insight_scale),
        )
        value_delta += insight_profile["value"] * insight_scale
        approach_delta += insight_profile["approach"] * insight_scale
        avoid_delta += insight_profile["avoid"] * insight_scale
        protect_delta += insight_profile["protect"] * insight_scale
        smoothing = float(np.clip(smoothing + insight_profile["smoothing"] * insight_scale, -1.0, 1.0))
        insight_strength = _driver_strength(
            insight_profile["value"] * insight_scale,
            insight_profile["approach"] * insight_scale,
            insight_profile["avoid"] * insight_scale,
            insight_profile["protect"] * insight_scale,
            width - insight_width_before,
            spread - insight_spread_before,
            insight_profile["smoothing"] * insight_scale,
        )
    local_strength = _driver_strength(
        base_value_delta,
        base_approach_delta,
        base_avoid_delta,
        base_protect_delta,
    )
    driver_scores = {
        "local_same_turn": round(_clamp01(local_strength), 4),
        "overnight_reweighting": round(_clamp01(reweighting_strength), 4),
        "commitment_carry": round(_clamp01(commitment_strength), 4),
        "insight_shape": round(_clamp01(insight_strength), 4),
    }
    winner_margin = _derive_winner_margin(driver_scores.values())
    dominant_inputs = _derive_terrain_dominant_inputs(
        safety_gain=safety_gain,
        strain_load=strain_load,
        bond_weight=bond_weight,
        unresolved_tension=unresolved_tension,
        terrain_reweighting_bias=terrain_reweighting_bias,
        memory_class_focus=memory_class_focus,
        commitment_scale=commitment_scale,
        commitment_target_focus=commitment_target_focus,
        insight_scale=insight_scale,
        insight_shape_reason=insight_terrain_shape_reason,
    )

    return TerrainPlasticityUpdate(
        center=center,
        width=float(width),
        spread=float(spread),
        value_delta=float(value_delta * dt),
        approach_delta=float(approach_delta * dt),
        avoid_delta=float(avoid_delta * dt),
        protect_delta=float(protect_delta * dt),
        confidence=confidence,
        reweighting_bias=float(terrain_reweighting_bias),
        memory_class_focus=memory_class_focus,
        smoothing=smoothing,
        commitment_shape_bias=float(commitment_scale),
        commitment_shape_target=commitment_shape_target if commitment_scale > 0.0 else "",
        insight_shape_bias=float(insight_scale),
        insight_shape_reason=insight_terrain_shape_reason,
        insight_shape_target=insight_shape_target if insight_scale > 0.0 else "",
        insight_anchor_center=insight_anchor_vector,
        insight_anchor_dispersion=max(0.0, float(insight_anchor_dispersion)),
        driver_scores=driver_scores,
        winner_margin=winner_margin,
        dominant_inputs=dominant_inputs,
    )


def apply_terrain_plasticity(
    terrain_state: AffectiveTerrainState,
    update: TerrainPlasticityUpdate,
) -> AffectiveTerrainState:
    if terrain_state.position_dim != update.center.shape[0]:
        raise ValueError("terrain and update position dimensions must match")

    distances = np.linalg.norm(terrain_state.centers - update.center[None, :], axis=1)
    patch_index = int(np.argmin(distances))
    spread = max(0.18, float(update.spread))
    influence = np.exp(-(np.square(distances.astype(np.float32))) / (2.0 * (spread ** 2))).astype(np.float32)
    influence[patch_index] = 1.0

    centers = terrain_state.centers.astype(np.float32).copy()
    widths = terrain_state.widths.astype(np.float32).copy()
    value_weights = terrain_state.value_weights.astype(np.float32).copy()
    approach_weights = terrain_state.approach_weights.astype(np.float32).copy()
    avoid_weights = terrain_state.avoid_weights.astype(np.float32).copy()
    protect_weights = terrain_state.protect_weights.astype(np.float32).copy()

    learning_rate = 0.035 + 0.085 * float(update.confidence)
    smoothing = float(update.smoothing)
    for index in range(terrain_state.patch_count):
        local_rate = learning_rate * float(influence[index])
        if local_rate <= 0.0:
            continue
        centers[index] = (
            (1.0 - local_rate) * centers[index] + local_rate * update.center
        ).astype(np.float32)
        target_width = float(update.width)
        width_bias = 0.0
        if smoothing > 0.0:
            width_bias = 0.18 * smoothing
        elif smoothing < 0.0:
            width_bias = 0.12 * smoothing
        widths[index] = max(
            0.1,
            float((1.0 - local_rate) * widths[index] + local_rate * target_width * (1.0 + width_bias)),
        )

        value_step = float(update.value_delta) * local_rate
        if smoothing > 0.0:
            value_step *= 1.0 - 0.35 * smoothing
        elif smoothing < 0.0:
            value_step *= 1.0 + 0.2 * abs(smoothing)
        value_weights[index] += np.float32(value_step)

        approach_step = float(update.approach_delta) * local_rate
        avoid_step = float(update.avoid_delta) * local_rate
        protect_step = float(update.protect_delta) * local_rate
        if smoothing > 0.0:
            avoid_step *= 1.0 - 0.22 * smoothing
            protect_step *= 1.0 - 0.12 * smoothing
        elif smoothing < 0.0:
            avoid_step *= 1.0 + 0.18 * abs(smoothing)
            protect_step *= 1.0 + 0.24 * abs(smoothing)

        approach_weights[index] = np.float32(_clamp01(float(approach_weights[index]) + approach_step))
        avoid_weights[index] = np.float32(_clamp01(float(avoid_weights[index]) + avoid_step))
        protect_weights[index] = np.float32(_clamp01(float(protect_weights[index]) + protect_step))

    return AffectiveTerrainState(
        centers=centers,
        widths=widths,
        value_weights=value_weights,
        approach_weights=approach_weights,
        avoid_weights=avoid_weights,
        protect_weights=protect_weights,
        anchor_labels=terrain_state.anchor_labels,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _reweighting_profile(memory_class_focus: str) -> Dict[str, float]:
    focus = str(memory_class_focus or "").strip()
    profiles: Dict[str, Dict[str, float]] = {
        "episodic": {
            "value": 0.08,
            "approach": 0.05,
            "avoid": 0.05,
            "protect": 0.05,
            "width_pull": 0.02,
            "spread_pull": 0.04,
            "smoothing": 0.04,
        },
        "body_risk": {
            "value": 0.02,
            "approach": -0.04,
            "avoid": 0.22,
            "protect": 0.24,
            "width_pull": -0.24,
            "spread_pull": -0.18,
            "smoothing": -0.38,
        },
        "bond_protection": {
            "value": 0.06,
            "approach": 0.08,
            "avoid": 0.04,
            "protect": 0.26,
            "width_pull": -0.08,
            "spread_pull": 0.16,
            "smoothing": -0.12,
        },
        "repair_trace": {
            "value": 0.04,
            "approach": 0.03,
            "avoid": 0.1,
            "protect": 0.2,
            "width_pull": -0.04,
            "spread_pull": 0.02,
            "smoothing": -0.08,
        },
        "unresolved_tension": {
            "value": -0.06,
            "approach": -0.04,
            "avoid": 0.18,
            "protect": 0.16,
            "width_pull": -0.16,
            "spread_pull": -0.1,
            "smoothing": -0.22,
        },
        "safe_repeat": {
            "value": 0.16,
            "approach": 0.14,
            "avoid": -0.04,
            "protect": 0.04,
            "width_pull": 0.28,
            "spread_pull": 0.22,
            "smoothing": 0.42,
        },
        "insight_trace": {
            "value": 0.1,
            "approach": 0.08,
            "avoid": -0.02,
            "protect": 0.02,
            "width_pull": 0.12,
            "spread_pull": 0.16,
            "smoothing": 0.18,
        },
    }
    return profiles.get(focus, profiles["episodic"])


def _coerce_anchor_center(value: Any, position_dim: int) -> Vector | None:
    vector = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    if vector.size != position_dim:
        return None
    return vector.astype(np.float32)


def _derive_insight_shape_scale(
    *,
    insight_terrain_shape_bias: float,
    insight_terrain_shape_reason: str,
    qualia_body_load: float,
    qualia_degraded: bool,
    memory_class_focus: str,
    protection_mode_name: str,
) -> float:
    reason = str(insight_terrain_shape_reason or "").strip()
    if not reason or insight_terrain_shape_bias <= 0.0:
        return 0.0
    scale = float(insight_terrain_shape_bias)
    if reason == "new_link_hypothesis":
        scale *= 0.18
    elif reason == "repeated_insight_trace":
        scale *= 0.44
    else:
        scale *= 0.62
    if qualia_degraded:
        scale *= 0.35
    scale *= max(0.18, 1.0 - 0.78 * qualia_body_load)
    if str(memory_class_focus or "").strip() == "body_risk":
        scale *= 0.28
    if protection_mode_name == "shield":
        scale *= 0.52
    elif protection_mode_name == "stabilize":
        scale *= 0.64
    elif protection_mode_name == "contain":
        scale *= 0.68
    return _clamp01(scale)


def _derive_commitment_shape_scale(
    *,
    commitment_carry_bias: float,
    commitment_target_focus: str,
    commitment_state_focus: str,
    qualia_body_load: float,
    qualia_degraded: bool,
    memory_class_focus: str,
    protection_mode_name: str,
) -> float:
    target = str(commitment_target_focus or "").strip()
    state = str(commitment_state_focus or "").strip()
    if not target or commitment_carry_bias <= 0.0 or state == "waver":
        return 0.0
    scale = float(commitment_carry_bias)
    if state == "commit":
        scale *= 0.3
    elif state == "settle":
        scale *= 0.16
    else:
        scale *= 0.06
    if target in {"repair", "bond_protect"}:
        scale *= 1.0
    elif target == "step_forward":
        scale *= 0.78
    elif target == "stabilize":
        scale *= 0.72
    elif target == "hold":
        scale *= 0.62
    else:
        scale = 0.0
    if qualia_degraded:
        scale *= 0.5
    scale *= max(0.18, 1.0 - 0.66 * qualia_body_load)
    if str(memory_class_focus or "").strip() == "body_risk":
        scale *= 0.34
    if protection_mode_name == "shield":
        scale *= 0.24
    elif protection_mode_name == "stabilize":
        scale *= 0.48
    elif protection_mode_name == "contain":
        scale *= 0.64
    return _clamp01(scale)


def _insight_shape_profile(
    reason: str,
    *,
    protection_mode_name: str,
) -> tuple[str, Dict[str, float]]:
    normalized_reason = str(reason or "").strip()
    protection_mode = str(protection_mode_name or "").strip()
    if normalized_reason == "reframed_relation":
        if protection_mode == "repair":
            return (
                "repair_basin",
                {
                    "value": 0.1,
                    "approach": 0.1,
                    "avoid": -0.012,
                    "protect": 0.028,
                    "width_pull": 0.14,
                    "spread_pull": 0.1,
                    "smoothing": 0.22,
                },
            )
        if protection_mode == "stabilize":
            return (
                "stabilize_basin",
                {
                    "value": 0.08,
                    "approach": 0.04,
                    "avoid": -0.01,
                    "protect": 0.05,
                    "width_pull": 0.1,
                    "spread_pull": 0.08,
                    "smoothing": 0.24,
                },
            )
        return (
            "soft_relation",
            {
                "value": 0.08,
                "approach": 0.07,
                "avoid": -0.01,
                "protect": 0.03,
                "width_pull": 0.12,
                "spread_pull": 0.08,
                "smoothing": 0.2,
            },
        )
    if normalized_reason == "repeated_insight_trace":
        if protection_mode == "repair":
            return (
                "repair_trace_basin",
                {
                    "value": 0.05,
                    "approach": 0.04,
                    "avoid": -0.006,
                    "protect": 0.016,
                    "width_pull": 0.06,
                    "spread_pull": 0.06,
                    "smoothing": 0.14,
                },
            )
        if protection_mode == "stabilize":
            return (
                "stabilize_trace_basin",
                {
                    "value": 0.045,
                    "approach": 0.022,
                    "avoid": -0.005,
                    "protect": 0.024,
                    "width_pull": 0.05,
                    "spread_pull": 0.05,
                    "smoothing": 0.15,
                },
            )
        return (
            "trace_basin",
            {
                "value": 0.04,
                "approach": 0.03,
                "avoid": -0.005,
                "protect": 0.015,
                "width_pull": 0.05,
                "spread_pull": 0.05,
                "smoothing": 0.12,
            },
        )
    return (
        "hypothesis_hold",
        {
            "value": 0.0,
            "approach": 0.0,
            "avoid": 0.0,
            "protect": 0.0,
            "width_pull": 0.02,
            "spread_pull": 0.03,
            "smoothing": 0.04,
        },
    )


def _commitment_shape_profile(target: str) -> tuple[str, Dict[str, float]]:
    normalized_target = str(target or "").strip()
    if normalized_target == "repair":
        return (
            "repair_commitment_basin",
            {
                "value": 0.05,
                "approach": 0.04,
                "avoid": -0.01,
                "protect": 0.03,
                "width_pull": -0.03,
                "spread_pull": 0.02,
                "smoothing": 0.02,
            },
        )
    if normalized_target == "bond_protect":
        return (
            "bond_commitment_basin",
            {
                "value": 0.03,
                "approach": 0.03,
                "avoid": 0.0,
                "protect": 0.05,
                "width_pull": -0.02,
                "spread_pull": 0.04,
                "smoothing": -0.01,
            },
        )
    if normalized_target == "step_forward":
        return (
            "forward_commitment_slope",
            {
                "value": 0.03,
                "approach": 0.05,
                "avoid": -0.02,
                "protect": -0.01,
                "width_pull": 0.04,
                "spread_pull": 0.06,
                "smoothing": 0.04,
            },
        )
    if normalized_target == "stabilize":
        return (
            "stabilize_commitment_basin",
            {
                "value": 0.02,
                "approach": 0.0,
                "avoid": 0.01,
                "protect": 0.04,
                "width_pull": -0.02,
                "spread_pull": 0.0,
                "smoothing": -0.03,
            },
        )
    return (
        "hold_commitment_basin",
        {
            "value": 0.0,
            "approach": -0.01,
            "avoid": 0.02,
            "protect": 0.03,
            "width_pull": -0.03,
            "spread_pull": -0.02,
            "smoothing": -0.02,
        },
    )


def _driver_strength(*values: float) -> float:
    return float(sum(abs(float(value)) for value in values))


def _derive_winner_margin(values: Any) -> float:
    ordered = sorted((_clamp01(value) for value in values), reverse=True)
    if not ordered:
        return 0.0
    top = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else 0.0
    return _clamp01(top - runner_up)


def _derive_terrain_dominant_inputs(
    *,
    safety_gain: float,
    strain_load: float,
    bond_weight: float,
    unresolved_tension: float,
    terrain_reweighting_bias: float,
    memory_class_focus: str,
    commitment_scale: float,
    commitment_target_focus: str,
    insight_scale: float,
    insight_shape_reason: str,
) -> tuple[str, ...]:
    inputs: list[tuple[str, float]] = [
        ("safety_gain", _clamp01(safety_gain)),
        ("strain_load", _clamp01(strain_load)),
        ("bond_weight", _clamp01(bond_weight)),
        ("unresolved_tension", _clamp01(unresolved_tension)),
    ]
    supplemental_labels: list[str] = []
    if terrain_reweighting_bias > 0.0:
        memory_label = f"memory_class:{str(memory_class_focus or '').strip() or 'episodic'}"
        inputs.append((memory_label, _clamp01(terrain_reweighting_bias)))
        supplemental_labels.append(memory_label)
    if commitment_scale > 0.0 and str(commitment_target_focus or "").strip():
        commitment_label = f"overnight_commitment:{str(commitment_target_focus).strip()}"
        inputs.append((commitment_label, _clamp01(commitment_scale)))
        supplemental_labels.append(commitment_label)
    if insight_scale > 0.0 and str(insight_shape_reason or "").strip():
        insight_label = f"insight:{str(insight_shape_reason).strip()}"
        inputs.append((insight_label, _clamp01(insight_scale)))
        supplemental_labels.append(insight_label)
    ordered = sorted(inputs, key=lambda item: item[1], reverse=True)
    dominant = [label for label, score in ordered if score > 0.0][:3]
    for label in supplemental_labels:
        if label not in dominant:
            dominant.append(label)
    return tuple(dominant)
