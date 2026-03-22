from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from .affective_terrain import TerrainReadout


@dataclass(frozen=True)
class ProtectionModeState:
    mode: str
    strength: float
    reasons: tuple[str, ...]
    scores: Dict[str, float] = field(default_factory=dict)
    winner_margin: float = 0.0
    dominant_inputs: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "strength": float(self.strength),
            "reasons": list(self.reasons),
            "scores": {str(key): float(value) for key, value in (self.scores or {}).items()},
            "winner_margin": float(self.winner_margin),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_protection_mode(
    *,
    terrain_readout: TerrainReadout | Mapping[str, Any] | None,
    affective_position: Mapping[str, Any] | None = None,
    self_state: Mapping[str, Any] | None = None,
    temperament_estimate: Mapping[str, Any] | None = None,
    workspace: Mapping[str, Any] | None = None,
    qualia_planner_view: Mapping[str, Any] | None = None,
    insight_reframing_bias: float = 0.0,
    insight_class_focus: str = "",
) -> ProtectionModeState:
    terrain = _coerce_terrain(terrain_readout)
    position = dict(affective_position or {})
    self_payload = dict(self_state or {})
    temperament = dict(temperament_estimate or {})
    workspace_payload = dict(workspace or {})
    qualia = dict(qualia_planner_view or {})

    workspace_mode = str(workspace_payload.get("workspace_mode") or "").strip()
    workspace_stability = _float01(workspace_payload.get("workspace_stability"))
    position_confidence = _float01(position.get("confidence"))
    source_weights = dict(position.get("source_weights") or {})
    position_memory_weight = _float01(source_weights.get("memory"))
    position_carryover_weight = _float01(source_weights.get("carryover"))
    self_stress = _float01(self_payload.get("stress"))
    self_recovery_need = _float01(self_payload.get("recovery_need"))
    self_continuity = _float01(self_payload.get("continuity_score"))
    self_recent_strain = _float01(self_payload.get("recent_strain"))
    commitment_carry_bias = _float01(self_payload.get("commitment_carry_bias"))
    commitment_mode_focus = str(self_payload.get("commitment_mode_focus") or "").strip()
    qualia_trust = _float01(qualia.get("trust"))
    qualia_degraded = bool(qualia.get("degraded", False))
    qualia_body_load = _float01(qualia.get("body_load"))
    qualia_protect = _float01(qualia.get("protection_bias"))
    risk_tolerance = _float01(temperament.get("risk_tolerance"))
    ambiguity_tolerance = _float01(temperament.get("ambiguity_tolerance"))
    curiosity_drive = _float01(temperament.get("curiosity_drive"))
    bond_drive = _float01(temperament.get("bond_drive"))
    recovery_discipline = _float01(temperament.get("recovery_discipline"))
    protect_floor = _float01(temperament.get("protect_floor"))
    insight_reframing_bias = _float01(insight_reframing_bias)
    insight_class_focus = str(insight_class_focus or "").strip()
    repair_prior, contain_prior = _insight_priors(
        insight_reframing_bias=insight_reframing_bias,
        insight_class_focus=insight_class_focus,
    )
    commitment_monitor_prior, commitment_contain_prior, commitment_stabilize_prior, commitment_repair_prior = _commitment_priors(
        commitment_carry_bias=commitment_carry_bias,
        commitment_mode_focus=commitment_mode_focus,
    )

    protect_pressure = _clamp01(
        0.55 * terrain.protect_bias
        + 0.2 * qualia_protect
        + 0.15 * qualia_body_load
        + 0.1 * (1.0 - qualia_trust)
        + 0.1 * self_stress
        + 0.06 * self_recent_strain
        + 0.06 * position_memory_weight
        + 0.04 * position_carryover_weight
        + contain_prior
        + commitment_contain_prior
        + 0.06 * protect_floor
        + 0.04 * recovery_discipline * max(self_recovery_need, self_stress)
        - 0.05 * risk_tolerance
    )
    stabilize_pressure = _clamp01(
        0.45 * protect_pressure
        + 0.25 * qualia_body_load
        + 0.2 * (1.0 - workspace_stability)
        + 0.1 * (1.0 if qualia_degraded else 0.0)
        + 0.16 * self_recovery_need
        + 0.08 * (1.0 - self_continuity)
        + 0.06 * (1.0 - position_confidence)
        + 0.04 * recovery_discipline
        + commitment_stabilize_prior
    )
    repair_opening = _clamp01(
        terrain.approach_bias
        + repair_prior
        + commitment_repair_prior
        + 0.06 * curiosity_drive
        + 0.05 * bond_drive
        + 0.04 * ambiguity_tolerance
        + 0.04 * risk_tolerance
        - 0.18 * terrain.protect_bias
        - 0.08 * qualia_body_load
        - 0.08 * (1.0 if qualia_degraded else 0.0)
        - 0.06 * self_stress
        - 0.05 * protect_floor
        - 0.05 * recovery_discipline * self_recovery_need
    )

    reasons: list[str] = []
    if terrain.protect_bias >= 0.45:
        reasons.append("terrain_protect_bias")
    if qualia_protect >= 0.12:
        reasons.append("qualia_protection_bias")
    if qualia_body_load >= 0.12:
        reasons.append("body_load")
    if workspace_mode == "guarded_foreground":
        reasons.append("guarded_workspace")
    if qualia_degraded:
        reasons.append("degraded_estimate")
    if self_stress >= 0.42:
        reasons.append("self_stress")
    if self_recovery_need >= 0.42:
        reasons.append("recovery_need")
    if position_memory_weight >= 0.24 or position_carryover_weight >= 0.24:
        reasons.append("carryover_pull")
    if position_confidence <= 0.38:
        reasons.append("low_position_confidence")
    if insight_reframing_bias >= 0.08:
        reasons.append(_insight_reason(insight_class_focus))
    if commitment_carry_bias >= 0.08 and commitment_mode_focus:
        reasons.append(_commitment_reason(commitment_mode_focus))
    if risk_tolerance >= 0.68 or curiosity_drive >= 0.64:
        reasons.append("temperament_forward_lean")
    if recovery_discipline >= 0.6 or protect_floor >= 0.6:
        reasons.append("temperament_guard_floor")

    severe_guard = (
        protect_pressure >= 0.82
        or (qualia_degraded and qualia_body_load >= 0.16)
        or (self_stress >= 0.72 and self_recovery_need >= 0.6)
    )
    contain_trigger = (
        protect_pressure >= 0.4
        or workspace_mode == "guarded_foreground"
        or position_memory_weight >= 0.34
        or self_stress >= 0.58
    )
    repair_window = (
        (terrain.approach_bias >= 0.42 or repair_opening >= 0.48)
        and terrain.avoid_bias <= 0.32
        and not qualia_degraded
        and self_stress <= 0.58
        and self_recovery_need <= 0.56
        and repair_opening >= 0.48
    )

    mode_scores = _mode_scores(
        protect_pressure=protect_pressure,
        stabilize_pressure=stabilize_pressure,
        repair_opening=repair_opening,
        workspace_stability=workspace_stability,
        workspace_mode=workspace_mode,
        position_confidence=position_confidence,
        position_memory_weight=position_memory_weight,
        self_stress=self_stress,
        self_recovery_need=self_recovery_need,
        self_continuity=self_continuity,
        qualia_trust=qualia_trust,
        qualia_degraded=qualia_degraded,
        qualia_body_load=qualia_body_load,
        terrain_approach_bias=terrain.approach_bias,
        terrain_avoid_bias=terrain.avoid_bias,
        contain_prior=contain_prior,
        repair_prior=repair_prior,
        commitment_monitor_prior=commitment_monitor_prior,
        commitment_contain_prior=commitment_contain_prior,
        commitment_stabilize_prior=commitment_stabilize_prior,
        commitment_repair_prior=commitment_repair_prior,
        risk_tolerance=risk_tolerance,
        ambiguity_tolerance=ambiguity_tolerance,
        curiosity_drive=curiosity_drive,
        bond_drive=bond_drive,
        recovery_discipline=recovery_discipline,
        protect_floor=protect_floor,
        severe_guard=severe_guard,
        contain_trigger=contain_trigger,
        repair_window=repair_window,
    )
    ranked_modes = sorted(mode_scores.items(), key=lambda item: item[1], reverse=True)
    mode, strength = ranked_modes[0]
    runner_up = ranked_modes[1][1] if len(ranked_modes) > 1 else 0.0
    winner_margin = _clamp01(strength - runner_up)
    dominant_inputs = _dominant_inputs_for_mode(
        mode=mode,
        protect_pressure=protect_pressure,
        stabilize_pressure=stabilize_pressure,
        repair_opening=repair_opening,
        workspace_stability=workspace_stability,
        workspace_mode=workspace_mode,
        position_confidence=position_confidence,
        position_memory_weight=position_memory_weight,
        position_carryover_weight=position_carryover_weight,
        self_stress=self_stress,
        self_recovery_need=self_recovery_need,
        self_continuity=self_continuity,
        qualia_trust=qualia_trust,
        qualia_degraded=qualia_degraded,
        qualia_body_load=qualia_body_load,
        qualia_protect=qualia_protect,
        terrain_approach_bias=terrain.approach_bias,
        terrain_avoid_bias=terrain.avoid_bias,
        terrain_protect_bias=terrain.protect_bias,
        contain_prior=contain_prior,
        repair_prior=repair_prior,
        commitment_monitor_prior=commitment_monitor_prior,
        commitment_contain_prior=commitment_contain_prior,
        commitment_stabilize_prior=commitment_stabilize_prior,
        commitment_repair_prior=commitment_repair_prior,
        risk_tolerance=risk_tolerance,
        curiosity_drive=curiosity_drive,
        bond_drive=bond_drive,
        recovery_discipline=recovery_discipline,
        protect_floor=protect_floor,
    )

    if mode == "repair":
        reasons.append("approach_opening")

    if not reasons:
        reasons.append("neutral_monitoring")

    return ProtectionModeState(
        mode=mode,
        strength=_clamp01(strength),
        reasons=tuple(reasons),
        scores={key: _clamp01(value) for key, value in mode_scores.items()},
        winner_margin=winner_margin,
        dominant_inputs=dominant_inputs,
    )


def _coerce_terrain(terrain_readout: TerrainReadout | Mapping[str, Any] | None) -> TerrainReadout:
    if isinstance(terrain_readout, TerrainReadout):
        return terrain_readout
    payload = dict(terrain_readout or {})
    return TerrainReadout(
        value=float(payload.get("value", 0.0) or 0.0),
        grad=_as_vector(payload.get("grad")),
        curvature=_as_vector(payload.get("curvature")),
        approach_bias=_float01(payload.get("approach_bias")),
        avoid_bias=_float01(payload.get("avoid_bias")),
        protect_bias=_float01(payload.get("protect_bias")),
        active_patch_index=int(payload.get("active_patch_index") or 0),
        active_patch_label=str(payload.get("active_patch_label") or ""),
    )


def _as_vector(value: Any) -> Any:
    import numpy as np

    vector = np.asarray(value or [], dtype=np.float32).reshape(-1)
    return vector.astype(np.float32)


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _insight_priors(
    *,
    insight_reframing_bias: float,
    insight_class_focus: str,
) -> tuple[float, float]:
    focus = str(insight_class_focus or "").strip()
    if focus == "reframed_relation":
        return (
            _clamp01(insight_reframing_bias * 0.18),
            _clamp01(insight_reframing_bias * 0.04),
        )
    if focus == "new_link_hypothesis":
        return (
            _clamp01(insight_reframing_bias * 0.04),
            _clamp01(insight_reframing_bias * 0.12),
        )
    if focus == "insight_trace":
        return (
            _clamp01(insight_reframing_bias * 0.06),
            _clamp01(insight_reframing_bias * 0.06),
        )
    return (0.0, 0.0)


def _insight_reason(insight_class_focus: str) -> str:
    focus = str(insight_class_focus or "").strip()
    if focus == "reframed_relation":
        return "insight_reframing_prior"
    if focus == "new_link_hypothesis":
        return "insight_link_prior"
    if focus == "insight_trace":
        return "insight_trace_prior"
    return "insight_prior"


def _commitment_priors(
    *,
    commitment_carry_bias: float,
    commitment_mode_focus: str,
) -> tuple[float, float, float, float]:
    bias = _clamp01(commitment_carry_bias)
    focus = str(commitment_mode_focus or "").strip()
    if focus == "repair":
        return (0.0, _clamp01(bias * 0.02), 0.0, _clamp01(bias * 0.1))
    if focus == "contain":
        return (0.0, _clamp01(bias * 0.08), _clamp01(bias * 0.02), 0.0)
    if focus == "stabilize":
        return (0.0, _clamp01(bias * 0.03), _clamp01(bias * 0.08), 0.0)
    if focus == "monitor":
        return (_clamp01(bias * 0.08), 0.0, 0.0, _clamp01(bias * 0.02))
    return (0.0, 0.0, 0.0, 0.0)


def _commitment_reason(commitment_mode_focus: str) -> str:
    focus = str(commitment_mode_focus or "").strip()
    if focus == "repair":
        return "overnight_commitment_repair_prior"
    if focus == "contain":
        return "overnight_commitment_hold_prior"
    if focus == "stabilize":
        return "overnight_commitment_stabilize_prior"
    if focus == "monitor":
        return "overnight_commitment_forward_prior"
    return "overnight_commitment_prior"


def _mode_scores(
    *,
    protect_pressure: float,
    stabilize_pressure: float,
    repair_opening: float,
    workspace_stability: float,
    workspace_mode: str,
    position_confidence: float,
    position_memory_weight: float,
    self_stress: float,
    self_recovery_need: float,
    self_continuity: float,
    qualia_trust: float,
    qualia_degraded: bool,
    qualia_body_load: float,
    terrain_approach_bias: float,
    terrain_avoid_bias: float,
    contain_prior: float,
    repair_prior: float,
    commitment_monitor_prior: float,
    commitment_contain_prior: float,
    commitment_stabilize_prior: float,
    commitment_repair_prior: float,
    risk_tolerance: float,
    ambiguity_tolerance: float,
    curiosity_drive: float,
    bond_drive: float,
    recovery_discipline: float,
    protect_floor: float,
    severe_guard: bool,
    contain_trigger: bool,
    repair_window: bool,
) -> Dict[str, float]:
    degraded_value = 1.0 if qualia_degraded else 0.0
    guarded_value = 1.0 if workspace_mode == "guarded_foreground" else 0.0
    shield_score = _clamp01(
        0.42 * protect_pressure
        + 0.18 * stabilize_pressure
        + 0.08 * degraded_value
        + 0.08 * self_stress
        + 0.08 * self_recovery_need
        + (0.34 if severe_guard else 0.0)
    )
    stabilize_score = _clamp01(
        0.78 * stabilize_pressure
        + 0.08 * self_recovery_need
        + 0.06 * qualia_body_load
        + 0.04 * degraded_value
        + commitment_stabilize_prior
    )
    contain_score = _clamp01(
        0.72 * protect_pressure
        + 0.12 * terrain_avoid_bias
        + 0.08 * guarded_value
        + 0.06 * position_memory_weight
        + contain_prior
        + commitment_contain_prior
        + 0.08 * protect_floor
        + 0.06 * recovery_discipline
        + (0.08 if contain_trigger else 0.0)
        - 0.08 * terrain_approach_bias
        - 0.06 * risk_tolerance
    )
    repair_score = _clamp01(
        0.74 * repair_opening
        + 0.08 * terrain_approach_bias
        + 0.06 * workspace_stability
        + 0.04 * qualia_trust
        + 0.1 * repair_prior
        + commitment_repair_prior
        + 0.08 * curiosity_drive
        + 0.06 * bond_drive
        + 0.04 * risk_tolerance
        + (0.06 if repair_window else 0.0)
        - 0.18 * protect_pressure
        - 0.12 * self_recovery_need
        - 0.08 * degraded_value
        - 0.04 * protect_floor
    )
    monitor_score = _clamp01(
        0.28
        + 0.18 * workspace_stability
        + 0.16 * position_confidence
        + 0.14 * qualia_trust
        + 0.12 * self_continuity
        + commitment_monitor_prior
        + 0.08 * risk_tolerance
        + 0.05 * ambiguity_tolerance
        - 0.34 * protect_pressure
        - 0.12 * repair_opening
        - 0.2 * self_stress
        - 0.16 * self_recovery_need
        - 0.04 * protect_floor
    )
    return {
        "monitor": monitor_score,
        "contain": contain_score,
        "stabilize": stabilize_score,
        "repair": repair_score,
        "shield": shield_score,
    }


def _dominant_inputs_for_mode(
    *,
    mode: str,
    protect_pressure: float,
    stabilize_pressure: float,
    repair_opening: float,
    workspace_stability: float,
    workspace_mode: str,
    position_confidence: float,
    position_memory_weight: float,
    position_carryover_weight: float,
    self_stress: float,
    self_recovery_need: float,
    self_continuity: float,
    qualia_trust: float,
    qualia_degraded: bool,
    qualia_body_load: float,
    qualia_protect: float,
    terrain_approach_bias: float,
    terrain_avoid_bias: float,
    terrain_protect_bias: float,
    contain_prior: float,
    repair_prior: float,
    commitment_monitor_prior: float,
    commitment_contain_prior: float,
    commitment_stabilize_prior: float,
    commitment_repair_prior: float,
    risk_tolerance: float,
    curiosity_drive: float,
    bond_drive: float,
    recovery_discipline: float,
    protect_floor: float,
) -> tuple[str, ...]:
    guarded_value = 1.0 if workspace_mode == "guarded_foreground" else 0.0
    degraded_value = 1.0 if qualia_degraded else 0.0
    components: Dict[str, float]
    if mode == "shield":
        components = {
            "protect_pressure": 0.76 * protect_pressure,
            "stabilize_pressure": 0.18 * stabilize_pressure,
            "degraded_estimate": 0.16 * degraded_value,
            "self_stress": 0.16 * self_stress,
            "recovery_need": 0.16 * self_recovery_need,
        }
    elif mode == "stabilize":
        components = {
            "stabilize_pressure": 0.78 * stabilize_pressure,
            "recovery_need": 0.08 * self_recovery_need,
            "body_load": 0.06 * qualia_body_load,
            "degraded_estimate": 0.04 * degraded_value,
            "overnight_commitment_prior": commitment_stabilize_prior,
        }
    elif mode == "contain":
        components = {
            "protect_pressure": 0.72 * protect_pressure,
            "terrain_avoid_bias": 0.12 * terrain_avoid_bias,
            "guarded_workspace": 0.08 * guarded_value,
            "memory_pull": 0.06 * position_memory_weight,
            "carryover_pull": 0.04 * position_carryover_weight,
            "insight_contain_prior": contain_prior,
            "overnight_commitment_prior": commitment_contain_prior,
            "terrain_protect_bias": 0.1 * terrain_protect_bias,
            "qualia_protection_bias": 0.08 * qualia_protect,
            "temperament_guard_floor": 0.08 * protect_floor + 0.06 * recovery_discipline,
        }
    elif mode == "repair":
        components = {
            "repair_opening": 0.74 * repair_opening,
            "terrain_approach_bias": 0.08 * terrain_approach_bias,
            "workspace_stability": 0.06 * workspace_stability,
            "qualia_trust": 0.04 * qualia_trust,
            "insight_repair_prior": repair_prior,
            "overnight_commitment_prior": commitment_repair_prior,
            "temperament_forward_lean": 0.08 * curiosity_drive + 0.06 * bond_drive + 0.04 * risk_tolerance,
        }
    else:
        components = {
            "workspace_stability": 0.18 * workspace_stability,
            "position_confidence": 0.16 * position_confidence,
            "qualia_trust": 0.14 * qualia_trust,
            "self_continuity": 0.12 * self_continuity,
            "overnight_commitment_prior": commitment_monitor_prior,
            "temperament_risk_tolerance": 0.08 * risk_tolerance,
        }
    ranked = sorted(
        ((name, value) for name, value in components.items() if value > 0.02),
        key=lambda item: item[1],
        reverse=True,
    )
    return tuple(name for name, _ in ranked[:3])
