from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from .affect_blend import AffectBlendState
from .scene_state import SceneState


@dataclass(frozen=True)
class ConstraintField:
    body_cost: float = 0.0
    boundary_pressure: float = 0.0
    repair_pressure: float = 0.0
    shared_world_pressure: float = 0.0
    protective_bias: float = 0.0
    disclosure_limit: str = "light"
    reportability_limit: str = "open"
    option_temperature: float = 1.0
    admissible_families: tuple[str, ...] = ()
    do_not_cross: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_constraint_field(
    *,
    scene_state: SceneState,
    affect_blend: AffectBlendState,
    stress: float = 0.0,
    recovery_need: float = 0.0,
    safety_bias: float = 0.0,
    recent_strain: float = 0.0,
    current_risks: Sequence[str] = (),
) -> ConstraintField:
    body_cost = _clamp01(
        stress * 0.52
        + recovery_need * 0.24
        + scene_state.environmental_load * 0.18
        + recent_strain * 0.1
    )
    boundary_pressure = _clamp01(
        affect_blend.defense * 0.3
        + scene_state.norm_pressure * 0.24
        + safety_bias * 0.2
        + recent_strain * 0.14
        + affect_blend.reverence * 0.08
    )
    repair_pressure = _clamp01(
        (0.34 if scene_state.scene_family == "repair_window" else 0.0)
        + min(affect_blend.care, affect_blend.defense) * 0.22
        + affect_blend.residual_tension * 0.16
        + recent_strain * 0.16
        + affect_blend.distress * 0.08
    )
    shared_world_pressure = _clamp01(
        affect_blend.shared_world_pull * 0.46
        + affect_blend.future_pull * 0.22
        + max(0.0, 1.0 - boundary_pressure) * 0.08
        + max(0.0, scene_state.privacy_level - 0.4) * 0.1
    )
    protective_bias = _clamp01(max(body_cost, boundary_pressure, safety_bias, affect_blend.defense))

    reportability_limit = "open"
    if body_cost >= 0.66 or boundary_pressure >= 0.68 or protective_bias >= 0.74:
        reportability_limit = "withhold"
    elif affect_blend.conflict_level >= 0.46 or affect_blend.residual_tension >= 0.44 or scene_state.norm_pressure >= 0.58:
        reportability_limit = "narrow"

    disclosure_limit = "light"
    if reportability_limit == "withhold" or "danger" in current_risks or scene_state.privacy_level <= 0.3:
        disclosure_limit = "minimal"
    elif shared_world_pressure >= 0.5 and scene_state.privacy_level >= 0.52 and boundary_pressure < 0.54:
        disclosure_limit = "medium"

    admissible_families = [
        "attune",
        "wait",
        "repair",
        "co_move",
        "contain",
        "reflect",
        "clarify",
        "withdraw",
    ]
    if "danger" in current_risks or body_cost >= 0.7:
        admissible_families = ["wait", "repair", "contain", "withdraw", "reflect"]
    elif boundary_pressure >= 0.72 and scene_state.privacy_level < 0.45:
        admissible_families = ["wait", "repair", "contain", "reflect", "withdraw"]
    elif reportability_limit == "withhold":
        admissible_families = ["wait", "repair", "contain", "reflect", "withdraw", "attune"]

    if shared_world_pressure >= 0.56 and body_cost < 0.6 and boundary_pressure < 0.58 and "co_move" not in admissible_families:
        admissible_families.append("co_move")

    option_temperature = max(
        0.58,
        min(
            1.28,
            0.82 + affect_blend.conflict_level * 0.22 + body_cost * 0.12 - scene_state.safety_margin * 0.1,
        ),
    )

    do_not_cross: list[str] = []
    if reportability_limit == "withhold":
        do_not_cross.append("force_reportability")
    if boundary_pressure >= 0.66:
        do_not_cross.append("flatten_boundary")
    if body_cost >= 0.64:
        do_not_cross.append("increase_load")
    if scene_state.norm_pressure >= 0.62:
        do_not_cross.append("violate_scene_norm")

    cues: list[str] = []
    if reportability_limit == "withhold":
        cues.append("constraint_withhold")
    if disclosure_limit == "minimal":
        cues.append("constraint_minimal_disclosure")
    if shared_world_pressure >= 0.5:
        cues.append("constraint_shared_world_ready")
    if repair_pressure >= 0.48:
        cues.append("constraint_repair_ready")

    return ConstraintField(
        body_cost=round(body_cost, 4),
        boundary_pressure=round(boundary_pressure, 4),
        repair_pressure=round(repair_pressure, 4),
        shared_world_pressure=round(shared_world_pressure, 4),
        protective_bias=round(protective_bias, 4),
        disclosure_limit=disclosure_limit,
        reportability_limit=reportability_limit,
        option_temperature=round(option_temperature, 4),
        admissible_families=tuple(dict.fromkeys(admissible_families)),
        do_not_cross=tuple(do_not_cross),
        cues=tuple(cues),
    )


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
