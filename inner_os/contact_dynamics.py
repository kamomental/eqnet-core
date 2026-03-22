from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class ContactDynamicPoint:
    point_id: str
    label: str
    source_modality: str
    base_intensity: float
    stabilized_activation: float
    reentry_gain: float
    decay_factor: float
    temporal_kernel_response: float
    binding_tags: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactDynamicsState:
    dynamics_mode: str = "fresh"
    carryover_strength: float = 0.0
    reentry_bias: float = 0.0
    protective_hold: float = 0.0
    stabilized_points: tuple[ContactDynamicPoint, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["stabilized_points"] = [point.to_dict() for point in self.stabilized_points]
        return payload


def advance_contact_dynamics(
    *,
    contact_field: Mapping[str, Any] | None,
    previous_dynamics: Mapping[str, Any] | None = None,
    previous_workspace: Mapping[str, Any] | None = None,
    previous_residue: float = 0.0,
) -> ContactDynamicsState:
    field = dict(contact_field or {})
    previous_dynamic_map = {
        str(point.get("label") or "").strip(): _clamp01(float(point.get("stabilized_activation", 0.0) or 0.0))
        for point in (previous_dynamics or {}).get("stabilized_points") or []
        if isinstance(point, Mapping) and str(point.get("label") or "").strip()
    }
    previous_workspace_labels = {
        str(slot.get("label") or "").strip()
        for slot in (previous_workspace or {}).get("active_slots") or []
        if isinstance(slot, Mapping) and str(slot.get("label") or "").strip()
    }
    raw_points = [dict(point) for point in field.get("points") or [] if isinstance(point, Mapping)]
    stabilized_points: list[ContactDynamicPoint] = []

    overlap_hits = 0
    for point in raw_points:
        label = str(point.get("label") or "").strip()
        if not label:
            continue
        source = str(point.get("source_modality") or point.get("source") or "ambient").strip()
        base_intensity = _clamp01(float(point.get("intensity", 0.0) or 0.0))
        temporal_kernel_response = _clamp01(float(point.get("temporal_kernel_response", 0.0) or 0.0))
        previous_activation = _clamp01(previous_dynamic_map.get(label, 0.0))
        reentry_gain = 0.0
        if label in previous_dynamic_map:
            reentry_gain += 0.18 + previous_activation * 0.34
            overlap_hits += 1
        if label in previous_workspace_labels:
            reentry_gain += 0.14 + _clamp01(previous_residue) * 0.18
            overlap_hits += 1
        reentry_gain = _clamp01(reentry_gain)

        defensive_salience = _clamp01(float(point.get("defensive_salience", 0.0) or 0.0))
        local_curvature = _clamp01(float(point.get("local_curvature", 0.0) or 0.0))
        decay_factor = _clamp01(0.12 + defensive_salience * 0.24 + local_curvature * 0.12)
        stabilized_activation = _clamp01(
            base_intensity * 0.62
            + previous_activation * 0.18
            + temporal_kernel_response * 0.12
            + _clamp01(previous_residue) * 0.08
            + reentry_gain * 0.14
            - decay_factor * 0.06
        )

        cues: list[str] = [f"dynamic:{source}"]
        if reentry_gain >= 0.3:
            cues.append("dynamic_reentry")
        if defensive_salience >= 0.44:
            cues.append("dynamic_protective")

        stabilized_points.append(
            ContactDynamicPoint(
                point_id=str(point.get("point_id") or label).strip(),
                label=label,
                source_modality=source,
                base_intensity=round(base_intensity, 4),
                stabilized_activation=round(stabilized_activation, 4),
                reentry_gain=round(reentry_gain, 4),
                decay_factor=round(decay_factor, 4),
                temporal_kernel_response=round(temporal_kernel_response, 4),
                binding_tags=tuple(str(item) for item in point.get("binding_tags") or [] if str(item).strip()),
                cues=tuple(cues),
            )
        )

    stabilized_points.sort(key=lambda item: item.stabilized_activation, reverse=True)
    carryover_strength = _clamp01(
        (_clamp01(previous_residue) * 0.34)
        + (min(overlap_hits, 4) / 4.0) * 0.42
        + max((point.reentry_gain for point in stabilized_points), default=0.0) * 0.24
    )
    protective_hold = _clamp01(
        max(
            float(field.get("protective_pressure", 0.0) or 0.0),
            max((point.reentry_gain for point in stabilized_points if point.source_modality == "protective"), default=0.0),
        )
    )
    reentry_bias = _clamp01(
        max((point.reentry_gain for point in stabilized_points), default=0.0) * 0.6
        + carryover_strength * 0.24
        + _clamp01(previous_residue) * 0.16
    )

    dynamics_mode = "fresh"
    if protective_hold >= 0.62 and reentry_bias >= 0.34:
        dynamics_mode = "guarded_reentry"
    elif reentry_bias >= 0.34:
        dynamics_mode = "reentrant"
    elif protective_hold >= 0.62:
        dynamics_mode = "guarded_fresh"

    cues: list[str] = [f"contact_dynamics:{dynamics_mode}"]
    if carryover_strength >= 0.34:
        cues.append("contact_carryover")
    if reentry_bias >= 0.24:
        cues.append("contact_reentry_bias")
    if protective_hold >= 0.56:
        cues.append("contact_protective_hold")

    return ContactDynamicsState(
        dynamics_mode=dynamics_mode,
        carryover_strength=round(carryover_strength, 4),
        reentry_bias=round(reentry_bias, 4),
        protective_hold=round(protective_hold, 4),
        stabilized_points=tuple(stabilized_points[:5]),
        cues=tuple(cues),
    )


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
