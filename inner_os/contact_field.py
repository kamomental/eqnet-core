from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class ContactPoint:
    point_id: str
    label: str
    source_modality: str
    intensity: float
    temporal_kernel_response: float
    local_terrain_gradient: float
    local_curvature: float
    relation_tag: str = ""
    scene_tag: str = ""
    uncertainty: float = 0.0
    ambiguity: float = 0.0
    defensive_salience: float = 0.0
    binding_tags: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactField:
    field_mode: str = "ambient"
    dominant_point: str = ""
    reportability_pressure: float = 0.0
    protective_pressure: float = 0.0
    points: tuple[ContactPoint, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["points"] = [point.to_dict() for point in self.points]
        return payload


def derive_contact_field(
    *,
    affect_blend_state: Mapping[str, Any] | None,
    constraint_field: Mapping[str, Any] | None,
    scene_state: Mapping[str, Any] | None,
    current_focus: str,
    reportable_facts: Sequence[str],
    current_risks: Sequence[str],
    related_person_ids: Sequence[str],
    memory_anchor: str = "",
    previous_residue: float = 0.0,
) -> ContactField:
    affect_blend = dict(affect_blend_state or {})
    constraint = dict(constraint_field or {})
    scene = dict(scene_state or {})
    scene_family = str(scene.get("scene_family") or "").strip()
    reportability_pressure = _clamp01(
        float(affect_blend.get("reportability_pressure", 0.0) or 0.0)
    )
    protective_pressure = _clamp01(
        max(
            float(affect_blend.get("defense", 0.0) or 0.0),
            float(constraint.get("protective_bias", 0.0) or 0.0),
            float(constraint.get("boundary_pressure", 0.0) or 0.0),
        )
    )
    local_gradient = _clamp01(
        max(
            float(affect_blend.get("conflict_level", 0.0) or 0.0),
            float(affect_blend.get("residual_tension", 0.0) or 0.0),
            reportability_pressure,
        )
    )
    local_curvature = _clamp01(
        max(
            float(constraint.get("body_cost", 0.0) or 0.0),
            float(constraint.get("boundary_pressure", 0.0) or 0.0),
            float(scene.get("norm_pressure", 0.0) or 0.0),
        )
    )
    temporal_kernel_response = _clamp01(
        max(
            float(affect_blend.get("future_pull", 0.0) or 0.0),
            float(affect_blend.get("shared_world_pull", 0.0) or 0.0),
            float(previous_residue or 0.0),
        )
    )
    relation_binding = tuple(
        f"person:{person_id}" for person_id in related_person_ids if str(person_id).strip()
    )
    scene_binding = tuple(
        item
        for item in (
            f"scene:{scene_family}" if scene_family else "",
            f"task:{scene.get('task_phase')}" if scene.get("task_phase") else "",
        )
        if item
    )

    points: list[ContactPoint] = []
    visible_focus = str(current_focus or "").strip() or str(memory_anchor or "").strip()
    if not visible_focus and reportable_facts:
        visible_focus = str(reportable_facts[0] or "").strip()
    if visible_focus and visible_focus != "ambient":
        points.append(
            ContactPoint(
                point_id="focus",
                label=visible_focus,
                source_modality="semantic",
                intensity=round(
                    _clamp01(0.34 + reportability_pressure * 0.34 + temporal_kernel_response * 0.12),
                    4,
                ),
                temporal_kernel_response=round(temporal_kernel_response, 4),
                local_terrain_gradient=round(local_gradient, 4),
                local_curvature=round(local_curvature, 4),
                relation_tag=relation_binding[0] if relation_binding else "",
                scene_tag=scene_family,
                uncertainty=round(max(0.0, 1.0 - reportability_pressure) * 0.46, 4),
                ambiguity=round(float(affect_blend.get("residual_tension", 0.0) or 0.0) * 0.48, 4),
                defensive_salience=round(protective_pressure * 0.42, 4),
                binding_tags=relation_binding + scene_binding,
                cues=("contact_focus",),
            )
        )

    if relation_binding:
        label = relation_binding[0]
        points.append(
            ContactPoint(
                point_id="relation",
                label=label,
                source_modality="social",
                intensity=round(
                    _clamp01(
                        0.28
                        + float(affect_blend.get("care", 0.0) or 0.0) * 0.24
                        + float(affect_blend.get("reverence", 0.0) or 0.0) * 0.16
                        + reportability_pressure * 0.12
                    ),
                    4,
                ),
                temporal_kernel_response=round(
                    _clamp01(
                        temporal_kernel_response * 0.68
                        + float(affect_blend.get("future_pull", 0.0) or 0.0) * 0.18
                    ),
                    4,
                ),
                local_terrain_gradient=round(local_gradient, 4),
                local_curvature=round(local_curvature, 4),
                relation_tag=label,
                scene_tag=scene_family,
                uncertainty=round(max(0.0, 1.0 - reportability_pressure) * 0.34, 4),
                ambiguity=round(float(affect_blend.get("conflict_level", 0.0) or 0.0) * 0.42, 4),
                defensive_salience=round(protective_pressure * 0.36, 4),
                binding_tags=relation_binding + scene_binding,
                cues=("contact_relation",),
            )
        )

    dominant_mode = str(affect_blend.get("dominant_mode") or "").strip()
    if dominant_mode:
        points.append(
            ContactPoint(
                point_id="affect",
                label=dominant_mode,
                source_modality="affective",
                intensity=round(
                    _clamp01(
                        max(
                            float(affect_blend.get("care", 0.0) or 0.0),
                            float(affect_blend.get("reverence", 0.0) or 0.0),
                            float(affect_blend.get("innocence", 0.0) or 0.0),
                            float(affect_blend.get("defense", 0.0) or 0.0),
                            float(affect_blend.get("future_pull", 0.0) or 0.0),
                            float(affect_blend.get("shared_world_pull", 0.0) or 0.0),
                            float(affect_blend.get("distress", 0.0) or 0.0),
                        )
                    ),
                    4,
                ),
                temporal_kernel_response=round(temporal_kernel_response, 4),
                local_terrain_gradient=round(local_gradient, 4),
                local_curvature=round(local_curvature, 4),
                relation_tag=relation_binding[0] if relation_binding else "",
                scene_tag=scene_family,
                uncertainty=round(max(0.0, 1.0 - reportability_pressure) * 0.42, 4),
                ambiguity=round(float(affect_blend.get("conflict_level", 0.0) or 0.0) * 0.5, 4),
                defensive_salience=round(protective_pressure * 0.52, 4),
                binding_tags=scene_binding,
                cues=(f"contact_affect:{dominant_mode}",),
            )
        )

    anchor = str(memory_anchor or "").strip()
    if anchor and anchor not in {visible_focus, "ambient"}:
        points.append(
            ContactPoint(
                point_id="memory",
                label=anchor,
                source_modality="memory",
                intensity=round(
                    _clamp01(0.22 + float(previous_residue or 0.0) * 0.28 + reportability_pressure * 0.14),
                    4,
                ),
                temporal_kernel_response=round(
                    _clamp01(temporal_kernel_response * 0.82 + float(previous_residue or 0.0) * 0.12),
                    4,
                ),
                local_terrain_gradient=round(local_gradient, 4),
                local_curvature=round(local_curvature, 4),
                relation_tag=relation_binding[0] if relation_binding else "",
                scene_tag=scene_family,
                uncertainty=round(0.32, 4),
                ambiguity=round(float(affect_blend.get("residual_tension", 0.0) or 0.0) * 0.34, 4),
                defensive_salience=round(protective_pressure * 0.24, 4),
                binding_tags=scene_binding,
                cues=("contact_memory",),
            )
        )

    if current_risks or protective_pressure >= 0.44:
        risk_label = str(current_risks[0] if current_risks else "protective_guard").strip()
        points.append(
            ContactPoint(
                point_id="protective",
                label=risk_label,
                source_modality="protective",
                intensity=round(
                    _clamp01(0.3 + protective_pressure * 0.48 + local_curvature * 0.12),
                    4,
                ),
                temporal_kernel_response=round(
                    _clamp01(temporal_kernel_response * 0.46 + protective_pressure * 0.28),
                    4,
                ),
                local_terrain_gradient=round(_clamp01(local_gradient * 0.72 + protective_pressure * 0.22), 4),
                local_curvature=round(_clamp01(local_curvature * 0.78 + protective_pressure * 0.16), 4),
                relation_tag=relation_binding[0] if relation_binding else "",
                scene_tag=scene_family,
                uncertainty=round(0.18, 4),
                ambiguity=round(float(affect_blend.get("conflict_level", 0.0) or 0.0) * 0.22, 4),
                defensive_salience=round(protective_pressure, 4),
                binding_tags=scene_binding,
                cues=("contact_protective",),
            )
        )

    points = sorted(points, key=lambda item: item.intensity, reverse=True)[:5]
    field_mode = "ambient"
    if protective_pressure >= 0.62:
        field_mode = "guarded"
    elif relation_binding:
        field_mode = "relational"
    elif visible_focus:
        field_mode = "focused"

    dominant_point = points[0].point_id if points else ""
    cues: list[str] = [f"contact_field:{field_mode}"]
    if dominant_point:
        cues.append(f"contact_dominant:{dominant_point}")
    if protective_pressure >= 0.44:
        cues.append("contact_protective")
    if relation_binding:
        cues.append("contact_relational")

    return ContactField(
        field_mode=field_mode,
        dominant_point=dominant_point,
        reportability_pressure=round(reportability_pressure, 4),
        protective_pressure=round(protective_pressure, 4),
        points=tuple(points),
        cues=tuple(cues),
    )


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
