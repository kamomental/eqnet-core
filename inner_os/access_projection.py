from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class AccessRegion:
    region_id: str
    label: str
    source: str
    activation: float
    reportable: bool
    withheld: bool
    actionable: bool
    binding_tags: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccessProjection:
    projection_mode: str = "ambient"
    dominant_region: str = ""
    reportable_slice: tuple[str, ...] = ()
    withheld_slice: tuple[str, ...] = ()
    actionable_slice: tuple[str, ...] = ()
    regions: tuple[AccessRegion, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["regions"] = [region.to_dict() for region in self.regions]
        return payload


def project_access_regions(
    *,
    contact_field: Mapping[str, Any] | None,
    contact_dynamics: Mapping[str, Any] | None = None,
    affect_blend_state: Mapping[str, Any] | None,
    constraint_field: Mapping[str, Any] | None,
    qualia_state: Mapping[str, Any] | None = None,
    terrain_readout: Mapping[str, Any] | None = None,
    insight_event: Mapping[str, Any] | None = None,
) -> AccessProjection:
    field = dict(contact_field or {})
    dynamics = dict(contact_dynamics or {})
    affect_blend = dict(affect_blend_state or {})
    constraint = dict(constraint_field or {})
    qualia = dict(qualia_state or {})
    terrain = dict(terrain_readout or {})
    insight = dict(insight_event or {})
    insight_regions = _insight_regions(insight)
    reportability_limit = str(constraint.get("reportability_limit") or "open").strip()
    protective_bias = _clamp01(float(constraint.get("protective_bias", 0.0) or 0.0))
    conflict_level = _clamp01(float(affect_blend.get("conflict_level", 0.0) or 0.0))
    residual_tension = _clamp01(float(affect_blend.get("residual_tension", 0.0) or 0.0))

    raw_points = [dict(point) for point in dynamics.get("stabilized_points") or [] if isinstance(point, Mapping)]
    if not raw_points:
        raw_points = [dict(point) for point in field.get("points") or [] if isinstance(point, Mapping)]
    scored_regions: list[AccessRegion] = []
    for index, point in enumerate(raw_points):
        label = str(point.get("label") or "").strip()
        if not label:
            continue
        source = str(point.get("source_modality") or point.get("source") or "ambient").strip()
        intensity = _clamp01(
            float(
                point.get("stabilized_activation", point.get("intensity", 0.0)) or 0.0
            )
        )
        temporal_kernel_response = _clamp01(float(point.get("temporal_kernel_response", 0.0) or 0.0))
        local_gradient = _clamp01(float(point.get("local_terrain_gradient", point.get("reentry_gain", 0.0)) or 0.0))
        defensive_salience = _clamp01(float(point.get("defensive_salience", 0.0) or 0.0))
        activation = _clamp01(
            intensity * 0.58
            + temporal_kernel_response * 0.16
            + local_gradient * 0.12
            + conflict_level * 0.08
            + residual_tension * 0.06
            + (0.08 if source in {"semantic", "social"} else 0.0)
            + (0.1 if source == "protective" else 0.0)
        )

        reportable = False
        withheld = False
        actionable = False
        if reportability_limit == "withhold":
            withheld = True
            actionable = source in {"semantic", "social", "protective", "memory"}
        elif reportability_limit == "narrow":
            reportable = source in {"semantic", "social"} and defensive_salience < 0.6 and index == 0
            withheld = not reportable
            actionable = source in {"semantic", "social", "memory", "option", "protective"}
        else:
            reportable = defensive_salience < 0.74 and source != "protective"
            withheld = not reportable
            actionable = source in {"semantic", "social", "memory", "protective"}

        if protective_bias >= 0.68 and source == "protective":
            actionable = True
            reportable = False
            withheld = True

        scored_regions.append(
            AccessRegion(
                region_id=f"{point.get('point_id') or source}:{index}",
                label=label,
                source=source,
                activation=round(activation, 4),
                reportable=reportable,
                withheld=withheld,
                actionable=actionable,
                binding_tags=tuple(str(item) for item in point.get("binding_tags") or [] if str(item).strip()),
                cues=tuple(str(item) for item in point.get("cues") or [] if str(item).strip()),
            )
        )

    scored_regions.extend(_qualia_regions(qualia))
    scored_regions.extend(_terrain_regions(terrain))
    scored_regions.extend(insight_regions)

    regions = sorted(scored_regions, key=lambda item: item.activation, reverse=True)[:4]
    reportable_slice = tuple(region.label for region in regions if region.reportable)[:2]
    withheld_slice = tuple(region.label for region in regions if region.withheld)[:3]
    actionable_slice = tuple(region.label for region in regions if region.actionable)[:2]
    if not actionable_slice and regions:
        actionable_slice = (regions[0].label,)

    projection_mode = "ambient_projection"
    if reportability_limit == "withhold":
        projection_mode = "guarded_projection"
    elif reportability_limit == "narrow":
        projection_mode = "narrow_projection"
    elif reportable_slice:
        projection_mode = "foreground_projection"

    dominant_region = regions[0].region_id if regions else ""
    cues: list[str] = [f"access_projection:{projection_mode}"]
    if reportable_slice:
        cues.append("access_reportable")
    if withheld_slice:
        cues.append("access_withheld")
    if actionable_slice:
        cues.append("access_actionable")
    if qualia:
        cues.append("access_qualia_input")
    if terrain:
        cues.append("access_terrain_input")
    if insight_regions:
        cues.append("access_insight_input")
    for item in dynamics.get("cues") or []:
        text = str(item).strip()
        if text:
            cues.append(text)

    return AccessProjection(
        projection_mode=projection_mode,
        dominant_region=dominant_region,
        reportable_slice=reportable_slice,
        withheld_slice=withheld_slice,
        actionable_slice=actionable_slice,
        regions=tuple(regions),
        cues=tuple(cues),
    )


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _qualia_regions(qualia_state: Mapping[str, Any]) -> list[AccessRegion]:
    qualia_values = _float_list(qualia_state.get("qualia"))
    gate_values = _float_list(qualia_state.get("gate"))
    body_coupling = _float_list(qualia_state.get("body_coupling"))
    value_grad = _float_list(qualia_state.get("value_grad"))
    habituation = _float_list(qualia_state.get("habituation"))
    axis_labels = [str(item).strip() for item in qualia_state.get("axis_labels") or [] if str(item).strip()]
    if not qualia_values or not gate_values:
        return []
    size = min(len(qualia_values), len(gate_values))
    trust = _clamp01(float(qualia_state.get("trust_applied", 1.0) or 1.0))
    degraded = bool(qualia_state.get("degraded", False))
    body_norm = _normalize(body_coupling, size)
    value_norm = _normalize(value_grad, size)
    habituation_norm = _normalize(habituation, size)
    qualia_strength = []
    for index in range(size):
        strength = abs(qualia_values[index]) * max(gate_values[index], 0.0) * max(trust, 0.12)
        qualia_strength.append((index, strength))
    qualia_strength.sort(key=lambda item: item[1], reverse=True)

    regions: list[AccessRegion] = []
    for rank, (index, strength) in enumerate(qualia_strength[:3]):
        if strength <= 1.0e-4:
            continue
        label = axis_labels[index] if index < len(axis_labels) else f"felt_axis_{index}"
        activation = _clamp01(
            strength * 0.52
            + _clamp01(gate_values[index]) * 0.24
            + body_norm[index] * 0.12
            + value_norm[index] * 0.12
            - habituation_norm[index] * 0.08
        )
        reportable = (
            not degraded
            and trust >= 0.42
            and _clamp01(gate_values[index]) >= 0.3
            and habituation_norm[index] < 0.84
        )
        withheld = not reportable
        actionable = activation >= 0.16
        cues = ["access_felt_candidate"]
        if degraded:
            cues.append("access_felt_degraded")
        if body_norm[index] >= 0.24:
            cues.append("access_felt_body_linked")
        if value_norm[index] >= 0.24:
            cues.append("access_felt_value_linked")
        regions.append(
            AccessRegion(
                region_id=f"felt:{index}:{rank}",
                label=label,
                source="felt",
                activation=round(activation, 4),
                reportable=reportable,
                withheld=withheld,
                actionable=actionable,
                binding_tags=("qualia",),
                cues=tuple(cues),
            )
        )
    return regions


def _terrain_regions(terrain_readout: Mapping[str, Any]) -> list[AccessRegion]:
    if not terrain_readout:
        return []
    label = str(terrain_readout.get("active_patch_label") or "").strip() or "terrain_patch"
    approach_bias = _clamp01(float(terrain_readout.get("approach_bias", 0.0) or 0.0))
    avoid_bias = _clamp01(float(terrain_readout.get("avoid_bias", 0.0) or 0.0))
    protect_bias = _clamp01(float(terrain_readout.get("protect_bias", 0.0) or 0.0))
    terrain_value = _clamp01(abs(float(terrain_readout.get("value", 0.0) or 0.0)))
    activation = _clamp01(
        terrain_value * 0.34
        + protect_bias * 0.34
        + max(approach_bias, avoid_bias) * 0.22
        + 0.1
    )
    if activation <= 1.0e-4:
        return []

    source = "protective" if protect_bias >= max(approach_bias, avoid_bias) else "terrain"
    reportable = approach_bias > avoid_bias and protect_bias < 0.45
    withheld = not reportable
    actionable = protect_bias >= 0.12 or max(approach_bias, avoid_bias) >= 0.18
    cues = ["access_terrain_candidate"]
    if protect_bias >= 0.18:
        cues.append("access_terrain_protective")
    if approach_bias > avoid_bias:
        cues.append("access_terrain_approach")
    elif avoid_bias > approach_bias:
        cues.append("access_terrain_avoid")

    return [
        AccessRegion(
            region_id=f"terrain:{terrain_readout.get('active_patch_index', 0)}",
            label=label,
            source=source,
            activation=round(activation, 4),
            reportable=reportable,
            withheld=withheld,
            actionable=actionable,
            binding_tags=("affective_terrain",),
            cues=tuple(cues),
        )
    ]


def _insight_regions(insight_event: Mapping[str, Any]) -> list[AccessRegion]:
    if not insight_event:
        return []
    score = dict(insight_event.get("score") or {})
    triggered = bool(insight_event.get("triggered", False))
    total = _clamp01(float(score.get("total", 0.0) or 0.0))
    if not triggered and total < 0.24:
        return []
    activation = _clamp01(
        total * 0.68
        + float(insight_event.get("orient_bias", 0.0) or 0.0) * 0.22
        + 0.1
    )
    if activation <= 1.0e-4:
        return []
    label = str(
        insight_event.get("dominant_seed_label")
        or insight_event.get("summary")
        or "new_connection"
    ).strip() or "new_connection"
    return [
        AccessRegion(
            region_id=f"insight:{str(insight_event.get('link_key') or 'candidate')}",
            label=label,
            source="insight",
            activation=round(activation, 4),
            reportable=triggered,
            withheld=not triggered,
            actionable=activation >= 0.18,
            binding_tags=("association", "insight"),
            cues=tuple(
                item
                for item in (
                    "access_insight_candidate",
                    "access_insight_triggered" if triggered else "",
                )
                if item
            ),
        )
    ]


def _float_list(values: Any) -> list[float]:
    if values is None:
        return []
    result: list[float] = []
    for item in values:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def _normalize(values: list[float], size: int) -> list[float]:
    if size <= 0:
        return []
    clipped = [max(0.0, float(item)) for item in values[:size]]
    if len(clipped) < size:
        clipped.extend([0.0] * (size - len(clipped)))
    scale = max(clipped) if clipped else 0.0
    if scale <= 1.0e-6:
        return [0.0] * size
    return [item / scale for item in clipped]
