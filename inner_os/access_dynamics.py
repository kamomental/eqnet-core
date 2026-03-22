from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class AccessDynamicRegion:
    region_id: str
    label: str
    source: str
    base_activation: float
    stabilized_activation: float
    inertia_gain: float
    gating_hysteresis: float
    reportable: bool
    withheld: bool
    actionable: bool
    binding_tags: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccessDynamicsState:
    dynamics_mode: str = "fresh_projection"
    membrane_inertia: float = 0.0
    gating_hysteresis: float = 0.0
    protective_filter: float = 0.0
    stabilized_regions: tuple[AccessDynamicRegion, ...] = ()
    reportable_slice: tuple[str, ...] = ()
    withheld_slice: tuple[str, ...] = ()
    actionable_slice: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["stabilized_regions"] = [region.to_dict() for region in self.stabilized_regions]
        return payload


def advance_access_dynamics(
    *,
    access_projection: Mapping[str, Any] | None,
    previous_access_dynamics: Mapping[str, Any] | None = None,
    previous_workspace: Mapping[str, Any] | None = None,
    previous_residue: float = 0.0,
    current_risks: Sequence[str] = (),
) -> AccessDynamicsState:
    projection = dict(access_projection or {})
    previous_dynamics = dict(previous_access_dynamics or {})
    previous_workspace = dict(previous_workspace or {})
    previous_region_map = {
        str(region.get("label") or "").strip(): dict(region)
        for region in previous_dynamics.get("stabilized_regions") or []
        if isinstance(region, Mapping) and str(region.get("label") or "").strip()
    }
    previous_reportable = {
        str(label).strip() for label in previous_dynamics.get("reportable_slice") or [] if str(label).strip()
    }
    previous_withheld = {
        str(label).strip() for label in previous_dynamics.get("withheld_slice") or [] if str(label).strip()
    }
    previous_actionable = {
        str(label).strip() for label in previous_dynamics.get("actionable_slice") or [] if str(label).strip()
    }
    previous_workspace_labels = {
        str(slot.get("label") or "").strip()
        for slot in previous_workspace.get("active_slots") or []
        if isinstance(slot, Mapping) and str(slot.get("label") or "").strip()
    }
    raw_regions = [dict(region) for region in projection.get("regions") or [] if isinstance(region, Mapping)]
    projection_mode = str(projection.get("projection_mode") or "ambient_projection").strip()
    risk_pressure = 0.74 if any(str(risk).strip() == "danger" for risk in current_risks) else 0.0
    if projection_mode == "guarded_projection":
        risk_pressure = max(risk_pressure, 0.62)

    stabilized_regions: list[AccessDynamicRegion] = []
    overlap_hits = 0
    max_inertia_gain = 0.0
    max_hysteresis = 0.0
    max_protective_seed = risk_pressure

    for index, region in enumerate(raw_regions):
        label = str(region.get("label") or "").strip()
        if not label:
            continue
        source = str(region.get("source") or "access").strip()
        base_activation = _clamp01(float(region.get("activation", 0.0) or 0.0))
        previous_region = previous_region_map.get(label, {})
        previous_activation = _clamp01(
            float(previous_region.get("stabilized_activation", previous_region.get("activation", 0.0)) or 0.0)
        )
        inertia_gain = 0.0
        if label in previous_region_map:
            inertia_gain += 0.16 + previous_activation * 0.34
            overlap_hits += 1
        if label in previous_workspace_labels:
            inertia_gain += 0.12 + _clamp01(previous_residue) * 0.18
            overlap_hits += 1
        inertia_gain = _clamp01(inertia_gain)

        hysteresis = 0.0
        if bool(region.get("reportable", False)) or label in previous_reportable:
            hysteresis += 0.18
        if bool(region.get("withheld", False)) or label in previous_withheld:
            hysteresis += 0.16
        if bool(region.get("actionable", False)) or label in previous_actionable:
            hysteresis += 0.12
        hysteresis = _clamp01(hysteresis + previous_activation * 0.14)

        protective_seed = risk_pressure
        if source == "protective":
            protective_seed = max(protective_seed, base_activation)
        max_protective_seed = max(max_protective_seed, protective_seed)
        stabilized_activation = _clamp01(
            base_activation * 0.64
            + previous_activation * 0.18
            + inertia_gain * 0.12
            + hysteresis * 0.1
            - protective_seed * (0.05 if source != "protective" else 0.02)
        )

        reportable = bool(region.get("reportable", False))
        withheld = bool(region.get("withheld", False))
        actionable = bool(region.get("actionable", False))

        if label in previous_reportable and not withheld and stabilized_activation >= 0.34 and protective_seed < 0.68:
            reportable = True
        if label in previous_withheld and not reportable and stabilized_activation >= 0.26:
            withheld = True
        if label in previous_actionable and stabilized_activation >= 0.24:
            actionable = True
        if projection_mode == "guarded_projection" and source == "protective":
            reportable = False
            withheld = True
            actionable = True
        if risk_pressure >= 0.62 and source == "affective" and reportable:
            reportable = False
            withheld = True

        cues: list[str] = [f"access_dynamic:{source}"]
        if inertia_gain >= 0.3:
            cues.append("access_dynamic_inertia")
        if hysteresis >= 0.28:
            cues.append("access_dynamic_hysteresis")
        if protective_seed >= 0.56:
            cues.append("access_dynamic_guarded")

        max_inertia_gain = max(max_inertia_gain, inertia_gain)
        max_hysteresis = max(max_hysteresis, hysteresis)
        stabilized_regions.append(
            AccessDynamicRegion(
                region_id=str(region.get("region_id") or f"{source}:{index}").strip(),
                label=label,
                source=source,
                base_activation=round(base_activation, 4),
                stabilized_activation=round(stabilized_activation, 4),
                inertia_gain=round(inertia_gain, 4),
                gating_hysteresis=round(hysteresis, 4),
                reportable=reportable,
                withheld=withheld,
                actionable=actionable,
                binding_tags=tuple(str(item) for item in region.get("binding_tags") or [] if str(item).strip()),
                cues=tuple(cues),
            )
        )

    stabilized_regions.sort(key=lambda item: item.stabilized_activation, reverse=True)
    membrane_inertia = _clamp01(
        _clamp01(previous_residue) * 0.24
        + (min(overlap_hits, 4) / 4.0) * 0.42
        + max_inertia_gain * 0.34
    )
    gating_hysteresis = _clamp01(
        max_hysteresis * 0.54
        + membrane_inertia * 0.24
        + (0.12 if previous_reportable or previous_withheld else 0.0)
        + _clamp01(previous_residue) * 0.1
    )
    protective_filter = _clamp01(
        max(
            max_protective_seed,
            max((region.stabilized_activation for region in stabilized_regions if region.source == "protective"), default=0.0),
        )
    )

    dynamics_mode = "fresh_projection"
    if protective_filter >= 0.62 and membrane_inertia >= 0.34:
        dynamics_mode = "guarded_inertial_projection"
    elif protective_filter >= 0.62:
        dynamics_mode = "guarded_projection"
    elif membrane_inertia >= 0.34:
        dynamics_mode = "inertial_projection"

    reportable_slice = tuple(region.label for region in stabilized_regions if region.reportable)[:2]
    withheld_slice = tuple(region.label for region in stabilized_regions if region.withheld)[:3]
    actionable_slice = tuple(region.label for region in stabilized_regions if region.actionable)[:2]
    if not actionable_slice and stabilized_regions:
        actionable_slice = (stabilized_regions[0].label,)

    cues: list[str] = [f"access_dynamics:{dynamics_mode}"]
    if membrane_inertia >= 0.34:
        cues.append("access_membrane_inertia")
    if gating_hysteresis >= 0.28:
        cues.append("access_gating_hysteresis")
    if protective_filter >= 0.56:
        cues.append("access_protective_filter")

    return AccessDynamicsState(
        dynamics_mode=dynamics_mode,
        membrane_inertia=round(membrane_inertia, 4),
        gating_hysteresis=round(gating_hysteresis, 4),
        protective_filter=round(protective_filter, 4),
        stabilized_regions=tuple(stabilized_regions[:4]),
        reportable_slice=reportable_slice,
        withheld_slice=withheld_slice,
        actionable_slice=actionable_slice,
        cues=tuple(cues),
    )


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
