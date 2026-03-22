from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Sequence

from .affect_blend import AffectBlendState
from .constraint_field import ConstraintField
from .ignition_loop import run_ignition_loop
from .reportability_gate import derive_reportability_gate


@dataclass(frozen=True)
class WorkspaceSlot:
    slot_id: str
    label: str
    source: str
    activation: float
    reportable: bool
    withheld: bool
    binding_tags: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConsciousWorkspace:
    workspace_mode: str = "preconscious"
    ignition_score: float = 0.0
    ignition_phase: str = "dormant"
    workspace_stability: float = 0.0
    recurrent_residue: float = 0.0
    dominant_slot: str = ""
    active_slots: tuple[WorkspaceSlot, ...] = ()
    reportable_slice: tuple[str, ...] = ()
    withheld_slice: tuple[str, ...] = ()
    actionable_slice: tuple[str, ...] = ()
    reportability_gate: Dict[str, Any] | None = None
    cues: tuple[str, ...] = ()
    slot_scores: Dict[str, float] = field(default_factory=dict)
    winner_margin: float = 0.0
    dominant_inputs: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["active_slots"] = [slot.to_dict() for slot in self.active_slots]
        payload["reportability_gate"] = dict(self.reportability_gate or {})
        return payload


def ignite_conscious_workspace(
    *,
    affect_blend: AffectBlendState,
    constraint_field: ConstraintField,
    current_focus: str,
    reportable_facts: Sequence[str],
    current_risks: Sequence[str],
    related_person_ids: Sequence[str],
    interaction_option_candidates: Sequence[Mapping[str, Any] | Any] = (),
    memory_anchor: str = "",
    scene_state: Mapping[str, Any] | None = None,
    contact_field: Mapping[str, Any] | None = None,
    contact_dynamics: Mapping[str, Any] | None = None,
    access_projection: Mapping[str, Any] | None = None,
    access_dynamics: Mapping[str, Any] | None = None,
    previous_workspace: Mapping[str, Any] | None = None,
) -> ConsciousWorkspace:
    option_family = _first_option_family(interaction_option_candidates)
    previous_workspace = dict(previous_workspace or {})
    previous_residue = _clamp01(float(previous_workspace.get("recurrent_residue", 0.0) or 0.0))
    slots: list[WorkspaceSlot] = _slots_from_access_state(
        access_dynamics=access_dynamics,
        access_projection=access_projection,
    )
    scene_family = str((scene_state or {}).get("scene_family") or "").strip()
    person_binding = tuple(f"person:{person_id}" for person_id in related_person_ids if person_id)

    visible_focus = current_focus or (reportable_facts[0] if reportable_facts else memory_anchor)
    if not slots and visible_focus and visible_focus != "ambient":
        focus_reportable = constraint_field.reportability_limit != "withhold"
        slots.append(
            WorkspaceSlot(
                slot_id="focus",
                label=visible_focus,
                source="focus",
                activation=round(
                    _clamp01(0.34 + affect_blend.reportability_pressure * 0.34 + (0.08 if visible_focus == current_focus else 0.0)),
                    4,
                ),
                reportable=focus_reportable,
                withheld=not focus_reportable,
                binding_tags=person_binding + ((f"scene:{scene_family}",) if scene_family else ()),
            )
        )

    if not slots and affect_blend.dominant_mode:
        affect_reportable = (
            constraint_field.reportability_limit == "open"
            and affect_blend.dominant_mode not in {"defense", "distress"}
        )
        slots.append(
            WorkspaceSlot(
                slot_id="affect",
                label=affect_blend.dominant_mode,
                source="affect",
                activation=round(
                    _clamp01(
                        max(
                            affect_blend.care,
                            affect_blend.reverence,
                            affect_blend.innocence,
                            affect_blend.defense,
                            affect_blend.future_pull,
                            affect_blend.shared_world_pull,
                            affect_blend.distress,
                        )
                    ),
                    4,
                ),
                reportable=affect_reportable,
                withheld=not affect_reportable,
                binding_tags=((f"scene:{scene_family}",) if scene_family else ()),
            )
        )

    if option_family:
        option_reportable = constraint_field.reportability_limit != "withhold" and option_family not in {"withdraw"}
        slots.append(
            WorkspaceSlot(
                slot_id="option",
                label=option_family,
                source="option",
                activation=round(_first_option_weight(interaction_option_candidates), 4),
                reportable=option_reportable,
                withheld=not option_reportable,
                binding_tags=((f"scene:{scene_family}",) if scene_family else ()) + person_binding,
            )
        )

    if not any(slot.label == memory_anchor for slot in slots) and memory_anchor and memory_anchor != visible_focus:
        slots.append(
            WorkspaceSlot(
                slot_id="memory",
                label=memory_anchor,
                source="memory",
                activation=round(_clamp01(0.24 + affect_blend.residual_tension * 0.18 + previous_residue * 0.16), 4),
                reportable=constraint_field.reportability_limit == "open",
                withheld=constraint_field.reportability_limit != "open",
                binding_tags=((f"scene:{scene_family}",) if scene_family else ()),
            )
        )

    slots = sorted(slots, key=lambda slot: slot.activation, reverse=True)[:3]
    ignition_state = run_ignition_loop(
        slot_candidates=slots,
        affect_blend=affect_blend,
        constraint_field=constraint_field,
        previous_workspace=previous_workspace,
        current_risks=current_risks,
    )
    reportability_gate = derive_reportability_gate(
        slot_candidates=slots,
        constraint_field=constraint_field,
        ignition_state=ignition_state,
        current_risks=current_risks,
    )

    reportable_slice = reportability_gate.reportable_slice
    withheld_slice = reportability_gate.withheld_slice
    actionable_slice = reportability_gate.actionable_slice
    if constraint_field.reportability_limit == "withhold" and reportable_facts:
        withheld_slice = tuple(dict.fromkeys(withheld_slice + tuple(reportable_facts[:2])))

    workspace_mode = "preconscious"
    if ignition_state.ignition_phase == "priming":
        workspace_mode = "latent_foreground"
    if reportable_slice:
        workspace_mode = "foreground"
    if ignition_state.ignition_phase == "guarded" or constraint_field.reportability_limit == "withhold":
        workspace_mode = "guarded_foreground"

    dominant_slot = slots[0].slot_id if slots else ""
    slot_scores = {slot.label: round(_clamp01(slot.activation), 4) for slot in slots}
    winner_margin = _derive_winner_margin(slots)
    dominant_inputs = _derive_workspace_dominant_inputs(
        slots=slots,
        constraint_field=constraint_field,
        ignition_state=ignition_state,
        reportable_slice=reportable_slice,
        withheld_slice=withheld_slice,
        actionable_slice=actionable_slice,
        current_risks=current_risks,
    )
    cues: list[str] = [f"workspace_{workspace_mode}", f"ignition_{ignition_state.ignition_phase}"]
    if reportable_slice:
        cues.append("workspace_reportable")
    if withheld_slice:
        cues.append("workspace_withheld")
    if actionable_slice:
        cues.append("workspace_actionable")
    if ignition_state.recurrent_residue >= 0.4:
        cues.append("workspace_residual")
    cues.extend(ignition_state.cues)
    cues.extend(reportability_gate.cues)
    for item in (contact_field or {}).get("cues") or []:
        text = str(item).strip()
        if text:
            cues.append(text)
    for item in (contact_dynamics or {}).get("cues") or []:
        text = str(item).strip()
        if text:
            cues.append(text)
    for item in (access_projection or {}).get("cues") or []:
        text = str(item).strip()
        if text:
            cues.append(text)
    for item in (access_dynamics or {}).get("cues") or []:
        text = str(item).strip()
        if text:
            cues.append(text)

    return ConsciousWorkspace(
        workspace_mode=workspace_mode,
        ignition_score=ignition_state.ignition_score,
        ignition_phase=ignition_state.ignition_phase,
        workspace_stability=ignition_state.workspace_stability,
        recurrent_residue=ignition_state.recurrent_residue,
        dominant_slot=dominant_slot,
        active_slots=tuple(slots),
        reportable_slice=reportable_slice,
        withheld_slice=withheld_slice,
        actionable_slice=actionable_slice,
        reportability_gate=reportability_gate.to_dict(),
        cues=tuple(cues),
        slot_scores=slot_scores,
        winner_margin=winner_margin,
        dominant_inputs=dominant_inputs,
    )


def _first_option_family(candidates: Sequence[Mapping[str, Any] | Any]) -> str:
    if not candidates:
        return ""
    first = candidates[0]
    if isinstance(first, Mapping):
        return str(first.get("family_id") or "").strip()
    return str(getattr(first, "family_id", "") or "").strip()


def _first_option_weight(candidates: Sequence[Mapping[str, Any] | Any]) -> float:
    if not candidates:
        return 0.0
    first = candidates[0]
    if isinstance(first, Mapping):
        return _clamp01(float(first.get("relative_weight", 0.0) or 0.0))
    return _clamp01(float(getattr(first, "relative_weight", 0.0) or 0.0))


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _derive_winner_margin(slots: Sequence[WorkspaceSlot]) -> float:
    if not slots:
        return 0.0
    top = _clamp01(slots[0].activation)
    runner_up = _clamp01(slots[1].activation) if len(slots) > 1 else 0.0
    return round(_clamp01(top - runner_up), 4)


def _derive_workspace_dominant_inputs(
    *,
    slots: Sequence[WorkspaceSlot],
    constraint_field: ConstraintField,
    ignition_state: Any,
    reportable_slice: Sequence[str],
    withheld_slice: Sequence[str],
    actionable_slice: Sequence[str],
    current_risks: Sequence[str],
) -> tuple[str, ...]:
    dominant_inputs: list[str] = []
    if slots:
        top_slot = slots[0]
        dominant_inputs.append(f"slot:{top_slot.source}")
        if _clamp01(top_slot.activation) >= 0.24:
            dominant_inputs.append("dominant_slot_activation")
    gate_mode = str(constraint_field.reportability_limit or "").strip()
    if gate_mode:
        dominant_inputs.append(f"reportability:{gate_mode}")
    ignition_phase = str(getattr(ignition_state, "ignition_phase", "") or "").strip()
    if ignition_phase:
        dominant_inputs.append(f"ignition:{ignition_phase}")
    if current_risks:
        dominant_inputs.append("risk_guard")
    if reportable_slice:
        dominant_inputs.append("reportable_slice")
    if withheld_slice:
        dominant_inputs.append("withheld_slice")
    if actionable_slice:
        dominant_inputs.append("actionable_slice")
    if _clamp01(float(getattr(ignition_state, "recurrent_residue", 0.0) or 0.0)) >= 0.4:
        dominant_inputs.append("recurrent_residue")
    deduped: list[str] = []
    for item in dominant_inputs:
        text = str(item).strip()
        if text and text not in deduped:
            deduped.append(text)
    return tuple(deduped)


def _slots_from_access_state(
    *,
    access_dynamics: Mapping[str, Any] | None,
    access_projection: Mapping[str, Any] | None,
) -> list[WorkspaceSlot]:
    dynamic_state = dict(access_dynamics or {})
    projection = dict(access_projection or {})
    slots: list[WorkspaceSlot] = []
    raw_regions = [
        dict(region)
        for region in dynamic_state.get("stabilized_regions") or []
        if isinstance(region, Mapping)
    ]
    if not raw_regions:
        raw_regions = [
            dict(region)
            for region in projection.get("regions") or []
            if isinstance(region, Mapping)
        ]
    for region in raw_regions:
        if not isinstance(region, Mapping):
            continue
        label = str(region.get("label") or "").strip()
        if not label:
            continue
        slots.append(
            WorkspaceSlot(
                slot_id=str(region.get("region_id") or label).strip(),
                label=label,
                source=str(region.get("source") or "access").strip(),
                activation=round(
                    _clamp01(
                        float(region.get("stabilized_activation", region.get("activation", 0.0)) or 0.0)
                    ),
                    4,
                ),
                reportable=bool(region.get("reportable", False)),
                withheld=bool(region.get("withheld", False)),
                binding_tags=tuple(
                    str(item) for item in region.get("binding_tags") or [] if str(item).strip()
                ),
            )
        )
    return slots
