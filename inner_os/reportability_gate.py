from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from .constraint_field import ConstraintField
from .ignition_loop import IgnitionLoopState


@dataclass(frozen=True)
class ReportabilityGate:
    gate_mode: str = "open"
    reportable_slice: tuple[str, ...] = ()
    withheld_slice: tuple[str, ...] = ()
    actionable_slice: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_reportability_gate(
    *,
    slot_candidates: Sequence[Any],
    constraint_field: ConstraintField,
    ignition_state: IgnitionLoopState,
    current_risks: Sequence[str] = (),
) -> ReportabilityGate:
    slots = sorted(slot_candidates, key=_slot_activation, reverse=True)
    gate_mode = constraint_field.reportability_limit or "open"
    if ignition_state.ignition_phase == "dormant" and gate_mode == "open":
        gate_mode = "narrow"

    reportable: list[str] = []
    withheld: list[str] = []
    actionable: list[str] = []

    for slot in slots:
        label = _slot_label(slot)
        if not label:
            continue
        slot_reportable = _slot_bool(slot, "reportable")
        slot_withheld = _slot_bool(slot, "withheld")
        source = _slot_source(slot)

        if gate_mode == "withhold":
            withheld.append(label)
        elif gate_mode == "narrow":
            if slot_reportable and not slot_withheld and len(reportable) < 1:
                reportable.append(label)
            else:
                withheld.append(label)
        else:
            if slot_reportable and not slot_withheld and len(reportable) < 2:
                reportable.append(label)
            elif slot_withheld:
                withheld.append(label)

        if source in {"option", "focus"} and len(actionable) < 2:
            actionable.append(label)
        elif gate_mode == "withhold" and len(actionable) < 2:
            actionable.append(label)

    if not actionable and reportable:
        actionable.extend(reportable[:1])
    if not actionable and withheld:
        actionable.extend(withheld[:1])

    if current_risks and gate_mode != "withhold":
        reportable = reportable[:1]
        withheld = list(dict.fromkeys(withheld + reportable[1:]))

    cues: list[str] = [f"reportability_{gate_mode}"]
    if actionable:
        cues.append("reportability_actionable")
    if withheld:
        cues.append("reportability_withheld")

    return ReportabilityGate(
        gate_mode=gate_mode,
        reportable_slice=tuple(dict.fromkeys(reportable)),
        withheld_slice=tuple(dict.fromkeys(withheld)),
        actionable_slice=tuple(dict.fromkeys(actionable)),
        cues=tuple(cues),
    )


def _slot_label(slot: Any) -> str:
    if isinstance(slot, Mapping):
        return str(slot.get("label") or "").strip()
    return str(getattr(slot, "label", "") or "").strip()


def _slot_source(slot: Any) -> str:
    if isinstance(slot, Mapping):
        return str(slot.get("source") or "").strip()
    return str(getattr(slot, "source", "") or "").strip()


def _slot_activation(slot: Any) -> float:
    if isinstance(slot, Mapping):
        return float(slot.get("activation", 0.0) or 0.0)
    return float(getattr(slot, "activation", 0.0) or 0.0)


def _slot_bool(slot: Any, field_name: str) -> bool:
    if isinstance(slot, Mapping):
        return bool(slot.get(field_name, False))
    return bool(getattr(slot, field_name, False))
