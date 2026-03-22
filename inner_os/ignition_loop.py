from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from .affect_blend import AffectBlendState
from .constraint_field import ConstraintField


@dataclass(frozen=True)
class IgnitionLoopState:
    ignition_score: float = 0.0
    workspace_stability: float = 0.0
    recurrent_residue: float = 0.0
    ignition_phase: str = "dormant"
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_ignition_loop(
    *,
    slot_candidates: Sequence[Any],
    affect_blend: AffectBlendState,
    constraint_field: ConstraintField,
    previous_workspace: Mapping[str, Any] | None = None,
    current_risks: Sequence[str] = (),
) -> IgnitionLoopState:
    previous_workspace = dict(previous_workspace or {})
    previous_stability = _clamp01(float(previous_workspace.get("workspace_stability", 0.0) or 0.0))
    previous_residue = _clamp01(float(previous_workspace.get("recurrent_residue", 0.0) or 0.0))
    previous_labels = {
        str(item.get("label") or "").strip()
        for item in previous_workspace.get("active_slots") or []
        if isinstance(item, Mapping) and str(item.get("label") or "").strip()
    }
    current_labels = {_slot_label(slot) for slot in slot_candidates if _slot_label(slot)}
    overlap_ratio = _overlap_ratio(current_labels, previous_labels)
    dominant_activation = max((_slot_activation(slot) for slot in slot_candidates), default=0.0)
    slot_pressure = _clamp01(sum(_slot_activation(slot) for slot in slot_candidates[:3]))

    ignition_score = _clamp01(
        0.18
        + affect_blend.reportability_pressure * 0.26
        + affect_blend.conflict_level * 0.12
        + affect_blend.residual_tension * 0.1
        + dominant_activation * 0.16
        + overlap_ratio * 0.1
        + previous_stability * 0.08
        + (0.08 if current_risks else 0.0)
    )
    workspace_stability = _clamp01(
        0.22
        + dominant_activation * 0.2
        + slot_pressure * 0.12
        + overlap_ratio * 0.18
        + previous_stability * 0.2
        + max(0.0, 1.0 - constraint_field.body_cost) * 0.1
        - affect_blend.residual_tension * 0.08
    )
    recurrent_residue = _clamp01(
        previous_residue * 0.42
        + affect_blend.residual_tension * 0.34
        + affect_blend.conflict_level * 0.16
        + overlap_ratio * 0.08
    )

    ignition_phase = "dormant"
    if ignition_score >= 0.32:
        ignition_phase = "priming"
    if ignition_score >= 0.56 and workspace_stability >= 0.48:
        ignition_phase = "ignited"
    if constraint_field.reportability_limit == "withhold" and ignition_score >= 0.32:
        ignition_phase = "guarded"

    cues: list[str] = [f"ignition_{ignition_phase}"]
    if overlap_ratio >= 0.34:
        cues.append("ignition_reentry")
    if recurrent_residue >= 0.4:
        cues.append("ignition_residual")

    return IgnitionLoopState(
        ignition_score=round(ignition_score, 4),
        workspace_stability=round(workspace_stability, 4),
        recurrent_residue=round(recurrent_residue, 4),
        ignition_phase=ignition_phase,
        cues=tuple(cues),
    )


def _slot_label(slot: Any) -> str:
    if isinstance(slot, Mapping):
        return str(slot.get("label") or "").strip()
    return str(getattr(slot, "label", "") or "").strip()


def _slot_activation(slot: Any) -> float:
    if isinstance(slot, Mapping):
        return _clamp01(float(slot.get("activation", 0.0) or 0.0))
    return _clamp01(float(getattr(slot, "activation", 0.0) or 0.0))


def _overlap_ratio(current: set[str], previous: set[str]) -> float:
    if not current or not previous:
        return 0.0
    return len(current & previous) / max(len(current | previous), 1)


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
