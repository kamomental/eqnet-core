from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class ContactReflectionState:
    state: str = "open_reflection"
    reflection_style: str = "reflect_then_question"
    transmit_share: float = 0.0
    reflect_share: float = 0.0
    absorb_share: float = 0.0
    block_share: float = 0.0
    dominant_inputs: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_contact_reflection_state(
    *,
    contact_field: Mapping[str, Any] | None,
    contact_dynamics: Mapping[str, Any] | None = None,
    access_projection: Mapping[str, Any] | None = None,
    constraint_field: Mapping[str, Any] | None = None,
    current_risks: Sequence[str] | None = None,
) -> ContactReflectionState:
    field = dict(contact_field or {})
    dynamics = dict(contact_dynamics or {})
    projection = dict(access_projection or {})
    constraint = dict(constraint_field or {})
    risks = [str(item).strip().lower() for item in current_risks or [] if str(item).strip()]

    reportable_slice = [
        str(item).strip()
        for item in projection.get("reportable_slice") or []
        if str(item).strip()
    ]
    withheld_slice = [
        str(item).strip()
        for item in projection.get("withheld_slice") or []
        if str(item).strip()
    ]
    actionable_slice = [
        str(item).strip()
        for item in projection.get("actionable_slice") or []
        if str(item).strip()
    ]

    reportability_pressure = _clamp01(field.get("reportability_pressure"))
    protective_pressure = _clamp01(field.get("protective_pressure"))
    reentry_bias = _clamp01(dynamics.get("reentry_bias"))
    protective_hold = _clamp01(dynamics.get("protective_hold"))
    reportability_limit = str(constraint.get("reportability_limit") or "").strip()

    reportable_signal = 1.0 if reportable_slice else 0.0
    withheld_signal = min(len(withheld_slice), 3) / 3.0
    actionable_signal = min(len(actionable_slice), 2) / 2.0
    risk_signal = 1.0 if risks else 0.0

    transmit_share = _clamp01(
        reportable_signal * 0.42
        + reportability_pressure * 0.18
        + actionable_signal * 0.12
        + reentry_bias * 0.08
        - protective_hold * 0.1
        - risk_signal * 0.08
    )
    reflect_share = _clamp01(
        reportable_signal * 0.26
        + actionable_signal * 0.2
        + reentry_bias * 0.18
        + reportability_pressure * 0.08
        + (0.08 if reportable_slice and withheld_slice else 0.0)
    )
    absorb_share = _clamp01(
        withheld_signal * 0.36
        + protective_hold * 0.22
        + protective_pressure * 0.16
        + (0.14 if reportability_limit == "withhold" else 0.0)
        + (0.08 if not reportable_slice and actionable_slice else 0.0)
    )
    block_share = _clamp01(
        risk_signal * 0.34
        + (0.18 if reportability_limit == "withhold" else 0.0)
        + max(protective_hold, protective_pressure) * 0.22
        + (0.12 if not actionable_slice and withheld_slice else 0.0)
    )

    state = "open_reflection"
    reflection_style = "reflect_then_question"
    if block_share >= 0.72:
        state = "blocked_contact"
        reflection_style = "boundary_only"
    elif absorb_share >= 0.46 and absorb_share >= reflect_share:
        state = "absorbing_contact"
        reflection_style = "reflect_only"
    elif protective_hold >= 0.58 or withheld_signal >= 0.66:
        state = "guarded_reflection"
        reflection_style = "reflect_only"

    dominant_inputs = tuple(
        item
        for item in (
            "reportable_slice" if reportable_slice else "",
            "withheld_slice" if withheld_slice else "",
            "actionable_slice" if actionable_slice else "",
            "contact_reentry_bias" if reentry_bias >= 0.24 else "",
            "contact_protective_hold" if protective_hold >= 0.34 else "",
            "contact_protective_pressure" if protective_pressure >= 0.34 else "",
            "reportability_withhold" if reportability_limit == "withhold" else "",
            "current_risks" if risks else "",
        )
        if item
    )
    cues = tuple(
        item
        for item in (
            f"contact_reflection:{state}",
            f"contact_reflection_style:{reflection_style}",
            "contact_reflection_withheld" if withheld_slice else "",
            "contact_reflection_reportable" if reportable_slice else "",
            "contact_reflection_blocked" if state == "blocked_contact" else "",
        )
        if item
    )
    return ContactReflectionState(
        state=state,
        reflection_style=reflection_style,
        transmit_share=round(transmit_share, 4),
        reflect_share=round(reflect_share, 4),
        absorb_share=round(absorb_share, 4),
        block_share=round(block_share, 4),
        dominant_inputs=dominant_inputs,
        cues=cues,
    )


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return float(numeric)
