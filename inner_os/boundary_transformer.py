from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


_SOFTEN_WHEN_NARROW = frozenset(
    {
        "gentle_extension",
        "offer_small_opening_line",
        "offer_small_opening_frame",
        "name_next_step_lightly",
        "offer_next_step",
    }
)
_WITHHOLD_WHEN_WITHHOLD = frozenset(
    {
        "clarify_question",
        "probe_detail",
        "name_hidden_detail",
        "explicit_advice",
        "offer_next_step",
    }
)
_WITHHOLD_FORCE_REPORTABILITY = frozenset(
    {
        "clarify_question",
        "probe_detail",
        "name_hidden_detail",
    }
)
_SOFTEN_WHEN_RISK_PRESENT = frozenset(
    {
        "offer_small_opening_line",
        "offer_small_opening_frame",
        "gentle_extension",
        "name_next_step_lightly",
        "offer_next_step",
        "seed_small_topic",
    }
)
_PROTECTIVE_ACTS = frozenset(
    {
        "respect_boundary",
        "quiet_presence",
        "leave_unfinished_closed",
        "leave_return_point",
        "protect_talking_room",
        "keep_choice_with_other_person",
    }
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


@dataclass(frozen=True)
class BoundaryCandidateDecision:
    act: str = ""
    decision: str = "allow"
    reasons: tuple[str, ...] = ()
    pressure: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryTransformResult:
    gate_mode: str = "open"
    authority_scope: str = "user"
    transformation_mode: str = "pass"
    allowed_acts: tuple[str, ...] = ()
    softened_acts: tuple[str, ...] = ()
    withheld_acts: tuple[str, ...] = ()
    deferred_topics: tuple[str, ...] = ()
    do_not_cross: tuple[str, ...] = ()
    candidate_decisions: tuple[BoundaryCandidateDecision, ...] = ()
    residual_pressure: float = 0.0
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["candidate_decisions"] = [
            item.to_dict() if isinstance(item, BoundaryCandidateDecision) else dict(item)
            for item in self.candidate_decisions
        ]
        return payload


def derive_boundary_transform_result(
    *,
    content_sequence: Sequence[Mapping[str, Any]] | None,
    interaction_constraints: Mapping[str, Any] | None,
    conversation_contract: Mapping[str, Any] | None,
    constraint_field: Mapping[str, Any] | None,
    reportability_gate: Mapping[str, Any] | None,
    current_risks: Sequence[str] = (),
) -> BoundaryTransformResult:
    constraints = dict(interaction_constraints or {})
    contract = dict(conversation_contract or {})
    field = dict(constraint_field or {})
    gate = dict(reportability_gate or {})
    gate_mode = _text(gate.get("gate_mode") or field.get("reportability_limit") or "open") or "open"
    do_not_cross = tuple(
        dict.fromkeys(
            _text(item)
            for item in field.get("do_not_cross") or ()
            if _text(item)
        )
    )
    deferred_topics = tuple(
        dict.fromkeys(
            _text(item)
            for item in contract.get("leave_closed_for_now") or ()
            if _text(item)
        )
    )
    allow_small_next_step = bool(constraints.get("allow_small_next_step", False))

    decisions: list[BoundaryCandidateDecision] = []
    allowed_acts: list[str] = []
    softened_acts: list[str] = []
    withheld_acts: list[str] = []

    for item in content_sequence or ():
        act = _text(item.get("act"))
        if not act:
            continue
        decision = "allow"
        reasons: list[str] = []
        pressure = 0.0

        if deferred_topics and act in _PROTECTIVE_ACTS:
            reasons.append("protect_deferred_topic")
            pressure = max(pressure, 0.24)

        if gate_mode == "narrow" and not allow_small_next_step and act in _SOFTEN_WHEN_NARROW:
            decision = "soften"
            reasons.append("narrow_reportability")
            pressure = max(pressure, 0.46)

        if gate_mode == "withhold" and act in _WITHHOLD_WHEN_WITHHOLD:
            decision = "withhold"
            reasons.append("withhold_reportability")
            pressure = max(pressure, 0.84)

        if "force_reportability" in do_not_cross and act in _WITHHOLD_FORCE_REPORTABILITY:
            decision = "withhold"
            reasons.append("force_reportability_block")
            pressure = max(pressure, 0.92)

        if current_risks and act in _SOFTEN_WHEN_RISK_PRESENT and decision == "allow":
            decision = "soften"
            reasons.append("current_risk_present")
            pressure = max(pressure, 0.4)

        if decision == "allow":
            allowed_acts.append(act)
        elif decision == "soften":
            softened_acts.append(act)
        else:
            withheld_acts.append(act)

        decisions.append(
            BoundaryCandidateDecision(
                act=act,
                decision=decision,
                reasons=tuple(dict.fromkeys(reasons)),
                pressure=round(pressure, 4),
            )
        )

    transformation_mode = "pass"
    if withheld_acts:
        transformation_mode = "withhold"
    elif softened_acts:
        transformation_mode = "soften"

    residual_pressure = _clamp01(
        max(
            0.76 if gate_mode == "withhold" else 0.44 if gate_mode == "narrow" else 0.0,
            len(softened_acts) * 0.14,
            len(withheld_acts) * 0.28,
            len(deferred_topics) * 0.18,
            len(do_not_cross) * 0.12,
        )
    )

    cues: list[str] = [f"boundary_gate_{gate_mode}"]
    if softened_acts:
        cues.append("boundary_softened_candidate")
    if withheld_acts:
        cues.append("boundary_withheld_candidate")
    if deferred_topics:
        cues.append("boundary_deferred_topic")

    authority_scope = "user"
    if gate_mode != "open" or current_risks:
        authority_scope = "user_guarded"

    return BoundaryTransformResult(
        gate_mode=gate_mode,
        authority_scope=authority_scope,
        transformation_mode=transformation_mode,
        allowed_acts=tuple(dict.fromkeys(allowed_acts)),
        softened_acts=tuple(dict.fromkeys(softened_acts)),
        withheld_acts=tuple(dict.fromkeys(withheld_acts)),
        deferred_topics=deferred_topics,
        do_not_cross=do_not_cross,
        candidate_decisions=tuple(decisions),
        residual_pressure=round(residual_pressure, 4),
        cues=tuple(dict.fromkeys(cues)),
    )
