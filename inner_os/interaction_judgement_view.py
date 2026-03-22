from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class ObservedSignal:
    signal_id: str
    signal_kind: str
    text: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferredSignal:
    signal_id: str
    signal_kind: str
    statement: str
    strength: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InteractionJudgementView:
    observed_signals: tuple[ObservedSignal, ...] = ()
    inferred_signals: tuple[InferredSignal, ...] = ()
    selected_object_labels: tuple[str, ...] = ()
    deferred_object_labels: tuple[str, ...] = ()
    active_operation_labels: tuple[str, ...] = ()
    intended_effect_labels: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observed_signals": [item.to_dict() for item in self.observed_signals],
            "inferred_signals": [item.to_dict() for item in self.inferred_signals],
            "selected_object_labels": list(self.selected_object_labels),
            "deferred_object_labels": list(self.deferred_object_labels),
            "active_operation_labels": list(self.active_operation_labels),
            "intended_effect_labels": list(self.intended_effect_labels),
            "cues": list(self.cues),
        }


def derive_interaction_judgement_view(
    *,
    current_text: str = "",
    reportable_facts: Sequence[str] = (),
    conversational_objects: Mapping[str, Any] | None = None,
    object_operations: Mapping[str, Any] | None = None,
    interaction_effects: Mapping[str, Any] | None = None,
    resonance_evaluation: Mapping[str, Any] | None = None,
) -> InteractionJudgementView:
    text = str(current_text or "").strip()
    reportable_facts = [str(item) for item in reportable_facts if str(item).strip()]
    objects_state = dict(conversational_objects or {})
    operations_plan = dict(object_operations or {})
    effects_plan = dict(interaction_effects or {})
    resonance = dict(resonance_evaluation or {})
    other_person_state = dict(resonance.get("estimated_other_person_state") or {})

    observed_signals: list[ObservedSignal] = []
    inferred_signals: list[InferredSignal] = []

    if text:
        observed_signals.append(
            ObservedSignal(
                signal_id="observed:0",
                signal_kind="user_text",
                text=text,
                confidence=1.0,
            )
        )
    for index, fact in enumerate(reportable_facts, start=len(observed_signals)):
        observed_signals.append(
            ObservedSignal(
                signal_id=f"observed:{index}",
                signal_kind="reportable_fact",
                text=fact,
                confidence=0.92,
            )
        )

    detail_room_level = str(other_person_state.get("detail_room_level") or "").strip()
    detail_room_score = _level_to_score(detail_room_level)
    if detail_room_level:
        inferred_signals.append(
            InferredSignal(
                signal_id="inferred:detail_room",
                signal_kind="detail_room",
                statement=f"system estimates that the other person has {detail_room_level} room for detail right now",
                strength=detail_room_score,
            )
        )
    pressure_sensitivity_level = str(other_person_state.get("pressure_sensitivity_level") or "").strip()
    pressure_sensitivity_score = _level_to_score(pressure_sensitivity_level)
    if pressure_sensitivity_level:
        inferred_signals.append(
            InferredSignal(
                signal_id="inferred:pressure_sensitivity",
                signal_kind="pressure_sensitivity",
                statement=f"system estimates that the other person may feel pushed if detail is requested too quickly",
                strength=pressure_sensitivity_score,
            )
        )
    acknowledgement_need_level = str(other_person_state.get("acknowledgement_need_level") or "").strip()
    if acknowledgement_need_level:
        inferred_signals.append(
            InferredSignal(
                signal_id="inferred:acknowledgement_need",
                signal_kind="acknowledgement_need",
                statement=f"system estimates that acknowledgement is {acknowledgement_need_level} priority right now",
                strength=_level_to_score(acknowledgement_need_level),
            )
        )

    question_pressure = _clamp01(float(operations_plan.get("question_pressure") or 0.0))
    if question_pressure > 0.0:
        inferred_signals.append(
            InferredSignal(
                signal_id="inferred:question_pressure",
                signal_kind="question_pressure",
                statement="system is leaning toward reducing questions in this turn",
                strength=question_pressure,
            )
        )
    defer_dominance = _clamp01(float(operations_plan.get("defer_dominance") or 0.0))
    if defer_dominance > 0.0:
        inferred_signals.append(
            InferredSignal(
                signal_id="inferred:defer_dominance",
                signal_kind="defer_dominance",
                statement="system is leaning toward holding or deferring rather than opening more detail",
                strength=defer_dominance,
            )
        )

    objects = [dict(item) for item in objects_state.get("objects") or [] if isinstance(item, Mapping)]
    deferred_ids = {str(item) for item in objects_state.get("deferred_object_ids") or [] if str(item).strip()}
    selected_object_labels = tuple(
        str(item.get("label") or "").strip()
        for item in objects
        if str(item.get("label") or "").strip()
        and str(item.get("object_id") or "").strip() not in deferred_ids
    )
    deferred_object_labels = tuple(
        str(item.get("label") or "").strip()
        for item in objects
        if str(item.get("label") or "").strip()
        and str(item.get("object_id") or "").strip() in deferred_ids
    )

    operations = [dict(item) for item in operations_plan.get("operations") or [] if isinstance(item, Mapping)]
    active_operation_labels = tuple(
        _format_operation_label(item)
        for item in operations
        if _format_operation_label(item)
    )

    effects = [dict(item) for item in effects_plan.get("effects") or [] if isinstance(item, Mapping)]
    intended_effect_labels = tuple(
        _format_effect_label(item)
        for item in effects
        if _format_effect_label(item)
    )

    cues = tuple(
        item
        for item in (
            f"observed:{len(observed_signals)}",
            f"inferred:{len(inferred_signals)}",
            f"objects:{len(selected_object_labels)}",
            f"deferred:{len(deferred_object_labels)}",
            f"operations:{len(active_operation_labels)}",
            f"effects:{len(intended_effect_labels)}",
        )
        if item
    )
    return InteractionJudgementView(
        observed_signals=tuple(observed_signals),
        inferred_signals=tuple(inferred_signals),
        selected_object_labels=selected_object_labels,
        deferred_object_labels=deferred_object_labels,
        active_operation_labels=active_operation_labels,
        intended_effect_labels=intended_effect_labels,
        cues=cues,
    )


def _format_operation_label(operation: Mapping[str, Any]) -> str:
    kind = str(operation.get("operation_kind") or "").strip()
    target = str(operation.get("target_label") or "").strip()
    if not kind:
        return ""
    if target:
        return f"{kind}:{target}"
    return kind


def _format_effect_label(effect: Mapping[str, Any]) -> str:
    kind = str(effect.get("effect_kind") or "").strip()
    target = str(effect.get("target_label") or "").strip()
    if not kind:
        return ""
    if target:
        return f"{kind}:{target}"
    return kind


def _level_to_score(level: str) -> float:
    normalized = str(level or "").strip().lower()
    if normalized == "low":
        return 0.24
    if normalized == "high":
        return 0.78
    return 0.52


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
