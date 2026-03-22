from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class InteractionAuditBundle:
    observed_lines: tuple[str, ...] = ()
    inferred_lines: tuple[str, ...] = ()
    selected_object_lines: tuple[str, ...] = ()
    deferred_object_lines: tuple[str, ...] = ()
    operation_lines: tuple[str, ...] = ()
    intended_effect_lines: tuple[str, ...] = ()
    scene_lines: tuple[str, ...] = ()
    relation_lines: tuple[str, ...] = ()
    memory_lines: tuple[str, ...] = ()
    integration_lines: tuple[str, ...] = ()
    inspection_lines: tuple[str, ...] = ()
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    report_lines: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_interaction_audit_bundle(
    *,
    interaction_judgement_summary: Mapping[str, Any] | None = None,
    interaction_condition_report: Mapping[str, Any] | None = None,
    interaction_inspection_report: Mapping[str, Any] | None = None,
    conversational_objects: Mapping[str, Any] | None = None,
    object_operations: Mapping[str, Any] | None = None,
    interaction_effects: Mapping[str, Any] | None = None,
    resonance_evaluation: Mapping[str, Any] | None = None,
) -> InteractionAuditBundle:
    summary = dict(interaction_judgement_summary or {})
    condition = dict(interaction_condition_report or {})
    inspection = dict(interaction_inspection_report or {})
    conversational = dict(conversational_objects or {})
    operations = dict(object_operations or {})
    effects = dict(interaction_effects or {})
    resonance = dict(resonance_evaluation or {})
    other_person_state = dict(resonance.get("estimated_other_person_state") or {})

    observed_lines = _tuple_lines(summary.get("observed_lines"))
    inferred_lines = _tuple_lines(summary.get("inferred_lines"))
    selected_object_lines = _tuple_lines(summary.get("selected_object_lines"))
    deferred_object_lines = _tuple_lines(summary.get("deferred_object_lines"))
    operation_lines = _tuple_lines(summary.get("operation_lines"))
    intended_effect_lines = _tuple_lines(summary.get("intended_effect_lines"))
    scene_lines = _tuple_lines(condition.get("scene_lines"))
    relation_lines = _tuple_lines(condition.get("relation_lines"))
    memory_lines = _tuple_lines(condition.get("memory_lines"))
    integration_lines = _tuple_lines(condition.get("integration_lines"))
    inspection_lines = _tuple_lines(inspection.get("report_lines"))

    key_metrics = {
        "pressure_balance": _as_float(conversational.get("pressure_balance")),
        "question_budget": _as_int(operations.get("question_budget")),
        "question_pressure": _as_float(operations.get("question_pressure")),
        "defer_dominance": _as_float(operations.get("defer_dominance")),
        "effect_count": len(tuple(effects.get("effects") or ())),
        "resonance_score": _as_float(resonance.get("resonance_score")),
        "recommended_family_id": str(resonance.get("recommended_family_id") or ""),
        "detail_room_level": str(other_person_state.get("detail_room_level") or ""),
        "acknowledgement_need_level": str(
            other_person_state.get("acknowledgement_need_level") or ""
        ),
        "pressure_sensitivity_level": str(
            other_person_state.get("pressure_sensitivity_level") or ""
        ),
        "next_step_room_level": str(other_person_state.get("next_step_room_level") or ""),
    }

    report_lines = (
        tuple(f"観測したこと: {line}" for line in observed_lines)
        + tuple(f"推測したこと: {line}" for line in inferred_lines)
        + tuple(f"今回扱う対象: {line}" for line in selected_object_lines)
        + tuple(f"今はまだ触れない対象: {line}" for line in deferred_object_lines)
        + tuple(f"今する操作: {line}" for line in operation_lines)
        + tuple(f"相手に起きてほしいこと: {line}" for line in intended_effect_lines)
        + tuple(f"場面が効いていること: {line}" for line in scene_lines)
        + tuple(f"相手との関係が効いていること: {line}" for line in relation_lines)
        + tuple(f"記憶が効いていること: {line}" for line in memory_lines)
        + tuple(f"統合した判断: {line}" for line in integration_lines)
        + tuple(f"確認メモ: {line}" for line in inspection_lines[:3])
    )

    return InteractionAuditBundle(
        observed_lines=observed_lines,
        inferred_lines=inferred_lines,
        selected_object_lines=selected_object_lines,
        deferred_object_lines=deferred_object_lines,
        operation_lines=operation_lines,
        intended_effect_lines=intended_effect_lines,
        scene_lines=scene_lines,
        relation_lines=relation_lines,
        memory_lines=memory_lines,
        integration_lines=integration_lines,
        inspection_lines=inspection_lines,
        key_metrics=key_metrics,
        report_lines=report_lines,
    )


def _tuple_lines(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(line) for line in value if str(line).strip())


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
