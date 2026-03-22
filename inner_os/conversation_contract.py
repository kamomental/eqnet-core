from __future__ import annotations

from typing import Any, Dict, Mapping


def build_conversation_contract(
    *,
    conversational_objects: Mapping[str, Any] | None,
    object_operations: Mapping[str, Any] | None,
    interaction_effects: Mapping[str, Any] | None,
    interaction_judgement_summary: Mapping[str, Any] | None = None,
    interaction_condition_report: Mapping[str, Any] | None = None,
    interaction_policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    conversational_objects = dict(conversational_objects or {})
    object_operations = dict(object_operations or {})
    interaction_effects = dict(interaction_effects or {})
    interaction_judgement_summary = dict(interaction_judgement_summary or {})
    interaction_condition_report = dict(interaction_condition_report or {})
    interaction_policy = dict(interaction_policy or {})

    payload: Dict[str, Any] = {}

    object_rows = [
        dict(item)
        for item in conversational_objects.get("objects") or []
        if isinstance(item, Mapping)
    ]
    primary_object_id = str(conversational_objects.get("primary_object_id") or "").strip()
    primary_object = next(
        (
            item
            for item in object_rows
            if str(item.get("object_id") or "").strip() == primary_object_id
        ),
        {},
    )
    selected_objects = [
        str(item.get("label") or "").strip()
        for item in object_rows[:3]
        if str(item.get("label") or "").strip()
    ]
    primary_label = str(primary_object.get("label") or "").strip()
    if primary_label:
        payload["primary_object"] = primary_label
        payload["focus_now"] = primary_label
    if selected_objects:
        payload["selected_objects"] = selected_objects

    deferred_object_ids = {
        str(item)
        for item in conversational_objects.get("deferred_object_ids") or []
        if str(item).strip()
    }
    deferred_objects = [
        str(item.get("label") or "").strip()
        for item in object_rows
        if str(item.get("object_id") or "").strip() in deferred_object_ids
        and str(item.get("label") or "").strip()
    ]
    if deferred_objects:
        payload["deferred_objects"] = deferred_objects[:3]
        payload["do_not_open_yet"] = deferred_objects[:3]
        payload["leave_closed_for_now"] = deferred_objects[:3]

    operation_rows = [
        dict(item)
        for item in object_operations.get("operations") or []
        if isinstance(item, Mapping)
    ]
    primary_operation_id = str(object_operations.get("primary_operation_id") or "").strip()
    primary_operation = next(
        (
            item
            for item in operation_rows
            if str(item.get("operation_id") or "").strip() == primary_operation_id
        ),
        {},
    )
    operation_focus: Dict[str, Any] = {}
    primary_operation_kind = str(primary_operation.get("operation_kind") or "").strip()
    if primary_operation_kind:
        operation_focus["primary_operation"] = primary_operation_kind
    primary_target = str(primary_operation.get("target_label") or "").strip()
    if primary_target:
        operation_focus["operation_target"] = primary_target
    ordered_operations = [
        str(item)
        for item in interaction_policy.get("ordered_operation_kinds") or []
        if str(item).strip()
    ]
    if ordered_operations:
        operation_focus["ordered_operations"] = ordered_operations
    if "question_budget" in object_operations:
        operation_focus["question_budget"] = int(object_operations.get("question_budget") or 0)
    if "question_pressure" in object_operations:
        operation_focus["question_pressure"] = round(float(object_operations.get("question_pressure", 0.0) or 0.0), 4)
    if "defer_dominance" in object_operations:
        operation_focus["defer_dominance"] = round(float(object_operations.get("defer_dominance", 0.0) or 0.0), 4)
    guarded_topics = [str(item) for item in object_operations.get("guarded_topics") or [] if str(item).strip()]
    if guarded_topics:
        operation_focus["guarded_topics"] = guarded_topics[:3]
    if operation_focus:
        payload["operation_focus"] = operation_focus
        payload["response_action_now"] = dict(operation_focus)

    effect_rows = [
        dict(item)
        for item in interaction_effects.get("effects") or []
        if isinstance(item, Mapping)
    ]
    intended_effects = []
    for item in effect_rows[:4]:
        effect_kind = str(item.get("effect_kind") or "").strip()
        if not effect_kind:
            continue
        intended_effects.append(
            {
                "effect": effect_kind,
                "target": str(item.get("target_label") or "").strip(),
                "intensity": round(float(item.get("intensity", 0.0) or 0.0), 4),
            }
        )
    if intended_effects:
        payload["intended_effects"] = intended_effects
        payload["wanted_effect_on_other"] = list(intended_effects)

    ordered_effects = [
        str(item)
        for item in interaction_policy.get("ordered_effect_kinds") or []
        if str(item).strip()
    ]
    if ordered_effects:
        payload["ordered_effects"] = ordered_effects

    observed_lines = [
        str(item).strip()
        for item in interaction_judgement_summary.get("observed_lines") or []
        if str(item).strip()
    ]
    inferred_lines = [
        str(item).strip()
        for item in interaction_judgement_summary.get("inferred_lines") or []
        if str(item).strip()
    ]
    operation_lines = [
        str(item).strip()
        for item in interaction_judgement_summary.get("operation_lines") or []
        if str(item).strip()
    ]
    effect_lines = [
        str(item).strip()
        for item in interaction_judgement_summary.get("intended_effect_lines") or []
        if str(item).strip()
    ]
    if observed_lines:
        payload["observed_user_content"] = observed_lines[:2]
    if inferred_lines:
        payload["inferred_other_state"] = inferred_lines[:3]
    if operation_lines:
        payload["operation_summary"] = operation_lines[:3]
    if effect_lines:
        payload["intended_effect_summary"] = effect_lines[:3]

    condition_lines = []
    for key in ("scene_lines", "relation_lines", "memory_lines", "integration_lines"):
        for item in interaction_condition_report.get(key) or []:
            text = str(item).strip()
            if text:
                condition_lines.append(text)
    if condition_lines:
        payload["condition_summary"] = condition_lines[:4]

    do_not_cross = [
        str(item)
        for item in interaction_policy.get("do_not_cross") or []
        if str(item).strip()
    ]
    if do_not_cross:
        payload["do_not_cross"] = do_not_cross

    return payload
