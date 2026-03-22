from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class ObjectOperation:
    operation_id: str
    object_id: str
    operation_kind: str
    target_label: str
    priority: float = 0.0
    operation_strength: float = 0.0
    question_budget_cost: int = 0
    burden_risk: float = 0.0
    connection_support: float = 0.0
    withheld: bool = False
    reason: str = ""
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectOperationPlan:
    primary_operation_id: str = ""
    operations: tuple[ObjectOperation, ...] = ()
    guarded_topics: tuple[str, ...] = ()
    question_budget: int = 0
    question_pressure: float = 0.0
    defer_dominance: float = 0.0
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_operation_id": self.primary_operation_id,
            "operations": [item.to_dict() for item in self.operations],
            "guarded_topics": list(self.guarded_topics),
            "question_budget": self.question_budget,
            "question_pressure": self.question_pressure,
            "defer_dominance": self.defer_dominance,
            "cues": list(self.cues),
        }


def derive_object_operations(
    *,
    conversational_objects: Mapping[str, Any] | None,
    scene_state: Mapping[str, Any] | None = None,
    relation_context: Mapping[str, Any] | None = None,
    memory_context: Mapping[str, Any] | None = None,
    resonance_evaluation: Mapping[str, Any] | None = None,
    constraint_field: Mapping[str, Any] | None = None,
    conscious_workspace: Mapping[str, Any] | None = None,
    interaction_option_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> ObjectOperationPlan:
    objects_state = dict(conversational_objects or {})
    scene = dict(scene_state or {})
    relation = dict(relation_context or {})
    memory = dict(memory_context or {})
    resonance = dict(resonance_evaluation or {})
    constraint = dict(constraint_field or {})
    workspace = dict(conscious_workspace or {})
    candidates = [dict(item) for item in interaction_option_candidates or [] if isinstance(item, Mapping)]

    objects = [dict(item) for item in objects_state.get("objects") or [] if isinstance(item, Mapping)]
    primary_object_id = str(objects_state.get("primary_object_id") or "").strip()
    primary_object = next((item for item in objects if str(item.get("object_id") or "") == primary_object_id), {})
    deferred_ids = {str(item) for item in objects_state.get("deferred_object_ids") or [] if str(item).strip()}
    reportability_gate = dict(workspace.get("reportability_gate") or {})
    detail_room_level = str(((resonance.get("estimated_other_person_state") or {}).get("detail_room_level")) or "medium").strip()
    pressure_sensitivity_level = str(((resonance.get("estimated_other_person_state") or {}).get("pressure_sensitivity_level")) or "medium").strip()
    next_step_room_level = str(((resonance.get("estimated_other_person_state") or {}).get("next_step_room_level")) or "medium").strip()
    reportability_limit = str(constraint.get("reportability_limit") or "").strip()
    top_family = str(candidates[0].get("family_id") or resonance.get("recommended_family_id") or "").strip()
    scene_family = str(scene.get("scene_family") or "").strip()
    scene_privacy = _clamp01(float(scene.get("privacy_level", 0.5) or 0.5))
    scene_norm = _clamp01(float(scene.get("norm_pressure", 0.0) or 0.0))
    scene_safety = _clamp01(float(scene.get("safety_margin", 0.5) or 0.5))
    relation_bias_strength = _clamp01(float(relation.get("relation_bias_strength", 0.0) or 0.0))
    recent_strain = _clamp01(float(relation.get("recent_strain", 0.0) or 0.0))
    trust_memory = _clamp01(float(relation.get("trust_memory", 0.0) or 0.0))
    familiarity = _clamp01(float(relation.get("familiarity", 0.0) or 0.0))
    attachment = _clamp01(float(relation.get("attachment", 0.0) or 0.0))
    partner_timing_hint = str(relation.get("partner_timing_hint") or "").strip()
    partner_stance_hint = str(relation.get("partner_stance_hint") or "").strip()
    relation_seed_summary = str(memory.get("relation_seed_summary") or "").strip()
    long_term_theme_summary = str(memory.get("long_term_theme_summary") or "").strip()
    conscious_residue_summary = str(memory.get("conscious_residue_summary") or "").strip()

    object_by_id = {
        str(item.get("object_id") or "").strip(): item
        for item in objects
        if str(item.get("object_id") or "").strip()
    }
    shared_thread_objects = [
        item for item in objects if str(item.get("object_kind") or "").strip() == "shared_thread"
    ]
    unfinished_objects = [
        item for item in objects if str(item.get("object_kind") or "").strip() == "unfinished_part"
    ]
    theme_objects = [
        item for item in objects if str(item.get("object_kind") or "").strip() == "theme_anchor"
    ]

    primary_touchability_score = _clamp01(float(primary_object.get("touchability_score", 0.0) or 0.0))
    primary_depth_room = _clamp01(float(primary_object.get("depth_room", 0.0) or 0.0))
    primary_defer_pressure = _clamp01(float(primary_object.get("defer_pressure", 0.0) or 0.0))
    primary_object_kind = str(primary_object.get("object_kind") or "").strip()

    question_room = _question_room(
        detail_room_level=detail_room_level,
        pressure_sensitivity_level=pressure_sensitivity_level,
        primary_depth_room=primary_depth_room,
        primary_touchability_score=primary_touchability_score,
        scene_privacy=scene_privacy,
        scene_norm=scene_norm,
        scene_safety=scene_safety,
        relation_bias_strength=relation_bias_strength,
        trust_memory=trust_memory,
        familiarity=familiarity,
        attachment=attachment,
        recent_strain=recent_strain,
        primary_defer_pressure=primary_defer_pressure,
    )

    question_budget = _question_budget_from_room(
        question_room=question_room,
        reportability_limit=reportability_limit,
        reportability_gate_mode=str(reportability_gate.get("gate_mode") or "").strip(),
        top_family=top_family,
    )

    operations: list[ObjectOperation] = []

    def add_operation(
        *,
        object_id: str,
        operation_kind: str,
        target_label: str,
        priority: float,
        operation_strength: float,
        question_budget_cost: int,
        burden_risk: float,
        connection_support: float,
        withheld: bool,
        reason: str,
        cues: Sequence[str] = (),
    ) -> str:
        operation_id = f"operation:{len(operations)}"
        operations.append(
            ObjectOperation(
                operation_id=operation_id,
                object_id=object_id,
                operation_kind=operation_kind,
                target_label=str(target_label or "").strip(),
                priority=_clamp01(priority),
                operation_strength=_clamp01(operation_strength),
                question_budget_cost=max(0, int(question_budget_cost)),
                burden_risk=_clamp01(burden_risk),
                connection_support=_clamp01(connection_support),
                withheld=bool(withheld),
                reason=str(reason or "").strip(),
                cues=tuple(str(item) for item in cues if str(item).strip()),
            )
        )
        return operation_id

    primary_operation_id = ""
    if primary_object:
        primary_touchability = str(primary_object.get("touchability") or "acknowledge_only").strip()
        primary_label = str(primary_object.get("label") or primary_object.get("explicit_text") or "current_topic").strip()
        if (
            primary_object_kind == "unfinished_part"
            or primary_defer_pressure >= 0.72
            or question_room <= 0.32
            or reportability_limit == "withhold"
        ):
            primary_operation_id = add_operation(
                object_id=str(primary_object.get("object_id") or ""),
                operation_kind="hold_without_probe",
                target_label=primary_label,
                priority=0.97,
                operation_strength=max(0.8, primary_defer_pressure),
                question_budget_cost=0,
                burden_risk=0.1,
                connection_support=0.82,
                withheld=False,
                reason="Stay with what is already present without making the other person unpack the unfinished part.",
                cues=("hold", "no_probe", "unfinished_safe"),
            )
        elif (
            top_family == "co_move"
            or primary_touchability == "small_next_step_ready"
            or (
                primary_object_kind in {"next_step", "theme_anchor"}
                and next_step_room_level != "low"
                and question_room >= 0.42
            )
            or (
                next_step_room_level == "high"
                and primary_defer_pressure <= 0.58
                and question_room >= 0.42
                and primary_object_kind not in {"unfinished_part", "strain"}
                and recent_strain <= 0.44
            )
        ):
            primary_operation_id = add_operation(
                object_id=str(primary_object.get("object_id") or ""),
                operation_kind="offer_small_next_step",
                target_label=primary_label,
                priority=0.9,
                operation_strength=max(0.68, primary_touchability_score, _level_to_score(next_step_room_level)),
                question_budget_cost=0,
                burden_risk=0.22,
                connection_support=0.74,
                withheld=False,
                reason="Handle only one small next step that should not increase the other person's burden.",
                cues=("next_step", "shared_room"),
            )
        elif (
            top_family == "clarify"
            and (
                (primary_touchability == "narrow_probe" and question_budget > 0)
                or (question_room >= 0.58 and primary_depth_room >= 0.5)
            )
        ):
            primary_operation_id = add_operation(
                object_id=str(primary_object.get("object_id") or ""),
                operation_kind="narrow_clarify",
                target_label=primary_label,
                priority=0.94,
                operation_strength=max(0.72, primary_touchability_score, question_room),
                question_budget_cost=1,
                burden_risk=0.42 if pressure_sensitivity_level == "high" else 0.28,
                connection_support=0.62,
                withheld=False,
                reason="Confirm only a narrow part of what the other person can currently talk about.",
                cues=("clarify", "bounded_scope"),
            )
        elif top_family in {"wait", "contain", "repair"} or question_budget == 0 or primary_touchability == "acknowledge_only":
            primary_operation_id = add_operation(
                object_id=str(primary_object.get("object_id") or ""),
                operation_kind="hold_without_probe",
                target_label=primary_label,
                priority=0.96,
                operation_strength=max(0.78, primary_defer_pressure),
                question_budget_cost=0,
                burden_risk=0.12,
                connection_support=0.82,
                withheld=False,
                reason="Receive what is already present without asking the other person for detail.",
                cues=("hold", "no_probe"),
            )
        else:
            primary_operation_id = add_operation(
                object_id=str(primary_object.get("object_id") or ""),
                operation_kind="acknowledge",
                target_label=primary_label,
                priority=0.88,
                operation_strength=max(0.66, primary_touchability_score),
                question_budget_cost=0,
                burden_risk=0.14,
                connection_support=0.76,
                withheld=False,
                reason="Acknowledge what the other person has already brought forward.",
                cues=("acknowledge",),
            )

    guarded_topics: list[str] = []
    for item in objects:
        object_id = str(item.get("object_id") or "").strip()
        label = str(item.get("label") or "").strip()
        if object_id in deferred_ids:
            guarded_topics.append(label)
            add_operation(
                object_id=object_id,
                operation_kind="defer_detail",
                target_label=label,
                priority=0.72,
                operation_strength=0.84,
                question_budget_cost=0,
                burden_risk=0.08,
                connection_support=0.58,
                withheld=True,
                reason="Keep this topic available for later without opening it further right now.",
                cues=("defer", "detail_hold"),
            )

    for item in shared_thread_objects:
        object_id = str(item.get("object_id") or "").strip()
        label = str(item.get("label") or "").strip()
        add_operation(
            object_id=object_id,
            operation_kind="anchor_shared_thread",
            target_label=label,
            priority=0.66,
            operation_strength=_clamp01(
                0.42
                + relation_bias_strength * 0.22
                + trust_memory * 0.14
                + familiarity * 0.1
                - recent_strain * 0.1
            ),
            question_budget_cost=0,
            burden_risk=0.12,
            connection_support=_clamp01(0.62 + relation_bias_strength * 0.18 + attachment * 0.1),
            withheld=False,
            reason="Keep the shared thread between this turn and the broader relationship visible.",
            cues=("shared_thread", "continuity"),
        )

    for item in unfinished_objects:
        object_id = str(item.get("object_id") or "").strip()
        label = str(item.get("label") or "").strip()
        guarded_topics.append(label)
        add_operation(
            object_id=object_id,
            operation_kind="protect_unfinished_part",
            target_label=label,
            priority=0.74,
            operation_strength=_clamp01(0.68 + recent_strain * 0.16 + primary_defer_pressure * 0.1),
            question_budget_cost=0,
            burden_risk=0.1,
            connection_support=0.64,
            withheld=True,
            reason="Keep the unfinished part present without forcing it open in this turn.",
            cues=("unfinished_part", "protect"),
        )

    for item in theme_objects:
        if question_budget == 0 or next_step_room_level == "low":
            continue
        object_id = str(item.get("object_id") or "").strip()
        label = str(item.get("label") or "").strip()
        add_operation(
            object_id=object_id,
            operation_kind="anchor_next_step_in_theme",
            target_label=label,
            priority=0.58,
            operation_strength=_clamp01(0.44 + _level_to_score(next_step_room_level) * 0.26),
            question_budget_cost=0,
            burden_risk=0.18,
            connection_support=0.68,
            withheld=False,
            reason="Keep the next small move connected to the longer thread that is already present.",
            cues=("theme_anchor", "next_step"),
        )

    if detail_room_level == "low" or top_family in {"wait", "repair"} or deferred_ids:
        fallback_target = str(primary_object.get("label") or "").strip() if primary_object else "conversation"
        add_operation(
            object_id=str(primary_object.get("object_id") or "") if primary_object else "",
            operation_kind="keep_return_point",
            target_label=fallback_target,
            priority=0.68,
            operation_strength=0.62,
            question_budget_cost=0,
            burden_risk=0.1,
            connection_support=0.7,
            withheld=False,
            reason="Leave room so the other person can return to this later if they want to.",
            cues=("return_point",),
        )

    if question_budget == 0 and (relation_seed_summary or long_term_theme_summary or conscious_residue_summary):
        fallback_object_id = str(primary_object.get("object_id") or "") if primary_object else ""
        add_operation(
            object_id=fallback_object_id,
            operation_kind="preserve_continuity_without_probe",
            target_label=str(primary_object.get("label") or relation_seed_summary or long_term_theme_summary or "conversation").strip(),
            priority=0.6,
            operation_strength=_clamp01(0.46 + relation_bias_strength * 0.16 + _memory_strength(memory) * 0.12),
            question_budget_cost=0,
            burden_risk=0.1,
            connection_support=0.7,
            withheld=False,
            reason="Keep continuity visible even while avoiding further pressure in this turn.",
            cues=("continuity", "no_probe"),
        )

    cues = tuple(
        item
        for item in (
            f"question_budget:{question_budget}",
            f"question_room:{round(question_room, 2)}",
            f"top_family:{top_family}" if top_family else "",
            f"scene_family:{scene_family}" if scene_family else "",
            f"detail_room:{detail_room_level}",
            f"pressure:{pressure_sensitivity_level}",
            f"relation_bias:{round(relation_bias_strength, 2)}",
            f"recent_strain:{round(recent_strain, 2)}",
        )
        if item
    )
    return ObjectOperationPlan(
        primary_operation_id=primary_operation_id,
        operations=tuple(operations),
        guarded_topics=tuple(dict.fromkeys(item for item in guarded_topics if item)),
        question_budget=question_budget,
        question_pressure=_question_pressure(
            question_room=question_room,
            detail_room_level=detail_room_level,
            pressure_sensitivity_level=pressure_sensitivity_level,
            scene_norm=scene_norm,
            recent_strain=recent_strain,
        ),
        defer_dominance=_defer_dominance(tuple(operations)),
        cues=cues,
    )


def _question_pressure(
    *,
    question_room: float,
    detail_room_level: str,
    pressure_sensitivity_level: str,
    scene_norm: float,
    recent_strain: float,
) -> float:
    pressure = 1.0 - _clamp01(question_room)
    if str(detail_room_level or "").strip().lower() == "low":
        pressure += 0.08
    if str(pressure_sensitivity_level or "").strip().lower() == "high":
        pressure += 0.12
    pressure += _clamp01(scene_norm) * 0.1
    pressure += _clamp01(recent_strain) * 0.12
    return _clamp01(pressure)


def _defer_dominance(operations: Sequence[ObjectOperation]) -> float:
    if not operations:
        return 0.0
    total = sum(item.operation_strength for item in operations)
    if total <= 0.0:
        return 0.0
    defer_total = sum(
        item.operation_strength
        for item in operations
        if item.operation_kind
        in {
            "hold_without_probe",
            "defer_detail",
            "keep_return_point",
            "protect_unfinished_part",
            "preserve_continuity_without_probe",
        }
    )
    return _clamp01(defer_total / total)


def _question_room(
    *,
    detail_room_level: str,
    pressure_sensitivity_level: str,
    primary_depth_room: float,
    primary_touchability_score: float,
    scene_privacy: float,
    scene_norm: float,
    scene_safety: float,
    relation_bias_strength: float,
    trust_memory: float,
    familiarity: float,
    attachment: float,
    recent_strain: float,
    primary_defer_pressure: float,
) -> float:
    room = (
        _level_to_score(detail_room_level) * 0.28
        + _clamp01(primary_depth_room) * 0.18
        + _clamp01(primary_touchability_score) * 0.12
        + _clamp01(scene_privacy) * 0.12
        + _clamp01(scene_safety) * 0.1
        + _clamp01(relation_bias_strength) * 0.08
        + _clamp01(trust_memory) * 0.05
        + _clamp01(familiarity) * 0.04
        + _clamp01(attachment) * 0.03
    )
    room -= _level_to_score(pressure_sensitivity_level) * 0.18
    room -= _clamp01(scene_norm) * 0.14
    room -= _clamp01(recent_strain) * 0.16
    room -= _clamp01(primary_defer_pressure) * 0.12
    return _clamp01(room)


def _question_budget_from_room(
    *,
    question_room: float,
    reportability_limit: str,
    reportability_gate_mode: str,
    top_family: str,
) -> int:
    if reportability_limit == "withhold" or reportability_gate_mode == "withhold":
        return 0
    if question_room <= 0.34:
        return 0
    if question_room >= 0.72 and top_family in {"clarify", "attune", "co_move"}:
        return 2
    return 1


def _level_to_score(level: str) -> float:
    normalized = str(level or "").strip().lower()
    if normalized == "low":
        return 0.24
    if normalized == "high":
        return 0.78
    return 0.52


def _memory_strength(memory_context: Mapping[str, Any]) -> float:
    score = 0.0
    if str(memory_context.get("relation_seed_summary") or "").strip():
        score += 0.36
    if str(memory_context.get("long_term_theme_summary") or "").strip():
        score += 0.28
    if str(memory_context.get("conscious_residue_summary") or "").strip():
        score += 0.32
    if str(memory_context.get("memory_anchor") or "").strip():
        score += 0.08
    return _clamp01(score)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
