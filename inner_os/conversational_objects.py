from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class ConversationalObject:
    object_id: str
    label: str
    object_kind: str = "topic"
    source: str = "implicit"
    emotional_tone: str = "neutral"
    touchability: str = "acknowledge_only"
    touchability_score: float = 0.0
    depth_limit: str = "shallow"
    depth_room: float = 0.0
    defer_pressure: float = 0.0
    salience: float = 0.0
    explicit_text: str = ""
    preferred_effects: tuple[str, ...] = ()
    blocked_moves: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationalObjectState:
    primary_object_id: str = ""
    objects: tuple[ConversationalObject, ...] = ()
    deferred_object_ids: tuple[str, ...] = ()
    active_labels: tuple[str, ...] = ()
    pressure_balance: float = 0.0
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_object_id": self.primary_object_id,
            "objects": [item.to_dict() for item in self.objects],
            "deferred_object_ids": list(self.deferred_object_ids),
            "active_labels": list(self.active_labels),
            "pressure_balance": self.pressure_balance,
            "cues": list(self.cues),
        }


def derive_conversational_objects(
    *,
    current_text: str = "",
    current_focus: str = "",
    reportable_facts: Sequence[str] = (),
    scene_state: Mapping[str, Any] | None = None,
    relation_context: Mapping[str, Any] | None = None,
    memory_context: Mapping[str, Any] | None = None,
    affect_blend_state: Mapping[str, Any] | None = None,
    constraint_field: Mapping[str, Any] | None = None,
    conscious_workspace: Mapping[str, Any] | None = None,
    resonance_evaluation: Mapping[str, Any] | None = None,
) -> ConversationalObjectState:
    scene = dict(scene_state or {})
    relation = dict(relation_context or {})
    memory = dict(memory_context or {})
    affect_blend = dict(affect_blend_state or {})
    constraint = dict(constraint_field or {})
    workspace = dict(conscious_workspace or {})
    resonance = dict(resonance_evaluation or {})
    other_person_state = dict(resonance.get("estimated_other_person_state") or {})
    expected_effects = tuple(
        str(item) for item in resonance.get("expected_effects") or [] if str(item).strip()
    )
    blocked_moves = tuple(
        str(item)
        for item in (
            tuple(str(item) for item in resonance.get("avoid_actions") or [] if str(item).strip())
            + tuple(str(item) for item in constraint.get("do_not_cross") or [] if str(item).strip())
        )
    )

    reportable_slice = [str(item) for item in workspace.get("reportable_slice") or [] if str(item).strip()]
    withheld_slice = [str(item) for item in workspace.get("withheld_slice") or [] if str(item).strip()]
    actionable_slice = [str(item) for item in workspace.get("actionable_slice") or [] if str(item).strip()]
    reportable_facts = [str(item) for item in reportable_facts if str(item).strip()]
    current_text = str(current_text or "").strip()
    current_focus = str(current_focus or "").strip()
    dominant_mode = str(affect_blend.get("dominant_mode") or "neutral").strip() or "neutral"
    detail_room_level = str(other_person_state.get("detail_room_level") or "medium").strip() or "medium"
    pressure_sensitivity_level = str(other_person_state.get("pressure_sensitivity_level") or "medium").strip() or "medium"
    next_step_room_level = str(other_person_state.get("next_step_room_level") or "medium").strip() or "medium"
    reportability_limit = str(constraint.get("reportability_limit") or "").strip()
    disclosure_limit = str(constraint.get("disclosure_limit") or "").strip()
    scene_family = str(scene.get("scene_family") or "").strip()
    scene_privacy = _clamp01(float(scene.get("privacy_level", 0.5) or 0.5))
    scene_norm = _clamp01(float(scene.get("norm_pressure", 0.0) or 0.0))
    scene_safety = _clamp01(float(scene.get("safety_margin", 0.5) or 0.5))
    scene_load = _clamp01(float(scene.get("environmental_load", 0.0) or 0.0))
    relation_bias_strength = _clamp01(float(relation.get("relation_bias_strength", 0.0) or 0.0))
    recent_strain = _clamp01(float(relation.get("recent_strain", 0.0) or 0.0))
    trust_memory = _clamp01(float(relation.get("trust_memory", 0.0) or 0.0))
    familiarity = _clamp01(float(relation.get("familiarity", 0.0) or 0.0))
    attachment = _clamp01(float(relation.get("attachment", 0.0) or 0.0))
    partner_timing_hint = str(relation.get("partner_timing_hint") or "").strip()
    partner_stance_hint = str(relation.get("partner_stance_hint") or "").strip()
    partner_social_interpretation = str(relation.get("partner_social_interpretation") or "").strip()
    relation_seed_summary = str(memory.get("relation_seed_summary") or "").strip()
    long_term_theme_summary = str(memory.get("long_term_theme_summary") or "").strip()
    conscious_residue_summary = str(memory.get("conscious_residue_summary") or "").strip()
    memory_anchor = str(memory.get("memory_anchor") or "").strip()
    guarding_pressure = _clamp01(
        scene_norm * 0.28
        + max(0.0, 0.52 - scene_safety) * 0.32
        + scene_load * 0.16
        + recent_strain * 0.24
        + (0.12 if partner_timing_hint == "delayed" else 0.0)
        + (0.1 if partner_stance_hint == "respectful" else 0.0)
    )
    relational_room = _clamp01(
        relation_bias_strength * 0.3
        + trust_memory * 0.24
        + familiarity * 0.18
        + attachment * 0.16
        + scene_privacy * 0.12
        - guarding_pressure * 0.22
    )
    memory_pull = _clamp01(
        (0.28 if relation_seed_summary else 0.0)
        + (0.22 if long_term_theme_summary else 0.0)
        + (0.26 if conscious_residue_summary else 0.0)
        + (0.08 if memory_anchor else 0.0)
    )

    objects: list[ConversationalObject] = []
    seen_labels: set[str] = set()
    deferred_ids: list[str] = []

    def add_object(
        *,
        label: str,
        source: str,
        touchability: str,
        touchability_score: float,
        depth_limit: str,
        depth_room: float,
        defer_pressure: float,
        salience: float,
        explicit_text: str = "",
        object_kind_override: str = "",
        preferred_effects_override: Sequence[str] = (),
        blocked_moves_override: Sequence[str] = (),
    ) -> str:
        clean_label = str(label or "").strip()
        if not clean_label or clean_label in seen_labels:
            return ""
        seen_labels.add(clean_label)
        object_id = f"object:{len(objects)}"
        cues = tuple(
            item
            for item in (
                f"source:{source}",
                f"touch:{touchability}",
                f"depth:{depth_limit}",
                f"scene:{scene_family}" if scene_family else "",
                f"blend:{dominant_mode}" if dominant_mode else "",
            )
            if item
        )
        objects.append(
            ConversationalObject(
                object_id=object_id,
                label=clean_label,
                object_kind=object_kind_override or _infer_object_kind(clean_label),
                source=source,
                emotional_tone=dominant_mode,
                touchability=touchability,
                touchability_score=_clamp01(touchability_score),
                depth_limit=depth_limit,
                depth_room=_clamp01(depth_room),
                defer_pressure=_clamp01(defer_pressure),
                salience=_clamp01(salience),
                explicit_text=str(explicit_text or "").strip(),
                preferred_effects=tuple(str(item) for item in preferred_effects_override if str(item).strip()) or expected_effects,
                blocked_moves=tuple(str(item) for item in blocked_moves_override if str(item).strip()) or blocked_moves,
                cues=cues,
            )
        )
        return object_id

    primary_label = (
        reportable_slice[0]
        if reportable_slice and not _is_internal_conversational_label(reportable_slice[0])
        else reportable_facts[0]
        if reportable_facts
        else current_focus
        if current_focus and current_focus != "ambient" and not _is_internal_conversational_label(current_focus)
        else current_text
    )
    primary_touchability = "acknowledge_only"
    primary_depth_limit = disclosure_limit or "shallow"
    primary_touchability_score = 0.38
    primary_depth_room = _depth_room_score(detail_room_level, disclosure_limit)
    primary_defer_pressure = _defer_pressure_score(
        reportability_limit=reportability_limit,
        detail_room_level=detail_room_level,
        pressure_sensitivity_level=pressure_sensitivity_level,
    )
    if reportability_limit == "withhold" or detail_room_level == "low":
        primary_touchability = "acknowledge_only"
        primary_depth_limit = "shallow"
        primary_touchability_score = 0.24
    elif next_step_room_level == "high" and actionable_slice:
        primary_touchability = "small_next_step_ready"
        primary_depth_limit = "bounded"
        primary_touchability_score = 0.68
    elif detail_room_level == "high":
        primary_touchability = "narrow_probe"
        primary_depth_limit = disclosure_limit or "bounded"
        primary_touchability_score = 0.74
    primary_touchability_score = _clamp01(
        primary_touchability_score
        + relational_room * 0.16
        - guarding_pressure * 0.18
        + (0.08 if scene_family == "attuned_presence" else 0.0)
        - (0.08 if scene_family == "guarded_boundary" else 0.0)
    )
    primary_depth_room = _clamp01(
        primary_depth_room
        + relational_room * 0.14
        - guarding_pressure * 0.18
        + (0.06 if detail_room_level == "high" and scene_privacy >= 0.6 else 0.0)
    )
    primary_defer_pressure = _clamp01(
        primary_defer_pressure
        + guarding_pressure * 0.18
        + recent_strain * 0.16
        - relational_room * 0.1
    )
    primary_object_id = add_object(
        label=primary_label or "current_topic",
        source="reportable" if reportable_slice or reportable_facts else "focus",
        touchability=primary_touchability,
        touchability_score=primary_touchability_score,
        depth_limit=primary_depth_limit,
        depth_room=primary_depth_room,
        defer_pressure=primary_defer_pressure,
        salience=0.82,
        explicit_text=current_text,
    )

    for item in withheld_slice:
        object_id = add_object(
            label=item,
            source="withheld",
            touchability="defer",
            touchability_score=0.12,
            depth_limit="hold",
            depth_room=0.08,
            defer_pressure=0.88,
            salience=0.56,
            blocked_moves_override=blocked_moves + ("probe_detail",),
        )
        if object_id:
            deferred_ids.append(object_id)

    for item in actionable_slice:
        add_object(
            label=item,
            source="actionable",
            touchability="act_gently",
            touchability_score=0.58 if next_step_room_level != "low" else 0.42,
            depth_limit="bounded",
            depth_room=_level_to_score(next_step_room_level),
            defer_pressure=0.22,
            salience=0.64,
            object_kind_override="next_step",
            preferred_effects_override=expected_effects + ("keep_room_for_next_step",),
        )

    if relation_seed_summary:
        add_object(
            label=relation_seed_summary,
            source="relation_memory",
            touchability="keep_visible",
            touchability_score=_clamp01(relational_room * 0.78 + memory_pull * 0.12),
            depth_limit="bounded" if relational_room >= 0.48 and guarding_pressure <= 0.52 else "shallow",
            depth_room=_clamp01(relational_room * 0.68 + scene_privacy * 0.12),
            defer_pressure=_clamp01(0.26 + guarding_pressure * 0.24 + recent_strain * 0.18),
            salience=_clamp01(0.4 + relation_bias_strength * 0.28 + trust_memory * 0.14),
            object_kind_override="shared_thread",
            preferred_effects_override=expected_effects + ("keep_connection_open",),
        )

    if long_term_theme_summary:
        add_object(
            label=long_term_theme_summary,
            source="theme_memory",
            touchability="bounded_theme_anchor",
            touchability_score=_clamp01(_level_to_score(next_step_room_level) * 0.52 + relational_room * 0.18),
            depth_limit="bounded",
            depth_room=_clamp01(_level_to_score(next_step_room_level) * 0.58 + scene_privacy * 0.08),
            defer_pressure=_clamp01(0.28 + guarding_pressure * 0.16),
            salience=_clamp01(0.34 + memory_pull * 0.26),
            object_kind_override="theme_anchor",
            preferred_effects_override=expected_effects + ("keep_next_turn_open",),
        )

    if conscious_residue_summary:
        residue_object_id = add_object(
            label=conscious_residue_summary,
            source="residue",
            touchability="defer",
            touchability_score=_clamp01(0.14 + relational_room * 0.06),
            depth_limit="hold",
            depth_room=_clamp01(0.08 + scene_privacy * 0.06),
            defer_pressure=_clamp01(0.72 + recent_strain * 0.18 + guarding_pressure * 0.12),
            salience=_clamp01(0.42 + recent_strain * 0.28 + memory_pull * 0.18),
            object_kind_override="unfinished_part",
            blocked_moves_override=blocked_moves + ("reopen_hidden_part_too_fast",),
        )
        if residue_object_id:
            deferred_ids.append(residue_object_id)

    if current_text and not objects:
        primary_object_id = add_object(
            label="current_topic",
            source="text",
            touchability="acknowledge_only",
            touchability_score=0.34,
            depth_limit="shallow",
            depth_room=primary_depth_room,
            defer_pressure=primary_defer_pressure,
            salience=0.5,
            explicit_text=current_text,
        )

    primary_object_id = _resolve_primary_object_id(
        objects=tuple(objects),
        fallback_object_id=primary_object_id,
        relation_bias_strength=relation_bias_strength,
        recent_strain=recent_strain,
        memory_pull=memory_pull,
    )

    cues = tuple(
        item
        for item in (
            f"detail_room:{detail_room_level}",
            f"next_step_room:{next_step_room_level}",
            f"reportability:{reportability_limit}" if reportability_limit else "",
            f"disclosure:{disclosure_limit}" if disclosure_limit else "",
            f"pressure:{pressure_sensitivity_level}",
            f"scene_family:{scene_family}" if scene_family else "",
            f"relation_room:{round(relational_room, 2)}",
            f"guarding_pressure:{round(guarding_pressure, 2)}",
            f"memory_pull:{round(memory_pull, 2)}",
            f"partner_social:{partner_social_interpretation}" if partner_social_interpretation else "",
            f"object_count:{len(objects)}",
        )
        if item
    )

    return ConversationalObjectState(
        primary_object_id=primary_object_id,
        objects=tuple(objects),
        deferred_object_ids=tuple(deferred_ids),
        active_labels=tuple(item.label for item in objects),
        pressure_balance=_pressure_balance(tuple(objects)),
        cues=cues,
    )


def _infer_object_kind(label: str) -> str:
    lowered = str(label or "").strip().lower()
    if not lowered:
        return "topic"
    if lowered.startswith("person:"):
        return "person"
    if any(token in lowered for token in ("next", "step", "future", "plan", "move")):
        return "next_step"
    if any(token in lowered for token in ("stress", "strain", "distress", "pain", "risk", "danger")):
        return "strain"
    if any(token in lowered for token in ("feeling", "mood", "care", "fear", "sad", "anger")):
        return "feeling"
    return "topic"


def _level_to_score(level: str) -> float:
    normalized = str(level or "").strip().lower()
    if normalized == "low":
        return 0.24
    if normalized == "high":
        return 0.78
    return 0.52


def _depth_room_score(detail_room_level: str, disclosure_limit: str) -> float:
    score = _level_to_score(detail_room_level)
    normalized_limit = str(disclosure_limit or "").strip().lower()
    if normalized_limit == "minimal":
        score -= 0.22
    elif normalized_limit == "light":
        score -= 0.1
    elif normalized_limit == "medium":
        score += 0.04
    return _clamp01(score)


def _defer_pressure_score(
    *,
    reportability_limit: str,
    detail_room_level: str,
    pressure_sensitivity_level: str,
) -> float:
    score = 0.22
    if str(reportability_limit or "").strip().lower() == "withhold":
        score += 0.34
    if str(detail_room_level or "").strip().lower() == "low":
        score += 0.2
    if str(pressure_sensitivity_level or "").strip().lower() == "high":
        score += 0.16
    return _clamp01(score)


def _pressure_balance(objects: Sequence[ConversationalObject]) -> float:
    if not objects:
        return 0.0
    total = sum(float(item.defer_pressure) for item in objects)
    return _clamp01(total / max(len(objects), 1))


def _resolve_primary_object_id(
    *,
    objects: Sequence[ConversationalObject],
    fallback_object_id: str,
    relation_bias_strength: float,
    recent_strain: float,
    memory_pull: float,
) -> str:
    if not objects:
        return fallback_object_id
    best_object = None
    best_score = -1.0
    for item in objects:
        score = float(item.salience)
        if item.object_kind == "shared_thread":
            score += relation_bias_strength * 0.18
        elif item.object_kind == "unfinished_part":
            score += recent_strain * 0.24 + float(item.defer_pressure) * 0.14
        elif item.object_kind == "theme_anchor":
            score += memory_pull * 0.12
        elif item.source in {"reportable", "focus"}:
            score += 0.08
        if score > best_score:
            best_score = score
            best_object = item
    return best_object.object_id if best_object is not None else fallback_object_id


def _is_internal_conversational_label(label: str) -> bool:
    normalized = str(label or "").strip().lower()
    return normalized in {
        "attune",
        "wait",
        "repair",
        "co_move",
        "contain",
        "reflect",
        "clarify",
        "withdraw",
        "ambient",
        "person",
        "social",
        "meaning",
        "place",
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
