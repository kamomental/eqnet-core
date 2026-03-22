from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class InteractionEffect:
    effect_id: str
    effect_kind: str
    target_person: str = "other_person"
    target_label: str = ""
    intensity: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InteractionEffectsPlan:
    primary_effect_ids: tuple[str, ...] = ()
    effects: tuple[InteractionEffect, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_effect_ids": list(self.primary_effect_ids),
            "effects": [item.to_dict() for item in self.effects],
            "cues": list(self.cues),
        }


def derive_interaction_effects(
    *,
    conversational_objects: Mapping[str, Any] | None,
    object_operations: Mapping[str, Any] | None,
    resonance_evaluation: Mapping[str, Any] | None = None,
    constraint_field: Mapping[str, Any] | None = None,
) -> InteractionEffectsPlan:
    objects_state = dict(conversational_objects or {})
    operations_plan = dict(object_operations or {})
    resonance = dict(resonance_evaluation or {})
    constraint = dict(constraint_field or {})
    other_person_state = dict(resonance.get("estimated_other_person_state") or {})
    expected_effects = [str(item) for item in resonance.get("expected_effects") or [] if str(item).strip()]
    operations = [dict(item) for item in operations_plan.get("operations") or [] if isinstance(item, Mapping)]
    primary_effect_ids: list[str] = []
    effects: list[InteractionEffect] = []

    primary_label = ""
    primary_object_id = str(objects_state.get("primary_object_id") or "").strip()
    for item in objects_state.get("objects") or []:
        if isinstance(item, Mapping) and str(item.get("object_id") or "").strip() == primary_object_id:
            primary_label = str(item.get("label") or "").strip()
            break

    acknowledgement_need = _level_to_score(str(other_person_state.get("acknowledgement_need_level") or "medium"))
    pressure_sensitivity = _level_to_score(str(other_person_state.get("pressure_sensitivity_level") or "medium"))
    next_step_room = _level_to_score(str(other_person_state.get("next_step_room_level") or "medium"))
    boundary_pressure = _clamp01(float(constraint.get("boundary_pressure", 0.0) or 0.0))

    def add_effect(effect_kind: str, intensity: float, reason: str, *, target_label: str = "") -> None:
        effect_id = f"effect:{len(effects)}"
        effects.append(
            InteractionEffect(
                effect_id=effect_id,
                effect_kind=effect_kind,
                target_person="other_person",
                target_label=target_label or primary_label,
                intensity=_clamp01(intensity),
                reason=str(reason or "").strip(),
            )
        )
        primary_effect_ids.append(effect_id)

    if any(item.get("operation_kind") in {"acknowledge", "hold_without_probe"} for item in operations):
        add_effect(
            "feel_received",
            max(0.58, acknowledgement_need),
            "Increase the chance that the other person feels their difficulty was received as it is.",
        )
    if any(item.get("operation_kind") in {"hold_without_probe", "defer_detail", "keep_return_point"} for item in operations):
        add_effect(
            "preserve_self_pacing",
            max(0.6, pressure_sensitivity),
            "Help the other person keep the ability to choose their own pace.",
        )
    if any(item.get("operation_kind") in {"anchor_shared_thread", "preserve_continuity_without_probe"} for item in operations):
        add_effect(
            "preserve_continuity",
            0.68,
            "Help the other person feel that this turn still belongs to the same ongoing relationship and thread.",
        )
    if any(item.get("operation_kind") == "keep_return_point" for item in operations):
        add_effect(
            "keep_connection_open",
            0.62,
            "Leave room so the other person can come back later even if they stop now.",
        )
    if any(item.get("operation_kind") == "offer_small_next_step" for item in operations):
        add_effect(
            "enable_small_next_step",
            max(0.52, next_step_room),
            "Make only one manageable next step visible without increasing burden.",
        )
    if any(item.get("operation_kind") == "defer_detail" for item in operations) or boundary_pressure >= 0.66:
        add_effect(
            "protect_boundary",
            max(0.56, boundary_pressure),
            "Prevent the exchange from spreading into areas the other person is not ready to touch.",
        )
    if any(item.get("operation_kind") == "protect_unfinished_part" for item in operations):
        add_effect(
            "avoid_forced_reopening",
            max(0.58, pressure_sensitivity),
            "Avoid making the other person reopen an unfinished part before they are ready.",
        )
    if any(item.get("operation_kind") == "anchor_next_step_in_theme" for item in operations):
        add_effect(
            "keep_next_step_connected",
            max(0.54, next_step_room),
            "Make the next small move feel connected to the longer thread instead of abrupt.",
        )

    for effect_name in expected_effects:
        if effect_name == "lower_pressure" and not any(item.effect_kind == "reduce_pressure" for item in effects):
            add_effect(
                "reduce_pressure",
                max(0.58, pressure_sensitivity),
                "Reduce the chance that the other person feels rushed.",
            )
        elif effect_name == "increase_felt_understanding" and not any(item.effect_kind == "feel_received" for item in effects):
            add_effect(
                "feel_received",
                max(0.58, acknowledgement_need),
                "Increase the chance that the other person feels understood.",
            )
        elif effect_name == "preserve_talk_choice" and not any(item.effect_kind == "preserve_self_pacing" for item in effects):
            add_effect(
                "preserve_self_pacing",
                max(0.6, pressure_sensitivity),
                "Preserve the other person's sense that continuing to talk is their own choice.",
            )

    cues = tuple(
        item
        for item in (
            f"effect_count:{len(effects)}",
            f"ack_need:{str(other_person_state.get('acknowledgement_need_level') or 'medium')}",
            f"pressure:{str(other_person_state.get('pressure_sensitivity_level') or 'medium')}",
        )
        if item
    )
    return InteractionEffectsPlan(
        primary_effect_ids=tuple(dict.fromkeys(primary_effect_ids)),
        effects=tuple(effects),
        cues=cues,
    )


def _level_to_score(level: str) -> float:
    normalized = str(level or "").strip().lower()
    if normalized == "low":
        return 0.28
    if normalized == "high":
        return 0.78
    return 0.52


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
