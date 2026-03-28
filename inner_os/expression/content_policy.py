from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence
from emot_terrain_lab.i18n.locale import lookup_text, lookup_value, normalize_locale

from ..anchor_normalization import normalize_anchor_hint
from .interaction_constraints import (
    InteractionConstraints,
    coerce_interaction_constraints,
    derive_interaction_constraints,
)
from .repetition_guard import (
    RepetitionGuard,
    coerce_repetition_guard,
    derive_repetition_guard,
)
from .turn_delta import TurnDelta, coerce_turn_delta, derive_turn_delta


_EMERGENCY_RISK_HINTS = (
    "danger",
    "violence",
    "weapon",
    "sharp",
    "forced_entry",
    "intrusion",
    "attack",
    "unsafe_person",
    "coercion",
    "threat",
    "escalation",
    "boundary_break",
    "harassment",
    "pursuit",
    "emergency",
    "injury",
    "collapse",
    "trapped",
    "panic",
)


def derive_content_sequence(
    *,
    current_text: str,
    interaction_policy: Optional[Mapping[str, Any]] = None,
    conscious_access: Optional[Mapping[str, Any]] = None,
    history: Optional[Sequence[str]] = None,
    interaction_constraints: Optional[Mapping[str, Any]] = None,
    repetition_guard: Optional[Mapping[str, Any]] = None,
    turn_delta: Optional[Mapping[str, Any]] = None,
    locale: Optional[str] = None,
) -> list[dict[str, str]]:
    text = str(current_text or "").strip()
    if not text:
        return []

    packet = dict(interaction_policy or {})
    constraints_model = (
        coerce_interaction_constraints(interaction_constraints)
        if interaction_constraints is not None
        else derive_interaction_constraints(packet)
    )
    turn_delta_model = (
        coerce_turn_delta(turn_delta)
        if turn_delta is not None
        else derive_turn_delta(packet, interaction_constraints=constraints_model)
    )
    repetition_guard_model = (
        coerce_repetition_guard(repetition_guard)
        if repetition_guard is not None
        else derive_repetition_guard(history)
    )

    def finalize_sequence(
        sequence: list[dict[str, str]],
        **_: object,
    ) -> list[dict[str, str]]:
        return _finalize_content_sequence(
            sequence,
            interaction_constraints=constraints_model,
            repetition_guard=repetition_guard_model,
            turn_delta=turn_delta_model,
            locale=locale,
        )
    contract = dict(packet.get("conversation_contract") or {})
    strategy = str(packet.get("response_strategy") or "").strip()
    dialogue_act = str(packet.get("dialogue_act") or (conscious_access or {}).get("intent") or "").strip()
    opening_move = str(packet.get("opening_move") or "").strip()
    followup_move = str(packet.get("followup_move") or "").strip()
    closing_move = str(packet.get("closing_move") or "").strip()
    grice_guard = dict(packet.get("grice_guard_state") or {})
    grice_state = str(grice_guard.get("state") or "").strip()
    contract_operation = dict(contract.get("response_action_now") or {})
    primary_object_label = str(
        packet.get("primary_conversational_object_label")
        or contract.get("focus_now")
        or contract.get("primary_object")
        or ""
    ).strip()
    if not primary_object_label:
        selected_objects = [
            str(item).strip()
            for item in contract.get("selected_objects") or []
            if str(item).strip()
        ]
        primary_object_label = selected_objects[0] if selected_objects else ""
    primary_operation = dict(packet.get("primary_object_operation") or {})
    operation_kind = str(
        primary_operation.get("operation_kind")
        or contract_operation.get("primary_operation")
        or ""
    ).strip()
    target_label = str(
        primary_operation.get("target_label")
        or contract_operation.get("operation_target")
        or primary_object_label
        or ""
    ).strip()
    operation_kinds = {
        str(item)
        for item in packet.get("object_operation_kinds") or []
        if str(item).strip()
    }
    if not operation_kinds:
        operation_kinds = {
            str(item).strip()
            for item in contract_operation.get("ordered_operations") or []
            if str(item).strip()
        }
    ordered_operation_kinds = [
        str(item)
        for item in packet.get("ordered_operation_kinds") or []
        if str(item).strip()
    ]
    if not ordered_operation_kinds:
        ordered_operation_kinds = [
            str(item).strip()
            for item in contract_operation.get("ordered_operations") or []
            if str(item).strip()
        ]
    effect_kinds = {
        str(item)
        for item in packet.get("interaction_effect_kinds") or []
        if str(item).strip()
    }
    if not effect_kinds:
        effect_kinds = {
            str(item.get("effect") or "").strip()
            for item in contract.get("wanted_effect_on_other") or contract.get("intended_effects") or []
            if isinstance(item, Mapping) and str(item.get("effect") or "").strip()
        }
    ordered_effect_kinds = [
        str(item)
        for item in packet.get("ordered_effect_kinds") or []
        if str(item).strip()
    ]
    if not ordered_effect_kinds:
        ordered_effect_kinds = [
            str(item).strip()
            for item in contract.get("ordered_effects") or []
            if str(item).strip()
        ]
    deferred_object_labels = [
        str(item)
        for item in packet.get("deferred_object_labels") or []
        if str(item).strip()
    ]
    if not deferred_object_labels:
        deferred_object_labels = [
            str(item).strip()
            for item in (
                contract.get("leave_closed_for_now")
                or contract.get("do_not_open_yet")
                or contract.get("deferred_objects")
                or []
            )
            if str(item).strip()
        ]
    dialogue_order = [
        str(item)
        for item in packet.get("dialogue_order") or []
        if str(item).strip()
    ]
    question_budget = int(
        packet.get("effective_question_budget")
        or contract_operation.get("effective_question_budget")
        or packet.get("question_budget")
        or contract_operation.get("question_budget")
        or 0
    )
    opening_request = _looks_like_opening_request(text, locale=locale)
    emergency_posture = dict(packet.get("emergency_posture") or {})
    emergency_posture_name = str(
        emergency_posture.get("state")
        or packet.get("emergency_posture_name")
        or ""
    ).strip()
    emergency_posture_score = float(emergency_posture.get("score") or 0.0)
    emergency_dialogue_permission = str(
        emergency_posture.get("dialogue_permission")
        or packet.get("emergency_dialogue_permission")
        or ""
    ).strip()
    emergency_primary_action = str(
        emergency_posture.get("primary_action")
        or packet.get("emergency_primary_action")
        or ""
    ).strip()
    current_risk_tokens = [
        str(item).strip().lower()
        for item in packet.get("current_risks") or []
        if str(item).strip()
    ]
    situation_risk_state = dict(packet.get("situation_risk_state") or {})
    situation_risk_name = str(
        situation_risk_state.get("state")
        or packet.get("situation_risk_state_name")
        or ""
    ).strip()
    situation_context_affordance = str(
        situation_risk_state.get("context_affordance")
        or packet.get("situation_risk_context_affordance")
        or ""
    ).strip()
    emergency_dominant_inputs = [
        str(item).strip()
        for item in emergency_posture.get("dominant_inputs") or []
        if str(item).strip()
    ]
    target_phrase = _target_phrase(target_label or primary_object_label)
    emergency_sequence = _derive_emergency_content_sequence(
        emergency_posture_name=emergency_posture_name,
        emergency_posture_score=emergency_posture_score,
        emergency_dialogue_permission=emergency_dialogue_permission,
        emergency_primary_action=emergency_primary_action,
        situation_risk_name=situation_risk_name,
        situation_context_affordance=situation_context_affordance,
        emergency_dominant_inputs=emergency_dominant_inputs,
        current_risk_tokens=current_risk_tokens,
    )
    if emergency_sequence:
        return finalize_sequence(
            emergency_sequence,
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    deep_disclosure_sequence = _derive_deep_disclosure_sequence(
        text=text,
        interaction_policy=packet,
        turn_delta=turn_delta_model.to_dict(),
        locale=locale,
    )
    if deep_disclosure_sequence:
        return finalize_sequence(
            deep_disclosure_sequence,
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if grice_state == "hold_obvious_advice":
        return finalize_sequence(
            [
            _segment("hold_known_thread", "I do not need to restate the part that is already clear here."),
            _segment(
                "stay_with_present_need",
                f"I can stay with {target_phrase or 'what still needs care'} without pushing the obvious part any harder.",
            ),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if grice_state == "attune_without_repeating":
        return finalize_sequence(
            [
            _segment("visible_anchor", "I can stay with what is already clear here without repeating it."),
            _segment(
                "gentle_extension",
                f"If anything new needs care, I can move only as far as {target_phrase or 'the next living part'} asks.",
            ),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if grice_state == "acknowledge_then_extend" and strategy in {"shared_world_next_step", "attune_then_extend"}:
        return finalize_sequence(
            [
            _segment("known_anchor", "I can keep the thread that is already clear in view."),
            _segment(
                "small_extension",
                f"If I add anything, I want it to be only the small part that is new around {target_phrase or 'this'}.",
            ),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if (
        dialogue_act == "clarify"
        and strategy not in {"repair_then_attune", "respectful_wait"}
        and question_budget > 0
        and operation_kind not in {"hold_without_probe", "offer_small_next_step"}
    ):
        return finalize_sequence(
            [
            _segment("clarify_gate", "Let me check one small thing before I go further."),
            _segment("visible_anchor", "I can stay with what is visible first."),
            _segment("clarify_followup", "Then I can answer a little more cleanly from there."),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if (
        dialogue_act == "clarify"
        and strategy == "contain_then_stabilize"
        and (operation_kind == "hold_without_probe" or opening_request)
    ):
        return finalize_sequence(
            [
            _segment("respect_boundary", "We do not have to press this right now."),
            _segment(
                "offer_small_opening_line",
                "If you want a first line, even 'Something has been catching on me lately, and I want help looking at it' is enough.",
            ),
            _segment(
                "quiet_presence",
                "I can stay nearby without leaning on it, and come back when it feels easier.",
            ),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    move_sequence = _derive_move_driven_sequence(
        opening_move=opening_move,
        followup_move=followup_move,
        closing_move=closing_move,
        target_label=target_label,
        operation_kind=operation_kind,
        operation_kinds=operation_kinds,
        ordered_operation_kinds=ordered_operation_kinds,
        effect_kinds=effect_kinds,
        ordered_effect_kinds=ordered_effect_kinds,
        deferred_object_labels=deferred_object_labels,
        dialogue_order=dialogue_order,
        question_budget=question_budget,
        opening_request=opening_request,
    )
    if move_sequence:
        return finalize_sequence(
            move_sequence,
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )

    if operation_kind:
        operation_sequence = _derive_operation_driven_sequence(
            operation_kind=operation_kind,
            target_label=target_label,
            operation_kinds=operation_kinds,
            ordered_operation_kinds=ordered_operation_kinds,
            effect_kinds=effect_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            question_budget=question_budget,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
            opening_request=opening_request,
        )
        if operation_sequence:
            return finalize_sequence(
                operation_sequence,
                interaction_constraints=constraints_model,
                turn_delta=turn_delta_model,
            )

    if strategy == "repair_then_attune":
        return finalize_sequence(
            [
            _segment("acknowledge_overreach", "I came in too fast there."),
            _segment("visible_anchor", "Let me slow this down and stay only with what is actually clear right now."),
            _segment("careful_reopen", "You do not have to carry the rest until it feels easier to reopen."),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if strategy == "respectful_wait":
        return finalize_sequence(
            [
            _segment("respect_boundary", "We do not have to press this right now."),
            _segment("quiet_presence", "I can stay nearby without leaning on it, and come back when it feels easier."),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if strategy == "shared_world_next_step":
        return finalize_sequence(
            [
            _segment("shared_anchor", "Here is the next step I can see from what is already clear."),
            _segment("pace_match", "I can keep that next move small enough for us to stay in step."),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if strategy == "contain_then_stabilize":
        if dialogue_act == "clarify" or opening_request:
            return finalize_sequence(
                [
                _segment("stabilize_boundary", "I can keep this inside what feels steady first."),
                _segment(
                    "offer_small_opening_line",
                    "If you want a first line, even 'Something has been catching on me lately, and I want help looking at it' is enough.",
                ),
                _segment("do_not_overread", "I do not want to read past what the scene can actually support."),
                ],
                interaction_constraints=constraints_model,
                turn_delta=turn_delta_model,
            )
        return finalize_sequence(
            [
            _segment("stabilize_boundary", "I can keep this inside what feels steady first."),
            _segment("do_not_overread", "I do not want to read past what the scene can actually support."),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if strategy == "reflect_without_settling":
        return finalize_sequence(
            [
            _segment("hold_meaning_open", "I can stay with what is here without settling the meaning too fast."),
            _segment("slow_reflection", "If we go further, I would rather keep the meaning open for a while."),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if strategy == "attune_then_extend":
        return finalize_sequence(
            [
            _segment("visible_anchor", "I'm here with you. We can start with what feels most visible."),
            _segment("gentle_extension", "You do not have to rush it; if it helps, we can move a little closer to what matters here."),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )

    if dialogue_act == "check_in" and opening_move == "stay_with_visible":
        return finalize_sequence(
            [
            _segment("visible_anchor", "I can stay with what is visible first."),
            _segment("gentle_extension", "Then I can go a little further if that helps."),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    if dialogue_act == "clarify" and followup_move == "invite_visible_state":
        return finalize_sequence(
            [
            _segment("clarify_anchor", "I can check what is visible first."),
            _segment("clarify_followup", "Then I can answer a little more cleanly from there."),
            ],
            interaction_constraints=constraints_model,
            turn_delta=turn_delta_model,
        )
    return finalize_sequence(
        [_segment("carry_text", text)],
        interaction_constraints=constraints_model,
        turn_delta=turn_delta_model,
    )


def derive_content_skeleton(
    *,
    current_text: str,
    interaction_policy: Optional[Mapping[str, Any]] = None,
    conscious_access: Optional[Mapping[str, Any]] = None,
    history: Optional[Sequence[str]] = None,
    interaction_constraints: Optional[Mapping[str, Any]] = None,
    repetition_guard: Optional[Mapping[str, Any]] = None,
    turn_delta: Optional[Mapping[str, Any]] = None,
) -> str:
    return render_content_sequence(
        derive_content_sequence(
            current_text=current_text,
            interaction_policy=interaction_policy,
            conscious_access=conscious_access,
            history=history,
            interaction_constraints=interaction_constraints,
            repetition_guard=repetition_guard,
            turn_delta=turn_delta,
        )
    )


def render_content_sequence(sequence: list[dict[str, str]]) -> str:
    return _join_lines(*(segment.get("text", "") for segment in sequence))


def localize_content_sequence(
    sequence: list[dict[str, str]],
    *,
    locale: Optional[str] = None,
) -> list[dict[str, str]]:
    return _localize_content_sequence(sequence, locale=locale)


def _segment(act: str, text: str) -> dict[str, str]:
    return {"act": act, "text": text}


def _derive_emergency_content_sequence(
    *,
    emergency_posture_name: str,
    emergency_posture_score: float,
    emergency_dialogue_permission: str,
    emergency_primary_action: str,
    situation_risk_name: str,
    situation_context_affordance: str,
    emergency_dominant_inputs: Sequence[str],
    current_risk_tokens: Sequence[str],
) -> list[dict[str, str]]:
    posture = str(emergency_posture_name or "").strip()
    posture_score = max(0.0, min(1.0, float(emergency_posture_score or 0.0)))
    permission = str(emergency_dialogue_permission or "").strip()
    primary_action = str(emergency_primary_action or "").strip()
    risk_name = str(situation_risk_name or "").strip()
    context_affordance = str(situation_context_affordance or "").strip()
    dominant_inputs = {
        str(item).strip()
        for item in emergency_dominant_inputs or []
        if str(item).strip()
    }
    concrete_risk_evidence = any(
        any(fragment in token for fragment in _EMERGENCY_RISK_HINTS)
        for token in current_risk_tokens or []
    )
    contextual_emergency_evidence = context_affordance == "shelter_breach"
    emergency_active = (
        posture_score >= 0.42
        or risk_name in {"acute_threat", "emergency", "unstable_contact"}
        or bool(
            dominant_inputs
            & {
                "acute_threat",
                "emergency_risk",
                "intrusion_signal",
                "relation_break",
            }
        )
        or primary_action in {"exit_space", "seek_help", "protect_immediately"}
    )
    if not emergency_active or not (concrete_risk_evidence or contextual_emergency_evidence):
        return []
    if posture in {"", "observe"} and permission not in {"boundary_only", "avoid_dialogue"}:
        return []

    if permission == "avoid_dialogue":
        if primary_action == "seek_help":
            return [
                _segment(
                    "emergency_seek_help_now",
                    "Safety comes first here. Please move toward help and make the risk visible right away.",
                )
            ]
        if primary_action == "protect_immediately":
            return [
                _segment(
                    "emergency_protect_now",
                    "Do not keep talking through this. Protect yourself first, create distance, and call for help now.",
                )
            ]
        return [
            _segment(
                "emergency_exit_now",
                "Do not keep negotiating this. Leave the space and prioritize safety first.",
            )
        ]

    if permission == "boundary_only":
        if primary_action == "create_distance":
            return [
                _segment(
                    "emergency_create_distance",
                    "I am not continuing this at close range. I am taking distance now.",
                )
            ]
        return [
            _segment(
                "emergency_deescalate_boundary",
                "I am stopping here. Step back and keep this brief.",
            )
        ]

    return []


def _derive_operation_driven_sequence(
    *,
    operation_kind: str,
    target_label: str,
    operation_kinds: set[str],
    ordered_operation_kinds: list[str],
    effect_kinds: set[str],
    ordered_effect_kinds: list[str],
    deferred_object_labels: list[str],
    dialogue_order: list[str],
    question_budget: int,
    opening_move: str,
    followup_move: str,
    closing_move: str,
    opening_request: bool,
) -> list[dict[str, str]]:
    target_phrase = _target_phrase(target_label)
    deferred_phrase = _target_phrase(deferred_object_labels[0]) if deferred_object_labels else ""
    if operation_kind == "hold_without_probe":
        if opening_move == "name_overreach_and_reduce_force":
            sequence = [
                _segment("acknowledge_overreach", "I came in too fast there."),
                _segment("visible_anchor", "Let me slow this down and stay only with what is actually clear right now."),
                _segment("careful_reopen", "You do not have to carry the rest until it feels easier to reopen."),
            ]
            return _append_dialogue_order_supporting_lines(
                sequence,
                operation_kind=operation_kind,
                ordered_operation_kinds=ordered_operation_kinds,
                ordered_effect_kinds=ordered_effect_kinds,
                deferred_object_labels=deferred_object_labels,
                dialogue_order=dialogue_order,
                opening_move=opening_move,
                followup_move=followup_move,
                closing_move=closing_move,
            )
        if opening_move == "acknowledge_without_probe":
            sequence = [
                _segment("respect_boundary", "We do not have to press this right now."),
            ]
            if opening_request:
                sequence.append(
                    _segment(
                        "offer_small_opening_line",
                        "If you want a first line, even 'Something has been catching on me lately, and I want help looking at it' is enough.",
                    )
                )
            sequence.append(
                _segment("quiet_presence", "I can stay nearby without leaning on it, and come back when it feels easier.")
            )
            if closing_move == "leave_unfinished_part_closed_for_now" or "protect_unfinished_part" in operation_kinds:
                sequence.append(
                    _segment("leave_unfinished_closed", "We can leave the unfinished part where it is for now, and come back only if it feels easier later.")
                )
            return _append_dialogue_order_supporting_lines(
                sequence,
                operation_kind=operation_kind,
                ordered_operation_kinds=ordered_operation_kinds,
                ordered_effect_kinds=ordered_effect_kinds,
                deferred_object_labels=deferred_object_labels,
                dialogue_order=dialogue_order,
                opening_move=opening_move,
                followup_move=followup_move,
                closing_move=closing_move,
            )
        sequence = [
            _segment("acknowledge_without_probe", f"I can stay with {target_phrase} without asking you to unpack it right now."),
        ]
        if deferred_phrase:
            sequence.append(
                _segment("leave_unfinished_closed", f"We can leave {deferred_phrase} where it is for now, and come back only if it feels easier later.")
            )
        elif "anchor_shared_thread" in operation_kinds:
            sequence.append(
                _segment("keep_shared_thread_visible", "I can keep the thread between us visible while leaving the rest untouched.")
            )
        else:
            sequence.append(
                _segment("protect_talking_room", f"You do not have to say more about {target_phrase} than feels manageable.")
            )
        if closing_move == "leave_unfinished_part_closed_for_now" or "protect_unfinished_part" in operation_kinds:
            sequence.append(
                _segment("leave_unfinished_closed", "We can leave the unfinished part where it is for now, and come back only if it feels easier later.")
            )
        elif "keep_return_point" in operation_kinds or "preserve_self_pacing" in effect_kinds:
            sequence.append(
                _segment("leave_return_point", f"We can come back to {target_phrase} when it feels easier, and only if you want to.")
            )
        return _append_dialogue_order_supporting_lines(
            sequence,
            operation_kind=operation_kind,
            ordered_operation_kinds=ordered_operation_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
        )

    if operation_kind == "narrow_clarify":
        sequence = [
            _segment("clarify_gate", "Let me check one small thing before I go further."),
            _segment("visible_anchor", f"I can stay with {target_phrase} first."),
            _segment(
                "clarify_followup",
                f"Then I can ask about only the part of {target_phrase} that feels easiest to name."
                if question_budget > 0
                else "Then I can answer a little more cleanly from there.",
            ),
        ]
        return _append_dialogue_order_supporting_lines(
            sequence,
            operation_kind=operation_kind,
            ordered_operation_kinds=ordered_operation_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
        )

    if operation_kind == "offer_small_next_step":
        sequence = [
            _segment("shared_anchor", f"Here is one next step I can see from {target_phrase} as it is now."),
            _segment("pace_match", "I can keep that next move small enough for us to stay in step."),
        ]
        if "keep_next_step_connected" in effect_kinds or "anchor_next_step_in_theme" in operation_kinds:
            sequence.append(
                _segment("keep_choice_with_other_person", "That next move can stay connected to the longer thread that is already here.")
            )
        return _append_dialogue_order_supporting_lines(
            sequence,
            operation_kind=operation_kind,
            ordered_operation_kinds=ordered_operation_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
        )

    if operation_kind == "acknowledge":
        if opening_move == "acknowledge_named_state" and followup_move in {"invite_visible_state", "keep_shared_thread_visible"}:
            sequence = [
                _segment("visible_anchor", "I'm here with you. We can start with what feels most visible."),
            ]
            if followup_move == "keep_shared_thread_visible" or "anchor_shared_thread" in operation_kinds:
                sequence.append(
                    _segment("gentle_extension", "I do not want to lose the thread that is already here between us while we stay with it.")
                )
            else:
                sequence.append(
                    _segment("gentle_extension", "You do not have to rush it; if it helps, we can move a little closer to what matters here.")
                )
            return _append_dialogue_order_supporting_lines(
                sequence,
                operation_kind=operation_kind,
                ordered_operation_kinds=ordered_operation_kinds,
                ordered_effect_kinds=ordered_effect_kinds,
                deferred_object_labels=deferred_object_labels,
                dialogue_order=dialogue_order,
                opening_move=opening_move,
                followup_move=followup_move,
                closing_move=closing_move,
            )
        sequence = [
            _segment("acknowledge_named_state", f"I can start by taking in {target_phrase} as it is."),
        ]
        if followup_move == "keep_shared_thread_visible" or "anchor_shared_thread" in operation_kinds:
            sequence.append(
                _segment("keep_shared_thread_visible", "I do not want to treat this as a separate moment from the thread that is already between us.")
            )
        elif followup_move == "protect_talking_room":
            sequence.append(
                _segment("protect_talking_room", "You do not have to say more than you have room to say right now.")
            )
        else:
            sequence.append(
                _segment("invite_visible_state", f"I can stay close to {target_phrase} before going any further.")
            )
        if "preserve_self_pacing" in effect_kinds:
            sequence.append(
                _segment("keep_choice_with_other_person", f"You can choose whether to keep talking about {target_phrase} or leave it here for the moment.")
            )
        return _append_dialogue_order_supporting_lines(
            sequence,
            operation_kind=operation_kind,
            ordered_operation_kinds=ordered_operation_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
        )

    return []


def _derive_move_driven_sequence(
    *,
    opening_move: str,
    followup_move: str,
    closing_move: str,
    target_label: str,
    operation_kind: str,
    operation_kinds: set[str],
    ordered_operation_kinds: list[str],
    effect_kinds: set[str],
    ordered_effect_kinds: list[str],
    deferred_object_labels: list[str],
    dialogue_order: list[str],
    question_budget: int,
    opening_request: bool,
) -> list[dict[str, str]]:
    target_phrase = _target_phrase(target_label)
    if opening_move == "name_overreach_and_reduce_force":
        sequence = [
            _segment("acknowledge_overreach", "I came in too fast there."),
            _segment("visible_anchor", "Let me slow this down and stay only with what is actually clear right now."),
            _segment("careful_reopen", "You do not have to carry the rest until it feels easier to reopen."),
        ]
        return _append_dialogue_order_supporting_lines(
            sequence,
            operation_kind=operation_kind,
            ordered_operation_kinds=ordered_operation_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
        )
    if opening_move == "acknowledge_without_probe":
        sequence = [
            _segment("respect_boundary", "We do not have to press this right now."),
        ]
        if opening_request:
            sequence.append(
                _segment(
                    "offer_small_opening_line",
                    "If you want a first line, even 'Something has been catching on me lately, and I want help looking at it' is enough.",
                )
            )
        sequence.append(
            _segment("quiet_presence", "I can stay nearby without leaning on it, and come back when it feels easier.")
        )
        if closing_move == "leave_unfinished_part_closed_for_now" or "protect_unfinished_part" in operation_kinds:
            sequence.append(
                _segment("leave_unfinished_closed", "We can leave the unfinished part where it is for now, and come back only if it feels easier later.")
            )
        return _append_dialogue_order_supporting_lines(
            sequence,
            operation_kind=operation_kind,
            ordered_operation_kinds=ordered_operation_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
        )
    if opening_move == "narrow_scope_first" or followup_move == "ask_one_bounded_part":
        sequence = [
            _segment("clarify_gate", "Let me check one small thing before I go further."),
            _segment("visible_anchor", f"I can stay with {target_phrase} first."),
            _segment(
                "clarify_followup",
                "Then I can ask about only the part that feels easiest to name."
                if question_budget > 0
                else "Then I can answer a little more cleanly from there.",
            ),
        ]
        return _append_dialogue_order_supporting_lines(
            sequence,
            operation_kind=operation_kind,
            ordered_operation_kinds=ordered_operation_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
        )
    if opening_move in {"anchor_visible_part", "anchor_shared_thread"} or followup_move == "offer_one_small_next_step":
        sequence = [
            _segment("shared_anchor", "Here is the next step I can see from what is already clear."),
            _segment("pace_match", "I can keep that next move small enough for us to stay in step."),
        ]
        return _append_dialogue_order_supporting_lines(
            sequence,
            operation_kind=operation_kind,
            ordered_operation_kinds=ordered_operation_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
        )
    if opening_move == "acknowledge_named_state":
        sequence = [
            _segment("visible_anchor", "I'm here with you. We can start with what feels most visible."),
            _segment("gentle_extension", "You do not have to rush it; if it helps, we can move a little closer to what matters here."),
        ]
        return _append_dialogue_order_supporting_lines(
            sequence,
            operation_kind=operation_kind,
            ordered_operation_kinds=ordered_operation_kinds,
            ordered_effect_kinds=ordered_effect_kinds,
            deferred_object_labels=deferred_object_labels,
            dialogue_order=dialogue_order,
            opening_move=opening_move,
            followup_move=followup_move,
            closing_move=closing_move,
        )
    return []


def _append_dialogue_order_supporting_lines(
    sequence: list[dict[str, str]],
    *,
    operation_kind: str,
    ordered_operation_kinds: list[str],
    ordered_effect_kinds: list[str],
    deferred_object_labels: list[str],
    dialogue_order: list[str],
    opening_move: str,
    followup_move: str,
    closing_move: str,
) -> list[dict[str, str]]:
    existing_acts = {segment.get("act", "") for segment in sequence}
    existing_texts = {segment.get("text", "") for segment in sequence}
    max_segments = 3 if len(sequence) >= 3 else 4
    for item in dialogue_order:
        if len(sequence) >= max_segments:
            break
        prefix, _, value = item.partition(":")
        value = value.strip()
        if not value:
            continue
        if prefix == "operate":
            if value == operation_kind:
                continue
            segment = _supporting_operation_segment(
                value,
                ordered_operation_kinds=ordered_operation_kinds,
                deferred_object_labels=deferred_object_labels,
            )
        elif prefix == "effect":
            segment = _supporting_effect_segment(value)
        elif prefix == "follow":
            segment = _supporting_followup_segment(value)
        elif prefix == "defer":
            segment = _supporting_defer_segment(deferred_object_labels)
        elif prefix == "close":
            segment = _supporting_close_segment(value)
        else:
            segment = None
        if not segment:
            continue
        if segment["act"] in existing_acts or segment["text"] in existing_texts:
            continue
        sequence.append(segment)
        existing_acts.add(segment["act"])
        existing_texts.add(segment["text"])

    for value in ordered_effect_kinds:
        if len(sequence) >= max_segments:
            break
        segment = _supporting_effect_segment(value)
        if not segment:
            continue
        if segment["act"] in existing_acts or segment["text"] in existing_texts:
            continue
        sequence.append(segment)
        existing_acts.add(segment["act"])
        existing_texts.add(segment["text"])

    for value in ordered_operation_kinds:
        if len(sequence) >= max_segments:
            break
        if value == operation_kind:
            continue
        segment = _supporting_operation_segment(
            value,
            ordered_operation_kinds=ordered_operation_kinds,
            deferred_object_labels=deferred_object_labels,
        )
        if not segment:
            continue
        if segment["act"] in existing_acts or segment["text"] in existing_texts:
            continue
        sequence.append(segment)
        existing_acts.add(segment["act"])
        existing_texts.add(segment["text"])

    if deferred_object_labels and len(sequence) < max_segments:
        segment = _supporting_defer_segment(deferred_object_labels)
        if segment and segment["act"] not in existing_acts and segment["text"] not in existing_texts:
            sequence.append(segment)
            existing_acts.add(segment["act"])
            existing_texts.add(segment["text"])

    if followup_move and len(sequence) < max_segments:
        segment = _supporting_followup_segment(followup_move)
        if segment and segment["act"] not in existing_acts and segment["text"] not in existing_texts:
            sequence.append(segment)
            existing_acts.add(segment["act"])
            existing_texts.add(segment["text"])

    if closing_move and len(sequence) < max_segments:
        segment = _supporting_close_segment(closing_move)
        if segment and segment["act"] not in existing_acts and segment["text"] not in existing_texts:
            sequence.append(segment)

    return sequence


def _supporting_operation_segment(
    operation_kind: str,
    *,
    ordered_operation_kinds: list[str],
    deferred_object_labels: list[str],
) -> dict[str, str] | None:
    if operation_kind == "anchor_shared_thread":
        return _segment("keep_shared_thread_visible", "I do not want to lose the thread that is already here between us while we stay with it.")
    if operation_kind in {"protect_unfinished_part", "defer_detail"}:
        return _segment("leave_unfinished_closed", "We can leave the unfinished part where it is for now, and come back only if it feels easier later.")
    if operation_kind == "keep_return_point":
        return _segment("leave_return_point", "We can come back to the rest when it feels easier, and only if you want to.")
    if operation_kind == "anchor_next_step_in_theme":
        return _segment("keep_choice_with_other_person", "That next move can stay connected to the longer thread that is already here.")
    if operation_kind == "preserve_continuity_without_probe":
        return _segment("keep_shared_thread_visible", "I can keep the thread between us visible while leaving the rest untouched.")
    return None


def _supporting_effect_segment(effect_kind: str) -> dict[str, str] | None:
    if effect_kind == "preserve_self_pacing":
        return _segment("keep_choice_with_other_person", "You can choose whether to keep talking or leave it here for the moment.")
    if effect_kind == "keep_connection_open":
        return _segment("leave_return_point", "We can come back to the rest when it feels easier, and only if you want to.")
    if effect_kind == "preserve_continuity":
        return _segment("keep_shared_thread_visible", "I do not want to treat this as separate from the thread that is already here between us.")
    if effect_kind == "avoid_forced_reopening":
        return _segment("leave_unfinished_closed", "We do not need to open the rest right now.")
    if effect_kind == "protect_boundary":
        return _segment("respect_boundary", "I do not want to push this past what feels steady right now.")
    if effect_kind == "keep_next_step_connected":
        return _segment("keep_choice_with_other_person", "That next move can stay connected to the longer thread that is already here.")
    if effect_kind == "reduce_pressure":
        return _segment("protect_talking_room", "You do not have to explain more than feels manageable.")
    return None


def _supporting_followup_segment(followup_move: str) -> dict[str, str] | None:
    if followup_move == "protect_talking_room":
        return _segment("protect_talking_room", "You do not have to explain more than feels manageable.")
    if followup_move == "keep_shared_thread_visible":
        return _segment("keep_shared_thread_visible", "I do not want to lose the thread that is already here between us while we stay with it.")
    if followup_move == "ask_one_bounded_part":
        return _segment("clarify_followup", "Then I can ask about only the part that feels easiest to name.")
    if followup_move == "offer_one_small_next_step":
        return _segment("pace_match", "I can keep that next move small enough for us to stay in step.")
    if followup_move == "invite_visible_state":
        return _segment("invite_visible_state", "I can stay close to the part that is already visible before going any further.")
    return None


def _supporting_defer_segment(deferred_object_labels: list[str]) -> dict[str, str] | None:
    if not deferred_object_labels:
        return None
    return _segment("leave_unfinished_closed", "We can leave the unfinished part where it is for now, and come back only if it feels easier later.")


def _supporting_close_segment(closing_move: str) -> dict[str, str] | None:
    if closing_move in {"leave_return_point", "leave_space"}:
        return _segment("leave_return_point", "We can come back to the rest when it feels easier, and only if you want to.")
    if closing_move == "leave_unfinished_part_closed_for_now":
        return _segment("leave_unfinished_closed", "We do not need to open the rest right now.")
    if closing_move == "keep_choice_with_other_person":
        return _segment("keep_choice_with_other_person", "You can choose whether to keep talking or leave it here for the moment.")
    if closing_move == "keep_pace_mutual":
        return _segment("keep_choice_with_other_person", "We can keep the next move connected from here without rushing it.")
    return None


def _target_phrase(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return "what is here"
    if text.lower() in {"social", "person", "ambient", "meaning", "place"}:
        return "what is here right now"
    return text


def _join_lines(*parts: str) -> str:
    values = [str(part).strip() for part in parts if str(part).strip()]
    return " ".join(values)


def _finalize_content_sequence(
    sequence: list[dict[str, str]],
    *,
    interaction_constraints: InteractionConstraints,
    repetition_guard: RepetitionGuard,
    turn_delta: TurnDelta,
    locale: Optional[str] = None,
) -> list[dict[str, str]]:
    candidates = [dict(item) for item in sequence if str(item.get("text") or "").strip()]
    updated: list[dict[str, str]] = []
    suppressed_candidates: list[dict[str, str]] = []
    for item in candidates:
        act = str(item.get("act") or "").strip()
        text = str(item.get("text") or "")
        localized_text = _localized_segment_text(act, text, locale=normalize_locale(locale or "en"))
        blocked = repetition_guard.blocks_text(text) or (
            localized_text != text and repetition_guard.blocks_text(localized_text)
        )
        if blocked:
            alternate = _alternate_repetition_segment(act)
            if alternate is not None:
                alternate_act = str(alternate.get("act") or "").strip()
                alternate_text = str(alternate.get("text") or "")
                localized_alternate_text = _localized_segment_text(
                    alternate_act,
                    alternate_text,
                    locale=normalize_locale(locale or "en"),
                )
                alternate_blocked = repetition_guard.blocks_text(alternate_text) or (
                    localized_alternate_text != alternate_text
                    and repetition_guard.blocks_text(localized_alternate_text)
                )
                if not alternate_blocked:
                    updated.append(dict(alternate))
                    continue
            suppressed_candidates.append(item)
            continue
        updated.append(item)
    if not updated and suppressed_candidates:
        updated.append(suppressed_candidates[0])
    recent_move_acts = _recent_move_acts(repetition_guard, locale=locale)
    updated = _swap_repeated_moves(
        updated,
        recent_move_acts=recent_move_acts,
        repetition_guard=repetition_guard,
        locale=locale,
    )
    existing_acts = {str(item.get("act") or "").strip() for item in updated}
    max_segments = 4 if len(updated) < 4 else len(updated)
    deep_reflection_acts = {
        "reflect_hidden_need",
        "reflect_self_blame",
        "reflect_fear_of_being_seen",
        "reflect_unspoken_weight",
    }
    suppress_delta_for_green_hold = (
        str(turn_delta.kind or "").strip() == "green_reflection_hold"
        and any(act in existing_acts for act in deep_reflection_acts)
    )

    delta_segment = None if suppress_delta_for_green_hold else _delta_segment(turn_delta, locale=locale)
    if (
        delta_segment
        and len(updated) < max_segments
        and delta_segment["act"] not in existing_acts
    ):
        if not repetition_guard.blocks_text(delta_segment["text"]):
            updated.append(delta_segment)
            existing_acts.add(delta_segment["act"])
        else:
            alternate_delta_segment = _alternate_delta_segment(
                turn_delta,
                locale=locale,
            )
            if (
                alternate_delta_segment
                and alternate_delta_segment["act"] not in existing_acts
                and not repetition_guard.blocks_text(alternate_delta_segment["text"])
            ):
                updated.append(alternate_delta_segment)
                existing_acts.add(alternate_delta_segment["act"])

    if (
        interaction_constraints.avoid_overclosure
        and not interaction_constraints.allow_small_next_step
        and "leave_return_point" not in existing_acts
        and any(act in existing_acts for act in {"shared_anchor", "pace_match", "gentle_extension"})
        and len(updated) < max_segments
    ):
        return_point_segment = _supporting_close_segment("leave_return_point")
        if return_point_segment and not repetition_guard.blocks_text(return_point_segment["text"]):
            updated.append(return_point_segment)

    return _localize_content_sequence(updated, locale=locale)


def _swap_repeated_moves(
    sequence: list[dict[str, str]],
    *,
    recent_move_acts: set[str],
    repetition_guard: RepetitionGuard,
    locale: Optional[str] = None,
) -> list[dict[str, str]]:
    swapped: list[dict[str, str]] = []
    for item in sequence:
        act = str(item.get("act") or "").strip()
        alternate = _alternate_repetition_segment(act)
        if act in recent_move_acts and alternate is not None:
            localized_alternate = _localize_content_sequence(
                [dict(alternate)],
                locale=locale,
            )[0]
            if not repetition_guard.blocks_text(str(localized_alternate.get("text") or "")):
                swapped.append(localized_alternate)
                continue
        swapped.append(item)
    return swapped


def _recent_move_acts(
    repetition_guard: RepetitionGuard,
    *,
    locale: Optional[str] = None,
) -> set[str]:
    normalized_locale = normalize_locale(locale or "en")
    recent_acts: set[str] = set()
    signatures = tuple(
        _normalize_signature(item)
        for item in repetition_guard.recent_text_signatures
        if _normalize_signature(item)
    )
    if not signatures:
        return recent_acts
    for act in (
        "offer_small_opening_line",
        "offer_small_opening_frame",
        "respect_boundary",
    ):
        probe_signatures = _move_history_signatures(act, locale=normalized_locale)
        if any(
            probe_signature and probe_signature in signature
            for probe_signature in probe_signatures
            for signature in signatures
        ):
            recent_acts.add(act)
    return recent_acts


def _move_history_signatures(
    act: str,
    *,
    locale: str,
) -> tuple[str, ...]:
    candidates: list[str] = []
    localized_text = _localized_segment_text(
        act,
        _default_segment_text_for_act(act),
        locale=locale,
    )
    if localized_text:
        candidates.append(localized_text)
    extras = lookup_value(locale, f"inner_os.content_policy_history_signatures.{act}")
    if isinstance(extras, list):
        candidates.extend(
            str(item).strip()
            for item in extras
            if str(item).strip()
        )
    normalized = [
        _normalize_signature(item)
        for item in candidates
        if _normalize_signature(item)
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for item in normalized:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return tuple(ordered)


def _alternate_repetition_segment(act: str) -> dict[str, str] | None:
    if act == "offer_small_opening_line":
        return _segment(
            "offer_small_opening_frame",
            "You can start with something like 'There is something catching on me lately, and I do not need to sort it before I say it.'",
        )
    if act == "offer_small_opening_frame":
        return _segment(
            "offer_small_opening_line",
            "If you want a first line, even 'Something has been catching on me lately, and I want help looking at it' is enough.",
        )
    if act == "respect_boundary":
        return _segment(
            "respect_boundary_soft",
            "We can let this stay unforced for now.",
        )
    return None


def _alternate_delta_segment(
    turn_delta: TurnDelta,
    *,
    locale: Optional[str] = None,
) -> dict[str, str] | None:
    preferred_act = str(turn_delta.preferred_act or "").strip()
    anchor_hint = _compact_anchor_hint(turn_delta.anchor_hint)
    normalized_locale = normalize_locale(locale or "en")
    if preferred_act == "reopen_from_anchor" and anchor_hint:
        return _segment(
            "reopen_from_anchor_soft",
            _discussion_template_text(
                "reopen_from_anchor_alt",
                anchor=anchor_hint,
                locale=normalized_locale,
                default_en=f"We can stay near '{anchor_hint}' again without forcing the whole thing open.",
            ),
        )
    if preferred_act == "leave_return_point_from_anchor" and anchor_hint:
        return _segment(
            "leave_return_point_from_anchor_soft",
            _discussion_template_text(
                "leave_return_point_from_anchor_alt",
                anchor=anchor_hint,
                locale=normalized_locale,
                default_en=f"We can keep '{anchor_hint}' in view and come back to it later if that feels easier.",
            ),
        )
    return None


def _default_segment_text_for_act(act: str) -> str:
    if act == "offer_small_opening_line":
        return "If you want a first line, even 'Something has been catching on me lately, and I want help looking at it' is enough."
    if act == "offer_small_opening_frame":
        return "You can start with something like 'There is something catching on me lately, and I do not need to sort it before I say it.'"
    if act == "respect_boundary":
        return "We do not have to press this right now."
    return ""


def _normalize_signature(text: str) -> str:
    normalized = str(text or "").strip().lower()
    for token in (
        "「",
        "」",
        "『",
        "』",
        '"',
        "'",
        "。",
        "、",
        ",",
        ".",
        "!",
        "?",
        "！",
        "？",
        "…",
        "（",
        "）",
        "(",
        ")",
    ):
        normalized = normalized.replace(token, "")
    return " ".join(normalized.split())


def _looks_like_opening_request(text: str, *, locale: Optional[str] = None) -> bool:
    return _contains_locale_cue(
        text,
        "inner_os.content_policy_cues.opening_request.contains_any",
        locale=locale,
    )


def _derive_deep_disclosure_sequence(
    *,
    text: str,
    interaction_policy: Optional[Mapping[str, Any]] = None,
    turn_delta: Optional[Mapping[str, Any]] = None,
    locale: Optional[str] = None,
) -> list[dict[str, str]]:
    body = str(text or "").strip()
    if not body:
        return []
    packet = dict(interaction_policy or {})
    contact_reflection_state = dict(packet.get("contact_reflection_state") or {})
    turn_delta_payload = dict(turn_delta or {})
    reflection_style = str(
        contact_reflection_state.get("reflection_style") or "reflect_then_question"
    ).strip() or "reflect_then_question"
    allow_question = reflection_style not in {"reflect_only", "boundary_only"}
    if str(turn_delta_payload.get("kind") or "").strip() == "green_reflection_hold":
        allow_question = _allow_followup_question_under_green_hold(
            interaction_policy=packet,
            contact_reflection_state=contact_reflection_state,
        )
    prefer_continuing_variant = _prefer_continuing_deep_question_variant(
        interaction_policy=packet,
    )
    normalized_locale = normalize_locale(locale or "ja-JP")
    has_unsaid = _contains_locale_cue(
        body,
        "inner_os.content_policy_cues.deep_disclosure.unsaid.contains_any",
        locale=normalized_locale,
    )
    has_help_request = _contains_locale_cue(
        body,
        "inner_os.content_policy_cues.deep_disclosure.help_request.contains_any",
        locale=normalized_locale,
    )
    has_fear = _contains_locale_cue(
        body,
        "inner_os.content_policy_cues.deep_disclosure.fear_of_view.contains_any",
        locale=normalized_locale,
    )
    has_self_blame = _contains_locale_cue(
        body,
        "inner_os.content_policy_cues.deep_disclosure.self_blame.contains_any",
        locale=normalized_locale,
    )

    if has_help_request and has_unsaid:
        sequence = [
            _segment(
                "reflect_hidden_need",
                "You wanted help then, and it is still staying inside unsaid.",
            ),
        ]
        if allow_question:
            sequence.append(
                _segment(
                    (
                        "gentle_question_hidden_need_continuing"
                        if prefer_continuing_variant
                        else "gentle_question_hidden_need"
                    ),
                    "Is the sharpest part still the moment you had to swallow those words?",
                )
            )
        return sequence
    if has_self_blame:
        sequence = [
            _segment(
                "reflect_self_blame",
                "That memory is still turning back toward self-blame.",
            ),
        ]
        if allow_question:
            sequence.append(
                _segment(
                    (
                        "gentle_question_self_blame_continuing"
                        if prefer_continuing_variant
                        else "gentle_question_self_blame"
                    ),
                    "Is the heaviest part right now the way you turn it against yourself?",
                )
            )
        return sequence
    if has_fear:
        sequence = [
            _segment(
                "reflect_fear_of_being_seen",
                "The fear of how you might be seen after saying it still feels strong.",
            ),
        ]
        if allow_question:
            sequence.append(
                _segment(
                    (
                        "gentle_question_fear_continuing"
                        if prefer_continuing_variant
                        else "gentle_question_fear"
                    ),
                    "Is that fear of how they would see you the part that catches most?",
                )
            )
        return sequence
    if has_unsaid:
        sequence = [
            _segment(
                "reflect_unspoken_weight",
                "There is something you still have not been able to say, and it is staying with you.",
            ),
        ]
        if allow_question:
            sequence.append(
                _segment(
                    (
                        "gentle_question_weight_continuing"
                        if prefer_continuing_variant
                        else "gentle_question_weight"
                    ),
                    "If you only touch one part, which part still feels heaviest?",
                )
            )
        return sequence
    return []


def _allow_followup_question_under_green_hold(
    *,
    interaction_policy: Mapping[str, Any] | None,
    contact_reflection_state: Mapping[str, Any] | None,
) -> bool:
    packet = dict(interaction_policy or {})
    contact = dict(contact_reflection_state or {})
    reflection_style = str(contact.get("reflection_style") or "").strip()
    if reflection_style == "boundary_only":
        return False
    try:
        block_share = float(contact.get("block_share") or 0.0)
    except (TypeError, ValueError):
        block_share = 0.0
    if block_share >= 0.46:
        return False

    recent_dialogue_state = dict(packet.get("recent_dialogue_state") or {})
    recent_kind = str(recent_dialogue_state.get("state") or "").strip()
    try:
        thread_carry = float(recent_dialogue_state.get("thread_carry") or 0.0)
    except (TypeError, ValueError):
        thread_carry = 0.0
    try:
        reopen_pressure = float(recent_dialogue_state.get("reopen_pressure") or 0.0)
    except (TypeError, ValueError):
        reopen_pressure = 0.0

    discussion_thread_state = dict(packet.get("discussion_thread_state") or {})
    discussion_kind = str(discussion_thread_state.get("state") or "").strip()
    try:
        revisit_readiness = float(discussion_thread_state.get("revisit_readiness") or 0.0)
    except (TypeError, ValueError):
        revisit_readiness = 0.0
    try:
        unresolved_pressure = float(discussion_thread_state.get("unresolved_pressure") or 0.0)
    except (TypeError, ValueError):
        unresolved_pressure = 0.0

    issue_state = dict(packet.get("issue_state") or {})
    try:
        question_pressure = float(issue_state.get("question_pressure") or 0.0)
    except (TypeError, ValueError):
        question_pressure = 0.0

    thread_ready = recent_kind in {"continuing_thread", "reopening_thread"} and (
        thread_carry >= 0.5 or reopen_pressure >= 0.3
    )
    discussion_ready = discussion_kind in {"revisit_issue", "active_issue", "fresh_issue"} and (
        revisit_readiness >= 0.4 or unresolved_pressure >= 0.22
    )
    issue_ready = question_pressure >= 0.24
    return thread_ready and (discussion_ready or issue_ready)


def _prefer_continuing_deep_question_variant(
    *,
    interaction_policy: Mapping[str, Any] | None,
) -> bool:
    packet = dict(interaction_policy or {})
    recent_dialogue_state = dict(packet.get("recent_dialogue_state") or {})
    recent_kind = str(recent_dialogue_state.get("state") or "").strip()
    try:
        thread_carry = float(recent_dialogue_state.get("thread_carry") or 0.0)
    except (TypeError, ValueError):
        thread_carry = 0.0
    return recent_kind in {"continuing_thread", "reopening_thread"} and thread_carry >= 0.5


def _contains_locale_cue(
    text: str,
    key: str,
    *,
    locale: Optional[str] = None,
) -> bool:
    body = str(text or "").strip()
    if not body:
        return False
    normalized = body.lower()
    patterns = lookup_value(locale or "ja-JP", key)
    if not isinstance(patterns, list):
        return False
    return any(
        pattern in normalized
        for pattern in (
            str(item).strip().lower()
            for item in patterns
        )
        if pattern
    )


def _localize_content_sequence(
    sequence: list[dict[str, str]],
    *,
    locale: Optional[str] = None,
) -> list[dict[str, str]]:
    normalized = normalize_locale(locale or "en")
    if not normalized.startswith("ja"):
        return sequence
    localized: list[dict[str, str]] = []
    for item in sequence:
        act = str(item.get("act") or "").strip()
        text = _localized_segment_text(act, str(item.get("text") or ""), locale=normalized)
        localized.append({"act": act, "text": text})
    return localized


def _localized_segment_text(
    act: str,
    text: str,
    *,
    locale: str,
) -> str:
    key = f"inner_os.content_policy_segments.{act}"
    localized = lookup_text(locale, key)
    if localized:
        return localized
    return text


def _delta_segment(
    turn_delta: TurnDelta,
    *,
    locale: Optional[str] = None,
) -> dict[str, str] | None:
    preferred_act = str(turn_delta.preferred_act or "").strip()
    anchor_hint = _compact_anchor_hint(turn_delta.anchor_hint)
    normalized_locale = normalize_locale(locale or "en")
    if preferred_act == "reopen_from_anchor":
        if anchor_hint:
            return _segment(
                "reopen_from_anchor",
                _discussion_template_text(
                    "reopen_from_anchor",
                    anchor=anchor_hint,
                    locale=normalized_locale,
                    default_en=f"We can pick up from '{anchor_hint}' without reopening the whole thing at once.",
                ),
            )
        return _supporting_effect_segment("preserve_continuity")
    if preferred_act == "leave_return_point_from_anchor":
        if anchor_hint:
            return _segment(
                "leave_return_point_from_anchor",
                _discussion_template_text(
                    "leave_return_point_from_anchor",
                    anchor=anchor_hint,
                    locale=normalized_locale,
                    default_en=f"If you want, we can come back to '{anchor_hint}' later instead of forcing it now.",
                ),
            )
        return _supporting_close_segment("leave_return_point")
    if preferred_act == "keep_shared_thread_visible":
        return _supporting_effect_segment("preserve_continuity")
    if preferred_act == "leave_return_point":
        return _supporting_close_segment("leave_return_point")
    if preferred_act == "pace_match":
        return _supporting_followup_segment("offer_one_small_next_step")
    if preferred_act == "stay_with_present_need":
        return _segment(
            "stay_with_present_need",
            "I can stay with what still needs care without pushing the obvious part any harder.",
        )
    if preferred_act == "protect_talking_room":
        return _supporting_followup_segment("protect_talking_room")
    return None


def _discussion_template_text(
    template_key: str,
    *,
    anchor: str,
    locale: str,
    default_en: str,
) -> str:
    template = lookup_text(locale, f"inner_os.content_policy_templates.{template_key}")
    if template and "{anchor}" in template:
        return template.replace("{anchor}", anchor)
    return default_en


def _compact_anchor_hint(value: str, *, limit: int = 32) -> str:
    normalized = normalize_anchor_hint(value, limit=limit)
    if normalized:
        return normalized
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"
