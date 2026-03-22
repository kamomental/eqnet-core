from __future__ import annotations

from typing import Any, Mapping, Optional


def derive_content_sequence(
    *,
    current_text: str,
    interaction_policy: Optional[Mapping[str, Any]] = None,
    conscious_access: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, str]]:
    text = str(current_text or "").strip()
    if not text:
        return []

    packet = dict(interaction_policy or {})
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
    target_phrase = _target_phrase(target_label or primary_object_label)
    if grice_state == "hold_obvious_advice":
        return [
            _segment("hold_known_thread", "I do not need to restate the part that is already clear here."),
            _segment(
                "stay_with_present_need",
                f"I can stay with {target_phrase or 'what still needs care'} without pushing the obvious part any harder.",
            ),
        ]
    if grice_state == "attune_without_repeating":
        return [
            _segment("visible_anchor", "I can stay with what is already clear here without repeating it."),
            _segment(
                "gentle_extension",
                f"If anything new needs care, I can move only as far as {target_phrase or 'the next living part'} asks.",
            ),
        ]
    if grice_state == "acknowledge_then_extend" and strategy in {"shared_world_next_step", "attune_then_extend"}:
        return [
            _segment("known_anchor", "I can keep the thread that is already clear in view."),
            _segment(
                "small_extension",
                f"If I add anything, I want it to be only the small part that is new around {target_phrase or 'this'}.",
            ),
        ]
    if (
        dialogue_act == "clarify"
        and strategy not in {"repair_then_attune", "respectful_wait"}
        and question_budget > 0
        and operation_kind not in {"hold_without_probe", "offer_small_next_step"}
    ):
        return [
            _segment("clarify_gate", "Let me check one small thing before I go further."),
            _segment("visible_anchor", "I can stay with what is visible first."),
            _segment("clarify_followup", "Then I can answer a little more cleanly from there."),
        ]
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
    )
    if move_sequence:
        return move_sequence

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
        )
        if operation_sequence:
            return operation_sequence

    if strategy == "repair_then_attune":
        return [
            _segment("acknowledge_overreach", "I came in too fast there."),
            _segment("visible_anchor", "Let me slow this down and stay only with what is actually clear right now."),
            _segment("careful_reopen", "You do not have to carry the rest until it feels easier to reopen."),
        ]
    if strategy == "respectful_wait":
        return [
            _segment("respect_boundary", "We do not have to press this right now."),
            _segment("quiet_presence", "I can stay nearby without leaning on it, and come back when it feels easier."),
        ]
    if strategy == "shared_world_next_step":
        return [
            _segment("shared_anchor", "Here is the next step I can see from what is already clear."),
            _segment("pace_match", "I can keep that next move small enough for us to stay in step."),
        ]
    if strategy == "contain_then_stabilize":
        return [
            _segment("stabilize_boundary", "I can keep this inside what feels steady first."),
            _segment("do_not_overread", "I do not want to read past what the scene can actually support."),
        ]
    if strategy == "reflect_without_settling":
        return [
            _segment("hold_meaning_open", "I can stay with what is here without settling the meaning too fast."),
            _segment("slow_reflection", "If we go further, I would rather keep the meaning open for a while."),
        ]
    if strategy == "attune_then_extend":
        return [
            _segment("visible_anchor", "I'm here with you. We can start with what feels most visible."),
            _segment("gentle_extension", "You do not have to rush it; if it helps, we can move a little closer to what matters here."),
        ]

    if dialogue_act == "check_in" and opening_move == "stay_with_visible":
        return [
            _segment("visible_anchor", "I can stay with what is visible first."),
            _segment("gentle_extension", "Then I can go a little further if that helps."),
        ]
    if dialogue_act == "clarify" and followup_move == "invite_visible_state":
        return [
            _segment("clarify_anchor", "I can check what is visible first."),
            _segment("clarify_followup", "Then I can answer a little more cleanly from there."),
        ]
    return [_segment("carry_text", text)]


def derive_content_skeleton(
    *,
    current_text: str,
    interaction_policy: Optional[Mapping[str, Any]] = None,
    conscious_access: Optional[Mapping[str, Any]] = None,
) -> str:
    return render_content_sequence(
        derive_content_sequence(
            current_text=current_text,
            interaction_policy=interaction_policy,
            conscious_access=conscious_access,
        )
    )


def render_content_sequence(sequence: list[dict[str, str]]) -> str:
    return _join_lines(*(segment.get("text", "") for segment in sequence))


def _segment(act: str, text: str) -> dict[str, str]:
    return {"act": act, "text": text}


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
                _segment("quiet_presence", "I can stay nearby without leaning on it, and come back when it feels easier."),
            ]
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
            _segment("quiet_presence", "I can stay nearby without leaning on it, and come back when it feels easier."),
        ]
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
