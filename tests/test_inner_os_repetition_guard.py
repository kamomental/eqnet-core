from inner_os.expression.content_policy import derive_content_sequence
from inner_os.expression.repetition_guard import derive_repetition_guard


def test_repetition_guard_blocks_exact_recent_text_reuse() -> None:
    guard = derive_repetition_guard(
        [
            "We can come back to the rest when it feels easier, and only if you want to.",
            "You do not have to explain more than feels manageable.",
        ]
    )

    assert guard.blocks_text("We can come back to the rest when it feels easier, and only if you want to.")
    assert guard.blocks_text("  you do not have to explain more than feels manageable. ")
    assert not guard.blocks_text("I can stay nearby without leaning on it, and come back when it feels easier.")


def test_content_sequence_suppresses_repeated_return_point_from_recent_history() -> None:
    sequence = derive_content_sequence(
        current_text="I want to stay with what feels difficult here.",
        history=["We can come back to the rest when it feels easier, and only if you want to."],
        interaction_policy={
            "dialogue_act": "check_in",
            "opening_move": "acknowledge_without_probe",
            "followup_move": "protect_talking_room",
            "closing_move": "leave_return_point",
            "primary_conversational_object_label": "what feels difficult here",
            "primary_object_operation": {
                "operation_kind": "hold_without_probe",
                "target_label": "what feels difficult here",
            },
            "ordered_operation_kinds": [
                "hold_without_probe",
                "protect_unfinished_part",
                "keep_return_point",
            ],
            "ordered_effect_kinds": [
                "keep_connection_open",
                "preserve_self_pacing",
            ],
            "deferred_object_labels": ["unfinished part"],
            "dialogue_order": [
                "open:acknowledge_without_probe",
                "focus:what feels difficult here",
                "operate:hold_without_probe",
                "operate:protect_unfinished_part",
                "follow:protect_talking_room",
                "effect:keep_connection_open",
                "effect:preserve_self_pacing",
                "defer:unfinished_part",
                "close:leave_return_point",
            ],
            "question_budget": 0,
        },
        conscious_access={"intent": "check_in"},
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert "respect_boundary" in acts
    assert "leave_return_point" not in acts
    assert "come back to the rest when it feels easier" not in text


def test_content_sequence_switches_opening_move_when_recent_opening_line_repeats() -> None:
    sequence = derive_content_sequence(
        current_text="今のしんどさを無理に整理せず、どう切り出せばよさそうですか。",
        history=[
            "いまは、ここを無理に押し進めなくて大丈夫です。 切り出すなら、「最近ちょっと引っかかっていることがあって、一緒に見てほしい」くらいで十分です。"
        ],
        interaction_policy={
            "dialogue_act": "clarify",
            "response_strategy": "contain_then_stabilize",
            "opening_move": "acknowledge_without_probe",
            "followup_move": "protect_talking_room",
            "closing_move": "leave_unfinished_part_closed_for_now",
            "primary_object_operation": {
                "operation_kind": "hold_without_probe",
                "target_label": "what feels difficult here",
            },
            "ordered_operation_kinds": [
                "hold_without_probe",
                "protect_unfinished_part",
            ],
            "ordered_effect_kinds": [
                "keep_connection_open",
                "preserve_self_pacing",
            ],
            "question_budget": 0,
        },
        conscious_access={"intent": "clarify"},
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert "offer_small_opening_line" not in acts
    assert "offer_small_opening_frame" in acts
    assert "まだうまく整理できない" in text


def test_content_sequence_switches_opening_move_even_when_quote_style_differs() -> None:
    sequence = derive_content_sequence(
        current_text="今のしんどさを無理に整理せず、どう切り出せばよさそうですか。",
        history=[
            "いまは、ここを無理に押し進めなくて大丈夫です。 切り出すなら、『最近ちょっと引っかかっていることがあって、一緒に見てほしい』くらいで十分です。"
        ],
        interaction_policy={
            "dialogue_act": "clarify",
            "response_strategy": "contain_then_stabilize",
            "opening_move": "acknowledge_without_probe",
            "followup_move": "protect_talking_room",
            "closing_move": "leave_unfinished_part_closed_for_now",
            "primary_object_operation": {
                "operation_kind": "hold_without_probe",
                "target_label": "what feels difficult here",
            },
            "ordered_operation_kinds": [
                "hold_without_probe",
                "protect_unfinished_part",
            ],
            "ordered_effect_kinds": [
                "keep_connection_open",
                "preserve_self_pacing",
            ],
            "question_budget": 0,
        },
        conscious_access={"intent": "clarify"},
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    assert "offer_small_opening_frame" in acts


def test_content_sequence_softens_repeated_boundary_opening() -> None:
    sequence = derive_content_sequence(
        current_text="I want to say this without making it too heavy.",
        history=["We do not have to press this right now."],
        interaction_policy={
            "dialogue_act": "clarify",
            "response_strategy": "contain_then_stabilize",
            "opening_move": "acknowledge_without_probe",
            "followup_move": "protect_talking_room",
            "closing_move": "leave_unfinished_part_closed_for_now",
            "primary_object_operation": {
                "operation_kind": "hold_without_probe",
                "target_label": "what feels difficult here",
            },
            "ordered_operation_kinds": [
                "hold_without_probe",
                "protect_unfinished_part",
            ],
            "ordered_effect_kinds": [
                "keep_connection_open",
                "preserve_self_pacing",
            ],
            "question_budget": 0,
        },
        conscious_access={"intent": "clarify"},
        locale="en-US",
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert "respect_boundary" not in acts
    assert "respect_boundary_soft" in acts
    assert "We do not have to press this right now." not in text
