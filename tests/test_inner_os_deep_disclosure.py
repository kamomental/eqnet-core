# -*- coding: utf-8 -*-

from inner_os.expression.content_policy import derive_content_sequence


def test_content_sequence_reflects_hidden_need_before_question() -> None:
    sequence = derive_content_sequence(
        current_text="本当は、あのとき助けてほしかったって、まだ言えていないんです。",
        interaction_policy={
            "dialogue_act": "check_in",
        },
        conscious_access={"intent": "check_in"},
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert acts[:2] == [
        "reflect_hidden_need",
        "gentle_question_hidden_need",
    ]
    assert "助けてほしかった" in text
    assert "飲み込んだ" in text


def test_content_sequence_reflects_fear_of_being_seen() -> None:
    sequence = derive_content_sequence(
        current_text="話すと、あのあとにどう見られるかが怖いんです。",
        interaction_policy={
            "dialogue_act": "check_in",
        },
        conscious_access={"intent": "check_in"},
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert acts[:2] == [
        "reflect_fear_of_being_seen",
        "gentle_question_fear",
    ]
    assert "どう見られるか" in text


def test_deep_disclosure_can_stop_at_reflection_when_contact_is_guarded() -> None:
    sequence = derive_content_sequence(
        current_text="本当は、あのとき助けてほしかったって、まだ言えていないんです。",
        interaction_policy={
            "dialogue_act": "check_in",
            "contact_reflection_state": {
                "state": "guarded_reflection",
                "reflection_style": "reflect_only",
            },
        },
        conscious_access={"intent": "check_in"},
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    assert acts == ["reflect_hidden_need"]


def test_deep_disclosure_can_stop_at_reflection_when_green_reflection_hold_is_selected() -> None:
    sequence = derive_content_sequence(
        current_text="本当は、あのとき助けてほしかったって、まだ言えていないんです。",
        interaction_policy={
            "dialogue_act": "check_in",
        },
        conscious_access={"intent": "check_in"},
        turn_delta={
            "kind": "green_reflection_hold",
            "preferred_act": "stay_with_present_need",
        },
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    assert acts == ["reflect_hidden_need"]


def test_deep_disclosure_can_restore_light_question_on_continuing_thread_under_green_hold() -> None:
    sequence = derive_content_sequence(
        current_text="本当は、あのとき助けてほしかったって、まだ言えていないんです。",
        interaction_policy={
            "dialogue_act": "check_in",
            "contact_reflection_state": {
                "state": "guarded_reflection",
                "reflection_style": "reflect_only",
                "block_share": 0.18,
            },
            "recent_dialogue_state": {
                "state": "continuing_thread",
                "thread_carry": 0.64,
                "reopen_pressure": 0.18,
            },
            "discussion_thread_state": {
                "state": "revisit_issue",
                "revisit_readiness": 0.54,
                "unresolved_pressure": 0.24,
            },
            "issue_state": {
                "state": "exploring_issue",
                "question_pressure": 0.32,
            },
        },
        conscious_access={"intent": "check_in"},
        turn_delta={
            "kind": "green_reflection_hold",
            "preferred_act": "stay_with_present_need",
        },
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    assert acts[:2] == [
        "reflect_hidden_need",
        "gentle_question_hidden_need_continuing",
    ]
