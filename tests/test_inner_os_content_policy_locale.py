# -*- coding: utf-8 -*-

from inner_os.expression.content_policy import derive_content_sequence


def test_content_sequence_localizes_guarded_sequence_for_ja() -> None:
    sequence = derive_content_sequence(
        current_text="まだ少し怖いです。",
        interaction_policy={
            "dialogue_act": "check_in",
            "opening_move": "acknowledge_without_probe",
            "followup_move": "protect_talking_room",
            "closing_move": "leave_unfinished_part_closed_for_now",
            "primary_object_operation": {
                "operation_kind": "hold_without_probe",
                "target_label": "まだ少し怖いところ",
            },
        },
        locale="ja-JP",
    )

    texts = [item["text"] for item in sequence]
    assert texts
    assert texts[0].startswith("いまは、ここを")
    assert all("We do not have to press this right now." not in text for text in texts)


def test_content_sequence_offers_small_opening_line_for_clarify_contain_strategy() -> None:
    sequence = derive_content_sequence(
        current_text="どう切り出せばよさそうですか。",
        interaction_policy={
            "dialogue_act": "clarify",
            "response_strategy": "contain_then_stabilize",
        },
        locale="ja-JP",
    )

    acts = [item["act"] for item in sequence]
    texts = [item["text"] for item in sequence]
    assert "offer_small_opening_line" in acts
    assert any("切り出すなら" in text for text in texts)


def test_content_sequence_offers_small_opening_line_for_opening_request_under_hold() -> None:
    sequence = derive_content_sequence(
        current_text="どう切り出せばよさそうですか。",
        interaction_policy={
            "response_strategy": "contain_then_stabilize",
            "opening_move": "acknowledge_without_probe",
            "closing_move": "leave_unfinished_part_closed_for_now",
            "primary_object_operation": {
                "operation_kind": "hold_without_probe",
            },
        },
        locale="ja-JP",
    )

    acts = [item["act"] for item in sequence]
    assert acts[:3] == [
        "respect_boundary",
        "offer_small_opening_line",
        "quiet_presence",
    ]


def test_content_sequence_prefers_emergency_distance_over_opening_support() -> None:
    sequence = derive_content_sequence(
        current_text="どう切り出せばいいですか。",
        interaction_policy={
            "dialogue_act": "clarify",
            "response_strategy": "contain_then_stabilize",
            "opening_move": "acknowledge_without_probe",
            "current_risks": ["danger", "unsafe_person"],
            "situation_risk_state": {
                "state": "acute_threat",
                "context_affordance": "public_exposure",
            },
            "emergency_posture": {
                "state": "create_distance",
                "score": 0.72,
                "dialogue_permission": "boundary_only",
                "primary_action": "create_distance",
            },
        },
        locale="ja-JP",
    )

    acts = [item["act"] for item in sequence]
    texts = [item["text"] for item in sequence]
    assert acts == ["emergency_create_distance"]
    assert all("切り出すなら" not in text for text in texts)
