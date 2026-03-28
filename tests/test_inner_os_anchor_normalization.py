# -*- coding: utf-8 -*-

from inner_os.anchor_normalization import normalize_anchor_hint, select_anchor_hint
from inner_os.expression.turn_delta import derive_turn_delta


def test_normalize_anchor_hint_prefers_quoted_topic() -> None:
    anchor = normalize_anchor_hint(
        "前に出ていた「雨での約束」のところを、いま話せるぶんだけ拾う感じで大丈夫です。"
    )

    assert anchor == "雨での約束"


def test_normalize_anchor_hint_strips_topic_suffix_inside_quotes() -> None:
    anchor = normalize_anchor_hint(
        "「約束のところなら」のところなら、いま話せる分だけで大丈夫です。"
    )

    assert anchor == "約束"


def test_normalize_anchor_hint_strips_reopen_frame_inside_quotes() -> None:
    anchor = normalize_anchor_hint(
        "「あの約束のところなら、いま話せるぶんだけで」のところなら、いま話せる分だけで大丈夫です。"
    )

    assert anchor == "あの約束"


def test_normalize_anchor_hint_compacts_surface_sentence_to_topic_core() -> None:
    anchor = normalize_anchor_hint(
        "前に少し引っかかっていた話の続きを、いま少しだけ戻したいです。"
    )

    assert anchor == "少し引っかかっていた話"


def test_turn_delta_prefers_more_anchor_like_candidate_over_long_surface_sentence() -> None:
    delta = derive_turn_delta(
        {
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.72,
                "reopen_pressure": 0.64,
                "recent_anchor": "前に少し引っかかっていた話の続きを、いま少しだけ戻したいです。",
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "promise_thread",
                "dominant_anchor": "雨での約束",
                "dominant_issue_state": "pausing_issue",
                "total_threads": 1,
            },
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "reopening_thread"
    assert delta["preferred_act"] == "reopen_from_anchor"
    assert delta["anchor_hint"] == "雨での約束"


def test_select_anchor_hint_prefers_quoted_topic_core() -> None:
    anchor = select_anchor_hint(
        (
            "前に少し引っかかっていた話の続きを、いま少しだけ戻したいです。",
            "前に出ていた「雨での約束」のところを、いま話せるぶんだけ拾う感じで大丈夫です。",
        )
    )

    assert anchor == "雨での約束"
