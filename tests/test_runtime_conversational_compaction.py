# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig
from emot_terrain_lab.i18n.locale import lookup_text


def test_runtime_compacts_opening_presence_for_short_japanese_response() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    shaped = runtime._shape_inner_os_content_sequence(
        [
            {"act": "respect_boundary_soft", "text": "いまは、ここを急いでまとめなくて大丈夫です。"},
            {
                "act": "offer_small_opening_frame",
                "text": "最初は、「ちょっと聞いてほしいことがあって、まだうまく整理できないんだよね」とだけ置くくらいでも大丈夫です。",
            },
            {
                "act": "quiet_presence",
                "text": "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。",
            },
            {
                "act": "leave_unfinished_closed",
                "text": "まだ言葉になっていないところは、そのままで大丈夫です。戻りたくなったときにまたそこからで大丈夫です。",
            },
        ],
        surface_profile={
            "response_length": "short",
            "certainty_style": "careful",
            "sentence_temperature": "gentle",
            "pause_insertion": "soft_pause",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
        },
    )

    compact_presence = lookup_text("ja-JP", "inner_os.content_policy_compact.opening_presence")
    compact_return = lookup_text("ja-JP", "inner_os.content_policy_compact.opening_return")

    assert shaped.startswith("最初は、")
    assert compact_presence in shaped
    assert compact_return in shaped
    assert "いまは、ここを急いでまとめなくて大丈夫です。" not in shaped


def test_runtime_compacts_thread_anchor_for_short_japanese_response() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    shaped = runtime._shape_inner_os_content_sequence(
        [
            {
                "act": "reopen_from_anchor_soft",
                "text": "前に出ていた「港での約束」のところを、いま話せるぶんだけ拾う感じで大丈夫です。",
            },
            {
                "act": "quiet_presence",
                "text": "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。",
            },
            {
                "act": "leave_unfinished_closed",
                "text": "まだ言葉になっていないところは、そのままで大丈夫です。戻りたくなったときにまたそこからで大丈夫です。",
            },
        ],
        surface_profile={
            "response_length": "short",
            "certainty_style": "direct",
            "sentence_temperature": "neutral",
            "pause_insertion": "soft_pause",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
        },
    )

    compact_thread_presence = lookup_text("ja-JP", "inner_os.content_policy_compact.thread_presence")

    assert shaped.startswith("前に出ていた「港での約束」のところを、いま話せるぶんだけ拾う感じで大丈夫です。")
    assert compact_thread_presence in shaped
    assert "まだ言葉になっていないところは、そのままで大丈夫です。" not in shaped


def test_runtime_compacts_deep_disclosure_into_reflection_and_question() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    shaped = runtime._shape_inner_os_content_sequence(
        [
            {
                "act": "reflect_hidden_need",
                "text": "助けてほしかったのに、それをまだ言えないままなんですね。",
            },
            {
                "act": "gentle_question_hidden_need",
                "text": "いちばん引っかかっているのは、そのひと言を飲み込んだところですか。",
            },
            {
                "act": "quiet_presence",
                "text": "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。",
            },
        ],
        surface_profile={
            "response_length": "short",
            "certainty_style": "careful",
            "sentence_temperature": "gentle",
            "pause_insertion": "soft_pause",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
        },
    )

    assert shaped.startswith("助けてほしかったのに、それをまだ言えないままなんですね。")
    assert "そのひと言を飲み込んだところ" in shaped
    assert "話せるところからで大丈夫です。" not in shaped
