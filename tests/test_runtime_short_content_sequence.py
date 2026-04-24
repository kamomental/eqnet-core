# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_runtime_short_sequence_keeps_quiet_presence_when_opening_line_exists() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    selected = runtime._select_short_inner_os_sequence(
        [
            {"act": "respect_boundary", "text": "We do not have to press this right now."},
            {
                "act": "offer_small_opening_line",
                "text": "If you want a first line, even 'Something has been catching on me lately' is enough.",
            },
            {
                "act": "quiet_presence",
                "text": "I can stay nearby without leaning on it, and come back when it feels easier.",
            },
            {
                "act": "leave_unfinished_closed",
                "text": "We can leave the unfinished part where it is for now, and come back only if it feels easier later.",
            },
        ]
    )

    acts = [str(item.get("act") or "").strip() for item in selected]
    assert acts == [
        "offer_small_opening_line",
        "quiet_presence",
        "leave_unfinished_closed",
    ]


def test_runtime_short_sequence_prefers_thread_visibility_before_generic_closing() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    selected = runtime._select_short_inner_os_sequence(
        [
            {"act": "respect_boundary", "text": "We do not have to press this right now."},
            {
                "act": "quiet_presence",
                "text": "I can stay nearby without leaning on it, and come back when it feels easier.",
            },
            {
                "act": "leave_unfinished_closed",
                "text": "We can leave the unfinished part where it is for now, and come back only if it feels easier later.",
            },
            {
                "act": "keep_shared_thread_visible",
                "text": "I do not want to lose the thread that is already here between us while we stay with it.",
            },
        ]
    )

    acts = [str(item.get("act") or "").strip() for item in selected]
    assert acts == [
        "respect_boundary",
        "keep_shared_thread_visible",
        "quiet_presence",
    ]


def test_runtime_short_sequence_keeps_soft_anchor_reopen_before_generic_closing() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    selected = runtime._select_short_inner_os_sequence(
        [
            {"act": "respect_boundary", "text": "We do not have to press this right now."},
            {
                "act": "quiet_presence",
                "text": "I can stay nearby without leaning on it, and come back when it feels easier.",
            },
            {
                "act": "leave_unfinished_closed",
                "text": "We can leave the unfinished part where it is for now, and come back only if it feels easier later.",
            },
            {
                "act": "reopen_from_anchor_soft",
                "text": "We can stay near 'harbor promise' again without forcing the whole thing open.",
            },
        ]
    )

    acts = [str(item.get("act") or "").strip() for item in selected]
    assert acts == [
        "reopen_from_anchor_soft",
        "quiet_presence",
        "leave_unfinished_closed",
    ]


def test_runtime_short_sequence_prioritizes_opening_frame_over_reflection_when_both_exist() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    selected = runtime._select_short_inner_os_sequence(
        [
            {
                "act": "reflect_hidden_need",
                "text": "You wanted help then, and it is still staying inside unsaid.",
            },
            {
                "act": "offer_small_opening_frame",
                "text": "If you want a first line, even 'Something has been catching on me lately' is enough.",
            },
            {
                "act": "quiet_presence",
                "text": "I can stay nearby without leaning on it, and come back when it feels easier.",
            },
        ]
    )

    acts = [str(item.get("act") or "").strip() for item in selected]
    assert acts[:2] == [
        "offer_small_opening_frame",
        "quiet_presence",
    ]


def test_runtime_short_sequence_prioritizes_deep_disclosure_reflection_and_question() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    selected = runtime._select_short_inner_os_sequence(
        [
            {"act": "respect_boundary_soft", "text": "We do not have to force this into words right now."},
            {
                "act": "reflect_hidden_need",
                "text": "You wanted help then, and it is still staying inside unsaid.",
            },
            {
                "act": "gentle_question_hidden_need",
                "text": "Is the sharpest part still the moment you had to swallow those words?",
            },
            {
                "act": "quiet_presence",
                "text": "I can stay nearby without leaning on it, and come back when it feels easier.",
            },
        ]
    )

    acts = [str(item.get("act") or "").strip() for item in selected]
    assert acts[:2] == [
        "reflect_hidden_need",
        "gentle_question_hidden_need",
    ]


def test_runtime_short_sequence_accepts_continuing_deep_disclosure_question_variant() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    selected = runtime._select_short_inner_os_sequence(
        [
            {
                "act": "reflect_hidden_need",
                "text": "You wanted help then, and it is still staying inside unsaid.",
            },
            {
                "act": "gentle_question_hidden_need_continuing",
                "text": "If it feels possible, we can start near the part where those words were swallowed.",
            },
            {
                "act": "quiet_presence",
                "text": "I can stay nearby without leaning on it, and come back when it feels easier.",
            },
        ]
    )

    acts = [str(item.get("act") or "").strip() for item in selected]
    assert acts[:2] == [
        "reflect_hidden_need",
        "gentle_question_hidden_need_continuing",
    ]


def test_runtime_short_sequence_can_hold_on_reflection_only_for_deep_disclosure() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    selected = runtime._select_short_inner_os_sequence(
        [
            {
                "act": "reflect_hidden_need",
                "text": "You wanted help then, and it is still staying inside unsaid.",
            },
            {
                "act": "quiet_presence",
                "text": "I can stay nearby without leaning on it, and come back when it feels easier.",
            },
        ]
    )

    acts = [str(item.get("act") or "").strip() for item in selected]
    assert acts == ["reflect_hidden_need", "quiet_presence"]


def test_runtime_compacts_reflection_only_with_quiet_presence_for_deep_disclosure() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    compact = runtime._compact_inner_os_sequence_text(
        [
            {
                "act": "reflect_hidden_need",
                "text": "助けてほしかったのに、それをまだ言えないままなんですね。",
            },
            {
                "act": "quiet_presence",
                "text": "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。",
            },
        ],
        surface_profile={"response_length": "short"},
    )

    assert compact == "助けてほしかったのに、それをまだ言えないままなんですね。 いまは、そこに触れただけでもいいです。"


def test_runtime_compacts_reflection_with_stay_line_for_deep_disclosure() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    compact = runtime._compact_inner_os_sequence_text(
        [
            {
                "act": "reflect_fear_of_being_seen",
                "text": "言ったあとにどう見られるか、その怖さがまだ強いんですね。",
            },
            {
                "act": "stay_with_present_need",
                "text": "いま必要なところだけに、そのまま付き添えます。",
            },
        ],
        surface_profile={"response_length": "short"},
    )

    assert compact == "言ったあとにどう見られるか、その怖さがまだ強いんですね。 いまは、そのままでいいよ。"


def test_runtime_compacts_continuity_opening_without_falling_back_to_three_explanatory_lines() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    compact = runtime._compact_inner_os_sequence_text(
        [
            {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
            {"act": "keep_shared_thread_visible", "text": "いまここにある流れを、切らさずに持っておきたいです。"},
            {"act": "quiet_presence", "text": "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。"},
        ],
        surface_profile={"response_length": "short"},
    )

    assert compact in {
        "いまここにある流れを、切らさずに持っておきたいです。 話せるところからで大丈夫です。",
        "いまは無理に広げなくて大丈夫です。話せるところからでいいです。",
        "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。",
        "うん、話せるところからで大丈夫です。急がなくていいです。",
    }


def test_runtime_compacts_thread_reopening_with_alternate_thread_presence_when_recent_history_matches() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    runtime._surface_response_history.append(  # type: ignore[attr-defined]
        "前に出ていた「港での約束」のところを、いま話せるぶんだけ拾う感じで大丈夫です。 残りは、また話せそうなときに。"
    )

    compact = runtime._compact_inner_os_sequence_text(
        [
            {
                "act": "reopen_from_anchor_soft",
                "text": "前に出ていた「港での約束」のところを、いま話せるぶんだけ拾う感じで大丈夫です。",
            },
            {
                "act": "leave_return_point",
                "text": "また話せそうなときに、そこから続けられます。",
            },
        ],
        surface_profile={"response_length": "short"},
    )

    assert "残りは、また話せそうなときに。" not in compact
    assert (
        "続きは、また話せそうなときに。" in compact
        or "その先は、また話せそうなときに。" in compact
        or "残りは、また話せそうなときで大丈夫です。" in compact
    )
