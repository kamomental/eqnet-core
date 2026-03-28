# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_runtime_compacts_reflection_only_disclosure_with_short_presence_line() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    shaped = runtime._shape_inner_os_content_sequence(
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
        surface_profile={
            "response_length": "short",
            "certainty_style": "careful",
            "sentence_temperature": "gentle",
            "pause_insertion": "soft_pause",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
        },
    )

    assert shaped == "助けてほしかったのに、それをまだ言えないままなんですね。 いまは、そこに触れただけでもいいよ。"


def test_runtime_compacts_reflection_only_disclosure_with_alternate_presence_when_recent_history_matches() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    runtime._surface_response_history.append(  # type: ignore[attr-defined]
        "助けてほしかったのに、それをまだ言えないままなんですね。 いまは、そこに触れただけでもいいです。"
    )

    shaped = runtime._shape_inner_os_content_sequence(
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
        surface_profile={
            "response_length": "short",
            "certainty_style": "careful",
            "sentence_temperature": "gentle",
            "pause_insertion": "soft_pause",
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
        },
    )

    assert "いまは、そこに触れただけでもいいです。" not in shaped
    assert (
        "いまは、そのひと言までで止めておいても大丈夫だよ。" in shaped
        or "ここでは、そこに触れられただけでも十分だよ。" in shaped
    )
