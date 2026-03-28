# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_runtime_surface_profile_softens_japanese_casual_register() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    shaped = runtime._shape_inner_os_surface_profile_text(
        "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。",
        surface_profile={
            "response_length": "short",
            "sentence_temperature": "warm",
            "pause_insertion": "none",
            "certainty_style": "direct",
            "cultural_register": "casual_shared",
            "group_register": "one_to_one",
        },
    )

    assert shaped == "話せるところからで大丈夫だよ。いま全部きれいに言わなくても大丈夫だよ。"


def test_runtime_surface_profile_keeps_polite_register_for_formal_character() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    shaped = runtime._shape_inner_os_surface_profile_text(
        "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。",
        surface_profile={
            "response_length": "short",
            "sentence_temperature": "gentle",
            "pause_insertion": "none",
            "certainty_style": "direct",
            "cultural_register": "careful_polite",
            "group_register": "one_to_one",
        },
    )

    assert shaped == "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。"


def test_runtime_content_sequence_prefers_casual_quiet_presence_variant_for_casual_register() -> None:
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
            "sentence_temperature": "warm",
            "pause_insertion": "none",
            "certainty_style": "direct",
            "cultural_register": "casual_shared",
            "group_register": "one_to_one",
        },
    )

    assert "いまは、そこに触れただけでもいいよ。" in shaped


def test_runtime_content_sequence_keeps_formal_light_question_for_formal_register() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    shaped = runtime._shape_inner_os_content_sequence(
        [
            {
                "act": "reflect_fear_of_being_seen",
                "text": "言ったあとにどう見られるか、その怖さがまだ強いんですね。",
            },
            {
                "act": "gentle_question_fear",
                "text": "その怖さ、どこでいちばん強くなりますか。",
            },
        ],
        surface_profile={
            "response_length": "short",
            "sentence_temperature": "gentle",
            "pause_insertion": "none",
            "certainty_style": "careful",
            "cultural_register": "careful_polite",
            "group_register": "one_to_one",
        },
    )

    assert "その怖さ、どこでいちばん強くなりますか。" in shaped
