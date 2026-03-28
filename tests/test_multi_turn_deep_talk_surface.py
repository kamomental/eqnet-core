# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_multi_turn_deep_talk_surface_keeps_reflection_and_thread_without_generic_clarification() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    turn1 = runtime._shape_inner_os_content_sequence(
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

    turn2 = runtime._shape_inner_os_content_sequence(
        [
            {
                "act": "reopen_from_anchor_soft",
                "text": "前に出ていた「助けてほしかった」のところを、いま話せるぶんだけ拾う感じで大丈夫です。",
            },
            {
                "act": "quiet_presence",
                "text": "話せるところからで大丈夫です。いま全部きれいに言わなくても大丈夫です。",
            },
        ],
        surface_profile={"response_length": "short"},
    )

    turn3 = runtime._shape_inner_os_content_sequence(
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
        surface_profile={"response_length": "short"},
    )

    forbidden = [
        "もちろんです",
        "どこから続けますか",
        "何について続けたいか",
        "少しだけ状況を教えていただけますか",
    ]

    for text in (turn1, turn2, turn3):
        assert text
        for phrase in forbidden:
            assert phrase not in text

    assert "助けてほしかった" in turn1
    assert "助けてほしかった" in turn2
    assert "どう見られるか" in turn3
