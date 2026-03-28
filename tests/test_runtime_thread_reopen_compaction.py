# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_runtime_compacts_thread_reopening_with_anchor_focused_surface() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    compact = runtime._compact_inner_os_sequence_text(
        [
            {
                "act": "reopen_from_anchor",
                "text": "「あの約束」のところなら、いま話せる分だけで大丈夫です。",
            },
            {
                "act": "leave_return_point",
                "text": "続きは、また話せそうなときに。",
            },
        ],
        surface_profile={"response_length": "short"},
    )

    assert compact == "あの約束のことなら、いま話せるところからでいいよ。 続きは、また話せそうなときに。"


def test_runtime_shapes_anchor_focused_thread_reopening_without_soft_pause() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    shaped = runtime._shape_inner_os_surface_profile_text(
        "あの約束のところなら、いま話せるぶんだけで。 続きは、また話せそうなときに。",
        surface_profile={
            "response_length": "short",
            "pause_insertion": "soft_pause",
            "opening_pace_windowed": "measured",
            "sentence_temperature": "neutral",
            "certainty_style": "direct",
            "return_gaze_expectation": "soft_return",
        },
    )

    assert not shaped.startswith("…")
    assert shaped.startswith("あの約束のところなら、")
