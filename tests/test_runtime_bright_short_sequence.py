from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_runtime_shape_uses_backchannel_channel_fallback_over_long_sequence() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    surface_context_packet = {
        "source_state": {
            "listener_action_state": "soft_ack",
            "listener_token_profile": "soft_ack",
        }
    }
    expected = runtime._shape_inner_os_surface_profile_text(
        runtime._render_inner_os_response_channel_text(
            response_channel="backchannel",
            surface_context_packet=surface_context_packet,
        ),
        surface_profile={
            "response_length": "short",
            "actuation_response_channel": "backchannel",
        },
    )

    shaped = runtime._shape_inner_os_content_sequence(
        [
            {"act": "reflect_hidden_need", "text": "This would normally become a longer reflective response."},
            {"act": "gentle_question_hidden_need", "text": "This follow-up question should be suppressed."},
        ],
        surface_profile={
            "response_length": "balanced",
            "actuation_response_channel": "backchannel",
        },
        surface_context_packet=surface_context_packet,
        fallback_text="fallback",
    )

    assert shaped == expected
    assert "longer reflective response" not in shaped


def test_runtime_shape_uses_hold_channel_fallback_over_long_sequence() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    expected = runtime._shape_inner_os_surface_profile_text(
        runtime._render_inner_os_response_channel_text(
            response_channel="hold",
            surface_context_packet={"source_state": {}},
        ),
        surface_profile={
            "response_length": "short",
            "actuation_response_channel": "hold",
            "opening_pace_windowed": "",
            "return_gaze_expectation": "",
        },
    )

    shaped = runtime._shape_inner_os_content_sequence(
        [
            {"act": "quiet_presence", "text": "This would normally remain as a quiet-presence sentence."},
            {"act": "leave_unfinished_closed", "text": "This closing sentence should be suppressed for hold."},
        ],
        surface_profile={
            "response_length": "balanced",
            "actuation_response_channel": "hold",
            "opening_pace_windowed": "held",
            "return_gaze_expectation": "soft_return",
        },
        surface_context_packet={"source_state": {}},
        fallback_text="fallback",
    )

    assert shaped == expected
    assert "quiet-presence sentence" not in shaped


def test_runtime_short_sequence_prefers_bright_opening_before_quiet_presence() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    selected = runtime._select_short_inner_os_sequence(
        [
            {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
            {"act": "quiet_presence", "text": "話せるところからで大丈夫です。"},
            {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
            {"act": "light_bounce", "text": "その感じ、ちょっと嬉しいね。"},
        ]
    )

    assert [str(item.get("act") or "").strip() for item in selected[:2]] == [
        "shared_delight",
        "light_bounce",
    ]


def test_runtime_compact_prefers_bright_text_when_light_playful_surface_has_bright_acts() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    compact = runtime._compact_inner_os_sequence_text(
        [
            {"act": "respect_boundary", "text": "いまは、無理にうまく話そうとしなくて大丈夫です。"},
            {"act": "quiet_presence", "text": "話せるところからで大丈夫です。"},
            {"act": "leave_unfinished_closed", "text": "また話せそうなときに、そこからで大丈夫です。"},
            {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
            {"act": "light_bounce", "text": "それは、ちょっと気持ちが軽くなるね。"},
        ],
        surface_profile={"response_length": "short", "voice_texture": "light_playful"},
    )

    assert compact == "それは、ちょっといい感じだね。 ちょっと気持ちが軽くなるね。"


def test_runtime_compact_trims_repeated_bright_opening_fragment() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    compact = runtime._compact_inner_os_sequence_text(
        [
            {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
            {"act": "light_bounce", "text": "それは、ちょっと気持ちが軽くなるね。"},
        ],
        surface_profile={"response_length": "short", "voice_texture": "light_playful"},
        discourse_shape={"shape_id": "bright_bounce"},
    )

    assert compact == "それは、ちょっといい感じだね。 ちょっと気持ちが軽くなるね。"


def test_runtime_compact_uses_laugh_cue_profile_for_bright_bounce() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    runtime._last_surface_user_text = "さっきの続きなんだけど、あのあとちょっと笑えることもあって。"  # type: ignore[attr-defined]

    compact = runtime._compact_inner_os_sequence_text(
        [
            {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
            {"act": "light_bounce", "text": "それは、ちょっと気持ちが軽くなるね。"},
        ],
        surface_profile={"response_length": "short", "voice_texture": "light_playful"},
        discourse_shape={"shape_id": "bright_bounce"},
    )

    assert compact == "それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
