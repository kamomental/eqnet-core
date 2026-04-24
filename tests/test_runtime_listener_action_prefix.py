from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_runtime_compact_uses_listener_action_prefix_for_bright_bounce() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    runtime._last_surface_user_text = (
        "\u3055\u3063\u304d\u306e\u7d9a\u304d\u306a\u3093\u3060\u3051\u3069\u3001"
        "\u3042\u306e\u3042\u3068\u3061\u3087\u3063\u3068\u7b11\u3048\u308b\u3053\u3068\u3082\u3042\u3063\u3066\u3002"
    )  # type: ignore[attr-defined]
    runtime._last_surface_context_packet = {  # type: ignore[attr-defined]
        "source_state": {
            "listener_action_state": "warm_laugh_ack",
            "listener_token_profile": "soft_laugh",
        }
    }

    compact = runtime._compact_inner_os_sequence_text(
        [
            {"act": "shared_delight", "text": "\u305d\u308c\u3001\u3061\u3087\u3063\u3068\u7b11\u3048\u308b\u3084\u3064\u3060\u306d\u3002"},
            {"act": "light_bounce", "text": "\u305d\u3046\u3044\u3046\u306e\u3042\u308b\u3068\u3001\u3061\u3087\u3063\u3068\u697d\u306b\u306a\u308b\u3088\u306d\u3002"},
        ],
        surface_profile={"response_length": "short", "voice_texture": "light_playful"},
        discourse_shape={"shape_id": "bright_bounce"},
    )

    assert (
        compact
        == "\u3075\u3075\u3063\u3001\u305d\u308c\u3001\u3061\u3087\u3063\u3068\u7b11\u3048\u308b\u3084\u3064\u3060\u306d\u3002 "
        "\u305d\u3046\u3044\u3046\u306e\u3042\u308b\u3068\u3001\u3061\u3087\u3063\u3068\u697d\u306b\u306a\u308b\u3088\u306d\u3002"
    )


def test_runtime_compact_prefers_current_surface_context_packet_over_stale_runtime_packet() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]
    runtime._last_surface_context_packet = {  # type: ignore[attr-defined]
        "source_state": {
            "listener_action_state": "soft_ack",
            "listener_token_profile": "soft_ack",
        }
    }

    compact = runtime._compact_inner_os_sequence_text(
        [
            {"act": "shared_delight", "text": "\u305d\u308c\u3001\u3061\u3087\u3063\u3068\u7b11\u3048\u308b\u3084\u3064\u3060\u306d\u3002"},
            {"act": "light_bounce", "text": "\u305d\u3046\u3044\u3046\u306e\u3042\u308b\u3068\u3001\u3061\u3087\u3063\u3068\u697d\u306b\u306a\u308b\u3088\u306d\u3002"},
        ],
        surface_profile={"response_length": "short", "voice_texture": "light_playful"},
        discourse_shape={"shape_id": "bright_bounce"},
        surface_context_packet={
            "source_state": {
                "listener_action_state": "warm_laugh_ack",
                "listener_token_profile": "soft_laugh",
            }
        },
    )

    assert compact.startswith("\u3075\u3075\u3063\u3001")
    assert not compact.startswith("\u3046\u3093\u3001")
