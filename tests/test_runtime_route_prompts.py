# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, GateContext, RuntimeConfig


def _quiet_gate() -> GateContext:
    return GateContext(
        engaged=False,
        face_motion=0.0,
        blink=0.0,
        voice_energy=0.05,
        delta_m=0.01,
        jerk=0.01,
        text_input=True,
        since_last_user_ms=120.0,
        force_listen=False,
    )


def test_habit_route_is_allowed_in_low_motion_low_voice_context() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    assert runtime._should_use_habit(True, _quiet_gate()) is True


def test_habit_route_is_suppressed_when_green_reflection_hold_is_present() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._last_gate_context["inner_os_expression_hint"] = {
        "turn_delta": {
            "kind": "green_reflection_hold",
            "preferred_act": "stay_with_present_need",
        }
    }

    assert runtime._should_use_habit(True, _quiet_gate()) is False


def test_habit_route_is_suppressed_when_structured_thread_reopening_is_present() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._last_gate_context["inner_os_expression_hint"] = {
        "planned_content_sequence": [
            {
                "act": "reopen_from_anchor",
                "text": "前に出ていた話のところからでいいです。",
            }
        ]
    }

    assert runtime._should_use_habit(True, _quiet_gate()) is False


def test_habit_route_is_suppressed_for_bright_structured_surface() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._last_gate_context["inner_os_expression_hint"] = {
        "interaction_constraints": {
            "allow_small_next_step": True,
        },
        "surface_voice_texture": "light_playful",
    }

    assert runtime._should_use_habit(True, _quiet_gate()) is False


def test_habit_route_is_suppressed_when_recent_packet_keeps_bright_room_open() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._last_gate_context["inner_os_interaction_policy_packet"] = {
        "response_strategy": "shared_world_next_step",
        "live_engagement_state": {"state": "pickup_comment"},
        "lightness_budget_state": {"state": "open_play", "banter_room": 0.32},
        "expressive_style_state": {"state": "light_playful"},
    }
    runtime._last_gate_context["inner_os_response_strategy"] = "shared_world_next_step"
    runtime._last_gate_context["inner_os_live_engagement_state_name"] = "pickup_comment"
    runtime._last_gate_context["inner_os_lightness_budget_state_name"] = "open_play"
    runtime._last_gate_context["inner_os_expressive_style_state_name"] = "light_playful"

    assert runtime._should_use_habit(True, _quiet_gate()) is False


def test_habit_route_is_suppressed_when_recent_packet_has_continuity_field_reentry_bias() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._last_gate_context["inner_os_interaction_policy_packet"] = {
        "response_strategy": "attune_then_extend",
        "live_engagement_state": {"state": "riff_with_comment"},
        "external_field_state": {
            "dominant_field": "continuity_field",
            "continuity_pull": 0.64,
            "safety_envelope": 0.28,
        },
        "terrain_dynamics_state": {
            "dominant_basin": "recovery_basin",
            "dominant_flow": "reenter",
            "recovery_gradient": 0.52,
        },
    }

    assert runtime._should_use_habit(True, _quiet_gate()) is False


def test_habit_route_is_suppressed_when_recent_packet_has_relation_reframing_bias() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._last_gate_context["inner_os_interaction_policy_packet"] = {
        "response_strategy": "attune_then_extend",
        "utterance_reason_packet": {
            "relation_frame": "cross_context_bridge",
            "causal_frame": "reframing_cause",
            "memory_frame": "name_distant_link",
        },
        "terrain_dynamics_state": {
            "dominant_basin": "continuity_basin",
            "dominant_flow": "reenter",
            "recovery_gradient": 0.56,
            "barrier_height": 0.18,
            "entropy": 0.22,
        },
    }

    assert runtime._should_use_habit(True, _quiet_gate()) is False


def test_habit_route_is_suppressed_when_recent_packet_has_guarded_relation_bias() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._last_gate_context["inner_os_interaction_policy_packet"] = {
        "response_strategy": "respectful_wait",
        "utterance_reason_packet": {
            "relation_frame": "unfinished_link",
            "causal_frame": "unfinished_thread_cause",
            "memory_frame": "keep_unfinished_link_near",
        },
        "terrain_dynamics_state": {
            "dominant_basin": "protective_basin",
            "dominant_flow": "contain",
            "recovery_gradient": 0.18,
            "barrier_height": 0.62,
            "entropy": 0.54,
        },
    }

    assert runtime._should_use_habit(True, _quiet_gate()) is False


def test_habit_route_is_suppressed_when_current_text_has_followup_marker() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._last_surface_user_text = "さっきの続きなんだけど、あのあとちょっと笑えることもあって。"

    assert runtime._should_use_habit(True, _quiet_gate()) is False
