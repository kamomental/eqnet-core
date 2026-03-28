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
