# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.llm_hub import HubResponse
from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def test_runtime_surface_disables_guarded_bridge_for_emergency_sequence() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    response = HubResponse(
        text="まずは落ち着いて説明してみてください。",
        model=None,
        trace_id="probe",
        latency_ms=0.0,
        controls_used={
            "inner_os_allow_guarded_narrative_bridge": True,
            "inner_os_llm_raw_text": "まずは事情を順番に説明してもらえると助かります。",
            "inner_os_planned_content_sequence": [
                {
                    "act": "emergency_create_distance",
                    "text": "I am not continuing this at close range. I am taking distance now.",
                }
            ],
        },
        safety={},
    )

    shaped = runtime._apply_inner_os_surface_profile(
        response,
        {
            "planned_content_sequence": [
                {
                    "act": "emergency_create_distance",
                    "text": "I am not continuing this at close range. I am taking distance now.",
                }
            ],
            "interaction_policy_packet": {
                "emergency_posture": {
                    "state": "create_distance",
                    "dialogue_permission": "boundary_only",
                    "primary_action": "create_distance",
                }
            },
            "interaction_policy_dialogue_act": "clarify",
            "surface_opening_delay": "measured",
            "surface_response_length": "balanced",
            "surface_sentence_temperature": "gentle",
            "surface_pause_insertion": "soft_pause",
            "surface_certainty_style": "tentative",
            "surface_banter_move": "none",
            "surface_lexical_variation_mode": "plain",
            "surface_group_register": "ambient",
        },
    )

    assert shaped is not None
    assert "いまは話を続けず、少し距離を取ります。" in shaped.text
    assert "事情を順番に説明" not in shaped.text
    assert shaped.controls_used["inner_os_guarded_narrative_bridge_used"] is False
    assert shaped.controls_used["inner_os_allow_guarded_narrative_bridge"] is False
