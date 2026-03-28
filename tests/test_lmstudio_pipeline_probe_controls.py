# -*- coding: utf-8 -*-

from eqnet_core.models.runtime_turn import RuntimeResponseSummary, RuntimeTurnResult

from emot_terrain_lab.hub.lmstudio_pipeline_probe import build_lmstudio_pipeline_probe


def test_build_lmstudio_pipeline_probe_prefers_surface_metadata_in_controls() -> None:
    result = RuntimeTurnResult(
        talk_mode="watch",
        response_route="habit",
        metrics={},
        persona_meta={
            "inner_os": {
                "interaction_policy_packet": {
                    "dialogue_act": "clarify",
                    "response_strategy": "contain_then_stabilize",
                }
            }
        },
        heart={},
        shadow=None,
        qualia_gate={},
        affect=None,
        response=RuntimeResponseSummary(
            text="final text",
            latency_ms=5.0,
            safety=None,
            controls={
                "inner_os_surface_profile": {
                    "opening_delay": "measured",
                    "response_length": "short",
                    "certainty_style": "careful",
                    "content_sequence_length": 2,
                },
                "inner_os_guarded_narrative_bridge_used": False,
                "inner_os_allow_guarded_narrative_bridge": False,
                "inner_os_planned_content_sequence": [
                    {"act": "respect_boundary", "text": "We do not have to press this right now."},
                    {
                        "act": "offer_small_opening_line",
                        "text": "If you want a first line, even 'Something has been catching on me lately, and I want help looking at it' is enough.",
                    },
                ],
            },
            retrieval_summary=None,
            perception_summary=None,
        ),
        memory_reference=None,
    )

    probe = build_lmstudio_pipeline_probe(
        result,
        current_text="どう切り出せばよさそうですか。",
    )

    assert probe.surface_profile["response_length"] == "short"
    assert probe.planned_content_sequence_present is True
    assert probe.allow_guarded_narrative_bridge is False
    assert probe.guarded_narrative_bridge_used is False
    assert any(item["act"] == "offer_small_opening_line" for item in probe.content_sequence)
