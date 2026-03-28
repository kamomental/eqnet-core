# -*- coding: utf-8 -*-

from eqnet_core.models.runtime_turn import RuntimeResponseSummary, RuntimeTurnResult

from emot_terrain_lab.hub.lmstudio_pipeline_probe import (
    build_lmstudio_pipeline_probe,
    render_lmstudio_pipeline_probe,
)


def test_build_lmstudio_pipeline_probe_extracts_runtime_shape() -> None:
    result = RuntimeTurnResult(
        talk_mode="ask",
        response_route="conscious",
        metrics={},
        persona_meta={
            "inner_os": {
                "route": "conscious",
                "talk_mode": "ask",
                "llm_model": "qwen-test",
                "llm_model_source": "live_list",
                "llm_bridge_called": True,
                "force_llm_bridge": True,
                "llm_raw_text": "まずは気になっている点を一つだけ置いてみましょう。",
                "llm_raw_model": "qwen-test",
                "llm_raw_model_source": "live_list",
                "llm_raw_differs_from_final": True,
                "qualia_gate_reason": "normal",
                "interaction_policy_packet": {
                    "dialogue_act": "check_in",
                    "response_strategy": "repair_then_attune",
                    "identity_arc_kind": "repairing_bond",
                    "temporal_membrane_mode": "reentry",
                    "grice_guard_state": {"state": "hold_obvious_advice"},
                    "agenda_window_state": {"state": "next_private_window"},
                    "relational_continuity_state": {"state": "holding_thread"},
                    "social_topology_state": {"state": "one_to_one"},
                },
                "actuation_primary_action": "repair_contact",
                "actuation_execution_mode": "repair_contact",
                "commitment_state": {"target": "repair"},
                "agenda_window_state": {"state": "next_private_window"},
                "continuity_summary": {
                    "same_turn": {
                        "commitment_target": "repair",
                        "agenda_window_state": "next_private_window",
                        "social_topology_state": "one_to_one",
                        "temporal_membrane_mode": "reentry",
                    },
                    "overnight": {
                        "temporal_membrane_focus": "same_group_reentry",
                        "temporal_reentry_bias": 0.22,
                    },
                },
                "dashboard_snapshot": {
                    "temporal_alignment": {
                        "focus_alignment": True,
                        "same_to_overnight_reentry_delta": -0.03,
                    }
                },
            }
        },
        heart={},
        shadow=None,
        qualia_gate={
            "allow": False,
            "suppress_narrative": True,
            "reason": "normal",
            "u_t": 0.71,
            "m_t": 0.18,
            "load_t": 1.0,
            "p_t": 0.31,
        },
        affect=None,
        response=RuntimeResponseSummary(
            text=".. Carefully, we do not have to press this right now.",
            latency_ms=12.5,
            safety=None,
            controls={
                "inner_os_surface_profile": {
                    "opening_delay": "measured",
                    "response_length": "short",
                    "certainty_style": "careful",
                    "content_sequence_length": 2,
                }
            },
            retrieval_summary=None,
            perception_summary=None,
        ),
        memory_reference=None,
    )

    probe = build_lmstudio_pipeline_probe(
        result,
        current_text="まだ少し怖いです。",
        history=["前に話したときも少し戸惑っていました。"],
    )

    assert probe.llm_model == "qwen-test"
    assert probe.llm_model_source == "live_list"
    assert probe.llm_bridge_called is True
    assert probe.force_llm_bridge is True
    assert probe.llm_raw_text.startswith("まずは気になっている点")
    assert probe.llm_raw_model == "qwen-test"
    assert probe.llm_raw_model_source == "live_list"
    assert probe.llm_raw_differs_from_final is True
    assert probe.response_strategy == "repair_then_attune"
    assert probe.commitment_target == "repair"
    assert probe.agenda_window_state == "next_private_window"
    assert probe.temporal_membrane_mode == "reentry"
    assert probe.temporal_membrane_focus == "same_group_reentry"
    assert probe.qualia_gate_allow is False
    assert probe.qualia_gate_suppress_narrative is True
    assert probe.qualia_gate_reason == "normal"
    assert probe.qualia_gate_details["u_t"] == 0.71
    assert probe.content_sequence
    assert probe.planned_content_sequence_present is False
    assert probe.allow_guarded_narrative_bridge is False
    assert probe.guarded_narrative_bridge_used is False
    assert probe.interaction_constraints["prefer_return_point"] is True
    assert probe.repetition_guard["recent_text_count"] == 1
    assert probe.turn_delta["preferred_act"]
    assert probe.temporal_alignment["focus_alignment"] is True


def test_render_lmstudio_pipeline_probe_contains_core_sections() -> None:
    result = RuntimeTurnResult(
        talk_mode="watch",
        response_route="conscious",
        metrics={},
        persona_meta={
            "inner_os": {
                "llm_raw_text": "生の LM 文面です。",
                "llm_raw_model": "probe-model",
                "llm_raw_model_source": "forced",
                "interaction_policy_packet": {},
            }
        },
        heart={},
        shadow=None,
        qualia_gate={},
        affect=None,
        response=RuntimeResponseSummary(
            text="最終応答です。",
            latency_ms=5.0,
            safety=None,
            controls={},
            retrieval_summary=None,
            perception_summary=None,
        ),
        memory_reference=None,
    )
    probe = build_lmstudio_pipeline_probe(result, current_text="大丈夫でしょうか。")
    rendered = render_lmstudio_pipeline_probe(probe)
    assert "LM Studio / EQNet パイプライン" in rendered
    assert "LM Raw Output" in rendered
    assert "raw_model: probe-model / raw_source: forced" in rendered
    assert "最終応答" in rendered
    assert "Content Sequence" in rendered
