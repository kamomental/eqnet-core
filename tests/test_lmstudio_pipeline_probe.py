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
                "gate_force_listen": True,
                "timing_guard": {
                    "active": True,
                    "reason": "interrupt_guard",
                    "response_channel": "hold",
                    "overlap_policy": "wait_for_release",
                    "emit_not_before_ms": 1420.0,
                    "interrupt_guard_until_ms": 1840.0,
                    "voice_conflict": True,
                },
                "llm_model": "qwen-test",
                "llm_model_source": "live_list",
                "llm_bridge_called": True,
                "force_llm_bridge": True,
                "llm_raw_text": "まずは気になっている点を一つだけ置いてみましょう。",
                "llm_raw_original_text": "縺ｾ縺壹・豌励↓縺ｪ縺｣縺ｦ縺・ｋ轤ｹ繧剃ｸ縺､縺縺醍ｽｮ縺・※縺ｿ縺ｾ縺励ｇ縺・・",
                "llm_raw_model": "qwen-test",
                "llm_raw_model_source": "live_list",
                "llm_raw_differs_from_final": True,
                "llm_raw_contract_ok": False,
                "llm_raw_contract_violations": [
                    "question_block_violation",
                    "assistant_attractor_violation",
                ],
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
                "actuation_response_channel": "hold",
                "actuation_wait_before_action": "extended",
                "actuation_turn_timing_hint": {
                    "entry_window": "held",
                    "pause_profile": "soft_pause",
                    "overlap_policy": "wait_for_release",
                    "interruptibility": "low",
                    "minimum_wait_ms": 420,
                    "interrupt_guard_ms": 420,
                },
                "actuation_emit_timing": {
                    "response_channel": "hold",
                    "entry_window": "held",
                    "pause_profile": "soft_pause",
                    "overlap_policy": "wait_for_release",
                    "interruptibility": "low",
                    "minimum_wait_ms": 420,
                    "interrupt_guard_ms": 420,
                    "effective_emit_delay_ms": 407.5,
                    "effective_latency_ms": 420.0,
                    "emit_not_before_ms": 1420.0,
                    "interrupt_guard_until_ms": 1840.0,
                    "wait_applied": False,
                    "wait_applied_ms": 0.0,
                },
                "reaction_contract": {
                    "stance": "hold",
                    "scale": "micro",
                    "initiative": "yield",
                    "question_budget": 0,
                    "interpretation_budget": "low",
                    "response_channel": "hold",
                    "timing_mode": "held_open",
                    "continuity_mode": "reopen",
                    "distance_mode": "guarded",
                    "closure_mode": "leave_open",
                    "reason_tags": ["repair_then_attune", "hold"],
                },
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
    assert probe.llm_raw_original_text.startswith("縺ｾ縺壹・豌励↓縺ｪ縺｣縺ｦ縺・ｋ轤ｹ")
    assert probe.llm_raw_model == "qwen-test"
    assert probe.llm_raw_model_source == "live_list"
    assert probe.llm_raw_differs_from_final is True
    assert probe.llm_raw_contract_ok is False
    assert "question_block_violation" in probe.llm_raw_contract_violations
    assert probe.response_strategy == "repair_then_attune"
    assert probe.actuation_response_channel == "hold"
    assert probe.actuation_wait_before_action == "extended"
    assert probe.actuation_turn_timing_hint["entry_window"] == "held"
    assert probe.actuation_emit_timing["effective_latency_ms"] == 420.0
    assert probe.actuation_emit_timing["interrupt_guard_until_ms"] == 1840.0
    assert probe.reaction_contract["stance"] == "hold"
    assert probe.reaction_contract["timing_mode"] == "held_open"
    assert probe.gate_force_listen is True
    assert probe.timing_guard["reason"] == "interrupt_guard"
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
    assert probe.discourse_shape["shape_id"]
    assert probe.temporal_alignment["focus_alignment"] is True


def test_render_lmstudio_pipeline_probe_contains_core_sections() -> None:
    result = RuntimeTurnResult(
        talk_mode="watch",
        response_route="conscious",
        metrics={},
        persona_meta={
            "inner_os": {
                "llm_raw_text": "生の LM 文面です。",
                "llm_raw_original_text": "逕溘・ LM 譁・擇縺ｧ縺吶・",
                "llm_raw_model": "probe-model",
                "llm_raw_model_source": "forced",
                "llm_raw_contract_ok": False,
                "llm_raw_contract_violations": ["question_block_violation"],
                "gate_force_listen": True,
                "timing_guard": {
                    "active": True,
                    "reason": "emit_delay",
                    "response_channel": "backchannel",
                    "overlap_policy": "allow_soft_overlap",
                    "emit_not_before_ms": 1040.0,
                    "interrupt_guard_until_ms": 1130.0,
                    "voice_conflict": False,
                },
                "actuation_response_channel": "backchannel",
                "actuation_wait_before_action": "brief",
                "actuation_turn_timing_hint": {
                    "entry_window": "ready",
                    "pause_profile": "none",
                    "overlap_policy": "allow_soft_overlap",
                    "interruptibility": "high",
                    "minimum_wait_ms": 40,
                    "interrupt_guard_ms": 90,
                },
                "actuation_emit_timing": {
                    "response_channel": "backchannel",
                    "entry_window": "ready",
                    "pause_profile": "none",
                    "overlap_policy": "allow_soft_overlap",
                    "interruptibility": "high",
                    "minimum_wait_ms": 40,
                    "interrupt_guard_ms": 90,
                    "effective_emit_delay_ms": 35.0,
                    "effective_latency_ms": 40.0,
                    "emit_not_before_ms": 1040.0,
                    "interrupt_guard_until_ms": 1130.0,
                    "wait_applied": True,
                    "wait_applied_ms": 35.0,
                },
                "reaction_contract": {
                    "stance": "join",
                    "scale": "micro",
                    "initiative": "receive",
                    "question_budget": 0,
                    "interpretation_budget": "none",
                    "response_channel": "backchannel",
                    "timing_mode": "quick_ack",
                    "continuity_mode": "open",
                    "distance_mode": "steady",
                    "closure_mode": "soft_close",
                    "reason_tags": ["backchannel"],
                },
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
    assert "llm_raw_contract_ok: False" in rendered
    assert "channel: backchannel" in rendered
    assert "gate_force_listen: true" in rendered
    assert "wait_before_action: brief" in rendered
    assert "turn_timing_hint:" in rendered
    assert "timing_guard:" in rendered
    assert "emit_timing:" in rendered
    assert "reaction_contract:" in rendered
    assert "timing_mode=quick_ack" in rendered
    assert "emit_not_before_ms=1040.0" in rendered
    assert "question_block_violation" in rendered
    assert "最終応答" in rendered
    assert "Discourse Shape" in rendered
    assert "Content Sequence" in rendered
