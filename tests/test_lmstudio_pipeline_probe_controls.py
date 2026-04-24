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
                "inner_os_reaction_contract": {
                    "stance": "witness",
                    "scale": "small",
                    "initiative": "receive",
                    "question_budget": 0,
                    "interpretation_budget": "low",
                    "response_channel": "speak",
                    "timing_mode": "brief_wait",
                    "continuity_mode": "fresh",
                    "distance_mode": "steady",
                    "closure_mode": "soft_open",
                    "reason_tags": ["contain_then_stabilize", "anchor_reopen"],
                },
                "inner_os_discourse_shape": {
                    "shape_id": "anchor_reopen",
                    "primary_move": "reopen",
                    "secondary_move": "return_point",
                    "sentence_budget": 2,
                    "question_budget": 0,
                    "anchor_mode": "explicit",
                    "closing_mode": "return_point",
                    "energy": "contained",
                    "brightness": 0.1,
                    "playfulness": 0.0,
                    "tempo": 0.2,
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
    assert probe.reaction_contract["question_budget"] == 0
    assert probe.reaction_contract["interpretation_budget"] == "low"
    assert probe.reaction_contract["continuity_mode"] == "fresh"
    assert probe.discourse_shape["shape_id"] == "anchor_reopen"
    assert probe.planned_content_sequence_present is True
    assert probe.allow_guarded_narrative_bridge is False
    assert probe.guarded_narrative_bridge_used is False
    assert any(item["act"] == "offer_small_opening_line" for item in probe.content_sequence)


def test_build_lmstudio_pipeline_probe_aligns_bright_sequence_to_final_response_text() -> None:
    result = RuntimeTurnResult(
        talk_mode="watch",
        response_route="conscious",
        metrics={},
        persona_meta={
            "inner_os": {
                "interaction_policy_packet": {
                    "dialogue_act": "check_in",
                    "response_strategy": "shared_world_next_step",
                }
            }
        },
        heart={},
        shadow=None,
        qualia_gate={},
        affect=None,
        response=RuntimeResponseSummary(
            text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
            latency_ms=5.0,
            safety=None,
            controls={
                "inner_os_discourse_shape": {
                    "shape_id": "bright_bounce",
                    "primary_move": "bounce",
                    "secondary_move": "glow",
                    "sentence_budget": 2,
                    "question_budget": 0,
                    "anchor_mode": "implicit",
                    "closing_mode": "open_light",
                    "energy": "bright",
                    "brightness": 0.52,
                    "playfulness": 0.44,
                    "tempo": 0.49,
                },
                "inner_os_planned_content_sequence": [
                    {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
                    {"act": "light_bounce", "text": "それは、ちょっと気持ちが軽くなるね。"},
                ],
            },
            retrieval_summary=None,
            perception_summary=None,
        ),
        memory_reference=None,
    )

    probe = build_lmstudio_pipeline_probe(
        result,
        current_text="さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
    )

    assert probe.discourse_shape["shape_id"] == "bright_bounce"
    assert probe.reaction_contract["scale"] == "small"
    assert probe.reaction_contract["question_budget"] == 0
    assert probe.reaction_contract["interpretation_budget"] == "none"
    assert probe.content_sequence[0]["text"].startswith("ふふっ、")
    assert "笑えるやつ" in probe.content_sequence[0]["text"]
    assert "ちょっと楽になるよね" in probe.content_sequence[1]["text"]


def test_build_lmstudio_pipeline_probe_post_reviews_live_like_raw_bright_moment() -> None:
    response = RuntimeResponseSummary(
        text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
        latency_ms=5.0,
        safety=None,
        controls={
            "inner_os_discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
                "anchor_mode": "implicit",
                "closing_mode": "open_light",
                "energy": "bright",
                "brightness": 0.52,
                "playfulness": 0.44,
                "tempo": 0.49,
            },
            "inner_os_planned_content_sequence": [
                {"act": "shared_delight", "text": "それは、ちょっといい感じだね。"},
                {"act": "light_bounce", "text": "それは、ちょっと気持ちが軽くなるね。"},
            ],
        },
        retrieval_summary=None,
        perception_summary=None,
    )
    response.controls_used = {
        "inner_os_llm_raw_text": (
            "あとの話ですね。少し笑えたとのこと、よかった。\n"
            "その後の流れは、まだ整理中でないかもしれませんね。\n"
            "ただ、そう思えた瞬間があったなら、そこには確かに何かあったのでしょう。\n"
            "ゆっくりと、でも確実に、あの続きが見えてくるでしょう。\n\n"
            "※推定信頼度: 0.78（中） / 不確実要因: 低"
        ),
        "inner_os_llm_raw_original_text": (
            "あとの話ですね。少し笑えたとのこと、よかった。\n"
            "その後の流れは、まだ整理中でないかもしれませんね。\n"
            "ただ、そう思えた瞬間があったなら、そこには確かに何かあったのでしょう。\n"
            "ゆっくりと、でも確実に、あの続きが見えてくるでしょう。\n\n"
            "※推定信頼度: 0.78（中） / 不確実要因: 低"
        ),
        "inner_os_llm_raw_contract_ok": False,
        "inner_os_llm_raw_contract_violations": ["uncertainty_meta_violation"],
        "inner_os_reaction_contract": {
            "stance": "join",
            "scale": "small",
            "initiative": "co_move",
            "question_budget": 0,
            "interpretation_budget": "none",
            "response_channel": "speak",
            "timing_mode": "immediate",
            "continuity_mode": "continue",
            "distance_mode": "near",
            "closure_mode": "open_light",
            "reason_tags": ["bright_bounce", "brief_shared_smile"],
        },
    }

    result = RuntimeTurnResult(
        talk_mode="talk",
        response_route="conscious",
        metrics={},
        persona_meta={
            "inner_os": {
                "interaction_policy_packet": {
                    "dialogue_act": "check_in",
                    "response_strategy": "shared_world_next_step",
                    "recent_dialogue_state": {"state": "continuing_thread"},
                    "turn_delta": {
                        "kind": "bright_continuity",
                        "preferred_act": "light_bounce",
                    },
                }
            }
        },
        heart={},
        shadow=None,
        qualia_gate={},
        affect=None,
        response=response,
        memory_reference=None,
    )

    probe = build_lmstudio_pipeline_probe(
        result,
        current_text="さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
    )

    assert probe.llm_raw_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert probe.reaction_contract["question_budget"] == 0
    assert probe.reaction_contract["interpretation_budget"] == "none"
    assert probe.llm_raw_contract_ok is False
    assert "assistant_attractor_violation" in probe.llm_raw_contract_violations
    assert "interpretive_bright_violation" in probe.llm_raw_contract_violations
    assert "uncertainty_meta_violation" in probe.llm_raw_contract_violations
