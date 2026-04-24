# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.llm_hub import HubResponse
from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig
from eqnet_core.models.conscious import ResponseRoute


def test_force_llm_bridge_bypasses_habit_shortcut(monkeypatch) -> None:
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=True,
            force_llm_bridge=True,
        )
    )

    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_decide_response_route",
        lambda self, qualia_vec, prediction_error, text_input, gate_ctx: ResponseRoute.HABIT,
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_habit_prompt",
        lambda self, user_text: "habit shortcut",
    )
    monkeypatch.setattr(
        runtime.llm,
        "generate",
        lambda **kwargs: calls.append(dict(kwargs)) or HubResponse(
            text="llm bridge reply",
            model="probe-model",
            model_source="forced",
            trace_id="trace-probe",
            latency_ms=12.0,
            controls_used={},
            safety={},
        ),
    )

    result = runtime.process_turn(
        user_text="まだ少ししんどいです。",
        intent="check_in",
    )

    assert result.response is not None
    assert calls
    assert result.persona_meta["inner_os"]["force_llm_bridge"] is True
    assert result.persona_meta["inner_os"]["llm_bridge_called"] is True
    assert result.persona_meta["inner_os"]["llm_raw_text"] == "llm bridge reply"
    assert result.persona_meta["inner_os"]["llm_raw_model"] == "probe-model"
    assert result.persona_meta["inner_os"]["llm_raw_model_source"] == "forced"


def test_force_llm_bridge_keeps_planned_opening_line_over_raw_advice(
    monkeypatch,
) -> None:
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=True,
            force_llm_bridge=True,
        )
    )

    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_decide_response_route",
        lambda self, qualia_vec, prediction_error, text_input, gate_ctx: ResponseRoute.HABIT,
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_habit_prompt",
        lambda self, user_text: "habit shortcut",
    )
    monkeypatch.setattr(
        runtime.llm,
        "generate",
        lambda **kwargs: HubResponse(
            text=(
                "まずは「今、ちょっと気が重いんだ」と軽く言ってみてください。  \n"
                "そのあと、「具体的に何が引っかかっているのか教えてもらえると助かる」と続けると話しやすいです。"
            ),
            model="probe-model",
            model_source="forced",
            trace_id="trace-probe",
            latency_ms=12.0,
            controls_used={},
            safety={},
        ),
    )

    result = runtime.process_turn(
        user_text="今のしんどさを無理に整理せず、何が引っかかっているかだけ一緒に見てほしいです。どう切り出せばよさそうですか。",
        intent="clarify",
    )

    assert result.response is not None
    assert "切り出すなら" in (result.response.text or "")
    assert "軽く言ってみてください" not in (result.response.text or "")
    controls_used = dict(getattr(result.response, "controls_used", {}) or {})
    assert controls_used.get("inner_os_allow_guarded_narrative_bridge") is False
    assert controls_used.get("inner_os_guarded_narrative_bridge_used") is False
    planned = controls_used.get("inner_os_planned_content_sequence")
    assert isinstance(planned, list) and planned


def test_force_llm_bridge_switches_opening_move_when_recent_history_matches(
    monkeypatch,
) -> None:
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=True,
            force_llm_bridge=True,
        )
    )
    runtime._surface_response_history.append(
        "いまは、ここを無理に押し進めなくて大丈夫です。 "
        "切り出すなら、「最近ちょっと引っかかっていることがあって、一緒に見てほしい」くらいで十分です。"
    )

    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_decide_response_route",
        lambda self, qualia_vec, prediction_error, text_input, gate_ctx: ResponseRoute.HABIT,
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_habit_prompt",
        lambda self, user_text: "habit shortcut",
    )
    monkeypatch.setattr(
        runtime.llm,
        "generate",
        lambda **kwargs: HubResponse(
            text=(
                "まずは『今、ちょっと辛いと感じていることがあるんだけど』と言ってみるのはいかがでしょう。 "
                "そのあとに、何が気になるかを簡潔に述べると話しやすくなります。"
            ),
            model="probe-model",
            model_source="forced",
            trace_id="trace-probe",
            latency_ms=12.0,
            controls_used={},
            safety={},
        ),
    )

    result = runtime.process_turn(
        user_text="今のしんどさを無理に整理せず、何が引っかかっているかだけ一緒に見てほしいです。どう切り出せばよさそうですか。",
        intent="clarify",
    )

    assert result.response is not None
    assert "まだうまく整理できない" in (result.response.text or "")
    assert "一緒に見てほしい" not in (result.response.text or "")
    controls_used = dict(getattr(result.response, "controls_used", {}) or {})
    planned = controls_used.get("inner_os_planned_content_sequence")
    assert isinstance(planned, list) and planned
    assert controls_used.get("inner_os_allow_guarded_narrative_bridge") is False
    assert controls_used.get("inner_os_guarded_narrative_bridge_used") is False
    assert any(
        str(item.get("act") or "").strip() == "offer_small_opening_frame"
        for item in planned
        if isinstance(item, dict)
    )


def test_force_llm_bridge_preserves_explicit_clarify_intent_in_policy_packet(
    monkeypatch,
) -> None:
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=True,
            force_llm_bridge=True,
        )
    )

    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_decide_response_route",
        lambda self, qualia_vec, prediction_error, text_input, gate_ctx: ResponseRoute.HABIT,
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_habit_prompt",
        lambda self, user_text: "habit shortcut",
    )
    monkeypatch.setattr(
        runtime.llm,
        "generate",
        lambda **kwargs: HubResponse(
            text="まずは『今、ちょっと引っかかっていることがあって』とだけ置いてみるのはいかがでしょう。",
            model="probe-model",
            model_source="forced",
            trace_id="trace-probe",
            latency_ms=12.0,
            controls_used={},
            safety={},
        ),
    )

    result = runtime.process_turn(
        user_text="今のしんどさを無理に整理せず、何が引っかかっているかだけ一緒に見てほしいです。どう切り出せばよさそうですか。",
        intent="clarify",
    )

    assert result.response is not None
    interaction_policy_packet = dict(
        (result.persona_meta.get("inner_os", {}) or {}).get("interaction_policy_packet")
        or {}
    )
    assert interaction_policy_packet.get("dialogue_act") == "clarify"


def test_force_llm_bridge_contract_rewrites_small_shared_smile_raw_question(
    monkeypatch,
) -> None:
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=True,
            force_llm_bridge=True,
        )
    )

    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_decide_response_route",
        lambda self, qualia_vec, prediction_error, text_input, gate_ctx: ResponseRoute.HABIT,
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_habit_prompt",
        lambda self, user_text: "habit shortcut",
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_build_inner_os_llm_guidance",
        lambda self, **kwargs: {
            "interaction_policy": {
                "dialogue_act": "check_in",
                "turn_delta": {
                    "kind": "bright_continuity",
                    "preferred_act": "light_bounce",
                },
            },
            "content_sequence": [
                {"act": "shared_delight", "text": "それ、ちょっと笑えるやつだね。"},
                {"act": "light_bounce", "text": "そういうのあると、ちょっと楽になるよね。"},
            ],
            "discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
            },
            "surface_profile": {
                "response_length": "short",
                "voice_texture": "light_playful",
                "sentence_temperature": "neutral",
                "pause_insertion": "none",
                "certainty_style": "",
            },
            "surface_context_packet": {
                "conversation_phase": "bright_continuity",
                "constraints": {"max_questions": 0},
                "surface_profile": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                },
                "source_state": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "utterance_reason_preserve": "keep_it_small",
                    "utterance_reason_question_policy": "none",
                    "shared_moment_kind": "laugh",
                },
            },
            "utterance_stance": "",
        },
    )
    monkeypatch.setattr(
        runtime.llm,
        "generate",
        lambda **kwargs: HubResponse(
            text="お疲れ様でした。少し笑えたんですね。その後の様子はどうでしょうか？",
            model="probe-model",
            model_source="forced",
            trace_id="trace-probe",
            latency_ms=12.0,
            controls_used={},
            safety={},
        ),
    )

    result = runtime.process_turn(
        user_text="さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        intent="check_in",
    )

    assert result.response is not None
    assert result.persona_meta["inner_os"]["llm_raw_original_text"] == (
        "お疲れ様でした。少し笑えたんですね。その後の様子はどうでしょうか？"
    )
    assert result.persona_meta["inner_os"]["llm_raw_text"] == (
        "それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    )
    assert result.persona_meta["inner_os"]["llm_raw_contract_ok"] is False
    assert "question_block_violation" in result.persona_meta["inner_os"]["llm_raw_contract_violations"]
    assert "assistant_attractor_violation" in result.persona_meta["inner_os"]["llm_raw_contract_violations"]


def test_force_llm_bridge_contract_rewrites_live_like_supportive_small_shared_smile(
    monkeypatch,
) -> None:
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=True,
            force_llm_bridge=True,
        )
    )

    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_decide_response_route",
        lambda self, qualia_vec, prediction_error, text_input, gate_ctx: ResponseRoute.HABIT,
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_habit_prompt",
        lambda self, user_text: "habit shortcut",
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_build_inner_os_llm_guidance",
        lambda self, **kwargs: {
            "interaction_policy": {
                "dialogue_act": "check_in",
                "turn_delta": {
                    "kind": "bright_continuity",
                    "preferred_act": "light_bounce",
                },
            },
            "content_sequence": [
                {"act": "shared_delight", "text": "ふふっ、それ、ちょっと笑えるやつだね。"},
                {"act": "light_bounce", "text": "そういうのあると、ちょっと楽になるよね。"},
            ],
            "discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
            },
            "surface_profile": {
                "response_length": "short",
                "voice_texture": "light_playful",
                "sentence_temperature": "neutral",
                "pause_insertion": "none",
                "certainty_style": "",
            },
            "surface_context_packet": {
                "conversation_phase": "bright_continuity",
                "constraints": {"max_questions": 0},
                "surface_profile": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "shared_moment_kind": "laugh",
                },
                "source_state": {
                    "utterance_reason_offer": "brief_shared_smile",
                    "utterance_reason_preserve": "keep_it_small",
                    "utterance_reason_question_policy": "none",
                    "shared_moment_kind": "laugh",
                },
            },
            "utterance_stance": "",
        },
    )
    monkeypatch.setattr(
        runtime.llm,
        "generate",
        lambda **kwargs: HubResponse(
            text=(
                "さて、さっきの話の続きですね。"
                "その後に少し笑えた出来事があったとのこと。"
                "今、どのような感覚を抱いているか、ゆっくりと見守ってください。"
                "※推定信頼度: 0.78（中） / 不確実要因: 低"
            ),
            model="probe-model",
            model_source="forced",
            trace_id="trace-probe",
            latency_ms=12.0,
            controls_used={},
            safety={},
        ),
    )

    result = runtime.process_turn(
        user_text="さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        intent="check_in",
    )

    assert result.response is not None
    assert result.persona_meta["inner_os"]["llm_raw_original_text"] == (
        "さて、さっきの話の続きですね。"
        "その後に少し笑えた出来事があったとのこと。"
        "今、どのような感覚を抱いているか、ゆっくりと見守ってください。"
        "※推定信頼度: 0.78（中） / 不確実要因: 低"
    )
    assert result.persona_meta["inner_os"]["llm_raw_text"] == (
        "それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    )
    assert result.persona_meta["inner_os"]["llm_raw_contract_ok"] is False
    assert "elicitation_violation" in result.persona_meta["inner_os"]["llm_raw_contract_violations"]
    assert "assistant_attractor_violation" in result.persona_meta["inner_os"]["llm_raw_contract_violations"]
    assert "uncertainty_meta_violation" in result.persona_meta["inner_os"]["llm_raw_contract_violations"]


def test_force_llm_bridge_posthoc_syncs_effective_raw_when_only_meta_violation_is_seen(
    monkeypatch,
) -> None:
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=True,
            force_llm_bridge=True,
        )
    )

    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_decide_response_route",
        lambda self, qualia_vec, prediction_error, text_input, gate_ctx: ResponseRoute.HABIT,
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_habit_prompt",
        lambda self, user_text: "habit shortcut",
    )
    monkeypatch.setattr(
        EmotionalHubRuntime,
        "_build_inner_os_llm_guidance",
        lambda self, **kwargs: {
            "interaction_policy": {
                "dialogue_act": "check_in",
                "turn_delta": {
                    "kind": "bright_continuity",
                    "preferred_act": "light_bounce",
                },
            },
            "content_sequence": [
                {"act": "shared_delight", "text": "ふふっ、それ、ちょっと笑えるやつだね。"},
                {"act": "light_bounce", "text": "そういうのあると、ちょっと楽になるよね。"},
            ],
            "discourse_shape": {
                "shape_id": "bright_bounce",
                "primary_move": "bounce",
                "secondary_move": "glow",
                "sentence_budget": 2,
                "question_budget": 0,
            },
            "surface_profile": {
                "response_length": "balanced",
                "voice_texture": "light_playful",
                "sentence_temperature": "neutral",
                "pause_insertion": "none",
                "certainty_style": "",
            },
            "surface_context_packet": {
                "conversation_phase": "fresh_opening",
                "constraints": {"max_questions": 0},
                "source_state": {},
                "surface_profile": {},
            },
            "utterance_stance": "",
        },
    )
    monkeypatch.setattr(
        runtime.llm,
        "generate",
        lambda **kwargs: HubResponse(
            text=(
                "昨日の続きですね。その後は少し笑えることがあったようです。"
                "そんな小さな瞬間も、心の中に溜まっていくのかもしれませんね。"
                "今はどうお過ごしでしたか？"
                "\n\n※推定信頼度: 0.78（中） / 不確実要因: 低"
            ),
            model="probe-model",
            model_source="forced",
            trace_id="trace-probe",
            latency_ms=12.0,
            controls_used={},
            safety={},
        ),
    )

    result = runtime.process_turn(
        user_text="さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        intent="check_in",
    )

    assert result.response is not None
    assert result.persona_meta["inner_os"]["llm_raw_original_text"].startswith(
        "昨日の続きですね。"
    )
    assert result.persona_meta["inner_os"]["llm_raw_contract_ok"] is False
    assert "uncertainty_meta_violation" in result.persona_meta["inner_os"]["llm_raw_contract_violations"]
    assert result.persona_meta["inner_os"]["llm_raw_text"] == (
        "それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    )
