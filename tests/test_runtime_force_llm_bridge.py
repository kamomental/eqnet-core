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
