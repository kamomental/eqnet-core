from scripts.core_llm_expression_eval import (
    evaluate_core_llm_expression,
    find_speech_act_analysis,
    load_speech_act_analysis_jsonl,
    save_eval_jsonl,
)
import json
import os
import scripts.core_llm_expression_eval as core_llm_expression_eval


def test_core_llm_expression_eval_dry_run_exposes_state_conditioned_request() -> None:
    result = evaluate_core_llm_expression(
        scenario_name="small_shared_moment",
        call_llm=False,
    )

    request = result["llm_expression_request"]
    assert result["called_llm"] is False
    assert result["run_metadata"]["model_label"] == "unconfigured"
    assert request["should_call_llm"] is True
    assert request["contract"]["response_channel"] == "speak"
    assert request["surface_policy"]["response_channel"] == "speak"
    assert "surface_policy" in request["user_prompt"]
    assert "reaction_contract" in request["user_prompt"]
    assert result["review"]["ok"] is True


def test_core_llm_expression_eval_skips_llm_for_hold_contract() -> None:
    result = evaluate_core_llm_expression(
        scenario_name="guarded_uncertainty",
        call_llm=True,
    )

    assert result["called_llm"] is False
    assert result["llm_expression_request"]["should_call_llm"] is False
    assert result["llm_expression_request"]["surface_policy"]["response_channel"] == "hold"
    assert result["llm_expression_request"]["surface_policy"]["max_sentences"] == 0
    assert result["final_action"]["type"] == "nonverbal"
    assert result["final_action"]["name"] == "presence_hold"


def test_core_llm_expression_eval_records_model_label() -> None:
    result = evaluate_core_llm_expression(
        scenario_name="small_shared_moment",
        call_llm=False,
        model_label="local-test-model",
    )

    assert result["run_metadata"]["model_label"] == "local-test-model"


def test_save_eval_jsonl_appends_record(tmp_path) -> None:
    output_path = tmp_path / "eval.jsonl"
    record = {"scenario_name": "small_shared_moment", "review": {"ok": True}}

    saved_path = save_eval_jsonl(output_path, record)

    assert saved_path == output_path
    assert output_path.read_text(encoding="utf-8").strip()


def test_core_llm_expression_eval_can_classify_output_before_review(monkeypatch) -> None:
    responses = iter(
        [
            "Share the next part when ready.",
            (
                '{"schema_version":"speech_act.v1","source":"test_classifier",'
                '"sentences":[{"text":"Share the next part when ready.",'
                '"labels":["information_request"],"confidence":0.92}]}'
            ),
        ]
    )

    def fake_chat_text(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(
        core_llm_expression_eval.terrain_llm,
        "chat_text",
        fake_chat_text,
    )

    result = evaluate_core_llm_expression(
        scenario_name="small_shared_moment",
        call_llm=True,
        classify_output=True,
    )

    assert result["run_metadata"]["classify_output"] is True
    assert result["run_metadata"]["classifier_model_label"] == "unconfigured"
    assert result["speech_act_analysis"]["source"] == "test_classifier"
    assert result["review"]["ok"] is False
    assert result["review"]["violations"][0]["code"] == "question_block_violation"


def test_core_llm_expression_eval_can_use_separate_classifier_model(monkeypatch) -> None:
    calls = []
    responses = iter(
        [
            "Share the next part when ready.",
            (
                '{"schema_version":"speech_act.v1","source":"separate_classifier",'
                '"sentences":[{"text":"Share the next part when ready.",'
                '"labels":["information_request"],"confidence":0.92}]}'
            ),
        ]
    )

    def fake_chat_text(*args, **kwargs):
        calls.append(os.environ.get("OPENAI_MODEL"))
        return next(responses)

    monkeypatch.setattr(
        core_llm_expression_eval.terrain_llm,
        "chat_text",
        fake_chat_text,
    )

    result = evaluate_core_llm_expression(
        scenario_name="small_shared_moment",
        call_llm=True,
        classify_output=True,
        classifier_model_label="classifier-audit-model",
        classifier_model="classifier-audit-model",
    )

    assert calls == [None, "classifier-audit-model"]
    assert result["run_metadata"]["classifier_model_label"] == "classifier-audit-model"
    assert result["speech_act_analysis"]["source"] == "separate_classifier"


def test_speech_act_jsonl_loader_can_find_human_label(tmp_path) -> None:
    path = tmp_path / "speech_act.jsonl"
    path.write_text(
        json.dumps(
            {
                "scenario_name": "small_shared_moment",
                "raw_text": "Share the next part when ready.",
                "source": "human_label",
                "sentences": [
                    {
                        "text": "Share the next part when ready.",
                        "labels": ["information_request"],
                        "confidence": 1.0,
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_speech_act_analysis_jsonl(path)
    analysis = find_speech_act_analysis(
        records,
        scenario_name="small_shared_moment",
        raw_text="Share the next part when ready.",
    )

    assert analysis is not None
    assert analysis["source"] == "human_label"
    assert analysis["sentences"][0]["labels"] == ["information_request"]
