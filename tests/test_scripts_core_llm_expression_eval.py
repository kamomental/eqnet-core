from scripts.core_llm_expression_eval import (
    evaluate_core_llm_expression,
    save_eval_jsonl,
)
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
    assert "reaction_contract" in request["user_prompt"]
    assert result["review"]["ok"] is True


def test_core_llm_expression_eval_skips_llm_for_hold_contract() -> None:
    result = evaluate_core_llm_expression(
        scenario_name="guarded_uncertainty",
        call_llm=True,
    )

    assert result["called_llm"] is False
    assert result["llm_expression_request"]["should_call_llm"] is False
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
    assert result["speech_act_analysis"]["source"] == "test_classifier"
    assert result["review"]["ok"] is False
    assert result["review"]["violations"][0]["code"] == "question_block_violation"
