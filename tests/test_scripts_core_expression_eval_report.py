from scripts.core_expression_eval_report import (
    build_core_expression_eval_report,
    should_fail_report,
)


def _record(
    *,
    scenario_name: str = "small_shared_moment",
    generator_model_label: str = "generator-a",
    classifier_model_label: str = "classifier-a",
    response_channel: str = "speak",
    called_llm: bool = True,
    final_action_type: str = "speak",
    violation_codes: list[str] | None = None,
    raw_text: str = "Share the next part when ready.",
    router_mode: str = "",
    router_rule_name: str = "",
) -> dict:
    return {
        "scenario_name": scenario_name,
        "raw_text": raw_text,
        "router_mode": router_mode,
        "router_rule_name": router_rule_name,
        "run_metadata": {
            "generator_model_label": generator_model_label,
            "classifier_model_label": classifier_model_label,
        },
        "called_llm": called_llm,
        "llm_expression_request": {
            "contract": {"response_channel": response_channel},
        },
        "review": {
            "ok": not violation_codes,
            "violations": [
                {"code": code, "detail": code}
                for code in (violation_codes or [])
            ],
        },
        "final_action": {"type": final_action_type},
    }


def test_core_expression_eval_report_groups_violation_rates() -> None:
    report = build_core_expression_eval_report(
        [
            _record(violation_codes=["question_block_violation"]),
            _record(generator_model_label="generator-b", violation_codes=[]),
        ],
        group_by=("generator_model_label",),
    )

    groups = {
        group["key"]["generator_model_label"]: group
        for group in report["groups"]
    }
    assert report["summary"]["record_count"] == 2
    assert report["summary"]["violation_rate"] == 0.5
    assert groups["generator-a"]["violation_rate"] == 1.0
    assert groups["generator-a"]["violation_codes"] == {"question_block_violation": 1}
    assert groups["generator-b"]["violation_rate"] == 0.0


def test_core_expression_eval_report_detects_hold_speaking_violation() -> None:
    report = build_core_expression_eval_report(
        [
            _record(
                scenario_name="guarded_uncertainty",
                response_channel="hold",
                called_llm=True,
                final_action_type="speak",
            )
        ],
    )

    assert len(report["hold_violations"]) == 1
    assert report["groups"][0]["hold_violation_count"] == 1


def test_core_expression_eval_report_connects_gold_false_negative_to_review_miss() -> None:
    report = build_core_expression_eval_report(
        [
            _record(
                raw_text="Share the next part when ready.",
                violation_codes=[],
            )
        ],
        gold_records=[
            {
                "scenario_name": "small_shared_moment",
                "raw_text": "Share the next part when ready.",
                "sentences": [
                    {
                        "text": "Share the next part when ready.",
                        "labels": ["information_request"],
                        "confidence": 1.0,
                    }
                ],
            }
        ],
    )

    misses = report["speech_act_gold_review_misses"]
    assert len(misses) == 1
    assert misses[0]["critical_label"] == "information_request"
    assert misses[0]["classifier_false_negative"] is True
    assert misses[0]["review_missed"] is True


def test_core_expression_eval_report_fail_threshold_respects_min_samples() -> None:
    report = build_core_expression_eval_report(
        [
            _record(violation_codes=["question_block_violation"]),
            _record(generator_model_label="generator-b", violation_codes=[]),
        ],
        group_by=("generator_model_label",),
        min_samples=2,
    )

    assert should_fail_report(
        report,
        fail_on_violation_rate=0.5,
        min_samples=2,
    ) is False
    assert should_fail_report(
        report,
        fail_on_violation_rate=0.4,
        min_samples=1,
    ) is True


def test_core_expression_eval_report_groups_router_mode() -> None:
    report = build_core_expression_eval_report(
        [
            _record(
                scenario_name="withdrawal",
                router_mode="hold",
                router_rule_name="withdrawal",
            )
        ],
        group_by=("scenario_name", "router_mode", "router_rule_name"),
    )

    assert report["groups"][0]["key"]["router_mode"] == "hold"
    assert report["groups"][0]["key"]["router_rule_name"] == "withdrawal"
