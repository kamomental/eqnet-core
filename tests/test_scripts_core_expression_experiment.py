import json

import scripts.core_expression_experiment as experiment
from scripts.core_expression_experiment import run_core_expression_experiment


def test_core_expression_experiment_writes_comparison_package(tmp_path, monkeypatch) -> None:
    responses = iter(
        [
            "baseline normal one",
            "baseline normal two",
            "baseline prompt one",
            "baseline prompt two",
            "baseline router one",
            "baseline router two",
            "eqnet two",
        ]
    )

    def fake_chat_text(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(experiment.terrain_llm, "chat_text", fake_chat_text)

    result = run_core_expression_experiment(
        input_records=[
            {
                "id": "case-1",
                "scenario": "vent_low",
                "core_scenario": "guarded_uncertainty",
                "input": "I am tired and still sorting it out.",
                "gold_speech_act": "support_offer",
            },
            {
                "id": "case-2",
                "scenario": "light_shared",
                "core_scenario": "small_shared_moment",
                "input": "That was a little funny.",
                "gold_speech_act": "small_shared_reaction",
            },
        ],
        out_dir=tmp_path,
        call_llm=True,
        generator_model="generator-test",
        generator_model_label="generator-test",
        classifier_model="classifier-test",
        classifier_model_label="classifier-test",
    )

    assert result["case_count"] == 2
    for filename in [
        "input.jsonl",
        "speech_act_gold.jsonl",
        "baseline_normal.jsonl",
        "baseline_prompt.jsonl",
        "baseline_router.jsonl",
        "eqnet.jsonl",
        "baseline_normal_contract_report.json",
        "baseline_prompt_contract_report.json",
        "baseline_router_contract_report.json",
        "eqnet_contract_report.json",
        "README.md",
    ]:
        assert (tmp_path / filename).exists()

    eqnet_rows = _read_jsonl(tmp_path / "eqnet.jsonl")
    assert eqnet_rows[0]["evaluation_mode"] == "eqnet"
    assert eqnet_rows[0]["run_metadata"]["generator_model_label"] == "generator-test"
    assert eqnet_rows[0]["run_metadata"]["classifier_model_label"] == "classifier-test"
    assert "surface_policy" in eqnet_rows[0]["llm_expression_request"]

    router_rows = _read_jsonl(tmp_path / "baseline_router.jsonl")
    assert router_rows[0]["evaluation_mode"] == "baseline_router"
    assert router_rows[0]["router_mode"]
    assert router_rows[0]["router_rule_name"]
    assert "router_should_call_llm" in router_rows[0]
    assert "router_constraints" in router_rows[0]
    assert "selected_response_channel" in router_rows[0]
    assert "expected_response_channel" in router_rows[0]
    assert "router_mode:" in router_rows[0]["baseline_system_prompt"]


def test_core_expression_experiment_dry_run_keeps_package_shape(tmp_path) -> None:
    result = run_core_expression_experiment(
        input_records=[
            {
                "id": "case-1",
                "scenario": "withdrawal",
                "core_scenario": "guarded_uncertainty",
                "input": "maybe later",
                "gold_speech_act": "other",
            }
        ],
        out_dir=tmp_path,
        call_llm=False,
    )

    assert result["case_count"] == 1
    assert (tmp_path / "baseline_normal_contract_report.json").exists()
    assert (tmp_path / "baseline_router_contract_report.json").exists()
    assert (tmp_path / "eqnet_contract_report.json").exists()


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
