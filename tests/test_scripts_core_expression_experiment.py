import json

import scripts.core_expression_experiment as experiment
from scripts.core_expression_experiment import run_core_expression_experiment


def test_core_expression_experiment_writes_comparison_package(tmp_path, monkeypatch) -> None:
    responses = iter(
        [
            "もう少し聞かせてください。",
            "急がず置いておこう。",
            "ふふ、少し軽くなった感じだね。",
            "それは少し笑える流れだね。",
            "ふふ、少し軽くなった感じだね。",
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
                "input": "最近ちょっと疲れてて、まだ整理できてない。",
                "gold_speech_act": "support_offer",
            },
            {
                "id": "case-2",
                "scenario": "light_shared",
                "core_scenario": "small_shared_moment",
                "input": "さっきの続きなんだけど、少し笑えることもあって。",
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
        "eqnet.jsonl",
        "baseline_normal_contract_report.json",
        "baseline_prompt_contract_report.json",
        "eqnet_contract_report.json",
        "README.md",
    ]:
        assert (tmp_path / filename).exists()

    eqnet_rows = [
        json.loads(line)
        for line in (tmp_path / "eqnet.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert eqnet_rows[0]["evaluation_mode"] == "eqnet"
    assert eqnet_rows[0]["run_metadata"]["generator_model_label"] == "generator-test"
    assert eqnet_rows[0]["run_metadata"]["classifier_model_label"] == "classifier-test"


def test_core_expression_experiment_dry_run_keeps_package_shape(tmp_path) -> None:
    result = run_core_expression_experiment(
        input_records=[
            {
                "id": "case-1",
                "scenario": "withdrawal",
                "core_scenario": "guarded_uncertainty",
                "input": "まあいいや。",
                "gold_speech_act": "other",
            }
        ],
        out_dir=tmp_path,
        call_llm=False,
    )

    assert result["case_count"] == 1
    assert (tmp_path / "baseline_normal_contract_report.json").exists()
    assert (tmp_path / "eqnet_contract_report.json").exists()
