import json
from pathlib import Path

from scripts.core_quickstart_demo import build_core_demo_result


INPUT_PATH = Path("docs/eval_runs/006/input_integrated_axes.jsonl")


def test_eval_run_006_has_thirty_integrated_axis_cases() -> None:
    records = _read_jsonl(INPUT_PATH)

    assert len(records) == 30
    assert len({record["scenario"] for record in records}) == 10
    for record in records:
        context = record["expression_context_state"]
        for key in (
            "memory",
            "green_kernel",
            "culture",
            "norm",
            "body",
            "homeostasis",
            "safety",
            "temperament",
            "qualia_structure_state",
            "qualia_state",
            "protective_trace",
            "sleep",
        ):
            assert key in context, record["id"]
        assert "normalization_stats" in context["qualia_state"]


def test_eval_run_006_projects_key_modes_without_leaking_hidden_trace_to_llm() -> None:
    records = {record["scenario"]: record for record in _read_jsonl(INPUT_PATH)}

    current_crisis = _run(records["protective_current_crisis"])
    rem_replay = _run(records["protective_rem_replay"])
    recovery = _run(records["protective_recovery_window"])
    qualia_fog = _run(records["qualia_novelty_fog"])

    assert current_crisis["protective_trace_palace"]["dominant_mode"] == "protective_hold"
    assert current_crisis["reaction_contract"]["response_channel"] == "hold"
    assert rem_replay["protective_trace_palace"]["dominant_mode"] == "restabilize"
    assert rem_replay["reaction_contract"]["response_channel"] == "hold"
    assert recovery["protective_trace_palace"]["dominant_mode"] in {
        "safe_reconsolidation",
        "recovery_opening",
    }
    assert recovery["reaction_contract"]["response_channel"] == "speak"
    assert qualia_fog["stimulus_history_influence"]["response_bias"] == "hold_for_clarity"
    assert qualia_fog["reaction_contract"]["response_channel"] == "hold"

    for result in (current_crisis, rem_replay, recovery, qualia_fog):
        prompt = result["llm_expression_request"]["user_prompt"]
        assert "protective_trace_" not in prompt
        assert "current_crisis_binding" not in prompt


def _run(record):
    return build_core_demo_result(
        scenario_name=record["core_scenario"],
        input_text=record["input"],
        expression_context_state=record["expression_context_state"],
    )


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
