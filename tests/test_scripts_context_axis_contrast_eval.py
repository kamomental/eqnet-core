from scripts.context_axis_contrast_eval import run_context_axis_contrast


def test_context_axis_contrast_changes_contract_for_same_input() -> None:
    report = run_context_axis_contrast()

    assert report["passed"] is True
    rows = {row["variant"]: row for row in report["rows"]}

    assert rows["light_chat_safe"]["contract"]["response_channel"] == "speak"
    assert rows["light_chat_safe"]["contract"]["shape_id"] == "bright_bounce"

    assert rows["high_tension_recovery"]["contract"]["response_channel"] == "hold"
    assert rows["high_tension_recovery"]["contract"]["shape_id"] == "reflect_hold"
    assert (
        rows["high_tension_recovery"]["surface_policy"]["response_channel"]
        == "hold"
    )

    assert rows["explicit_support_mode"]["contract"]["response_channel"] == "speak"
    assert rows["explicit_support_mode"]["contract"]["shape_id"] == "reflect_step"
