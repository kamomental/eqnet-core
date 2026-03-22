from inner_os.headless_runtime import HeadlessInnerOSRuntime


def test_headless_runtime_returns_action_execution_plan() -> None:
    runtime = HeadlessInnerOSRuntime()
    result = runtime.step(
        actuation_plan={
            "execution_mode": "repair_contact",
            "primary_action": "soft_repair",
            "action_queue": ["name_overreach", "reduce_force", "reopen_carefully"],
            "reply_permission": "speak_briefly",
            "wait_before_action": "measured",
            "repair_window_commitment": "active",
            "outcome_goal": "restore_contact_without_pressure",
            "boundary_mode": "softened",
            "attention_target": "person:user",
            "memory_write_priority": "relation_episode",
            "do_not_cross": ["overinterpret"],
        }
    )
    assert result.execution_mode == "repair_contact"
    assert result.primary_action == "soft_repair"
    assert result.action_queue == ["name_overreach", "reduce_force", "reopen_carefully"]
    assert result.reply_permission == "speak_briefly"
    assert result.wait_before_action == "measured"
    assert result.to_dict()["attention_target"] == "person:user"
