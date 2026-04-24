from inner_os.headless_runtime import HeadlessInnerOSRuntime


def test_headless_runtime_returns_action_execution_plan() -> None:
    runtime = HeadlessInnerOSRuntime()
    result = runtime.step(
        actuation_plan={
            "execution_mode": "repair_contact",
            "primary_action": "soft_repair",
            "action_queue": ["name_overreach", "reduce_force", "reopen_carefully"],
            "response_channel": "backchannel",
            "response_channel_score": 0.61,
            "reply_permission": "speak_briefly",
            "wait_before_action": "measured",
            "repair_window_commitment": "active",
            "outcome_goal": "restore_contact_without_pressure",
            "boundary_mode": "softened",
            "attention_target": "person:user",
            "memory_write_priority": "relation_episode",
            "nonverbal_response_state": {
                "state": "soft_ack_presence",
                "response_kind": "backchannel",
                "timing_bias": "gentle_overlap",
            },
            "presence_hold_state": {
                "state": "backchannel_ready_hold",
                "hold_room": 0.42,
                "backchannel_room": 0.67,
            },
            "do_not_cross": ["overinterpret"],
        }
    )
    assert result.execution_mode == "repair_contact"
    assert result.primary_action == "soft_repair"
    assert result.action_queue == ["name_overreach", "reduce_force", "reopen_carefully"]
    assert result.response_channel == "backchannel"
    assert result.response_channel_score == 0.61
    assert result.reply_permission == "speak_briefly"
    assert result.wait_before_action == "measured"
    assert result.nonverbal_response_state["state"] == "soft_ack_presence"
    assert result.presence_hold_state["state"] == "backchannel_ready_hold"
    assert result.turn_timing_hint.response_channel == "backchannel"
    assert result.turn_timing_hint.entry_window == "ready"
    assert result.turn_timing_hint.overlap_policy in {"allow_soft_overlap", "yield_to_user_release"}
    assert result.turn_timing_hint.minimum_wait_ms <= 180
    assert result.to_dict()["attention_target"] == "person:user"
    assert result.to_dict()["response_channel"] == "backchannel"
    assert result.to_dict()["turn_timing_hint"]["interrupt_guard_ms"] >= 80
