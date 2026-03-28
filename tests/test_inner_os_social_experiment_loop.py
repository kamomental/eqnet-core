from inner_os.social_experiment_loop import derive_social_experiment_loop_state


def test_social_experiment_prefers_repair_signal_probe() -> None:
    state = derive_social_experiment_loop_state(
        learning_mode_state={"state": "repair_probe", "score": 0.72, "probe_room": 0.58, "update_bias": 0.44},
        commitment_state={"state": "commit", "target": "repair", "score": 0.76},
        agenda_state={"state": "repair", "score": 0.64},
        agenda_window_state={"state": "next_private_window"},
        body_recovery_guard={"state": "open", "score": 0.16},
        protection_mode={"mode": "repair", "strength": 0.52},
        grice_guard_state={"state": "attune_without_repeating"},
        relational_continuity_state={"state": "reopening", "score": 0.66},
        social_topology_state={"state": "one_to_one", "visibility_pressure": 0.08, "hierarchy_pressure": 0.06},
        self_state={"degraded": False},
        identity_arc_kind="repairing_bond",
    )
    assert state.state == "repair_signal_probe"
    assert state.hypothesis == "small_repair_softens_contact"


def test_social_experiment_prefers_hold_probe_when_window_is_long_hold() -> None:
    state = derive_social_experiment_loop_state(
        learning_mode_state={"state": "hold_and_wait", "score": 0.82, "probe_room": 0.18, "update_bias": 0.22},
        commitment_state={"state": "commit", "target": "hold", "score": 0.62},
        agenda_state={"state": "hold", "score": 0.42},
        agenda_window_state={"state": "long_hold"},
        body_recovery_guard={"state": "recovery_first", "score": 0.82},
        protection_mode={"mode": "stabilize", "strength": 0.74},
        grice_guard_state={"state": "hold_obvious_advice"},
        relational_continuity_state={"state": "holding_thread", "score": 0.44},
        social_topology_state={"state": "public_visible", "visibility_pressure": 0.52, "hierarchy_pressure": 0.24},
        self_state={"degraded": True},
        identity_arc_kind="holding_thread",
    )
    assert state.state == "hold_probe"
    assert state.stop_rule in {"until_thread_reopens_naturally", "until_private_window"}


def test_social_experiment_reads_overnight_carry_as_weak_prior() -> None:
    state = derive_social_experiment_loop_state(
        learning_mode_state={"state": "test_small", "score": 0.56, "probe_room": 0.54, "update_bias": 0.38},
        commitment_state={"state": "settle", "target": "step_forward", "score": 0.52},
        agenda_state={"state": "revisit", "score": 0.36},
        agenda_window_state={"state": "now"},
        body_recovery_guard={"state": "open", "score": 0.14},
        protection_mode={"mode": "monitor", "strength": 0.34},
        grice_guard_state={"state": "acknowledge_then_extend"},
        relational_continuity_state={"state": "reopening", "score": 0.4},
        social_topology_state={"state": "one_to_one", "visibility_pressure": 0.06, "hierarchy_pressure": 0.04},
        self_state={
            "degraded": False,
            "social_experiment_focus": "confirm_shared_direction",
            "social_experiment_carry_bias": 0.22,
        },
        identity_arc_kind="growing_edge",
    )
    assert state.state in {"test_small_step", "confirm_shared_direction"}
    assert "overnight_social_experiment_carry" in state.dominant_inputs
