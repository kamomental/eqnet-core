from inner_os.learning_mode_state import derive_learning_mode_state


def test_learning_mode_prefers_hold_and_wait_under_recovery_first() -> None:
    state = derive_learning_mode_state(
        self_state={"stress": 0.58, "recovery_need": 0.62},
        body_recovery_guard={"state": "recovery_first", "score": 0.84},
        body_homeostasis_state={"state": "recovering", "score": 0.56},
        homeostasis_budget_state={"state": "depleted", "score": 0.64},
        protection_mode={"mode": "stabilize", "strength": 0.72},
        initiative_readiness={"state": "hold", "score": 0.18},
        agenda_state={"state": "hold", "score": 0.32},
        agenda_window_state={"state": "long_hold", "deferral_budget": 0.74},
        commitment_state={"state": "commit", "target": "hold", "score": 0.62},
        attention_regulation_state={"state": "reflex_guard"},
        grice_guard_state={"state": "hold_obvious_advice"},
        relational_continuity_state={"state": "holding_thread", "score": 0.42},
        social_topology_state={"state": "ambient", "score": 0.12},
        insight_event={"triggered": False, "orient_bias": 0.0},
        identity_arc_kind="stabilizing_self",
        identity_arc_phase="holding",
    )
    assert state.state == "hold_and_wait"
    assert state.probe_room < 0.4


def test_learning_mode_prefers_repair_probe_for_reopening_repair_line() -> None:
    state = derive_learning_mode_state(
        self_state={"stress": 0.24, "recovery_need": 0.18},
        body_recovery_guard={"state": "open", "score": 0.16},
        body_homeostasis_state={"state": "steady", "score": 0.2},
        homeostasis_budget_state={"state": "steady", "score": 0.18},
        protection_mode={"mode": "repair", "strength": 0.54},
        initiative_readiness={"state": "tentative", "score": 0.48},
        agenda_state={"state": "repair", "score": 0.66},
        agenda_window_state={"state": "next_private_window", "deferral_budget": 0.38},
        commitment_state={"state": "commit", "target": "repair", "score": 0.72},
        attention_regulation_state={"state": "selective_hold"},
        grice_guard_state={"state": "attune_without_repeating"},
        relational_continuity_state={"state": "reopening", "score": 0.68},
        social_topology_state={"state": "one_to_one", "score": 0.32},
        insight_event={"triggered": False, "orient_bias": 0.0},
        identity_arc_kind="repairing_bond",
        identity_arc_phase="integrating",
    )
    assert state.state == "repair_probe"
    assert "commitment_state" in state.dominant_inputs


def test_learning_mode_reads_overnight_carry_as_weak_prior() -> None:
    state = derive_learning_mode_state(
        self_state={
            "stress": 0.2,
            "recovery_need": 0.16,
            "learning_mode_focus": "test_small",
            "learning_mode_carry_bias": 0.24,
        },
        body_recovery_guard={"state": "open", "score": 0.14},
        body_homeostasis_state={"state": "steady", "score": 0.18},
        homeostasis_budget_state={"state": "steady", "score": 0.16},
        protection_mode={"mode": "monitor", "strength": 0.36},
        initiative_readiness={"state": "tentative", "score": 0.44},
        agenda_state={"state": "revisit", "score": 0.34},
        agenda_window_state={"state": "now", "deferral_budget": 0.12},
        commitment_state={"state": "settle", "target": "step_forward", "score": 0.46},
        attention_regulation_state={"state": "selective_hold"},
        grice_guard_state={"state": "acknowledge_then_extend"},
        relational_continuity_state={"state": "reopening", "score": 0.42},
        social_topology_state={"state": "one_to_one", "score": 0.24},
        insight_event={"triggered": False, "orient_bias": 0.0},
        identity_arc_kind="growing_edge",
        identity_arc_phase="holding",
    )
    assert state.state == "test_small"
    assert "overnight_learning_mode_carry" in state.dominant_inputs
