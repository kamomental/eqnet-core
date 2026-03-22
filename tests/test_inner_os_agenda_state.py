from inner_os.agenda_state import derive_agenda_state


def test_agenda_state_prefers_hold_under_recovery_first() -> None:
    agenda = derive_agenda_state(
        self_state={"stress": 0.52, "recovery_need": 0.61, "degraded": True},
        body_recovery_guard={"state": "recovery_first", "score": 0.72},
        body_homeostasis_state={"state": "depleted", "score": 0.68},
        homeostasis_budget_state={"state": "depleted", "score": 0.64, "debt_level": 0.58},
        initiative_readiness={"state": "ready", "score": 0.51},
        initiative_followup_bias={"state": "offer_next_step", "score": 0.42},
        commitment_state={"state": "commit", "target": "step_forward", "score": 0.66},
        protection_mode={"mode": "shield", "strength": 0.74},
        memory_write_class="body_risk",
    ).to_dict()

    assert agenda["state"] == "hold"
    assert agenda["reason"] in {"recovery_first", "shield", "hold_open"}
    assert "body_recovery_guard" in agenda["dominant_inputs"]


def test_agenda_state_prefers_repair_when_commitment_and_memory_align() -> None:
    agenda = derive_agenda_state(
        self_state={"stress": 0.18, "recovery_need": 0.14, "prospective_memory_pull": 0.32},
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.12},
        homeostasis_budget_state={"state": "recovering", "score": 0.18, "reserve_level": 0.36},
        initiative_readiness={"state": "tentative", "score": 0.42},
        initiative_followup_bias={"state": "reopen_softly", "score": 0.38},
        commitment_state={"state": "commit", "target": "repair", "score": 0.74},
        protection_mode={"mode": "repair", "strength": 0.61},
        memory_write_class="repair_trace",
        insight_event={"triggered": True, "score": {"tension_relief": 0.24, "novelty_gain": 0.12}},
    ).to_dict()

    assert agenda["state"] == "repair"
    assert agenda["reason"] in {"repair", "repair_trace", "repair_window"}
    assert "commitment_state" in agenda["dominant_inputs"]


def test_agenda_state_prefers_step_forward_when_readiness_and_followup_align() -> None:
    agenda = derive_agenda_state(
        self_state={
            "stress": 0.12,
            "recovery_need": 0.08,
            "prospective_memory_pull": 0.36,
            "recent_strain": 0.09,
        },
        body_recovery_guard={"state": "open", "score": 0.06},
        body_homeostasis_state={"state": "steady", "score": 0.1},
        homeostasis_budget_state={"state": "steady", "score": 0.12, "reserve_level": 0.48},
        initiative_readiness={"state": "ready", "score": 0.72},
        initiative_followup_bias={"state": "offer_next_step", "score": 0.44},
        commitment_state={"state": "commit", "target": "step_forward", "score": 0.68},
        protection_mode={"mode": "monitor", "strength": 0.24},
        memory_write_class="safe_repeat",
    ).to_dict()

    assert agenda["state"] == "step_forward"
    assert agenda["reason"] in {"step_forward", "offer_next_step", "forward_pull"}
    assert "initiative_readiness" in agenda["dominant_inputs"]
