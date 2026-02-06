from __future__ import annotations

from eqnet.hub.output_control import apply_policy_prior
from eqnet.hub.repair_fsm import RepairEvent, RepairSnapshot, RepairState
from eqnet.runtime.policy import PolicyPrior


def test_apply_policy_prior_repair_overlay_active_sets_repair_mode() -> None:
    prior = PolicyPrior(warmth=0.7, directness=0.6, self_disclosure=0.5, calmness=0.7)
    snap = RepairSnapshot(
        state=RepairState.RECOGNIZE,
        since_ts=1.0,
        reason_codes=["USER_DISTRESS"],
        last_event=RepairEvent.TRIGGER,
        cooldown_until=0.0,
    )
    out = apply_policy_prior(
        prior,
        day_key="2026-02-06",
        episode_id="ep-x",
        repair_snapshot=snap,
    )
    assert out.response_style_mode == "repair"
    assert out.repair_state == "RECOGNIZE"
    assert out.recall_budget_override <= 1


def test_apply_policy_prior_repair_overlay_next_step_restores_normal_mode() -> None:
    prior = PolicyPrior(warmth=0.6, directness=0.5, self_disclosure=0.5, calmness=0.6)
    snap = RepairSnapshot(
        state=RepairState.NEXT_STEP,
        since_ts=1.0,
        reason_codes=["USER_DISTRESS"],
        last_event=RepairEvent.COMMIT,
        cooldown_until=0.0,
    )
    out = apply_policy_prior(
        prior,
        day_key="2026-02-06",
        episode_id="ep-y",
        repair_snapshot=snap,
    )
    assert out.response_style_mode != "repair"
    assert out.repair_state == "NEXT_STEP"

