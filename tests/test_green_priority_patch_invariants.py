from __future__ import annotations

from eqnet.runtime.nightly.green_bridge import apply_green_priority_patch


def test_green_priority_patch_state_guards_keep_priority_unchanged() -> None:
    policy_on = {
        "schema_version": "green_bridge_policy_v0",
        "policy_version": "green_bridge_policy_v0",
        "policy_source": "configs/green_bridge_policy_v0.yaml",
        "enabled": True,
        "priority_patch": {"enabled": True, "alpha": 0.25, "max_delta": 0.08},
    }
    green_snapshot = {
        "enabled": True,
        "green_response_score": 0.9,
    }
    base = 0.4

    blocked = apply_green_priority_patch(
        base_priority_score=base,
        green_snapshot=green_snapshot,
        policy=policy_on,
        blocked=True,
    )
    assert blocked["priority_score"] == base
    assert blocked["priority_patch_applied"] is False

    suppressed = apply_green_priority_patch(
        base_priority_score=base,
        green_snapshot=green_snapshot,
        policy=policy_on,
        suppressed=True,
    )
    assert suppressed["priority_score"] == base
    assert suppressed["priority_patch_applied"] is False

    unknown = apply_green_priority_patch(
        base_priority_score=base,
        green_snapshot=green_snapshot,
        policy=policy_on,
        unknown=True,
    )
    assert unknown["priority_score"] == base
    assert unknown["priority_patch_applied"] is False

    policy_off = {
        **policy_on,
        "enabled": False,
        "priority_patch": {"enabled": False, "alpha": 0.25, "max_delta": 0.08},
    }
    green_off = apply_green_priority_patch(
        base_priority_score=base,
        green_snapshot={"enabled": False, "green_response_score": 0.9},
        policy=policy_off,
    )
    assert green_off["priority_score"] == base
    assert green_off["priority_patch_applied"] is False


def test_green_priority_patch_respects_clip_max_delta() -> None:
    policy_on = {
        "schema_version": "green_bridge_policy_v0",
        "policy_version": "green_bridge_policy_v0",
        "policy_source": "configs/green_bridge_policy_v0.yaml",
        "enabled": True,
        "priority_patch": {"enabled": True, "alpha": 2.0, "max_delta": 0.05},
    }
    green_snapshot = {
        "enabled": True,
        "green_response_score": 0.95,
    }
    base = 0.3
    out = apply_green_priority_patch(
        base_priority_score=base,
        green_snapshot=green_snapshot,
        policy=policy_on,
    )
    assert out["priority_patch_applied"] is True
    assert out["priority_patch_delta"] == 0.05
    assert out["priority_score"] == 0.35
