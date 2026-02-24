from __future__ import annotations

from pathlib import Path

from eqnet.runtime.companion_policy import (
    companion_policy_meta,
    load_lifelong_companion_policy,
    validate_lifelong_companion_policy,
)


def test_lifelong_companion_policy_contract_is_valid() -> None:
    path = Path("configs/lifelong_companion_policy_v0.yaml")
    policy = load_lifelong_companion_policy(path)
    ok, reasons = validate_lifelong_companion_policy(policy)
    assert ok, reasons
    meta = companion_policy_meta(policy)
    assert str(meta.get("policy_version") or "") == "lifelong_companion_policy_v0"
    assert str(meta.get("policy_source") or "").endswith("configs/lifelong_companion_policy_v0.yaml")
    assert len(str(meta.get("policy_fingerprint") or "")) == 16


def test_lifelong_companion_policy_rejects_missing_safety_guards() -> None:
    path = Path("configs/lifelong_companion_policy_v0.yaml")
    policy = load_lifelong_companion_policy(path)
    policy["principles"]["agency"]["requires_approval_for_actions"] = False
    policy["principles"]["safety"]["non_isolation_required"] = False
    ok, reasons = validate_lifelong_companion_policy(policy)
    assert not ok
    assert "APPROVAL_GATE_REQUIRED" in reasons
    assert "NON_ISOLATION_REQUIRED" in reasons
