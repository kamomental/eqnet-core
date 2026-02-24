from __future__ import annotations

import json
from pathlib import Path

from eqnet.runtime.adaptive_fsm import (
    AdaptiveMode,
    load_fsm_policy,
    policy_fingerprint,
    reduce_latest_mode,
    reduce_mode_sequence,
)


def test_reduce_mode_sequence_matches_golden_fixture() -> None:
    fixture_path = Path("tests/fixtures/golden/fsm_mode_sequence_v0.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    policy = load_fsm_policy()
    snapshots = reduce_mode_sequence(fixture["events"], policy=policy)
    got_modes = [snap.mode.value for snap in snapshots]
    got_reason_codes = [snap.reason_codes for snap in snapshots]
    assert got_modes == fixture["expected_modes"]
    assert got_reason_codes == fixture["expected_reason_codes"]
    expected_fp = policy_fingerprint(policy)
    assert all(snap.policy_fingerprint == expected_fp for snap in snapshots)
    assert all(snap.policy_version == "fsm_policy_v0" for snap in snapshots)
    assert all("fsm_policy_v0.yaml" in snap.policy_source for snap in snapshots)


def test_reduce_latest_mode_defaults_to_stable_on_empty_events() -> None:
    policy = load_fsm_policy()
    snapshot = reduce_latest_mode([], policy=policy)
    assert snapshot.mode == AdaptiveMode.STABLE
    assert snapshot.evidence == {}
    assert snapshot.reason_codes == []
    assert snapshot.policy_fingerprint == policy_fingerprint(policy)
    assert "fsm_policy_v0.yaml" in snapshot.policy_source
