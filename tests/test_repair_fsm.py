from __future__ import annotations

from eqnet.hub.repair_fsm import (
    RepairEvent,
    RepairSnapshot,
    RepairState,
    apply_repair_event,
    repair_fingerprint,
)


def test_repair_snapshot_initial_contract() -> None:
    snap = RepairSnapshot.initial()
    payload = snap.to_payload()
    assert payload["state"] == RepairState.RECOGNIZE.value
    assert payload["last_event"] == RepairEvent.NONE.value
    assert isinstance(payload["reason_codes"], list)
    assert isinstance(payload["since_ts"], float)
    assert isinstance(payload["cooldown_until"], float)


def test_repair_fingerprint_stable_for_same_snapshot() -> None:
    snap = RepairSnapshot(
        state=RepairState.NON_BLAME,
        since_ts=1.0,
        reason_codes=["USER_DISTRESS"],
        last_event=RepairEvent.TRIGGER,
        cooldown_until=2.0,
    )
    assert repair_fingerprint(snap) == repair_fingerprint(snap)


def test_repair_transition_sequence_reaches_next_step() -> None:
    snap = RepairSnapshot.initial()
    snap = apply_repair_event(
        snap,
        event=RepairEvent.TRIGGER,
        reason_codes=["USER_DISTRESS"],
        now_ts=100.0,
    )
    assert snap.state == RepairState.RECOGNIZE
    snap = apply_repair_event(snap, event=RepairEvent.ACK, reason_codes=[], now_ts=101.0)
    assert snap.state == RepairState.NON_BLAME
    snap = apply_repair_event(snap, event=RepairEvent.CALM, reason_codes=[], now_ts=102.0)
    assert snap.state == RepairState.ACCEPT
    snap = apply_repair_event(snap, event=RepairEvent.COMMIT, reason_codes=[], now_ts=103.0)
    assert snap.state == RepairState.NEXT_STEP


def test_repair_undefined_transition_keeps_state() -> None:
    snap = RepairSnapshot.initial()
    next_snap = apply_repair_event(snap, event=RepairEvent.COMMIT, reason_codes=[], now_ts=2.0)
    assert next_snap.state == RepairState.RECOGNIZE
