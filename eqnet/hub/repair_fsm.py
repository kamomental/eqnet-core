from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Tuple


class RepairState(str, Enum):
    RECOGNIZE = "RECOGNIZE"
    NON_BLAME = "NON_BLAME"
    ACCEPT = "ACCEPT"
    NEXT_STEP = "NEXT_STEP"


class RepairEvent(str, Enum):
    NONE = "NONE"
    TRIGGER = "TRIGGER"
    ACK = "ACK"
    CALM = "CALM"
    COMMIT = "COMMIT"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class RepairSnapshot:
    state: RepairState
    since_ts: float
    reason_codes: List[str]
    last_event: RepairEvent
    cooldown_until: float

    @classmethod
    def initial(cls) -> "RepairSnapshot":
        now = datetime.now(timezone.utc).timestamp()
        return cls(
            state=RepairState.RECOGNIZE,
            since_ts=now,
            reason_codes=[],
            last_event=RepairEvent.NONE,
            cooldown_until=0.0,
        )

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["last_event"] = self.last_event.value
        payload["reason_codes"] = list(self.reason_codes)
        payload["since_ts"] = float(self.since_ts)
        payload["cooldown_until"] = float(self.cooldown_until)
        return payload


def repair_fingerprint(snapshot: RepairSnapshot) -> str:
    raw = json.dumps(snapshot.to_payload(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_TRANSITIONS: dict[Tuple[RepairState, RepairEvent], RepairState] = {
    (RepairState.RECOGNIZE, RepairEvent.ACK): RepairState.NON_BLAME,
    (RepairState.NON_BLAME, RepairEvent.CALM): RepairState.ACCEPT,
    (RepairState.ACCEPT, RepairEvent.COMMIT): RepairState.NEXT_STEP,
}


def apply_repair_event(
    snapshot: RepairSnapshot,
    *,
    event: RepairEvent,
    reason_codes: List[str],
    now_ts: float | None = None,
) -> RepairSnapshot:
    ts = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())
    if event is RepairEvent.NONE:
        return snapshot
    if event is RepairEvent.TRIGGER:
        return RepairSnapshot(
            state=RepairState.RECOGNIZE,
            since_ts=ts,
            reason_codes=list(reason_codes) if reason_codes else list(snapshot.reason_codes),
            last_event=event,
            cooldown_until=snapshot.cooldown_until,
        )
    next_state = _TRANSITIONS.get((snapshot.state, event), snapshot.state)
    return RepairSnapshot(
        state=next_state,
        since_ts=snapshot.since_ts,
        reason_codes=list(reason_codes) if reason_codes else list(snapshot.reason_codes),
        last_event=event,
        cooldown_until=snapshot.cooldown_until,
    )
