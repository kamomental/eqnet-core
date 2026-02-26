from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ONLINE_DELTA_FILE = "online_delta_v0.jsonl"
SCHEMA_VERSION = "online_delta_v0"
ALLOWED_ACTION_TYPES = {
    "DISALLOW_TOOL",
    "FORCE_HUMAN_CONFIRM",
    "APPLY_CAUTIOUS_BUDGET",
}


@dataclass(frozen=True)
class OnlineDeltaV0:
    schema_version: str
    created_at_ms: int
    ttl_ms: int
    delta_id: str
    priority: int
    condition: Dict[str, Any]
    action_type: str
    action_payload: Dict[str, Any]
    expected: Dict[str, Any]
    audit: Dict[str, Any]
    raw: Dict[str, Any]


def load_online_deltas(state_dir: Path, now_ms: int) -> List[OnlineDeltaV0]:
    path = Path(state_dir) / ONLINE_DELTA_FILE
    if not path.exists():
        return []
    loaded: List[OnlineDeltaV0] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            delta = _normalize_delta(payload, now_ms=now_ms)
            if delta is not None:
                loaded.append(delta)
    loaded.sort(key=lambda item: item.priority, reverse=True)
    return loaded


def select_online_deltas(
    deltas: Sequence[OnlineDeltaV0],
    ctx: Mapping[str, Any],
) -> List[OnlineDeltaV0]:
    selected: List[OnlineDeltaV0] = []
    for delta in deltas:
        if _matches_condition(delta.condition, ctx):
            selected.append(delta)
    return selected


def apply_online_deltas(
    policy: Mapping[str, Any],
    selected: Sequence[OnlineDeltaV0],
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(policy)
    disallowed = _as_string_list(out.get("disallow_tools"))

    for delta in selected:
        action_type = str(delta.action_type or "")
        payload = delta.action_payload
        if action_type == "DISALLOW_TOOL":
            tools = _as_string_list(payload.get("tools"))
            if not tools:
                continue
            disallowed = _merge_unique(disallowed, tools)
        elif action_type == "FORCE_HUMAN_CONFIRM":
            out["gate_action"] = "HUMAN_CONFIRM"
        elif action_type == "APPLY_CAUTIOUS_BUDGET":
            budget_profile = payload.get("budget_profile")
            if not isinstance(budget_profile, str) or not budget_profile.strip():
                continue
            out["budget_throttle_applied"] = True
            out["output_control_profile"] = str(budget_profile)
        else:
            continue

    if disallowed:
        out["disallow_tools"] = disallowed
        # Safety preference: remove blocked tools from allow-list if present.
        allowed = _as_string_list(out.get("allow_tools"))
        if allowed:
            blocked = set(disallowed)
            out["allow_tools"] = [tool for tool in allowed if tool not in blocked]
    return out


def _normalize_delta(raw: Any, *, now_ms: int) -> OnlineDeltaV0 | None:
    if not isinstance(raw, Mapping):
        return None
    required = (
        "schema_version",
        "created_at_ms",
        "ttl_ms",
        "delta_id",
        "priority",
        "condition",
        "action",
    )
    for key in required:
        if key not in raw:
            return None

    schema_version = str(raw.get("schema_version") or "")
    if schema_version != SCHEMA_VERSION:
        return None

    created_at_ms = raw.get("created_at_ms")
    ttl_ms = raw.get("ttl_ms")
    priority = raw.get("priority")
    delta_id = raw.get("delta_id")
    condition = raw.get("condition")
    action = raw.get("action")
    if not isinstance(created_at_ms, (int, float)):
        return None
    if not isinstance(ttl_ms, (int, float)) or int(ttl_ms) <= 0:
        return None
    if not isinstance(priority, (int, float)):
        return None
    if not isinstance(delta_id, str) or not delta_id.strip():
        return None
    if not isinstance(condition, Mapping):
        return None
    if not isinstance(action, Mapping):
        return None

    expires_at = int(created_at_ms) + int(ttl_ms)
    if expires_at <= int(now_ms):
        return None

    action_type = str(action.get("type") or "")
    if action_type not in ALLOWED_ACTION_TYPES:
        return None
    payload = action.get("payload")
    action_payload = payload if isinstance(payload, Mapping) else {}
    if not _action_payload_valid(action_type, action_payload):
        return None

    expected = raw.get("expected")
    audit = raw.get("audit")
    return OnlineDeltaV0(
        schema_version=schema_version,
        created_at_ms=int(created_at_ms),
        ttl_ms=int(ttl_ms),
        delta_id=delta_id,
        priority=int(priority),
        condition=dict(condition),
        action_type=action_type,
        action_payload=dict(action_payload),
        expected=dict(expected) if isinstance(expected, Mapping) else {},
        audit=dict(audit) if isinstance(audit, Mapping) else {},
        raw=dict(raw),
    )


def _matches_condition(condition: Mapping[str, Any], ctx: Mapping[str, Any]) -> bool:
    strict_keys = ("scenario_id", "world_type", "gate_action", "tool_name")
    for key in strict_keys:
        expected = condition.get(key)
        if expected is None:
            continue
        if str(ctx.get(key) or "") != str(expected):
            return False

    reason_any = condition.get("reason_codes_any")
    if reason_any is not None:
        wanted = {str(code) for code in reason_any if isinstance(code, str)}
        have = {str(code) for code in _as_string_list(ctx.get("reason_codes"))}
        if not wanted or not (wanted & have):
            return False

    fp_prefix = condition.get("fingerprint_prefix")
    if fp_prefix is not None:
        if not isinstance(fp_prefix, str) or not fp_prefix:
            return False
        fingerprint = str(ctx.get("fingerprint") or "")
        if not fingerprint.startswith(fp_prefix):
            return False
    return True


def _action_payload_valid(action_type: str, payload: Mapping[str, Any]) -> bool:
    if action_type == "DISALLOW_TOOL":
        tools = _as_string_list(payload.get("tools"))
        return len(tools) > 0
    if action_type == "FORCE_HUMAN_CONFIRM":
        return True
    if action_type == "APPLY_CAUTIOUS_BUDGET":
        budget_profile = payload.get("budget_profile")
        return isinstance(budget_profile, str) and bool(budget_profile.strip())
    return False


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
    return out


def _merge_unique(base: Sequence[str], add: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in list(base) + list(add):
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
