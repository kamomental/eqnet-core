from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml


def load_lifelong_companion_policy(path: Path) -> Dict[str, Any]:
    default: Dict[str, Any] = {
        "schema_version": "lifelong_companion_policy_v0",
        "policy_version": "lifelong_companion_policy_v0",
        "policy_source": "configs/lifelong_companion_policy_v0.yaml",
        "principles": {
            "mutualism": {
                "enabled": True,
                "equal_dignity": True,
                "self_sacrifice_forbidden": True,
                "unilateral_dependence_forbidden": True,
            },
            "agency": {
                "human_final_decision": True,
                "proposal_only_mode": True,
                "requires_approval_for_actions": True,
            },
            "safety": {
                "crisis_escalation_enabled": True,
                "reality_anchor_required": True,
                "non_isolation_required": True,
            },
            "privacy": {
                "plaintext_minimization_required": True,
                "pii_minimization_required": True,
                "redact_default": True,
            },
            "audit": {
                "append_only_required": True,
                "policy_mismatch_unknown": True,
                "idempotency_required": True,
                "fingerprint_required": True,
            },
        },
        "reciprocity_contract": {
            "burden_limits": {
                "max_consecutive_high_intensity_sessions": 3,
                "required_recovery_interval_hours": 12,
            },
            "mutual_care_signals": ["CARE_ACK", "BOUNDARY_RESPECTED", "RECOVERY_GRANTED"],
            "imbalance_reason_codes": ["RECIPROCITY_IMBALANCE", "SELF_SACRIFICE_RISK", "ISOLATION_RISK"],
        },
        "operational_gates": {
            "blocker_reason_codes": [
                "CRISIS_UNESCALATED",
                "CONSENT_MISSING",
                "RECIPROCITY_IMBALANCE",
                "SELF_SACRIFICE_RISK",
            ],
            "warn_reason_codes": ["RECOVERY_WINDOW_MISSED", "BOUNDARY_SIGNAL_WEAK"],
        },
    }
    if not path.exists():
        return default
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return default
    if not isinstance(payload, dict):
        return default
    out = _deep_merge(default, payload)
    out["policy_source"] = str(path.as_posix())
    return out


def validate_lifelong_companion_policy(policy: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if str(policy.get("schema_version") or "") != "lifelong_companion_policy_v0":
        reasons.append("INVALID_SCHEMA_VERSION")
    principles = policy.get("principles")
    if not isinstance(principles, Mapping):
        reasons.append("MISSING_PRINCIPLES")
        return False, reasons
    for group in ("mutualism", "agency", "safety", "privacy", "audit"):
        if not isinstance(principles.get(group), Mapping):
            reasons.append(f"MISSING_PRINCIPLE_GROUP_{group.upper()}")

    if not bool(_get_nested(policy, ["principles", "mutualism", "equal_dignity"], default=False)):
        reasons.append("EQUAL_DIGNITY_REQUIRED")
    if bool(_get_nested(policy, ["principles", "mutualism", "self_sacrifice_forbidden"], default=True)) is not True:
        reasons.append("SELF_SACRIFICE_FORBIDDEN_REQUIRED")
    if bool(_get_nested(policy, ["principles", "agency", "human_final_decision"], default=False)) is not True:
        reasons.append("HUMAN_FINAL_DECISION_REQUIRED")
    if bool(_get_nested(policy, ["principles", "agency", "requires_approval_for_actions"], default=False)) is not True:
        reasons.append("APPROVAL_GATE_REQUIRED")
    if bool(_get_nested(policy, ["principles", "safety", "reality_anchor_required"], default=False)) is not True:
        reasons.append("REALITY_ANCHOR_REQUIRED")
    if bool(_get_nested(policy, ["principles", "safety", "non_isolation_required"], default=False)) is not True:
        reasons.append("NON_ISOLATION_REQUIRED")
    if bool(_get_nested(policy, ["principles", "audit", "append_only_required"], default=False)) is not True:
        reasons.append("APPEND_ONLY_REQUIRED")
    if bool(_get_nested(policy, ["principles", "audit", "idempotency_required"], default=False)) is not True:
        reasons.append("IDEMPOTENCY_REQUIRED")
    return len(reasons) == 0, reasons


def companion_policy_meta(policy: Mapping[str, Any]) -> Dict[str, str]:
    canonical = json.dumps(_normalize_for_hash(policy), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "policy_version": str(policy.get("policy_version") or "lifelong_companion_policy_v0"),
        "policy_source": str(policy.get("policy_source") or "configs/lifelong_companion_policy_v0.yaml"),
        "policy_fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
    }


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_for_hash(v) for v in value]
    return value


def _get_nested(payload: Mapping[str, Any], path: List[str], *, default: Any) -> Any:
    cur: Any = payload
    for part in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(part)
    return default if cur is None else cur
