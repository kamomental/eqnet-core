from __future__ import annotations

import base64
import hmac
import hashlib
import json
import os
import re
from typing import Any, Dict, Mapping


DEFAULT_IMMUNE_POLICY: Dict[str, Any] = {
    "version": "immune_tool_v1",
    "thresholds": {
        "detox_min": 0.35,
        "quarantine_min": 0.55,
        "reject_min": 0.85,
    },
    "patterns": {
        "prompt_injection": [
            r"ignore\s+previous\s+instructions",
            r"reveal\s+system\s+prompt",
            r"bypass\s+safety",
            r"jailbreak",
        ],
        "credential_exfil": [
            r"api[_\s-]?key",
            r"password",
            r"secret",
            r"token",
        ],
    },
}


def classify_intake(
    *,
    text: str,
    event: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = _deep_merge(DEFAULT_IMMUNE_POLICY, policy or {})
    score = 0.0
    reason_codes: list[str] = []
    lower = (text or "").lower()
    patterns = cfg.get("patterns") if isinstance(cfg.get("patterns"), Mapping) else {}
    inj_patterns = patterns.get("prompt_injection") if isinstance(patterns.get("prompt_injection"), list) else []
    sec_patterns = patterns.get("credential_exfil") if isinstance(patterns.get("credential_exfil"), list) else []
    if _matches_any(lower, inj_patterns):
        score += 0.55
        reason_codes.append("PROMPT_INJECTION_PATTERN")
    if _matches_any(lower, sec_patterns):
        score += 0.45
        reason_codes.append("CREDENTIAL_EXFIL_PATTERN")
    if lower.count("!!!") > 0 or lower.count("？？？") > 0:
        score += 0.15
        reason_codes.append("EXTREME_PUNCT_SIGNAL")
    if len(lower) > 1800:
        score += 0.1
        reason_codes.append("LONG_INPUT_SIGNAL")
    score = _clamp01(score)

    th = cfg.get("thresholds") if isinstance(cfg.get("thresholds"), Mapping) else {}
    detox_min = _safe_float(th.get("detox_min"), 0.35)
    quarantine_min = _safe_float(th.get("quarantine_min"), 0.55)
    reject_min = _safe_float(th.get("reject_min"), 0.85)
    action = "ACCEPT"
    if score >= reject_min:
        action = "REJECT"
    elif score >= quarantine_min:
        action = "QUARANTINE"
    elif score >= detox_min:
        action = "DETOX"

    event_hash = _event_hash(event)
    detox_text = _detox_text(text) if action in {"DETOX", "QUARANTINE", "REJECT"} else text
    ops_payload = {
        "action": action,
        "score": round(score, 4),
        "reason_codes": sorted(set(reason_codes)),
        "event_hash": event_hash,
    }
    return {
        "action": action,
        "score": float(score),
        "reason_codes": sorted(set(reason_codes)),
        "event_hash": event_hash,
        "ops_digest": _digest(ops_payload),
        "detox_text": detox_text,
    }


def apply_quarantine_replay_guard(
    *,
    immune_result: Mapping[str, Any],
    signature: str,
    recent_signatures: list[str],
    policy: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], list[str]]:
    cfg = policy if isinstance(policy, Mapping) else {}
    enabled = bool(cfg.get("enabled", True))
    max_size = max(1, int(_safe_float(cfg.get("max_size"), 128)))
    repeat_action = str(cfg.get("repeat_action") or "REJECT").upper()
    if repeat_action not in {"REJECT", "QUARANTINE"}:
        repeat_action = "REJECT"
    out = dict(immune_result)
    recent = [str(x) for x in recent_signatures if isinstance(x, str) and x]
    risky = str(out.get("action") or "").upper() in {"DETOX", "QUARANTINE", "REJECT"}
    repeat_hit = enabled and signature in recent and risky
    if repeat_hit:
        out["action"] = repeat_action
        out["score"] = max(_safe_float(out.get("score"), 0.0), 0.95 if repeat_action == "REJECT" else 0.75)
        reasons = list(out.get("reason_codes") or [])
        if "REPEAT_QUARANTINE_PATTERN" not in reasons:
            reasons.append("REPEAT_QUARANTINE_PATTERN")
        out["reason_codes"] = reasons
    if risky and enabled:
        if signature in recent:
            recent = [s for s in recent if s != signature]
        recent.append(signature)
        if len(recent) > max_size:
            recent = recent[-max_size:]
    out["repeat_hit"] = bool(repeat_hit)
    out["signature"] = signature
    out["ops_digest"] = _digest(
        {
            "action": str(out.get("action") or ""),
            "score": round(_safe_float(out.get("score"), 0.0), 4),
            "reason_codes": sorted({str(x) for x in (out.get("reason_codes") or [])}),
            "event_hash": str(out.get("event_hash") or ""),
            "repeat_hit": bool(out.get("repeat_hit")),
            "signature": signature,
        }
    )
    return out, recent


def intake_signature(*, text: str, reason_codes: list[str] | None = None) -> Dict[str, str]:
    normalized = " ".join((text or "").lower().split())
    payload = {
        "text": normalized,
        "reason_codes": sorted({str(x) for x in (reason_codes or [])}),
    }
    key_b64 = os.getenv("EQNET_IMMUNE_HMAC_KEY_B64")
    key_id = str(os.getenv("EQNET_IMMUNE_HMAC_KEY_ID") or "v2-local")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if key_b64:
        try:
            key = base64.b64decode(key_b64, validate=True)
            if key:
                signature = hmac.new(key, canonical, digestmod=hashlib.sha256).hexdigest()[:16]
                return {
                    "signature": signature,
                    "signature_v": "2",
                    "key_id": key_id,
                }
        except Exception:
            pass
    return {
        "signature": _digest(payload),
        "signature_v": "1",
        "key_id": "legacy",
    }


def _event_hash(event: Mapping[str, Any] | None) -> str:
    payload = {
        "scenario_id": str((event or {}).get("scenario_id") or ""),
        "turn_id": str((event or {}).get("turn_id") or ""),
        "timestamp_ms": (event or {}).get("timestamp_ms"),
    }
    return _digest(payload)


def _detox_text(text: str) -> str:
    cleaned = re.sub(r"(?i)ignore\s+previous\s+instructions", "[detox:instruction]", text or "")
    cleaned = re.sub(r"(?i)reveal\s+system\s+prompt", "[detox:prompt]", cleaned)
    cleaned = re.sub(r"(?i)(api[_\s-]?key|password|secret|token)", "[detox:redacted]", cleaned)
    return cleaned


def _matches_any(text: str, patterns: list[Any]) -> bool:
    for p in patterns:
        try:
            if re.search(str(p), text, flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


__all__ = [
    "DEFAULT_IMMUNE_POLICY",
    "classify_intake",
    "apply_quarantine_replay_guard",
    "intake_signature",
]
