from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from emot_terrain_lab.mind.green import green_response


def load_green_bridge_policy(path: Path) -> Dict[str, Any]:
    default: Dict[str, Any] = {
        "schema_version": "green_bridge_policy_v0",
        "policy_version": "green_bridge_policy_v0",
        "policy_source": "configs/green_bridge_policy_v0.yaml",
        "enabled": False,
        "culture_resonance": 0.3,
        "tau_rate": 1.0,
        "score_weights": {
            "delta": 0.65,
            "control": 0.35,
        },
        "priority_patch": {
            "enabled": False,
            "alpha": 0.0,
            "max_delta": 0.0,
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


def green_bridge_policy_meta(policy: Mapping[str, Any]) -> Dict[str, str]:
    canonical = json.dumps(_normalize_for_hash(policy), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "kind": "green_bridge_policy",
        "policy_version": str(policy.get("policy_version") or "green_bridge_policy_v0"),
        "policy_source": str(policy.get("policy_source") or "configs/green_bridge_policy_v0.yaml"),
        "policy_fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
    }


def compute_green_bridge_snapshot(
    *,
    current_metrics: Mapping[str, Any],
    fsm_mode: str,
    companion_policy_valid: bool,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    policy_meta = green_bridge_policy_meta(policy)
    enabled = bool(policy.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "green_mode": "OFF",
            "green_response_score": 0.0,
            "green_quality": 0.0,
            "green_decay_tau": 0.0,
            "reason_codes": ["GREEN_BRIDGE_DISABLED"],
            "policy_meta": policy_meta,
        }

    sat = _to_float(current_metrics.get("sat_p95"), default=0.0)
    low = _to_float(current_metrics.get("low_ratio"), default=0.0)
    contract = _to_float(current_metrics.get("mecpe_contract_error_ratio"), default=0.0)
    score_weights = policy.get("score_weights") if isinstance(policy.get("score_weights"), Mapping) else {}
    w_delta = _to_float(score_weights.get("delta"), default=0.65)
    w_control = _to_float(score_weights.get("control"), default=0.35)
    culture_resonance = _to_float(policy.get("culture_resonance"), default=0.3)
    tau_rate = _to_float(policy.get("tau_rate"), default=1.0)

    qualia = {
        "tone": _tone_from_metrics(sat, low),
        "tempo": _tempo_from_metrics(sat),
        "semantic_valence": _clamp01((0.5 - sat) * 2.0),
        "coherence": _clamp01((0.5 - low) * 2.0),
        "trust_hint": 0.6 if companion_policy_valid else -0.2,
        "sensor_intensity": _clamp01(sat - 0.5),
    }
    response = green_response(
        qualia,
        culture_resonance=culture_resonance,
        tau_rate=tau_rate,
    )
    delta_energy = sum(abs(_to_float(v, default=0.0)) for v in (response.delta_mood or {}).values())
    ctrl_energy = sum(abs(_to_float(v, default=0.0)) for v in (response.controls or {}).values())
    ctrl_norm = min(1.0, ctrl_energy / 400.0)
    response_score = max(0.0, min(1.0, (w_delta * min(1.0, delta_energy)) + (w_control * ctrl_norm)))
    green_quality = max(0.0, min(1.0, 1.0 - min(1.0, low + contract * 4.0)))
    green_decay_tau = round(max(0.05, tau_rate / (1.0 + sat + low)), 6)

    mode = "QUIET"
    if response_score >= 0.66:
        mode = "ACTIVE"
    elif response_score >= 0.33:
        mode = "COUPLED"
    reason_codes = [
        f"FSM_MODE_{str(fsm_mode or 'STABLE').upper()}",
        "GREEN_RESPONSE_ACTIVE" if response_score >= 0.33 else "GREEN_RESPONSE_QUIET",
    ]

    return {
        "enabled": True,
        "green_mode": mode,
        "green_response_score": round(response_score, 6),
        "green_quality": round(green_quality, 6),
        "green_decay_tau": green_decay_tau,
        "reason_codes": reason_codes,
        "policy_meta": policy_meta,
    }


def apply_green_priority_patch(
    *,
    base_priority_score: float,
    green_snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
    blocked: bool = False,
    suppressed: bool = False,
    unknown: bool = False,
) -> Dict[str, Any]:
    base = _clip01(base_priority_score)
    policy_meta = green_bridge_policy_meta(policy)
    patch = policy.get("priority_patch") if isinstance(policy.get("priority_patch"), Mapping) else {}
    patch_enabled = bool(patch.get("enabled", False))
    alpha = _to_float(patch.get("alpha"), default=0.0)
    max_delta = max(0.0, _to_float(patch.get("max_delta"), default=0.0))
    green_enabled = bool(green_snapshot.get("enabled", False))
    green_score = _clip01(_to_float(green_snapshot.get("green_response_score"), default=0.0))

    reasons = []
    applied = False
    delta = 0.0

    if blocked or suppressed or unknown:
        reasons.append("GREEN_PRIORITY_PATCH_SKIPPED_STATE_GUARD")
    elif not patch_enabled:
        reasons.append("GREEN_PRIORITY_PATCH_DISABLED")
    elif not green_enabled:
        reasons.append("GREEN_PRIORITY_PATCH_GREEN_DISABLED")
    elif alpha == 0.0 or max_delta == 0.0:
        reasons.append("GREEN_PRIORITY_PATCH_ZERO_EFFECT")
    else:
        raw_delta = alpha * green_score
        delta = max(-max_delta, min(max_delta, raw_delta))
        applied = abs(delta) > 0.0
        if applied:
            reasons.append("GREEN_PRIORITY_PATCH_APPLIED")
        else:
            reasons.append("GREEN_PRIORITY_PATCH_ZERO_EFFECT")

    patched = _clip01(base + delta)
    return {
        "priority_score": round(patched, 6),
        "priority_patch_delta": round(delta, 6),
        "priority_patch_applied": bool(applied),
        "priority_patch_reason_codes": reasons,
        "policy_meta": policy_meta,
    }


def _tone_from_metrics(sat: float, low: float) -> str:
    if sat >= 0.75:
        return "sharp"
    if low >= 0.35:
        return "soft"
    if sat <= 0.25:
        return "warm"
    return "neutral"


def _tempo_from_metrics(sat: float) -> str:
    if sat >= 0.75:
        return "fast"
    if sat <= 0.25:
        return "slow"
    return "steady"


def _clamp01(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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
