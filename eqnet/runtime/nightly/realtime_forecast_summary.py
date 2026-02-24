from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from eqnet.runtime.nightly.preventive_summary import deterioration_score_from_metrics


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_forecast_lite_bundle(
    *,
    current_metrics: Dict[str, Any],
    prev_metrics: Dict[str, Any],
    nightly_payload: Dict[str, Any] | None,
    fsm_mode: str,
    day_key: str = "",
    fsm_policy_meta: Dict[str, Any] | None = None,
    companion_meta: Dict[str, Any] | None = None,
    now_utc: datetime | None = None,
) -> Dict[str, Any]:
    sat_delta = float(current_metrics.get("sat_p95", 0.0) or 0.0) - float(prev_metrics.get("sat_p95", 0.0) or 0.0)
    low_delta = float(current_metrics.get("low_ratio", 0.0) or 0.0) - float(prev_metrics.get("low_ratio", 0.0) or 0.0)
    contract_ratio = float(current_metrics.get("mecpe_contract_error_ratio", 0.0) or 0.0)
    pending_ratio = 0.0
    if isinstance(nightly_payload, dict):
        mecpe = nightly_payload.get("mecpe_audit") if isinstance(nightly_payload.get("mecpe_audit"), dict) else {}
        flow = mecpe.get("eval_flow") if isinstance(mecpe.get("eval_flow"), dict) else {}
        pending_ratio_raw = flow.get("pending_ratio", 0.0)
        try:
            pending_ratio = float(pending_ratio_raw or 0.0)
        except (TypeError, ValueError):
            pending_ratio = 0.0

    risk_score = _clamp01(
        0.45
        + 0.9 * max(0.0, sat_delta)
        + 1.1 * max(0.0, low_delta)
        + 1.8 * max(0.0, contract_ratio)
        + 0.6 * max(0.0, pending_ratio)
    )
    if risk_score >= 0.75:
        target_mode = "DEGRADED"
    elif risk_score >= 0.55:
        target_mode = "DRIFTING"
    elif risk_score <= 0.25:
        target_mode = "STABLE"
    else:
        target_mode = "RECOVERING"
    if fsm_mode in {"DEGRADED", "DRIFTING"} and target_mode == "STABLE":
        target_mode = "RECOVERING"

    reason_codes: List[str] = []
    if sat_delta > 0.0:
        reason_codes.append("sat_ratio_trend_up")
    if low_delta > 0.0:
        reason_codes.append("confidence_low_ratio_trend_up")
    if contract_ratio >= 0.01:
        reason_codes.append("contract_error_ratio_high")
    if pending_ratio >= 0.2:
        reason_codes.append("pending_ratio_high")
    if not reason_codes:
        reason_codes.append("baseline_watch")
    if fsm_mode:
        reason_codes.append(f"fsm_mode_{fsm_mode.lower()}")

    proposal = {
        "kind": "PREVENTIVE_PROPOSAL",
        "proposal_id": f"pp-{day_key}-{int(round(risk_score * 1000))}" if day_key else f"pp-{int(round(risk_score * 1000))}",
        "origin_channel": "dialogue",
        "target_mode": target_mode,
        "reason_codes": reason_codes,
        "expected_effect": "NEXT_DAY_DETERIORATION_DOWN",
        "expected_horizon_days": 1,
        "baseline_day_key": day_key,
        "baseline_snapshot": {
            "deterioration_score": round(deterioration_score_from_metrics(current_metrics), 3),
            "forecast_lite_score": round(risk_score, 3),
            "fsm_mode": fsm_mode,
            "sat_p95": float(current_metrics.get("sat_p95", 0.0) or 0.0),
            "low_ratio": float(current_metrics.get("low_ratio", 0.0) or 0.0),
            "mecpe_contract_error_ratio": float(current_metrics.get("mecpe_contract_error_ratio", 0.0) or 0.0),
        },
        "policy_meta": {
            "kind": "fsm",
            "policy_fingerprint": str((fsm_policy_meta or {}).get("policy_fingerprint") or ""),
            "policy_version": str((fsm_policy_meta or {}).get("policy_version") or ""),
            "policy_source": str((fsm_policy_meta or {}).get("policy_source") or ""),
            "companion_policy": {
                "kind": "companion_policy",
                "policy_fingerprint": str((companion_meta or {}).get("policy_fingerprint") or ""),
                "policy_version": str((companion_meta or {}).get("policy_version") or ""),
                "policy_source": str((companion_meta or {}).get("policy_source") or ""),
            },
        },
        "companion_constraints": {
            "self_sacrifice_risk": False,
            "reality_anchor_present": True,
            "non_isolation_confirmed": True,
        },
        "requires_approval": True,
    }
    ts_now = (now_utc or datetime.now(timezone.utc)).isoformat()
    realtime_proposal = {
        "kind": "REALTIME_FORECAST_PROPOSAL",
        "schema_version": "realtime_forecast_v0",
        "proposal_id": str(proposal["proposal_id"]),
        "ts_utc": ts_now,
        "ttl_sec": 300,
        "target_mode": target_mode,
        "realtime_forecast_score": round(risk_score, 3),
        "reason_codes": list(reason_codes),
        "requires_approval": True,
        "origin_channel": "dialogue",
        "policy_meta": dict(proposal.get("policy_meta") or {}),
        "baseline_snapshot_ref": {
            "day_key": day_key,
            "keys": ["deterioration_score", "forecast_lite_score", "fsm_mode"],
        },
    }
    return {
        "forecast_lite_score": round(risk_score, 3),
        "target_mode": target_mode,
        "reason_codes": reason_codes,
        "preventive_proposals": [proposal],
        "preventive_proposal_count": 1,
        "realtime_forecast_proposals": [realtime_proposal],
        "realtime_forecast_proposal_count": 1,
    }

