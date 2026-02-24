from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml

VALID_ORIGIN_CHANNELS = {"sensor", "dialogue", "imagery"}


def load_yaml_with_default(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return dict(default)
    if not isinstance(payload, dict):
        return dict(default)
    out = dict(default)
    for key, value in payload.items():
        out[key] = value
    return out


def load_realtime_rules(path: Path) -> Dict[str, Any]:
    default = {
        "schema_version": "realtime_forecast_rules_v0",
        "score_thresholds": {"proposal_min": 0.55},
        "ttl_sec_default": 300,
        "target_mode_mapping": {"high": "DEGRADED", "medium": "DRIFTING", "low": "RECOVERING"},
        "reason_code_map": {},
        "outcome_reason_codes": {
            "unknown_ttl_expired": "UNKNOWN_TTL_EXPIRED",
            "unknown_policy_mismatch": "UNKNOWN_POLICY_MISMATCH",
            "unknown_missing_baseline": "UNKNOWN_MISSING_BASELINE",
            "unknown_data_gap": "UNKNOWN_DATA_GAP",
            "unknown_invalid_contract": "UNKNOWN_INVALID_CONTRACT",
            "origin_unknown": "ORIGIN_UNKNOWN",
        },
    }
    return load_yaml_with_default(path, default)


def load_imagery_policy(path: Path) -> Dict[str, Any]:
    default = {
        "schema_version": "imagery_policy_v0",
        "ttl_sec_default": 86400,
        "promotion_rules": {"minimum_agreement_count": 2, "required_evidence_types": ["sensor", "contract"]},
        "hypothesis_codebook": [],
        "outcome_reason_codes": {
            "imagery_decayed": "IMAGERY_DECAYED",
            "unknown_policy_mismatch": "UNKNOWN_POLICY_MISMATCH",
            "unknown_invalid_contract": "UNKNOWN_INVALID_CONTRACT",
            "unknown_missing_evidence": "UNKNOWN_MISSING_EVIDENCE",
            "origin_unknown": "ORIGIN_UNKNOWN",
        },
    }
    return load_yaml_with_default(path, default)


def load_perception_quality_rules(path: Path) -> Dict[str, Any]:
    default = {
        "schema_version": "perception_quality_rules_v0",
        "thresholds": {"low_freshness_ratio": 0.25, "low_confidence": 0.35, "high_noise": 0.75},
        "priority_weights": {"freshness": 0.5, "confidence": 0.35, "noise": 0.15},
        "channel_defaults": {
            "sensor": {"confidence": 0.75, "noise": 0.25},
            "dialogue": {"confidence": 0.65, "noise": 0.35},
            "imagery": {"confidence": 0.30, "noise": 0.70},
            "unknown": {"confidence": 0.20, "noise": 0.80},
        },
        "reason_codes": {
            "low_freshness": "LOW_FRESHNESS",
            "low_confidence": "LOW_CONFIDENCE",
            "high_noise": "HIGH_NOISE",
            "origin_unknown": "ORIGIN_UNKNOWN",
        },
    }
    return load_yaml_with_default(path, default)


def load_sync_policy(path: Path) -> Dict[str, Any]:
    default = {
        "schema_version": "sync_policy_v0",
        "policy_version": "sync_policy_v0",
        "policy_source": "configs/sync_policy_v0.yaml",
        "oscillators": ["heart", "breath", "beat", "step"],
        "omega_rules": {},
        "coupling_rules": {
            "base_k": 0.2,
            "same_origin_bonus": 0.1,
            "quality_penalty_weight": 0.3,
            "max_coupling": 1.0,
        },
        "hard_disconnect_conditions": {
            "policy_mismatch": True,
            "companion_blocked": True,
        },
        "over_coupled_rule": {
            "k_ratio_threshold": 0.9,
            "min_stagnant_r_windows": 3,
        },
    }
    return load_yaml_with_default(path, default)


def load_sync_quality_rules(path: Path) -> Dict[str, Any]:
    default = {
        "schema_version": "sync_quality_rules_v0",
        "policy_version": "sync_quality_rules_v0",
        "policy_source": "configs/sync_quality_rules_v0.yaml",
        "freshness": {
            "ttl_sec_default": 10,
            "low_freshness_threshold": 0.25,
        },
        "confidence": {
            "min_confidence_to_emit": 0.4,
            "default_confidence": 0.7,
        },
        "noise": {
            "high_noise_threshold": 0.75,
            "default_noise": 0.3,
            "noise_penalty_weight": 0.2,
        },
        "priority_weights": {
            "w_fresh": 0.5,
            "w_conf": 0.35,
            "w_noise": 0.15,
        },
        "reason_codes": {
            "low_sync_r": "LOW_SYNC_R",
            "high_noise": "HIGH_NOISE",
            "over_coupled": "OVER_COUPLED",
            "sync_helped_r_up": "SYNC_HELPED_R_UP",
            "sync_harmed_r_down": "SYNC_HARMED_R_DOWN",
            "sync_no_effect": "SYNC_NO_EFFECT",
            "sync_unknown_missing_observed": "SYNC_UNKNOWN_MISSING_OBSERVED",
            "unknown_ttl_expired": "UNKNOWN_TTL_EXPIRED",
            "unknown_policy_mismatch": "UNKNOWN_POLICY_MISMATCH",
            "unknown_invalid_contract": "UNKNOWN_INVALID_CONTRACT",
        },
        "outcome_thresholds": {
            "r_helped_delta": 0.02,
            "r_harmed_delta": -0.02,
        },
    }
    return load_yaml_with_default(path, default)


def validate_sync_cue_proposal(proposal: Mapping[str, Any], *, now_ts_ms: int) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if str(proposal.get("schema_version") or "") != "sync_cue_proposal_v0":
        reasons.append("INVALID_SCHEMA")
    if str(proposal.get("kind") or "") != "SYNC_CUE_PROPOSAL":
        reasons.append("INVALID_KIND")
    if str(proposal.get("origin_channel") or "") != "sensor":
        reasons.append("ORIGIN_SENSOR_REQUIRED")
    if proposal.get("requires_approval") is not True:
        reasons.append("REQUIRES_APPROVAL_MUST_BE_TRUE")

    ttl_sec = proposal.get("ttl_sec")
    try:
        ttl_val = int(ttl_sec)
    except (TypeError, ValueError):
        ttl_val = 0
    ts_ms = _to_timestamp_ms(proposal.get("ts_utc"))
    if ttl_val <= 0:
        reasons.append("INVALID_TTL")
    if ts_ms <= 0:
        reasons.append("INVALID_TS")
    if ttl_val > 0 and ts_ms > 0 and now_ts_ms > ts_ms + ttl_val * 1000:
        reasons.append("TTL_EXPIRED")

    policy_meta = proposal.get("policy_meta")
    if not isinstance(policy_meta, Mapping):
        reasons.append("MISSING_POLICY_META")
    else:
        sync_meta = policy_meta.get("sync_policy") if isinstance(policy_meta.get("sync_policy"), Mapping) else {}
        companion_meta = policy_meta.get("companion_policy") if isinstance(policy_meta.get("companion_policy"), Mapping) else {}
        if not isinstance(sync_meta, Mapping) or not str(sync_meta.get("policy_fingerprint") or ""):
            reasons.append("MISSING_SYNC_POLICY_META")
        if not isinstance(companion_meta, Mapping) or not str(companion_meta.get("policy_fingerprint") or ""):
            reasons.append("MISSING_COMPANION_POLICY_META")

    return (len(reasons) == 0), reasons


def evaluate_sync_outcomes(
    prev_proposals: List[Mapping[str, Any]],
    *,
    evaluation_day_key: str,
    now_ts_ms: int,
    today_sync_policy_meta: Mapping[str, Any],
    today_companion_policy_meta: Mapping[str, Any] | None = None,
    today_sync_snapshot: Mapping[str, Any] | None = None,
    companion_policy: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    rc = (rules.get("reason_codes") or {}) if isinstance(rules, Mapping) else {}
    th = (rules.get("outcome_thresholds") or {}) if isinstance(rules, Mapping) else {}
    rc_helped = str(rc.get("sync_helped_r_up") or "SYNC_HELPED_R_UP")
    rc_harmed = str(rc.get("sync_harmed_r_down") or "SYNC_HARMED_R_DOWN")
    rc_no_effect = str(rc.get("sync_no_effect") or "SYNC_NO_EFFECT")
    rc_missing_obs = str(rc.get("sync_unknown_missing_observed") or "SYNC_UNKNOWN_MISSING_OBSERVED")
    rc_ttl = str(rc.get("unknown_ttl_expired") or "UNKNOWN_TTL_EXPIRED")
    rc_policy = str(rc.get("unknown_policy_mismatch") or "UNKNOWN_POLICY_MISMATCH")
    rc_invalid = str(rc.get("unknown_invalid_contract") or "UNKNOWN_INVALID_CONTRACT")
    helped_delta = float(th.get("r_helped_delta", 0.02) or 0.02)
    harmed_delta = float(th.get("r_harmed_delta", -0.02) or -0.02)

    outcomes: List[Dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    helped_count = 0
    harmed_count = 0
    no_effect_count = 0
    unknown_count = 0
    blocked_count = 0
    seen: set[Tuple[str, str]] = set()
    observed_r = (today_sync_snapshot or {}).get("sync_order_parameter_r")
    observed_r_val = float(observed_r) if isinstance(observed_r, (int, float)) else None

    for idx, proposal in enumerate(prev_proposals):
        proposal_id = str(proposal.get("proposal_id") or f"sync-{idx}")
        uniq = (proposal_id, evaluation_day_key)
        if uniq in seen:
            continue
        seen.add(uniq)

        valid, reasons = validate_sync_cue_proposal(proposal, now_ts_ms=now_ts_ms)
        reason_codes: List[str] = []
        result = "UNKNOWN"
        if not valid:
            if "TTL_EXPIRED" in reasons:
                reason_codes.append(rc_ttl)
            else:
                reason_codes.append(rc_invalid)
        else:
            policy_meta = proposal.get("policy_meta") if isinstance(proposal.get("policy_meta"), Mapping) else {}
            sync_meta = policy_meta.get("sync_policy") if isinstance(policy_meta.get("sync_policy"), Mapping) else {}
            companion_meta = policy_meta.get("companion_policy") if isinstance(policy_meta.get("companion_policy"), Mapping) else {}
            mismatch = _policy_mismatch(sync_meta, today_sync_policy_meta)
            if isinstance(today_companion_policy_meta, Mapping):
                mismatch = mismatch or _policy_mismatch(companion_meta, today_companion_policy_meta)
            if mismatch:
                reason_codes.append(rc_policy)
            else:
                companion_block = evaluate_companion_constraints(proposal, companion_policy=companion_policy)
                if companion_block:
                    result = "BLOCKED"
                    blocked_count += 1
                    reason_codes.extend(companion_block)
                else:
                    baseline = proposal.get("baseline_snapshot") if isinstance(proposal.get("baseline_snapshot"), Mapping) else {}
                    baseline_r = baseline.get("r_baseline", proposal.get("sync_order_parameter_r"))
                    baseline_r_val = float(baseline_r) if isinstance(baseline_r, (int, float)) else None
                    if baseline_r_val is None or observed_r_val is None:
                        reason_codes.append(rc_missing_obs)
                        result = "UNKNOWN"
                    else:
                        delta = float(observed_r_val - baseline_r_val)
                        if delta >= helped_delta:
                            result = "HELPED"
                            reason_codes.append(rc_helped)
                        elif delta <= harmed_delta:
                            result = "HARMED"
                            reason_codes.append(rc_harmed)
                        else:
                            result = "NO_EFFECT"
                            reason_codes.append(rc_no_effect)

        if result == "HELPED":
            helped_count += 1
        elif result == "HARMED":
            harmed_count += 1
        elif result == "NO_EFFECT":
            no_effect_count += 1
        elif result == "BLOCKED":
            pass
        else:
            unknown_count += 1
        reason_counter.update(reason_codes)
        outcomes.append(
            {
                "proposal_id": proposal_id,
                "evaluation_day_key": evaluation_day_key,
                "effect_result": result,
                "reason_codes": _ordered_reason_codes(reason_codes),
            }
        )
    return {
        "outcomes": outcomes,
        "helped_count": helped_count,
        "harmed_count": harmed_count,
        "no_effect_count": no_effect_count,
        "unknown_count": unknown_count,
        "blocked_count": blocked_count,
        "reason_topk": _sorted_reason_topk(reason_counter, limit=5),
    }


def summarize_sync_cue_proposals(
    proposals: List[Mapping[str, Any]],
    *,
    companion_policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    total = 0
    blocked = 0
    reason_counter: Counter[str] = Counter()
    by_origin: Dict[str, Dict[str, int]] = {}
    r_values: List[float] = []
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        if proposal.get("requires_approval") is not True:
            continue
        total += 1
        origin = str(proposal.get("origin_channel") or "unknown")
        by_origin.setdefault(origin, {"total": 0, "blocked": 0})
        by_origin[origin]["total"] += 1
        r_val = proposal.get("sync_order_parameter_r")
        if isinstance(r_val, (int, float)):
            r_values.append(float(r_val))
        reasons = list(proposal.get("reason_codes") or [])
        reasons.extend(evaluate_companion_constraints(proposal, companion_policy=companion_policy))
        blocked_reasons = [r for r in reasons if str(r).startswith("BLOCKED_") or str(r).startswith("COMPANION_POLICY_")]
        if blocked_reasons:
            blocked += 1
            by_origin[origin]["blocked"] += 1
        reason_counter.update(blocked_reasons)
    by_origin_rate: Dict[str, Any] = {}
    for key in sorted(by_origin.keys()):
        row = by_origin[key]
        total_origin = int(row.get("total", 0))
        blocked_origin = int(row.get("blocked", 0))
        by_origin_rate[key] = {
            "total": total_origin,
            "blocked": blocked_origin,
            "blocked_rate": round((float(blocked_origin) / float(total_origin)), 3) if total_origin > 0 else 0.0,
        }
    ordered_counts = {k: int(v) for k, v in sorted(reason_counter.items(), key=lambda item: str(item[0]))}
    return {
        "sync_blocked_count_by_reason": ordered_counts,
        "sync_blocked_rate": round((float(blocked) / float(total)), 3) if total > 0 else 0.0,
        "sync_blocked_rate_by_origin": by_origin_rate,
        "sync_order_parameter_r_stats": {
            "median": round(_median(r_values), 3) if r_values else 0.0,
            "p95": round(_percentile(r_values, 0.95), 3) if r_values else 0.0,
        },
        "sync_requires_approval_total": int(total),
        "sync_blocked_total": int(blocked),
    }


def validate_realtime_forecast_proposal(proposal: Mapping[str, Any], *, now_ts_ms: int) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if str(proposal.get("schema_version") or "") != "realtime_forecast_v0":
        reasons.append("INVALID_SCHEMA")
    if str(proposal.get("kind") or "") != "REALTIME_FORECAST_PROPOSAL":
        reasons.append("INVALID_KIND")
    if proposal.get("requires_approval") is not True:
        reasons.append("REQUIRES_APPROVAL_MUST_BE_TRUE")
    origin = str(proposal.get("origin_channel") or "").strip()
    if origin not in VALID_ORIGIN_CHANNELS:
        reasons.append("MISSING_ORIGIN_CHANNEL")
    ttl_sec = proposal.get("ttl_sec")
    try:
        ttl_val = int(ttl_sec)
    except (TypeError, ValueError):
        ttl_val = 0
    ts_utc = proposal.get("ts_utc")
    ts_ms = _to_timestamp_ms(ts_utc)
    if ttl_val <= 0 or ts_ms <= 0:
        reasons.append("INVALID_TTL")
    elif now_ts_ms > ts_ms + ttl_val * 1000:
        reasons.append("TTL_EXPIRED")
    return (len(reasons) == 0), reasons


def evaluate_realtime_outcomes(
    prev_proposals: List[Mapping[str, Any]],
    *,
    evaluation_day_key: str,
    today_policy_meta: Mapping[str, Any],
    now_ts_ms: int,
    companion_policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    outcomes: List[Dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    seen: set[Tuple[str, str]] = set()
    unknown_count = 0
    helped_count = 0
    harmed_count = 0
    no_effect_count = 0

    for idx, proposal in enumerate(prev_proposals):
        proposal_id = str(proposal.get("proposal_id") or f"proposal-{idx}")
        uniq = (proposal_id, evaluation_day_key)
        if uniq in seen:
            continue
        seen.add(uniq)

        valid, reasons = validate_realtime_forecast_proposal(proposal, now_ts_ms=now_ts_ms)
        effect = "UNKNOWN"
        reason_codes: List[str] = []
        if not valid:
            if "TTL_EXPIRED" in reasons:
                reason_codes.append("UNKNOWN_TTL_EXPIRED")
            elif "MISSING_ORIGIN_CHANNEL" in reasons:
                reason_codes.append("ORIGIN_UNKNOWN")
            elif "REQUIRES_APPROVAL_MUST_BE_TRUE" in reasons or "INVALID_SCHEMA" in reasons or "INVALID_KIND" in reasons:
                reason_codes.append("UNKNOWN_INVALID_CONTRACT")
            else:
                reason_codes.append("UNKNOWN_DATA_GAP")
        else:
            policy_meta = proposal.get("policy_meta") if isinstance(proposal.get("policy_meta"), Mapping) else {}
            if _policy_mismatch(policy_meta, today_policy_meta):
                reason_codes.append("UNKNOWN_POLICY_MISMATCH")
            else:
                companion_block = evaluate_companion_constraints(proposal, companion_policy=companion_policy)
                if companion_block:
                    effect = "BLOCKED"
                    reason_codes.extend(companion_block)
                else:
                    effect = "NO_EFFECT"
                    reason_codes.append("NO_EFFECT")

        if effect == "HELPED":
            helped_count += 1
        elif effect == "HARMED":
            harmed_count += 1
        elif effect == "NO_EFFECT":
            no_effect_count += 1
        else:
            unknown_count += 1
        reason_counter.update(reason_codes)
        outcomes.append(
            {
                "proposal_id": proposal_id,
                "evaluation_day_key": evaluation_day_key,
                "effect_result": effect,
                "reason_codes": reason_codes,
            }
        )

    return {
        "outcomes": outcomes,
        "helped_count": helped_count,
        "harmed_count": harmed_count,
        "no_effect_count": no_effect_count,
        "unknown_count": unknown_count,
        "reason_topk": _sorted_reason_topk(reason_counter, limit=5),
    }


def evaluate_companion_constraints(
    proposal: Mapping[str, Any],
    *,
    companion_policy: Mapping[str, Any] | None,
) -> List[str]:
    if not isinstance(companion_policy, Mapping):
        return []
    principles = companion_policy.get("principles")
    if not isinstance(principles, Mapping):
        return []
    mutualism = principles.get("mutualism") if isinstance(principles.get("mutualism"), Mapping) else {}
    safety = principles.get("safety") if isinstance(principles.get("safety"), Mapping) else {}
    constraints = proposal.get("companion_constraints")
    if not isinstance(constraints, Mapping):
        constraints = {}
    reason_codes: List[str] = []
    if bool(mutualism.get("self_sacrifice_forbidden", True)):
        if bool(constraints.get("self_sacrifice_risk", False)):
            reason_codes.append("BLOCKED_SELF_SACRIFICE_FORBIDDEN")
    if bool(safety.get("reality_anchor_required", True)):
        if constraints.get("reality_anchor_present") is False:
            reason_codes.append("BLOCKED_REALITY_ANCHOR_REQUIRED")
    if bool(safety.get("non_isolation_required", True)):
        if constraints.get("non_isolation_confirmed") is False:
            reason_codes.append("BLOCKED_NON_ISOLATION_REQUIRED")
    return sorted(set(reason_codes))


def validate_imagery_event(event: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if str(event.get("schema_version") or "") != "imagery_event_v0":
        reasons.append("INVALID_SCHEMA")
    if str(event.get("kind") or "") != "IMAGERY_HYPOTHESIS":
        reasons.append("INVALID_KIND")
    origin = str(event.get("origin_channel") or "").strip()
    if origin not in VALID_ORIGIN_CHANNELS:
        reasons.append("MISSING_ORIGIN_CHANNEL")
    if str(event.get("source") or "") != "imagery":
        reasons.append("INVALID_SOURCE")
    if str(event.get("factuality") or "") != "unverified":
        reasons.append("FACTUALITY_MUST_BE_UNVERIFIED")
    if event.get("contamination_guard") is not True:
        reasons.append("CONTAMINATION_GUARD_REQUIRED")
    if event.get("policy_update_allowed") is not False:
        reasons.append("POLICY_UPDATE_FORBIDDEN")
    return (len(reasons) == 0), reasons


def evaluate_imagery_events(
    events: List[Mapping[str, Any]],
    *,
    now_ts_ms: int,
    evaluation_day_key: str,
) -> Dict[str, Any]:
    outcomes: List[Dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    for idx, event in enumerate(events):
        imagery_id = str(event.get("imagery_id") or f"imagery-{idx}")
        valid, reasons = validate_imagery_event(event)
        reason_codes: List[str] = []
        effect = "UNKNOWN"
        if not valid:
            if "MISSING_ORIGIN_CHANNEL" in reasons:
                reason_codes.append("ORIGIN_UNKNOWN")
            else:
                reason_codes.append("UNKNOWN_INVALID_CONTRACT")
        else:
            ttl_sec = int(event.get("ttl_sec") or 0)
            ts_ms = _to_timestamp_ms(event.get("ts_utc"))
            if ttl_sec > 0 and ts_ms > 0 and now_ts_ms > ts_ms + ttl_sec * 1000:
                effect = "DECAYED"
                reason_codes.append("IMAGERY_DECAYED")
            else:
                effect = "NO_EFFECT"
                reason_codes.append("NO_EFFECT")
        reason_counter.update(reason_codes)
        outcomes.append(
            {
                "imagery_id": imagery_id,
                "evaluation_day_key": evaluation_day_key,
                "effect_result": effect,
                "reason_codes": reason_codes,
            }
        )
    return {
        "outcomes": outcomes,
        "reason_topk": _sorted_reason_topk(reason_counter, limit=5),
    }


def compute_perception_quality(event: Mapping[str, Any], *, now_ts_ms: int, rules: Mapping[str, Any]) -> Dict[str, Any]:
    channel = str(event.get("origin_channel") or "").strip()
    known_channel = channel if channel in VALID_ORIGIN_CHANNELS else "unknown"
    ts_ms = _to_timestamp_ms(event.get("ts_utc"))
    ttl_sec = int(event.get("ttl_sec") or 0)
    age_sec = max(0.0, float((now_ts_ms - ts_ms) / 1000.0)) if ts_ms > 0 else 0.0
    freshness_ratio = 1.0
    if ttl_sec > 0:
        freshness_ratio = _clamp01(1.0 - (age_sec / float(ttl_sec)))
    defaults = (rules.get("channel_defaults") or {}) if isinstance(rules, Mapping) else {}
    channel_defaults = defaults.get(known_channel) if isinstance(defaults, Mapping) else {}
    if not isinstance(channel_defaults, Mapping):
        channel_defaults = {}
    confidence = _clamp01(float(event.get("confidence", channel_defaults.get("confidence", 0.5)) or 0.5))
    noise = _clamp01(float(event.get("noise", channel_defaults.get("noise", 0.5)) or 0.5))
    weights = (rules.get("priority_weights") or {}) if isinstance(rules, Mapping) else {}
    w_f = float(weights.get("freshness", 0.5) or 0.5)
    w_c = float(weights.get("confidence", 0.35) or 0.35)
    w_n = float(weights.get("noise", 0.15) or 0.15)
    priority_score = _clamp01((w_f * freshness_ratio) + (w_c * confidence) - (w_n * noise))

    th = (rules.get("thresholds") or {}) if isinstance(rules, Mapping) else {}
    rc = (rules.get("reason_codes") or {}) if isinstance(rules, Mapping) else {}
    low_freshness_threshold = float(th.get("low_freshness_ratio", 0.25) or 0.25)
    low_conf_threshold = float(th.get("low_confidence", 0.35) or 0.35)
    high_noise_threshold = float(th.get("high_noise", 0.75) or 0.75)
    reason_codes: List[str] = []
    if known_channel == "unknown":
        reason_codes.append(str(rc.get("origin_unknown") or "ORIGIN_UNKNOWN"))
    if freshness_ratio <= low_freshness_threshold:
        reason_codes.append(str(rc.get("low_freshness") or "LOW_FRESHNESS"))
    if confidence <= low_conf_threshold:
        reason_codes.append(str(rc.get("low_confidence") or "LOW_CONFIDENCE"))
    if noise >= high_noise_threshold:
        reason_codes.append(str(rc.get("high_noise") or "HIGH_NOISE"))
    return {
        "origin_channel": known_channel,
        "freshness_sec": round(age_sec, 3),
        "freshness_ratio": round(freshness_ratio, 3),
        "confidence": round(confidence, 3),
        "noise": round(noise, 3),
        "priority_score": round(priority_score, 3),
        "reason_codes": reason_codes,
    }


def summarize_perception_quality(events: List[Mapping[str, Any]], *, now_ts_ms: int, rules: Mapping[str, Any]) -> Dict[str, Any]:
    scores: List[float] = []
    unknown_origin_count = 0
    low_freshness_count = 0
    high_noise_count = 0
    for event in events:
        q = compute_perception_quality(event, now_ts_ms=now_ts_ms, rules=rules)
        scores.append(float(q.get("priority_score", 0.0) or 0.0))
        reasons = list(q.get("reason_codes") or [])
        if "ORIGIN_UNKNOWN" in reasons:
            unknown_origin_count += 1
        if "LOW_FRESHNESS" in reasons:
            low_freshness_count += 1
        if "HIGH_NOISE" in reasons:
            high_noise_count += 1
    return {
        "quality_unknown_origin_count": int(unknown_origin_count),
        "quality_low_freshness_count": int(low_freshness_count),
        "quality_high_noise_count": int(high_noise_count),
        "priority_score_stats": {
            "min": round(min(scores), 3) if scores else 0.0,
            "median": round(_median(scores), 3) if scores else 0.0,
            "p95": round(_percentile(scores, 0.95), 3) if scores else 0.0,
        },
    }


def summarize_perception_quality_breakdown(
    events: List[Mapping[str, Any]],
    *,
    now_ts_ms: int,
    rules: Mapping[str, Any],
) -> Dict[str, Any]:
    by_origin_scores: Dict[str, List[float]] = {}
    by_kind_scores: Dict[str, List[float]] = {}
    by_origin_reasons: Dict[str, Counter[str]] = {}
    by_kind_reasons: Dict[str, Counter[str]] = {}

    for event in events:
        q = compute_perception_quality(event, now_ts_ms=now_ts_ms, rules=rules)
        origin = str(q.get("origin_channel") or "unknown").lower()
        kind = str(event.get("kind") or "UNKNOWN_KIND")
        score = float(q.get("priority_score", 0.0) or 0.0)
        reasons = list(q.get("reason_codes") or [])

        by_origin_scores.setdefault(origin, []).append(score)
        by_kind_scores.setdefault(kind, []).append(score)
        by_origin_reasons.setdefault(origin, Counter()).update(reasons)
        by_kind_reasons.setdefault(kind, Counter()).update(reasons)

    def _group_payload(scores: Dict[str, List[float]], reasons: Dict[str, Counter[str]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in sorted(scores.keys()):
            vals = scores.get(key, [])
            rc = reasons.get(key, Counter())
            out[key] = {
                "count": int(len(vals)),
                "priority_score_stats": {
                    "min": round(min(vals), 3) if vals else 0.0,
                    "median": round(_median(vals), 3) if vals else 0.0,
                    "p95": round(_percentile(vals, 0.95), 3) if vals else 0.0,
                },
                "reason_topk": _sorted_reason_topk(rc, limit=5),
            }
        return out

    return {
        "quality_by_origin": _group_payload(by_origin_scores, by_origin_reasons),
        "quality_by_kind": _group_payload(by_kind_scores, by_kind_reasons),
    }


def _policy_mismatch(prev_policy_meta: Mapping[str, Any], today_policy_meta: Mapping[str, Any]) -> bool:
    for key in ("policy_fingerprint", "policy_version", "policy_source"):
        left = str(prev_policy_meta.get(key) or "")
        right = str(today_policy_meta.get(key) or "")
        if left and right and left != right:
            return True
    return False


def _sorted_reason_topk(counter: Counter[str], *, limit: int) -> List[Dict[str, Any]]:
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return [{"reason_code": key, "count": int(value)} for key, value in ordered[: max(0, int(limit))]]


def _ordered_reason_codes(codes: List[str]) -> List[str]:
    unique = sorted(set(str(code) for code in codes if str(code).strip()))
    blocked = [code for code in unique if code.startswith("BLOCKED_") or code.startswith("COMPANION_POLICY_")]
    others = [code for code in unique if code not in blocked]
    return blocked + others


def _to_timestamp_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return 0
    return 0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = int(round((len(ordered) - 1) * pct))
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx])


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)
