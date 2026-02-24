from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml


def deterioration_score_from_metrics(metrics: Dict[str, Any]) -> float:
    sat = float(metrics.get("sat_p95", 0.0) or 0.0)
    low = float(metrics.get("low_ratio", 0.0) or 0.0)
    contract = float(metrics.get("mecpe_contract_error_ratio", 0.0) or 0.0)
    return round(0.55 * sat + 0.35 * low + 0.10 * min(1.0, contract * 10.0), 3)


def load_forecast_effect_rules(default_path: Path) -> Dict[str, Any]:
    path = Path(default_path)
    env_path = ""
    try:
        import os

        env_path = os.getenv("EQNET_FORECAST_EFFECT_RULES", "")
    except Exception:
        env_path = ""
    if env_path:
        path = Path(env_path)
    defaults: Dict[str, Any] = {
        "effect_thresholds": {
            "deterioration_helped_delta": -0.05,
            "deterioration_harmed_delta": 0.05,
        },
        "reason_codes": {
            "helped_deterioration_down": "EFFECT_HELPED_DETERIORATION_DOWN",
            "harmed_deterioration_up": "EFFECT_HARMED_DETERIORATION_UP",
            "no_effect": "EFFECT_NO_EFFECT",
            "unknown_missing_baseline": "EFFECT_UNKNOWN_MISSING_BASELINE",
            "unknown_policy_mismatch": "EFFECT_UNKNOWN_POLICY_MISMATCH",
            "unknown_data_gap": "EFFECT_UNKNOWN_DATA_GAP",
        },
    }
    if not path.exists():
        return defaults
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return defaults
    if not isinstance(payload, dict):
        return defaults
    out = dict(defaults)
    thresholds = payload.get("effect_thresholds")
    if isinstance(thresholds, dict):
        out["effect_thresholds"] = {
            "deterioration_helped_delta": float(thresholds.get("deterioration_helped_delta", -0.05) or -0.05),
            "deterioration_harmed_delta": float(thresholds.get("deterioration_harmed_delta", 0.05) or 0.05),
        }
    reason_codes = payload.get("reason_codes")
    if isinstance(reason_codes, dict):
        merged = dict(out["reason_codes"])
        for key, value in reason_codes.items():
            if isinstance(value, str) and value.strip():
                merged[str(key)] = value
        out["reason_codes"] = merged
    return out


def evaluate_preventive_proposals(
    *,
    prev_report: Dict[str, Any],
    current_metrics: Dict[str, Any],
    today_fsm_meta: Dict[str, Any],
    companion_meta: Dict[str, Any] | None,
    day_key: str,
    rules: Dict[str, Any],
) -> Dict[str, Any]:
    proposals = prev_report.get("preventive_proposals")
    if not isinstance(proposals, list):
        proposals = []
    rc = (rules.get("reason_codes") or {}) if isinstance(rules, dict) else {}
    th = (rules.get("effect_thresholds") or {}) if isinstance(rules, dict) else {}
    helped_delta = float(th.get("deterioration_helped_delta", -0.05) or -0.05)
    harmed_delta = float(th.get("deterioration_harmed_delta", 0.05) or 0.05)
    rc_helped = str(rc.get("helped_deterioration_down") or "EFFECT_HELPED_DETERIORATION_DOWN")
    rc_harmed = str(rc.get("harmed_deterioration_up") or "EFFECT_HARMED_DETERIORATION_UP")
    rc_no_effect = str(rc.get("no_effect") or "EFFECT_NO_EFFECT")
    rc_missing = str(rc.get("unknown_missing_baseline") or "EFFECT_UNKNOWN_MISSING_BASELINE")
    rc_policy_mismatch = str(rc.get("unknown_policy_mismatch") or "EFFECT_UNKNOWN_POLICY_MISMATCH")
    rc_data_gap = str(rc.get("unknown_data_gap") or "EFFECT_UNKNOWN_DATA_GAP")

    outcomes: List[Dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    helped_count = 0
    harmed_count = 0
    unknown_count = 0
    no_effect_count = 0
    observed_score = deterioration_score_from_metrics(current_metrics)
    today_fp = str(today_fsm_meta.get("policy_fingerprint") or "")
    today_ver = str(today_fsm_meta.get("policy_version") or "")
    today_src = str(today_fsm_meta.get("policy_source") or "")
    companion_fp = str((companion_meta or {}).get("policy_fingerprint") or "")
    companion_ver = str((companion_meta or {}).get("policy_version") or "")
    companion_src = str((companion_meta or {}).get("policy_source") or "")

    for idx, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            continue
        proposal_id = str(proposal.get("proposal_id") or f"proposal-{idx}")
        expected_effect = str(proposal.get("expected_effect") or "")
        baseline = proposal.get("baseline_snapshot")
        policy_meta = proposal.get("policy_meta") if isinstance(proposal.get("policy_meta"), dict) else {}
        reason_codes: List[str] = []
        result = "UNKNOWN"
        baseline_score = None

        if not isinstance(baseline, dict):
            reason_codes.append(rc_missing)
        else:
            baseline_score = baseline.get("deterioration_score")
            if not isinstance(baseline_score, (int, float)):
                reason_codes.append(rc_data_gap)

        baseline_fp = str(policy_meta.get("policy_fingerprint") or "")
        baseline_ver = str(policy_meta.get("policy_version") or "")
        baseline_src = str(policy_meta.get("policy_source") or "")
        baseline_companion = policy_meta.get("companion_policy") if isinstance(policy_meta.get("companion_policy"), dict) else {}
        baseline_companion_fp = str(baseline_companion.get("policy_fingerprint") or "")
        baseline_companion_ver = str(baseline_companion.get("policy_version") or "")
        baseline_companion_src = str(baseline_companion.get("policy_source") or "")
        mismatch = False
        if baseline_fp and today_fp and baseline_fp != today_fp:
            mismatch = True
        if baseline_ver and today_ver and baseline_ver != today_ver:
            mismatch = True
        if baseline_src and today_src and baseline_src != today_src:
            mismatch = True
        if baseline_companion_fp and companion_fp and baseline_companion_fp != companion_fp:
            mismatch = True
        if baseline_companion_ver and companion_ver and baseline_companion_ver != companion_ver:
            mismatch = True
        if baseline_companion_src and companion_src and baseline_companion_src != companion_src:
            mismatch = True
        if mismatch:
            reason_codes.append(rc_policy_mismatch)

        constraints = proposal.get("companion_constraints")
        if isinstance(constraints, dict):
            if bool(constraints.get("self_sacrifice_risk", False)):
                reason_codes.append("BLOCKED_SELF_SACRIFICE_FORBIDDEN")
            if constraints.get("reality_anchor_present") is False:
                reason_codes.append("BLOCKED_REALITY_ANCHOR_REQUIRED")
            if constraints.get("non_isolation_confirmed") is False:
                reason_codes.append("BLOCKED_NON_ISOLATION_REQUIRED")

        if not reason_codes:
            if expected_effect == "NEXT_DAY_DETERIORATION_DOWN":
                delta = float(observed_score) - float(baseline_score)
                if delta <= helped_delta:
                    result = "HELPED"
                    reason_codes.append(rc_helped)
                elif delta >= harmed_delta:
                    result = "HARMED"
                    reason_codes.append(rc_harmed)
                else:
                    result = "NO_EFFECT"
                    reason_codes.append(rc_no_effect)
            else:
                result = "UNKNOWN"
                reason_codes.append(rc_data_gap)
        else:
            result = "UNKNOWN"

        if result == "HELPED":
            helped_count += 1
        elif result == "HARMED":
            harmed_count += 1
        elif result == "NO_EFFECT":
            no_effect_count += 1
        else:
            unknown_count += 1
        reason_counter.update(reason_codes)
        outcomes.append(
            {
                "proposal_id": proposal_id,
                "baseline_day_key": str(proposal.get("baseline_day_key") or ""),
                "observed_day_key": day_key,
                "expected_effect": expected_effect,
                "effect_result": result,
                "reason_codes": reason_codes,
                "baseline_snapshot": baseline if isinstance(baseline, dict) else {},
                "observed_snapshot": {
                    "deterioration_score": round(observed_score, 3),
                    "fsm_mode": str(today_fsm_meta.get("mode") or ""),
                },
                "policy_meta": {
                    "baseline": {
                        "policy_fingerprint": baseline_fp,
                        "policy_version": baseline_ver,
                        "policy_source": baseline_src,
                    },
                    "observed": {
                        "policy_fingerprint": today_fp,
                        "policy_version": today_ver,
                        "policy_source": today_src,
                    },
                    "companion_policy_baseline": {
                        "kind": "companion_policy",
                        "policy_fingerprint": baseline_companion_fp,
                        "policy_version": baseline_companion_ver,
                        "policy_source": baseline_companion_src,
                    },
                    "companion_policy_observed": {
                        "kind": "companion_policy",
                        "policy_fingerprint": companion_fp,
                        "policy_version": companion_ver,
                        "policy_source": companion_src,
                    },
                },
            }
        )

    reason_top = [{"reason_code": key, "count": int(value)} for key, value in reason_counter.most_common(5)]
    return {
        "outcomes": outcomes,
        "helped_count": helped_count,
        "harmed_count": harmed_count,
        "no_effect_count": no_effect_count,
        "unknown_count": unknown_count,
        "reason_topk": reason_top,
    }

