from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping
import yaml

from eqnet.runtime.future_contracts import (
    compute_perception_quality,
    evaluate_imagery_events,
    evaluate_realtime_outcomes,
    evaluate_sync_outcomes,
    load_imagery_policy,
    load_perception_quality_rules,
    load_realtime_rules,
    load_sync_policy,
    load_sync_quality_rules,
    summarize_sync_cue_proposals,
    summarize_perception_quality,
    summarize_perception_quality_breakdown,
)
from eqnet.runtime.companion_policy import companion_policy_meta, load_lifelong_companion_policy
from eqnet.runtime.nightly.green_bridge import (
    apply_green_priority_patch,
    compute_green_bridge_snapshot,
    load_green_bridge_policy,
)
from eqnet.runtime.nightly.behavior_change import (
    compute_behavior_change_snapshot,
    load_behavior_change_policy,
)
from eqnet.runtime.replay.loader import read_jsonl, resolve_trace_files
from eqnet.runtime.replay.selectors import group_by_day, select_day_range
from scripts.run_nightly_audit import (
    _adaptive_fsm_snapshot,
    _evaluate_preventive_proposals,
    _extract_nightly_metrics,
    _forecast_lite,
    _fsm_feature_events_from_trace,
    _load_forecast_effect_rules,
)


@dataclass(frozen=True)
class ReplayConfig:
    trace_path: Path
    start_day_key: str | None = None
    end_day_key: str | None = None
    config_set: str | None = None
    config_root: Path = Path("configs")


def run_replay(config: ReplayConfig) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in resolve_trace_files(config.trace_path):
        rows.extend(read_jsonl(path))
    grouped = group_by_day(rows)
    selected = select_day_range(grouped, start_day_key=config.start_day_key, end_day_key=config.end_day_key)
    day_keys = sorted(selected.keys())
    if not day_keys:
        return {"daily": [], "aggregate": {"day_count": 0}}

    replay_policy = _load_replay_policy(config.config_root / "replay_policy_v0.yaml")
    config_dir = _resolve_config_dir(config.config_root, config.config_set, replay_policy)
    _validate_required_files(config_dir, replay_policy)
    config_meta = _config_meta(config_dir, replay_policy)
    effect_rules = _load_forecast_effect_rules(config_dir / "forecast_effect_rules_v0.yaml")
    realtime_rules = load_realtime_rules(config_dir / "realtime_forecast_rules_v0.yaml")
    imagery_policy = load_imagery_policy(config_dir / "imagery_policy_v0.yaml")
    quality_rules = load_perception_quality_rules(config_dir / "perception_quality_rules_v0.yaml")
    companion_policy = load_lifelong_companion_policy(config_dir / "lifelong_companion_policy_v0.yaml")
    companion_meta = companion_policy_meta(companion_policy)
    sync_policy = load_sync_policy(config_dir / "sync_policy_v0.yaml")
    sync_quality_rules = load_sync_quality_rules(config_dir / "sync_quality_rules_v0.yaml")
    green_bridge_policy = load_green_bridge_policy(config_dir / "green_bridge_policy_v0.yaml")
    behavior_change_policy = load_behavior_change_policy(config_dir / "behavior_change_v0.yaml")
    sync_policy_meta = companion_policy_meta(sync_policy)

    daily: List[Dict[str, Any]] = []
    prev_report: Dict[str, Any] = {}
    prev_metrics: Dict[str, Any] = {"sat_p95": 0.0, "low_ratio": 0.0, "mecpe_contract_error_ratio": 0.0}
    for day_key in day_keys:
        day_rows = list(selected.get(day_key) or [])
        day_rows.sort(key=lambda row: int(row.get("timestamp_ms") or 0))
        adaptive_fsm = _adaptive_fsm_snapshot(day_rows)
        current_metrics = _metrics_from_trace(day_rows)
        eval_ts_ms = _eval_ts_ms_for_day(day_key)

        preventive_outcomes = _evaluate_preventive_proposals(
            prev_report=prev_report,
            current_metrics=current_metrics,
            today_fsm_meta=adaptive_fsm,
            companion_meta=companion_meta,
            day_key=day_key,
            rules=effect_rules,
        )

        forecast = _forecast_lite(
            current_metrics=current_metrics,
            prev_metrics=prev_metrics,
            nightly_payload=None,
            fsm_mode=str(adaptive_fsm.get("mode") or "STABLE"),
            day_key=day_key,
            fsm_policy_meta=adaptive_fsm,
            companion_meta=companion_meta,
        )
        green_bridge = compute_green_bridge_snapshot(
            current_metrics=current_metrics,
            fsm_mode=str(adaptive_fsm.get("mode") or "STABLE"),
            companion_policy_valid=True,
            policy=green_bridge_policy,
        )
        realtime_proposals = list(forecast.get("realtime_forecast_proposals") or [])
        ttl_default = int((realtime_rules.get("ttl_sec_default") or 300))
        for proposal in realtime_proposals:
            proposal["ttl_sec"] = int(proposal.get("ttl_sec") or ttl_default)
            proposal["ts_utc"] = _iso_from_ms(eval_ts_ms)
            quality = compute_perception_quality(proposal, now_ts_ms=eval_ts_ms, rules=quality_rules)
            proposal["perception_quality"] = {
                "freshness_sec": quality.get("freshness_sec"),
                "freshness_ratio": quality.get("freshness_ratio"),
                "confidence": quality.get("confidence"),
                "noise": quality.get("noise"),
            }
            base_priority = float(quality.get("priority_score", 0.0) or 0.0)
            patch = apply_green_priority_patch(
                base_priority_score=base_priority,
                green_snapshot=green_bridge,
                policy=green_bridge_policy,
                blocked=False,
                suppressed=False,
                unknown=False,
            )
            proposal["priority_score"] = patch.get("priority_score")
            proposal["green_priority_patch"] = {
                "applied": bool(patch.get("priority_patch_applied", False)),
                "delta": float(patch.get("priority_patch_delta", 0.0) or 0.0),
                "reason_codes": list(patch.get("priority_patch_reason_codes") or []),
                "policy_meta": dict(patch.get("policy_meta") or {}),
            }
            proposal["priority_reason_codes"] = quality.get("reason_codes")

        realtime_outcomes = evaluate_realtime_outcomes(
            prev_report.get("realtime_forecast_proposals") if isinstance(prev_report.get("realtime_forecast_proposals"), list) else [],
            evaluation_day_key=day_key,
            today_policy_meta={
                "policy_fingerprint": str(adaptive_fsm.get("policy_fingerprint") or ""),
                "policy_version": str(adaptive_fsm.get("policy_version") or ""),
                "policy_source": str(adaptive_fsm.get("policy_source") or ""),
            },
            now_ts_ms=eval_ts_ms,
            companion_policy=companion_policy,
        )
        sync_proposals: List[Dict[str, Any]] = []
        sync_summary = summarize_sync_cue_proposals(sync_proposals, companion_policy=companion_policy)
        sync_outcomes = evaluate_sync_outcomes(
            prev_report.get("sync_cue_proposals") if isinstance(prev_report.get("sync_cue_proposals"), list) else [],
            evaluation_day_key=day_key,
            now_ts_ms=eval_ts_ms,
            today_sync_policy_meta={
                "policy_fingerprint": str(sync_policy_meta.get("policy_fingerprint") or ""),
                "policy_version": str(sync_policy_meta.get("policy_version") or ""),
                "policy_source": str(sync_policy_meta.get("policy_source") or ""),
            },
            today_companion_policy_meta={
                "policy_fingerprint": str(companion_meta.get("policy_fingerprint") or ""),
                "policy_version": str(companion_meta.get("policy_version") or ""),
                "policy_source": str(companion_meta.get("policy_source") or ""),
            },
            today_sync_snapshot={
                "sync_order_parameter_r": float((sync_summary.get("sync_order_parameter_r_stats") or {}).get("median") or 0.0),
            },
            companion_policy=companion_policy,
            rules=sync_quality_rules,
        )
        imagery_outcomes = evaluate_imagery_events(
            prev_report.get("imagery_events") if isinstance(prev_report.get("imagery_events"), list) else [],
            now_ts_ms=eval_ts_ms,
            evaluation_day_key=day_key,
        )
        quality_summary = summarize_perception_quality(realtime_proposals, now_ts_ms=eval_ts_ms, rules=quality_rules)
        quality_breakdown = summarize_perception_quality_breakdown(realtime_proposals, now_ts_ms=eval_ts_ms, rules=quality_rules)
        green_patch_applied_count = 0
        green_patch_skipped_by_guard_count = 0
        green_patch_deltas: List[float] = []
        for proposal in realtime_proposals:
            patch = proposal.get("green_priority_patch") if isinstance(proposal.get("green_priority_patch"), dict) else {}
            if not isinstance(patch, dict):
                continue
            if bool(patch.get("applied", False)):
                green_patch_applied_count += 1
            reasons = patch.get("reason_codes") if isinstance(patch.get("reason_codes"), list) else []
            if "GREEN_PRIORITY_PATCH_SKIPPED_STATE_GUARD" in [str(x) for x in reasons]:
                green_patch_skipped_by_guard_count += 1
            try:
                green_patch_deltas.append(float(patch.get("delta", 0.0) or 0.0))
            except (TypeError, ValueError):
                green_patch_deltas.append(0.0)

        day_payload = {
            "day_key": day_key,
            "trace_count": len(day_rows),
            "metrics": current_metrics,
            "adaptive_fsm": adaptive_fsm,
            "forecast_lite_score": float(forecast.get("forecast_lite_score") or 0.0),
            "preventive_proposals": list(forecast.get("preventive_proposals") or []),
            "realtime_forecast_proposals": realtime_proposals,
            "preventive_proposal_outcomes": preventive_outcomes.get("outcomes", []),
            "preventive_outcome_helped_count": int(preventive_outcomes.get("helped_count", 0) or 0),
            "preventive_outcome_harmed_count": int(preventive_outcomes.get("harmed_count", 0) or 0),
            "preventive_outcome_unknown_count": int(preventive_outcomes.get("unknown_count", 0) or 0),
            "preventive_outcome_no_effect_count": int(preventive_outcomes.get("no_effect_count", 0) or 0),
            "realtime_forecast_outcomes": realtime_outcomes.get("outcomes", []),
            "realtime_outcome_helped_count": int(realtime_outcomes.get("helped_count", 0) or 0),
            "realtime_outcome_harmed_count": int(realtime_outcomes.get("harmed_count", 0) or 0),
            "realtime_outcome_unknown_count": int(realtime_outcomes.get("unknown_count", 0) or 0),
            "realtime_outcome_no_effect_count": int(realtime_outcomes.get("no_effect_count", 0) or 0),
            "imagery_outcomes": imagery_outcomes.get("outcomes", []),
            "quality_unknown_origin_count": int(quality_summary.get("quality_unknown_origin_count", 0) or 0),
            "quality_low_freshness_count": int(quality_summary.get("quality_low_freshness_count", 0) or 0),
            "quality_high_noise_count": int(quality_summary.get("quality_high_noise_count", 0) or 0),
            "priority_score_stats": quality_summary.get("priority_score_stats", {}),
            "quality_by_origin": quality_breakdown.get("quality_by_origin", {}),
            "quality_by_kind": quality_breakdown.get("quality_by_kind", {}),
            "green_bridge": green_bridge,
            "green_response_score": float(green_bridge.get("green_response_score", 0.0) or 0.0),
            "green_quality": float(green_bridge.get("green_quality", 0.0) or 0.0),
            "green_decay_tau": float(green_bridge.get("green_decay_tau", 0.0) or 0.0),
            "green_mode": str(green_bridge.get("green_mode") or "OFF"),
            "green_priority_patch_applied_count": int(green_patch_applied_count),
            "green_priority_patch_skipped_by_guard_count": int(green_patch_skipped_by_guard_count),
            "green_priority_patch_delta_stats": {
                "median": round(_percentile(green_patch_deltas, 0.5), 6) if green_patch_deltas else 0.0,
                "p95": round(_percentile(green_patch_deltas, 0.95), 6) if green_patch_deltas else 0.0,
            },
            "rule_versions": {
                "effect": str(effect_rules.get("schema_version") or ""),
                "realtime": str(realtime_rules.get("schema_version") or ""),
                "imagery": str(imagery_policy.get("schema_version") or ""),
                "quality": str(quality_rules.get("schema_version") or ""),
                "sync_policy": str(sync_policy.get("schema_version") or ""),
                "sync_quality": str(sync_quality_rules.get("schema_version") or ""),
            },
            "sync_cue_proposals": sync_proposals,
            "sync_blocked_count_by_reason": sync_summary.get("sync_blocked_count_by_reason", {}),
            "sync_order_parameter_r_stats": sync_summary.get("sync_order_parameter_r_stats", {}),
            "sync_blocked_rate": float(sync_summary.get("sync_blocked_rate", 0.0) or 0.0),
            "sync_blocked_rate_by_origin": sync_summary.get("sync_blocked_rate_by_origin", {}),
            "sync_outcomes": sync_outcomes.get("outcomes", []),
            "sync_outcome_helped_count": int(sync_outcomes.get("helped_count", 0) or 0),
            "sync_outcome_harmed_count": int(sync_outcomes.get("harmed_count", 0) or 0),
            "sync_outcome_no_effect_count": int(sync_outcomes.get("no_effect_count", 0) or 0),
            "sync_outcome_unknown_count": int(sync_outcomes.get("unknown_count", 0) or 0),
            "sync_outcome_blocked_count": int(sync_outcomes.get("blocked_count", 0) or 0),
            "sync_outcome_reason_topk": sync_outcomes.get("reason_topk", []),
            "sync_micro_helped_count": 0,
            "sync_micro_harmed_count": 0,
            "sync_micro_unknown_count": 0,
            "sync_micro_no_effect_count": 0,
            "sync_downshift_applied_count": 0,
            "sync_emit_suppressed_time_ratio": 0.0,
        }
        behavior_change = compute_behavior_change_snapshot(
            current_payload=day_payload,
            prev_payload=prev_report if isinstance(prev_report, dict) else {},
            telemetry_dir=Path("telemetry"),
            day=datetime.strptime(day_key, "%Y-%m-%d").date(),
            policy=behavior_change_policy,
        )
        day_payload["behavior_change"] = behavior_change
        day_payload["behavior_change_priority_shift"] = float(behavior_change.get("proposal_priority_shift", 0.0) or 0.0)
        day_payload["behavior_change_reject_rate_delta"] = float(behavior_change.get("reject_rate_delta", 0.0) or 0.0)
        day_payload["behavior_change_harmed_rate_delta"] = float(behavior_change.get("harmed_rate_delta", 0.0) or 0.0)
        gate = behavior_change.get("gate") if isinstance(behavior_change.get("gate"), dict) else {}
        day_payload["behavior_change_gate_status"] = str(gate.get("status") or "PASS")
        day_payload["behavior_change_gate_reason_codes"] = list(gate.get("reason_codes") or [])
        sig = behavior_change.get("signature") if isinstance(behavior_change.get("signature"), dict) else {}
        sig_support = behavior_change.get("signature_support") if isinstance(behavior_change.get("signature_support"), dict) else {}
        tol = behavior_change.get("tolerance") if isinstance(behavior_change.get("tolerance"), dict) else {}
        day_payload["behavior_change_signature_key"] = str(sig.get("key") or "")
        day_payload["behavior_change_sig_support_count"] = int(sig_support.get("compare_events", 0) or 0)
        day_payload["behavior_change_mix_weight_sig_effective"] = float(tol.get("mix_weight_sig_effective", 0.0) or 0.0)
        day_payload["behavior_change_trust_source"] = str(tol.get("trust_source") or "global")
        day_payload["behavior_change_active_preset"] = str(tol.get("active_preset") or "default")
        day_payload["behavior_change_preset_source"] = str(tol.get("preset_source") or "manual")
        daily.append(day_payload)
        prev_report = day_payload
        prev_metrics = dict(current_metrics)

    aggregate = _aggregate_daily(daily, behavior_change_policy=behavior_change_policy)
    return {"daily": daily, "aggregate": aggregate, "config_meta": config_meta}


def _resolve_config_dir(config_root: Path, config_set: str | None, replay_policy: Mapping[str, Any]) -> Path:
    if config_set:
        patterns = replay_policy.get("config_set_search_paths")
        if isinstance(patterns, list):
            for pattern in patterns:
                text = str(pattern or "").strip()
                if not text:
                    continue
                candidate = Path(text.replace("{name}", str(config_set)))
                if not candidate.is_absolute():
                    candidate = (Path.cwd() / candidate).resolve()
                if candidate.exists():
                    return candidate
        candidate = config_root / "config_sets" / str(config_set)
        if candidate.exists():
            return candidate
    return config_root


def _load_replay_policy(path: Path) -> Dict[str, Any]:
    default = {
        "schema_version": "replay_policy_v0",
        "required_files": [
            "fsm_policy_v0.yaml",
            "forecast_effect_rules_v0.yaml",
            "realtime_forecast_rules_v0.yaml",
            "imagery_policy_v0.yaml",
            "perception_quality_rules_v0.yaml",
            "lifelong_companion_policy_v0.yaml",
            "sync_policy_v0.yaml",
            "sync_quality_rules_v0.yaml",
            "realtime_downshift_policy_v0.yaml",
            "green_bridge_policy_v0.yaml",
            "behavior_change_v0.yaml",
        ],
        "config_set_search_paths": ["configs/config_sets/{name}"],
        "emit_config_fingerprints": True,
    }
    if not path.exists():
        return default
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return default
    if not isinstance(payload, dict):
        return default
    out = dict(default)
    out.update(payload)
    return out


def _validate_required_files(config_dir: Path, replay_policy: Mapping[str, Any]) -> None:
    required = replay_policy.get("required_files")
    if not isinstance(required, list):
        return
    missing: List[str] = []
    for name in required:
        fn = str(name or "").strip()
        if not fn:
            continue
        if not (config_dir / fn).exists():
            missing.append(fn)
    if missing:
        raise FileNotFoundError(f"missing required replay config files in {config_dir}: {missing}")


def _config_meta(config_dir: Path, replay_policy: Mapping[str, Any]) -> Dict[str, Any]:
    required = replay_policy.get("required_files")
    files = [str(name or "").strip() for name in required] if isinstance(required, list) else []
    files = [name for name in files if name]
    include_fp = bool(replay_policy.get("emit_config_fingerprints", True))
    entries: Dict[str, Any] = {}
    for name in sorted(files):
        path = config_dir / name
        if not path.exists():
            continue
        version = ""
        fingerprint = ""
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(payload, dict):
                version = str(payload.get("schema_version") or "")
                if include_fp:
                    canonical = json.dumps(_normalize_for_hash(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        except Exception:
            pass
        entries[name] = {
            "source": str(path.as_posix()),
            "version": version,
            "fingerprint": fingerprint,
        }
    return {"config_dir": str(config_dir.as_posix()), "files": entries}


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_for_hash(v) for v in value]
    return value


def _metrics_from_trace(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    features = _fsm_feature_events_from_trace([dict(row) for row in records])
    if not features:
        return {"sat_p95": 0.0, "low_ratio": 0.0, "mecpe_contract_error_ratio": 0.0}
    pending = [float(row.get("pending_ratio", 0.0) or 0.0) for row in features]
    contract = [float(row.get("contract_errors_ratio", 0.0) or 0.0) for row in features]
    sat_p95 = _percentile(pending, 0.95)
    low_ratio = float(sum(1 for v in pending if v >= 0.25)) / float(len(pending))
    contract_ratio = float(sum(contract) / float(len(contract)))
    return {
        "sat_p95": round(sat_p95, 3),
        "low_ratio": round(low_ratio, 3),
        "mecpe_contract_error_ratio": round(contract_ratio, 3),
    }


def _aggregate_daily(
    daily: List[Mapping[str, Any]],
    *,
    behavior_change_policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if not daily:
        return {"day_count": 0}
    day_count = int(len(daily))
    helped = sum(int(row.get("preventive_outcome_helped_count", 0) or 0) for row in daily)
    harmed = sum(int(row.get("preventive_outcome_harmed_count", 0) or 0) for row in daily)
    unknown = sum(int(row.get("preventive_outcome_unknown_count", 0) or 0) for row in daily)
    realtime_unknown = sum(int(row.get("realtime_outcome_unknown_count", 0) or 0) for row in daily)
    realtime_helped = sum(int(row.get("realtime_outcome_helped_count", 0) or 0) for row in daily)
    realtime_harmed = sum(int(row.get("realtime_outcome_harmed_count", 0) or 0) for row in daily)
    quality_unknown = sum(int(row.get("quality_unknown_origin_count", 0) or 0) for row in daily)
    freshness_low = sum(int(row.get("quality_low_freshness_count", 0) or 0) for row in daily)
    noise_high = sum(int(row.get("quality_high_noise_count", 0) or 0) for row in daily)
    medians = [float(((row.get("priority_score_stats") or {}).get("median")) or 0.0) for row in daily]
    green_scores = [float(row.get("green_response_score", 0.0) or 0.0) for row in daily]
    green_quality = [float(row.get("green_quality", 0.0) or 0.0) for row in daily]
    green_tau = [float(row.get("green_decay_tau", 0.0) or 0.0) for row in daily]
    green_patch_applied = sum(int(row.get("green_priority_patch_applied_count", 0) or 0) for row in daily)
    green_patch_skipped = sum(int(row.get("green_priority_patch_skipped_by_guard_count", 0) or 0) for row in daily)
    green_patch_delta_medians = [float(((row.get("green_priority_patch_delta_stats") or {}).get("median")) or 0.0) for row in daily]
    green_patch_delta_p95 = [float(((row.get("green_priority_patch_delta_stats") or {}).get("p95")) or 0.0) for row in daily]
    sync_helped = sum(int(row.get("sync_outcome_helped_count", 0) or 0) for row in daily)
    sync_harmed = sum(int(row.get("sync_outcome_harmed_count", 0) or 0) for row in daily)
    sync_unknown = sum(int(row.get("sync_outcome_unknown_count", 0) or 0) for row in daily)
    sync_blocked = sum(int(row.get("sync_outcome_blocked_count", 0) or 0) for row in daily)
    sync_requires = sum(int(row.get("sync_requires_approval_total", 0) or 0) for row in daily)
    sync_r_medians = [float(((row.get("sync_order_parameter_r_stats") or {}).get("median")) or 0.0) for row in daily]
    sync_r_p95 = [float(((row.get("sync_order_parameter_r_stats") or {}).get("p95")) or 0.0) for row in daily]
    sync_micro_helped = sum(int(row.get("sync_micro_helped_count", 0) or 0) for row in daily)
    sync_micro_harmed = sum(int(row.get("sync_micro_harmed_count", 0) or 0) for row in daily)
    sync_micro_unknown = sum(int(row.get("sync_micro_unknown_count", 0) or 0) for row in daily)
    sync_micro_no_effect = sum(int(row.get("sync_micro_no_effect_count", 0) or 0) for row in daily)
    sync_downshift_applied = sum(int(row.get("sync_downshift_applied_count", 0) or 0) for row in daily)
    sync_emit_suppressed_ratios = [float(row.get("sync_emit_suppressed_time_ratio", 0.0) or 0.0) for row in daily]
    bc_priority_shift = [float(row.get("behavior_change_priority_shift", 0.0) or 0.0) for row in daily]
    bc_reject_delta = [float(row.get("behavior_change_reject_rate_delta", 0.0) or 0.0) for row in daily]
    bc_harmed_delta = [float(row.get("behavior_change_harmed_rate_delta", 0.0) or 0.0) for row in daily]
    bc_gate_fail_count = sum(1 for row in daily if str(row.get("behavior_change_gate_status") or "PASS").upper() == "FAIL")
    bc_gate_warn_count = sum(1 for row in daily if str(row.get("behavior_change_gate_status") or "PASS").upper() == "WARN")
    bc_sig_active_keys = {
        str(row.get("behavior_change_signature_key") or "")
        for row in daily
        if str(row.get("behavior_change_signature_key") or "").strip()
    }
    bc_sig_fallback_count = sum(
        1
        for row in daily
        if str(row.get("behavior_change_trust_source") or "").startswith("global_fallback")
    )
    bc_sig_support_rows: List[tuple[str, int]] = []
    for row in daily:
        key = str(row.get("behavior_change_signature_key") or "").strip()
        if not key:
            continue
        support = int(row.get("behavior_change_sig_support_count", 0) or 0)
        bc_sig_support_rows.append((key, support))
    bc_sig_support_topk = [
        {"signature_key": key, "support_count": int(support)}
        for key, support in sorted(
            bc_sig_support_rows,
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:3]
    ]
    bc_mix_weight_effective = [
        float(row.get("behavior_change_mix_weight_sig_effective", 0.0) or 0.0)
        for row in daily
    ]
    bc_active_presets = [str(row.get("behavior_change_active_preset") or "").strip() for row in daily]
    bc_preset_sources = [str(row.get("behavior_change_preset_source") or "").strip() for row in daily]
    bc_preset_change_count = 0
    bc_preset_change_reason_counts: Dict[str, int] = {}
    prev_preset = ""
    for idx, preset in enumerate(bc_active_presets):
        if not preset:
            continue
        if not prev_preset:
            prev_preset = preset
            continue
        if preset != prev_preset:
            bc_preset_change_count += 1
            source = bc_preset_sources[idx] if idx < len(bc_preset_sources) else ""
            reason = source if source else "unknown"
            bc_preset_change_reason_counts[reason] = int(bc_preset_change_reason_counts.get(reason, 0)) + 1
            prev_preset = preset
    bc_preset_change_reasons_topk = [
        {"reason": reason, "count": int(count)}
        for reason, count in sorted(
            bc_preset_change_reason_counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:3]
    ]

    harmed_delta_by_preset_sum: Dict[str, float] = {}
    harmed_delta_by_preset_count: Dict[str, int] = {}
    reject_delta_by_preset_sum: Dict[str, float] = {}
    reject_delta_by_preset_count: Dict[str, int] = {}
    mix_effective_by_preset_sum: Dict[str, float] = {}
    mix_effective_by_preset_count: Dict[str, int] = {}
    for row in daily:
        preset = str(row.get("behavior_change_active_preset") or "").strip()
        if not preset:
            continue
        harmed_val = float(row.get("behavior_change_harmed_rate_delta", 0.0) or 0.0)
        reject_val = float(row.get("behavior_change_reject_rate_delta", 0.0) or 0.0)
        mix_val = float(row.get("behavior_change_mix_weight_sig_effective", 0.0) or 0.0)

        harmed_delta_by_preset_sum[preset] = float(harmed_delta_by_preset_sum.get(preset, 0.0)) + harmed_val
        harmed_delta_by_preset_count[preset] = int(harmed_delta_by_preset_count.get(preset, 0)) + 1
        reject_delta_by_preset_sum[preset] = float(reject_delta_by_preset_sum.get(preset, 0.0)) + reject_val
        reject_delta_by_preset_count[preset] = int(reject_delta_by_preset_count.get(preset, 0)) + 1
        mix_effective_by_preset_sum[preset] = float(mix_effective_by_preset_sum.get(preset, 0.0)) + mix_val
        mix_effective_by_preset_count[preset] = int(mix_effective_by_preset_count.get(preset, 0)) + 1

    harmed_delta_avg_by_preset = {
        preset: round(float(harmed_delta_by_preset_sum.get(preset, 0.0)) / float(max(1, harmed_delta_by_preset_count.get(preset, 0))), 6)
        for preset in sorted(harmed_delta_by_preset_sum.keys())
    }
    reject_delta_avg_by_preset = {
        preset: round(float(reject_delta_by_preset_sum.get(preset, 0.0)) / float(max(1, reject_delta_by_preset_count.get(preset, 0))), 6)
        for preset in sorted(reject_delta_by_preset_sum.keys())
    }
    mix_effective_avg_by_preset = {
        preset: round(float(mix_effective_by_preset_sum.get(preset, 0.0)) / float(max(1, mix_effective_by_preset_count.get(preset, 0))), 6)
        for preset in sorted(mix_effective_by_preset_sum.keys())
    }
    signature_cfg = behavior_change_policy.get("signature") if isinstance(behavior_change_policy, Mapping) and isinstance(behavior_change_policy.get("signature"), Mapping) else {}
    health_cfg = signature_cfg.get("health") if isinstance(signature_cfg.get("health"), Mapping) else {}
    max_keys = int(signature_cfg.get("max_keys", 128) or 128) if isinstance(signature_cfg, Mapping) else 128
    fallback_ratio_warn = float(health_cfg.get("fallback_ratio_warn", 0.9) or 0.9)
    active_keys_near_cap_ratio = float(health_cfg.get("active_keys_near_cap_ratio", 0.9) or 0.9)
    topk_support_min = int(health_cfg.get("topk_support_min", 2) or 2)
    warn_consecutive_weeks = max(1, int(health_cfg.get("warn_consecutive_weeks", 1) or 1))
    recover_consecutive_weeks = max(1, int(health_cfg.get("recover_consecutive_weeks", 1) or 1))
    reason_action_map = health_cfg.get("reason_action_map") if isinstance(health_cfg.get("reason_action_map"), Mapping) else {}
    actions_mode = str(health_cfg.get("actions_mode") or "advisory_only")
    actions_version = str(health_cfg.get("reason_action_map_version") or "v1")
    sync_helped_rate = (float(sync_helped) / float(day_count)) if day_count > 0 else 0.0
    sync_harmed_rate = (float(sync_harmed) / float(day_count)) if day_count > 0 else 0.0
    sync_unknown_rate = (float(sync_unknown) / float(day_count)) if day_count > 0 else 0.0
    sync_blocked_rate = (float(sync_blocked) / float(sync_requires)) if sync_requires > 0 else 0.0
    sync_micro_helped_rate = (float(sync_micro_helped) / float(day_count)) if day_count > 0 else 0.0
    sync_micro_harmed_rate = (float(sync_micro_harmed) / float(day_count)) if day_count > 0 else 0.0
    sync_micro_unknown_rate = (float(sync_micro_unknown) / float(day_count)) if day_count > 0 else 0.0
    sync_downshift_applied_rate = (float(sync_downshift_applied) / float(day_count)) if day_count > 0 else 0.0
    realtime_helped_rate = (float(realtime_helped) / float(day_count)) if day_count > 0 else 0.0
    realtime_harmed_rate = (float(realtime_harmed) / float(day_count)) if day_count > 0 else 0.0
    realtime_unknown_rate = (float(realtime_unknown) / float(day_count)) if day_count > 0 else 0.0
    preventive_helped_rate = (float(helped) / float(day_count)) if day_count > 0 else 0.0
    preventive_harmed_rate = (float(harmed) / float(day_count)) if day_count > 0 else 0.0
    preventive_unknown_rate = (float(unknown) / float(day_count)) if day_count > 0 else 0.0
    bc_sig_fallback_ratio = round(float(bc_sig_fallback_count) / float(day_count), 6) if day_count > 0 else 0.0
    week_health = _build_weekly_sig_health(
        daily=daily,
        max_keys=max_keys,
        fallback_ratio_warn=fallback_ratio_warn,
        active_keys_near_cap_ratio=active_keys_near_cap_ratio,
        topk_support_min=topk_support_min,
    )
    latest_week = week_health[-1] if week_health else {"week_key": "", "raw_warn": False, "reason_codes": []}
    warn_streak = _streak_from_tail(week_health, value=True)
    ok_streak = _streak_from_tail(week_health, value=False)
    latest_raw_warn = bool(latest_week.get("raw_warn", False))
    latest_raw_reasons = list(latest_week.get("reason_codes") or [])
    bc_sig_health_status = "OK"
    bc_sig_health_reasons: List[str] = []
    if latest_raw_warn:
        if warn_streak >= warn_consecutive_weeks:
            bc_sig_health_status = "WARN"
            bc_sig_health_reasons = list(latest_raw_reasons)
        else:
            bc_sig_health_status = "FYI"
            bc_sig_health_reasons = list(latest_raw_reasons) + ["BC_SIG_HEALTH_WARN_PENDING"]
    else:
        previous_warn = any(bool(entry.get("raw_warn", False)) for entry in week_health[:-1])
        if previous_warn and ok_streak < recover_consecutive_weeks:
            bc_sig_health_status = "WARN"
            bc_sig_health_reasons = ["BC_SIG_HEALTH_RECOVERY_PENDING"]
    recommended_actions_struct = _recommended_actions_from_reasons(
        reason_codes=bc_sig_health_reasons,
        reason_action_map=reason_action_map,
    )
    recommended_actions = [str(item.get("action") or "") for item in recommended_actions_struct if str(item.get("action") or "").strip()]
    recommended_scopes = sorted({str(item.get("scope") or "") for item in recommended_actions_struct if str(item.get("scope") or "").strip()})
    recommended_targets = sorted({str(item.get("target") or "") for item in recommended_actions_struct if str(item.get("target") or "").strip()})
    return {
        "day_count": day_count,
        "preventive_helped_total": int(helped),
        "preventive_harmed_total": int(harmed),
        "preventive_unknown_total": int(unknown),
        "preventive_helped_rate": round(preventive_helped_rate, 6),
        "preventive_harmed_rate": round(preventive_harmed_rate, 6),
        "preventive_unknown_rate": round(preventive_unknown_rate, 6),
        "realtime_helped_total": int(realtime_helped),
        "realtime_harmed_total": int(realtime_harmed),
        "realtime_unknown_total": int(realtime_unknown),
        "realtime_helped_rate": round(realtime_helped_rate, 6),
        "realtime_harmed_rate": round(realtime_harmed_rate, 6),
        "realtime_unknown_rate": round(realtime_unknown_rate, 6),
        "quality_unknown_origin_total": int(quality_unknown),
        "quality_low_freshness_total": int(freshness_low),
        "quality_high_noise_total": int(noise_high),
        "priority_median_avg": round(sum(medians) / float(len(medians)), 6) if medians else 0.0,
        "green_response_score_avg": round(sum(green_scores) / float(len(green_scores)), 6) if green_scores else 0.0,
        "green_quality_avg": round(sum(green_quality) / float(len(green_quality)), 6) if green_quality else 0.0,
        "green_decay_tau_avg": round(sum(green_tau) / float(len(green_tau)), 6) if green_tau else 0.0,
        "green_priority_patch_applied_total": int(green_patch_applied),
        "green_priority_patch_skipped_by_guard_total": int(green_patch_skipped),
        "green_priority_patch_delta_median_avg": round(sum(green_patch_delta_medians) / float(len(green_patch_delta_medians)), 6) if green_patch_delta_medians else 0.0,
        "green_priority_patch_delta_p95_avg": round(sum(green_patch_delta_p95) / float(len(green_patch_delta_p95)), 6) if green_patch_delta_p95 else 0.0,
        "sync_outcome_helped_total": int(sync_helped),
        "sync_outcome_harmed_total": int(sync_harmed),
        "sync_outcome_unknown_total": int(sync_unknown),
        "sync_outcome_blocked_total": int(sync_blocked),
        "sync_requires_approval_total": int(sync_requires),
        "sync_outcome_helped_rate": round(sync_helped_rate, 6),
        "sync_outcome_harmed_rate": round(sync_harmed_rate, 6),
        "sync_outcome_unknown_rate": round(sync_unknown_rate, 6),
        "sync_blocked_rate": round(sync_blocked_rate, 6),
        "sync_r_median_avg": round(sum(sync_r_medians) / float(len(sync_r_medians)), 6) if sync_r_medians else 0.0,
        "sync_r_p95_avg": round(sum(sync_r_p95) / float(len(sync_r_p95)), 6) if sync_r_p95 else 0.0,
        "sync_micro_helped_total": int(sync_micro_helped),
        "sync_micro_harmed_total": int(sync_micro_harmed),
        "sync_micro_unknown_total": int(sync_micro_unknown),
        "sync_micro_no_effect_total": int(sync_micro_no_effect),
        "sync_micro_helped_rate": round(sync_micro_helped_rate, 6),
        "sync_micro_harmed_rate": round(sync_micro_harmed_rate, 6),
        "sync_micro_unknown_rate": round(sync_micro_unknown_rate, 6),
        "sync_downshift_applied_total": int(sync_downshift_applied),
        "sync_downshift_applied_rate": round(sync_downshift_applied_rate, 6),
        "sync_emit_suppressed_time_ratio_avg": round(sum(sync_emit_suppressed_ratios) / float(len(sync_emit_suppressed_ratios)), 6) if sync_emit_suppressed_ratios else 0.0,
        "behavior_change_priority_shift_avg": round(sum(bc_priority_shift) / float(len(bc_priority_shift)), 6) if bc_priority_shift else 0.0,
        "behavior_change_reject_rate_delta_avg": round(sum(bc_reject_delta) / float(len(bc_reject_delta)), 6) if bc_reject_delta else 0.0,
        "behavior_change_harmed_rate_delta_avg": round(sum(bc_harmed_delta) / float(len(bc_harmed_delta)), 6) if bc_harmed_delta else 0.0,
        "behavior_change_gate_fail_total": int(bc_gate_fail_count),
        "behavior_change_gate_warn_total": int(bc_gate_warn_count),
        "behavior_change_sig_active_keys": int(len(bc_sig_active_keys)),
        "behavior_change_sig_fallback_ratio": bc_sig_fallback_ratio,
        "behavior_change_sig_topk_support": bc_sig_support_topk,
        "behavior_change_mix_weight_sig_effective_avg": round(sum(bc_mix_weight_effective) / float(len(bc_mix_weight_effective)), 6) if bc_mix_weight_effective else 0.0,
        "behavior_change_active_preset_latest": next((value for value in reversed(bc_active_presets) if value), "default"),
        "behavior_change_preset_source_latest": next((value for value in reversed(bc_preset_sources) if value), "manual"),
        "behavior_change_active_presets": sorted({value for value in bc_active_presets if value}),
        "behavior_change_preset_sources": sorted({value for value in bc_preset_sources if value}),
        "behavior_change_preset_change_count_weekly": int(bc_preset_change_count),
        "behavior_change_preset_change_reasons_topk": bc_preset_change_reasons_topk,
        "behavior_change_harmed_rate_delta_avg_by_preset": harmed_delta_avg_by_preset,
        "behavior_change_reject_rate_delta_avg_by_preset": reject_delta_avg_by_preset,
        "behavior_change_mix_weight_sig_effective_avg_by_preset": mix_effective_avg_by_preset,
        "behavior_change_sig_health_status": bc_sig_health_status,
        "behavior_change_sig_health_reason_codes": sorted(set(bc_sig_health_reasons)),
        "behavior_change_sig_health_recommended_actions_mode": actions_mode,
        "behavior_change_sig_health_recommended_actions_version": actions_version,
        "behavior_change_sig_health_recommended_actions": recommended_actions,
        "behavior_change_sig_health_recommended_actions_scope": recommended_scopes,
        "behavior_change_sig_health_recommended_actions_target": recommended_targets,
        "behavior_change_sig_health_recommended_actions_details": recommended_actions_struct,
        "behavior_change_sig_health_latest_week": str(latest_week.get("week_key") or ""),
        "behavior_change_sig_health_warn_streak": int(warn_streak),
        "behavior_change_sig_health_ok_streak": int(ok_streak),
    }


def _build_weekly_sig_health(
    *,
    daily: List[Mapping[str, Any]],
    max_keys: int,
    fallback_ratio_warn: float,
    active_keys_near_cap_ratio: float,
    topk_support_min: int,
) -> List[Dict[str, Any]]:
    week_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for row in daily:
        day_key = str(row.get("day_key") or "").strip()
        if not day_key:
            continue
        week_key = _iso_week_key(day_key)
        if not week_key:
            continue
        week_rows.setdefault(week_key, []).append(row)
    out: List[Dict[str, Any]] = []
    for week_key in sorted(week_rows.keys()):
        rows = week_rows.get(week_key) or []
        days = len(rows)
        if days <= 0:
            continue
        fallback_count = sum(
            1 for row in rows if str(row.get("behavior_change_trust_source") or "").startswith("global_fallback")
        )
        fallback_ratio = float(fallback_count) / float(days)
        active_keys = {
            str(row.get("behavior_change_signature_key") or "")
            for row in rows
            if str(row.get("behavior_change_signature_key") or "").strip()
        }
        top_support = 0
        for row in rows:
            support = int(row.get("behavior_change_sig_support_count", 0) or 0)
            if support > top_support:
                top_support = support
        reasons: List[str] = []
        if fallback_ratio > fallback_ratio_warn:
            reasons.append("BC_SIG_HEALTH_FALLBACK_RATIO_HIGH")
        if max_keys > 0 and len(active_keys) >= int(max_keys * active_keys_near_cap_ratio):
            reasons.append("BC_SIG_HEALTH_ACTIVE_KEYS_NEAR_CAP")
        if top_support < topk_support_min:
            reasons.append("BC_SIG_HEALTH_TOP_SUPPORT_LOW")
        out.append(
            {
                "week_key": week_key,
                "raw_warn": len(reasons) > 0,
                "reason_codes": sorted(set(reasons)),
            }
        )
    return out


def _iso_week_key(day_key: str) -> str:
    try:
        dt = datetime.strptime(day_key, "%Y-%m-%d")
    except Exception:
        return ""
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def _streak_from_tail(week_health: List[Mapping[str, Any]], *, value: bool) -> int:
    streak = 0
    for row in reversed(week_health):
        raw_warn = bool(row.get("raw_warn", False))
        if raw_warn == value:
            streak += 1
            continue
        break
    return streak


def _recommended_actions_from_reasons(
    *,
    reason_codes: List[str],
    reason_action_map: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    mapping = reason_action_map if isinstance(reason_action_map, Mapping) else {}
    for reason in reason_codes:
        raw = mapping.get(reason)
        if isinstance(raw, str):
            text = str(raw).strip()
            if text:
                actions.append(
                    {
                        "action": text,
                        "scope": "",
                        "target": "",
                        "effort": "",
                        "risk": "",
                        "expected_effect": "",
                    }
                )
            continue
        if isinstance(raw, Mapping):
            item = _normalize_action_item(raw)
            if item:
                actions.append(item)
            continue
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, str):
                    text = str(entry).strip()
                    if text:
                        actions.append(
                            {
                                "action": text,
                                "scope": "",
                                "target": "",
                                "effort": "",
                                "risk": "",
                                "expected_effect": "",
                            }
                        )
                    continue
                if isinstance(entry, Mapping):
                    item = _normalize_action_item(entry)
                    if item:
                        actions.append(item)
    dedup: Dict[str, Dict[str, Any]] = {}
    for item in actions:
        key = "|".join(
            [
                str(item.get("action") or ""),
                str(item.get("scope") or ""),
                str(item.get("target") or ""),
                str(item.get("effort") or ""),
                str(item.get("risk") or ""),
                str(item.get("expected_effect") or ""),
            ]
        )
        dedup[key] = item
    return [dedup[key] for key in sorted(dedup.keys())]


def _normalize_action_item(entry: Mapping[str, Any]) -> Dict[str, Any] | None:
    action = str(entry.get("action") or "").strip()
    if not action:
        return None
    scope = str(entry.get("scope") or "").strip()
    target = str(entry.get("target") or "").strip()
    effort = str(entry.get("effort") or "").strip()
    risk = str(entry.get("risk") or "").strip()
    expected_effect = str(entry.get("expected_effect") or "").strip()
    return {
        "action": action,
        "scope": scope,
        "target": target,
        "effort": effort,
        "risk": risk,
        "expected_effect": expected_effect,
    }


def _eval_ts_ms_for_day(day_key: str) -> int:
    dt = datetime.strptime(day_key, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=23, minute=59, second=0)
    return int(dt.timestamp() * 1000)


def _iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx])
