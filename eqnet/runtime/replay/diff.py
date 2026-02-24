from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml


def load_diff_ranking_policy(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    default = {
        "ranking_version": "diff_ranking_policy_v0",
        "weights": {
            "harmed_rate": 100.0,
            "unknown_rate": 50.0,
            "blocked_rate": 40.0,
            "helped_rate": 10.0,
            "sync_r_metric": 15.0,
        },
        "direction": {
            "harmed_rate": "increase_is_worse",
            "unknown_rate": "increase_is_worse",
            "blocked_rate": "increase_is_worse",
            "helped_rate": "increase_is_better",
            "sync_r_metric": "increase_is_better",
        },
        "modes": {"regressions_first": True, "top_k": 20},
    }
    source = str(path.as_posix())
    if path.exists():
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(payload, dict):
                policy = _merge_policy(default, payload)
            else:
                policy = dict(default)
        except Exception:
            policy = dict(default)
    else:
        policy = dict(default)

    canonical = json.dumps(_normalize_for_hash(policy), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    meta = {
        "source": source,
        "version": str(policy.get("ranking_version") or "diff_ranking_policy_v0"),
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
    }
    return policy, meta


def build_diff_summary(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    comparison_scope: Mapping[str, Any] | None = None,
    ranking_policy: Mapping[str, Any] | None = None,
    ranking_policy_meta: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    agg_a = summary_a.get("aggregate") if isinstance(summary_a.get("aggregate"), dict) else {}
    agg_b = summary_b.get("aggregate") if isinstance(summary_b.get("aggregate"), dict) else {}
    keys = sorted(set(agg_a.keys()) | set(agg_b.keys()))
    deltas: Dict[str, float] = {}
    for key in keys:
        va = _to_float(agg_a.get(key))
        vb = _to_float(agg_b.get(key))
        deltas[key] = round(vb - va, 6)

    policy = dict(ranking_policy or {})
    if not policy:
        policy, _ = load_diff_ranking_policy(Path("configs/diff_ranking_policy_v0.yaml"))
    top_k = int(((policy.get("modes") or {}).get("top_k")) or 20)
    regressions_first = bool(((policy.get("modes") or {}).get("regressions_first")) if isinstance(policy.get("modes"), Mapping) else True)
    regressions, improvements = _rank_changes(deltas, policy=policy, top_k=top_k)
    if regressions_first:
        top_changes = list(regressions) + list(improvements)
    else:
        top_changes = list(improvements) + list(regressions)
    top_changes = top_changes[: max(0, top_k)]

    scope = dict(comparison_scope or {})
    inputs_fp = _ranking_inputs_fingerprint(agg_a, agg_b, scope)
    return {
        "config_set_a_meta": summary_a.get("config_meta") if isinstance(summary_a.get("config_meta"), dict) else {},
        "config_set_b_meta": summary_b.get("config_meta") if isinstance(summary_b.get("config_meta"), dict) else {},
        "comparison_scope": scope,
        "ranking_policy_meta": dict(ranking_policy_meta or {}),
        "ranking_inputs_fingerprint": inputs_fp,
        "aggregate_delta": deltas,
        "top_regressions": regressions,
        "top_improvements": improvements,
        "top_changes": top_changes,
    }


def _rank_changes(deltas: Dict[str, float], *, policy: Mapping[str, Any], top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    regressions: List[Tuple[str, float, float, str]] = []
    improvements: List[Tuple[str, float, float, str]] = []
    for metric_key, delta in deltas.items():
        family = _metric_family(metric_key)
        if family is None:
            continue
        weight = float(((policy.get("weights") or {}).get(family)) or 0.0)
        direction = str(((policy.get("direction") or {}).get(family)) or "increase_is_worse")
        reg_score, imp_score = _scores_for_delta(delta, direction=direction, weight=weight)
        segment = _metric_segment(metric_key)
        if reg_score > 0.0:
            regressions.append((metric_key, float(delta), float(reg_score), segment))
        if imp_score > 0.0:
            improvements.append((metric_key, float(delta), float(imp_score), segment))

    regressions_sorted = sorted(regressions, key=lambda item: (-item[2], str(item[0]), str(item[3])))
    improvements_sorted = sorted(improvements, key=lambda item: (-item[2], str(item[0]), str(item[3])))
    return (
        [{"metric": m, "delta": d, "severity_score": s, "segment": seg} for m, d, s, seg in regressions_sorted[: max(0, top_k)]],
        [{"metric": m, "delta": d, "severity_score": s, "segment": seg} for m, d, s, seg in improvements_sorted[: max(0, top_k)]],
    )


def _metric_family(metric_key: str) -> str | None:
    key = str(metric_key).lower()
    if "sync_r_" in key:
        return "sync_r_metric"
    if "downshift_applied_rate" in key or "suppressed_time_ratio" in key:
        return "blocked_rate"
    if "blocked_rate" in key:
        return "blocked_rate"
    if "harmed" in key:
        return "harmed_rate"
    if "reject_rate" in key:
        return "unknown_rate"
    if "unknown" in key:
        return "unknown_rate"
    if "helped" in key:
        return "helped_rate"
    return None


def _metric_segment(metric_key: str) -> str:
    key = str(metric_key)
    if "." in key:
        return key.split(".", 1)[0]
    if "_" in key:
        return key.split("_", 1)[0]
    return "global"


def _scores_for_delta(delta: float, *, direction: str, weight: float) -> Tuple[float, float]:
    if direction == "increase_is_better":
        regression = max(0.0, -float(delta)) * float(weight)
        improvement = max(0.0, float(delta)) * float(weight)
        return regression, improvement
    regression = max(0.0, float(delta)) * float(weight)
    improvement = max(0.0, -float(delta)) * float(weight)
    return regression, improvement


def _ranking_inputs_fingerprint(agg_a: Mapping[str, Any], agg_b: Mapping[str, Any], scope: Mapping[str, Any]) -> str:
    payload = {"aggregate_a": dict(agg_a), "aggregate_b": dict(agg_b), "comparison_scope": dict(scope)}
    canonical = json.dumps(_normalize_for_hash(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _merge_policy(default: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(default)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            merged = dict(out[key])  # type: ignore[index]
            merged.update(dict(value))
            out[key] = merged
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


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
