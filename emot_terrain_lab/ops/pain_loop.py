from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Mapping

from emot_terrain_lab.utils.jsonl_io import append_jsonl, read_jsonl, tail_jsonl
from emot_terrain_lab.utils.validate import validate_pain_record

PAIN_DIR = Path("logs/pain")
PAIN_LOG = PAIN_DIR / "pain_events.jsonl"
FORGIVE_LOG = PAIN_DIR / "forgiveness_events.jsonl"
STATS_LOG = PAIN_DIR / "pain_empathy_stats.jsonl"
POLICY_LOG = PAIN_DIR / "policy_updates.jsonl"
VALUE_INFLUENCE_LOG = PAIN_DIR / "value_influence.jsonl"

_RECENT_IDEMPOTENCY: deque[str] = deque(maxlen=256)


@dataclass
class PainEvent:
    ts_ms: int
    kind: str  # isolation | energy_depletion | value_conflict | other
    delta_aff: float
    reasons: List[str]
    hp: Dict[str, float]
    context: Dict[str, Any]
    labels: List[str]


def _resolve_log_path(base: Path, rotate_daily: bool = False) -> str:
    if not rotate_daily:
        return str(base)
    stem = base.stem
    suffix = base.suffix or ".jsonl"
    today = datetime.utcnow().strftime("%Y%m%d")
    return str(base.parent / f"{stem}-{today}{suffix}")


def _record_seen(idempotency_key: str) -> bool:
    if idempotency_key in _RECENT_IDEMPOTENCY:
        return True
    _RECENT_IDEMPOTENCY.append(idempotency_key)
    return False


def _pearson_corr(vec1: Sequence[float], vec2: Sequence[float]) -> Optional[float]:
    if len(vec1) != len(vec2) or not vec1:
        return None
    mean1 = sum(vec1) / len(vec1)
    mean2 = sum(vec2) / len(vec2)
    num = sum((a - mean1) * (b - mean2) for a, b in zip(vec1, vec2))
    den1 = math.sqrt(sum((a - mean1) ** 2 for a in vec1))
    den2 = math.sqrt(sum((b - mean2) ** 2 for b in vec2))
    denom = den1 * den2
    if denom == 0:
        return None
    return max(-1.0, min(1.0, num / denom))


def _compute_replay_fidelity(event: Dict[str, Any], replay_eval: bool) -> Optional[float]:
    if not replay_eval:
        return None
    context = event.get("context") or {}
    replay_vec = context.get("replay_vector")
    emotion_vec = context.get("emotion_vector")
    if not isinstance(replay_vec, Sequence) or not isinstance(emotion_vec, Sequence):
        return None
    try:
        replay = [float(x) for x in replay_vec]
        emotions = [float(x) for x in emotion_vec]
    except (TypeError, ValueError):
        return None
    return _pearson_corr(replay, emotions)


def _recent_severities(window: int) -> Sequence[float]:
    records = tail_jsonl(str(FORGIVE_LOG), window)
    result: List[float] = []
    for rec in records:
        sev = rec.get("severity")
        if isinstance(sev, (int, float)) and math.isfinite(sev):
            result.append(float(sev))
    return result


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[idx])


def adaptive_forgive_threshold(
    severities: Sequence[float],
    base: float,
    *,
    min_threshold: float,
    max_threshold: float,
    quantile: float,
    min_samples: int,
) -> float:
    if len(severities) < max(1, min_samples):
        return base
    sorted_values = sorted(severities)
    candidate = _quantile(sorted_values, quantile)
    return max(min_threshold, min(max_threshold, candidate))


def _smooth(prev: float, target: float, alpha: float) -> float:
    return prev + alpha * (target - prev)


def _apply_budgets(delta_p: float, delta_e: float, l1_budget: float, l_inf: float) -> Tuple[float, float]:
    delta_p = max(-l_inf, min(l_inf, delta_p))
    delta_e = max(-l_inf, min(l_inf, delta_e))
    norm = abs(delta_p) + abs(delta_e)
    if norm > l1_budget > 0:
        scale = l1_budget / norm
        delta_p *= scale
        delta_e *= scale
    return delta_p, delta_e


def _latest_policy_record() -> Optional[Dict[str, Any]]:
    records = tail_jsonl(str(POLICY_LOG), 1)
    return records[-1] if records else None


def _find_stats(nightly_id: str) -> Optional[Dict[str, Any]]:
    records = tail_jsonl(str(STATS_LOG), 200)
    for rec in reversed(records):
        if rec.get("nightly_id") == nightly_id:
            return rec
    return None


def log_pain_event(
    kind: str,
    delta_aff: float,
    reasons: List[str],
    hp: Dict[str, float],
    context: Dict[str, Any],
    labels: Optional[List[str]] = None,
    *,
    rotate_daily: bool = False,
    replay_eval: bool = False,
) -> Dict[str, Any]:
    """Record a pain event (Δaff < 0) into the jsonl log."""
    event = {
        "ts_ms": int(time.time() * 1000),
        "kind": kind,
        "delta_aff": float(delta_aff),
        "reasons": list(reasons or []),
        "hp": {
            "emotional": float(hp.get("emotional", 0.0)),
            "metabolic": float(hp.get("metabolic", 0.0)),
            "ethical": float(hp.get("ethical", 0.0)),
        },
        "context": dict(context or {}),
        "labels": list(labels or []),
    }
    ok, reason = validate_pain_record(event)
    if not ok:
        raise ValueError(f"invalid pain event: {reason}")
    raw = json.dumps(
        {
            "kind": kind,
            "delta_aff": float(delta_aff),
            "reasons": list(reasons or []),
            "hp": {
                "emotional": event["hp"]["emotional"],
                "metabolic": event["hp"]["metabolic"],
                "ethical": event["hp"]["ethical"],
            },
            "context": event["context"],
        },
        sort_keys=True,
    )
    event["idempotency_key"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    if _record_seen(event["idempotency_key"]):
        return event
    fidelity = _compute_replay_fidelity(event, replay_eval)
    if fidelity is not None:
        event["replay_fidelity"] = fidelity
    append_jsonl(_resolve_log_path(PAIN_LOG, rotate_daily=rotate_daily), event)
    return event


def _iter_pain_events(rotate_daily: bool = False) -> Iterable[Dict[str, Any]]:
    paths: List[str] = []
    if rotate_daily:
        stem = PAIN_LOG.stem
        for candidate in sorted(PAIN_DIR.glob(f"{stem}-*.jsonl")):
            paths.append(str(candidate))
    paths.append(str(PAIN_LOG))
    events: List[Dict[str, Any]] = []
    for path in paths:
        events.extend(list(read_jsonl(path)))
    return events


def _replay_like_sim(event: Dict[str, Any]) -> Dict[str, Any]:
    """Proxy for Replay: produce a quick severity score."""
    hp = event.get("hp", {})
    weight = hp.get("emotional", 0.0) * 0.5 + hp.get("metabolic", 0.0) * 0.3 + hp.get("ethical", 0.0) * 0.7
    severity = min(1.0, max(0.0, -float(event.get("delta_aff", 0.0)))) * (0.25 + weight)
    return {"severity": severity}


def _value_committee(event: Dict[str, Any], sim: Dict[str, Any]) -> Dict[str, Any]:
    reasons = event.get("reasons", [])
    harmed: List[str] = []
    if "rudeness" in reasons or event.get("kind") == "value_conflict":
        harmed.append("dignity")
    if "overload" in reasons or event.get("kind") == "energy_depletion":
        harmed.append("care_of_self")
    if "isolation" in reasons or event.get("kind") == "isolation":
        harmed.append("belonging")
    return {"harmed_values": list(dict.fromkeys(harmed)), "severity": sim["severity"]}


def _reason_tag(reasons: Sequence[str], harmed_values: Sequence[str]) -> str:
    for candidate in reasons:
        if candidate:
            return str(candidate)
    for harmed in harmed_values:
        if harmed:
            return str(harmed)
    return "unspecified"


def _next_action(can_forgive: bool, harmed_values: Sequence[str]) -> str:
    if not harmed_values:
        return "listen_first_2s" if can_forgive else "stand_guard"
    primary = harmed_values[0]
    if primary == "belonging":
        return "stay_near_and_listen" if can_forgive else "call_trusted_friend"
    if primary == "care_of_self":
        return "offer_rest" if can_forgive else "enforce_rest"
    if primary == "dignity":
        return "acknowledge_feelings" if can_forgive else "set_boundary"
    return "listen_first_2s" if can_forgive else "stand_guard"


def evaluate_and_forgive(
    nightly_id: str,
    base_threshold: float = 0.35,
    *,
    adaptive: bool = True,
    min_threshold: float = 0.2,
    max_threshold: float = 0.6,
    quantile: float = 0.35,
    min_samples: int = 50,
    severity_window: int = 200,
    rotate_daily: bool = False,
    max_events: Optional[int] = None,
    care_targets: Optional[Sequence[str]] = None,
    comfort_gain_base: float = 0.15,
    protection_bias: float = 0.3,
    growth_reward: float = 0.2,
    patience_budget: float = 0.5,
    replay_eval: bool = False,
) -> Tuple[int, int, Dict[str, Any]]:
    total = 0
    forgiven = 0
    events = _iter_pain_events(rotate_daily=rotate_daily)
    if max_events is not None:
        events = events[-max_events:]
    threshold = base_threshold
    if adaptive:
        recent = _recent_severities(severity_window)
        threshold = adaptive_forgive_threshold(
            recent,
            base_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            quantile=quantile,
            min_samples=min_samples,
        )
    by_kind: Dict[str, Dict[str, int]] = {}
    by_target: Dict[str, Dict[str, int]] = {}
    care_breakdown: Dict[str, Dict[str, float]] = {}
    fidelity_sum = 0.0
    fidelity_count = 0
    events_detail: List[Dict[str, Any]] = []
    care_targets_set = {t for t in (care_targets or []) if isinstance(t, str) and t}
    care_stats: Dict[str, Any] = {
        "targets": {},
        "comfort_gain_applied": 0.0,
        "protection_signal": 0.0,
        "growth_signal": 0.0,
        "patience_used": 0.0,
        "patience_budget": max(0.0, patience_budget),
        "interventions": 0,
        "watch_only": 0,
    } if care_targets_set else {}
    for event in events:
        fidelity = _compute_replay_fidelity(event, replay_eval)
        if fidelity is not None:
            fidelity_sum += fidelity
            fidelity_count += 1
        sim = _replay_like_sim(event)
        committee = _value_committee(event, sim)
        severity = committee["severity"]
        can_forgive = bool(severity <= threshold)
        target = event.get("context", {}).get("target", "unknown")
        is_care_target = bool(care_targets_set and target in care_targets_set)
        record = {
            "ts_ms": int(time.time() * 1000),
            "nightly_id": nightly_id,
            "pain_ts_ms": event.get("ts_ms"),
            "forgiven": can_forgive,
            "severity": severity,
            "harmed_values": committee["harmed_values"],
            "notes": "auto" if can_forgive else "needs_attention",
            "replay_fidelity": fidelity,
            "target": target,
            "care_target": is_care_target,
        }
        reasons = event.get("reasons") or []
        harmed = committee["harmed_values"] or []
        reason_tag = _reason_tag(reasons, harmed)
        next_action = _next_action(can_forgive, harmed)
        if can_forgive:
            record["narrative"] = (
                f"I felt {event.get('kind')} due to {', '.join(reasons) or 'an event'}, "
                f"but realized {', '.join(harmed) or 'my core values'} mattered more, so I chose to forgive."
            )
        else:
            record["narrative"] = (
                f"I felt {event.get('kind')} due to {', '.join(reasons) or 'an event'}, "
                f"yet {', '.join(harmed) or 'these values'} still ache, so forgiveness must wait."
            )
        record["reason_tag"] = reason_tag
        record["next_action"] = next_action
        append_jsonl(str(FORGIVE_LOG), record)
        total += 1
        if can_forgive:
            forgiven += 1
        kind = event.get("kind", "unknown")
        bucket = by_kind.setdefault(kind, {"total": 0, "forgiven": 0})
        bucket["total"] += 1
        bucket["forgiven"] += int(can_forgive)
        target_bucket = by_target.setdefault(target, {"total": 0, "forgiven": 0})
        target_bucket["total"] += 1
        target_bucket["forgiven"] += int(can_forgive)
        if is_care_target and care_stats:
            care_bucket = care_breakdown.setdefault(
                target,
                {"total": 0.0, "forgiven": 0.0, "severity_sum": 0.0},
            )
            care_bucket["total"] += 1.0
            care_bucket["forgiven"] += float(can_forgive)
            care_bucket["severity_sum"] += float(severity)
            care_stats["patience_used"] += float(severity)
            if can_forgive:
                care_stats["watch_only"] += 1
            else:
                care_stats["interventions"] += 1
            if can_forgive:
                care_stats["comfort_gain_applied"] += float(comfort_gain_base) * max(0.0, 1.0 - float(severity))
                care_stats["growth_signal"] += float(growth_reward) * max(0.0, threshold - float(severity))
            else:
                care_stats["protection_signal"] += float(protection_bias) * float(severity)
        events_detail.append(
            {
                "kind": kind,
                "harmed_values": committee["harmed_values"],
                "severity": severity,
                "replay_fidelity": fidelity,
                "target": target,
                "care_target": is_care_target,
                "reason_tag": reason_tag,
                "next_action": next_action,
                "intervention": bool(is_care_target and not can_forgive),
            }
        )

    def _finalise_breakdown(data: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, counts in data.items():
            denom = counts["total"] or 1
            result[key] = {
                "total": counts["total"],
                "forgiven": counts["forgiven"],
                "forgive_rate": counts["forgiven"] / denom,
            }
        return result

    if care_stats:
        targets_summary: Dict[str, Any] = {}
        for target, info in care_breakdown.items():
            total_care = info.get("total", 0.0)
            forgiven_care = info.get("forgiven", 0.0)
            severity_sum = info.get("severity_sum", 0.0)
            denom = total_care or 1.0
            targets_summary[target] = {
                "total": int(total_care),
                "forgiven": int(forgiven_care),
                "forgive_rate": float(forgiven_care / denom),
                "avg_severity": float(severity_sum / denom),
            }
        care_stats["targets"] = targets_summary
        care_stats["comfort_gain_applied"] = float(care_stats["comfort_gain_applied"])
        care_stats["protection_signal"] = float(care_stats["protection_signal"])
        care_stats["growth_signal"] = float(care_stats["growth_signal"])
        care_stats["patience_used"] = float(care_stats["patience_used"])
        care_stats["interventions"] = int(care_stats["interventions"])
        care_stats["watch_only"] = int(care_stats["watch_only"])
        care_stats["comfort_score"] = care_stats["comfort_gain_applied"]

    avg_fidelity = fidelity_sum / fidelity_count if fidelity_count else None
    stats = {
        "ts_ms": int(time.time() * 1000),
        "nightly_id": nightly_id,
        "total": total,
        "forgiven": forgiven,
        "forgive_rate": float(forgiven / total) if total else 0.0,
        "forgive_threshold": threshold,
        "breakdown": {
            "by_kind": _finalise_breakdown(by_kind),
            "by_target": _finalise_breakdown(by_target),
        },
        "avg_replay_fidelity": avg_fidelity,
        "events_detail": events_detail,
    }
    if care_stats:
        stats["care_stats"] = care_stats
    append_jsonl(str(STATS_LOG), stats)
    return total, forgiven, stats


def policy_update_from_forgiveness(
    nightly_id: str,
    base_threshold: float = 0.5,
    empathy_gain_base: float = 0.1,
    *,
    ema_alpha: float = 0.4,
    l1_budget: float = 0.8,
    l_inf_budget: float = 0.3,
    min_threshold: float = 0.2,
    max_threshold: float = 0.6,
    max_empathy: float = 0.5,
    events_detail: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    stats = _find_stats(nightly_id)
    forgive_rate = stats.get("forgive_rate", 0.0) if stats else 0.0
    prev = _latest_policy_record()
    prev_threshold = float(prev.get("policy_feedback_threshold", base_threshold)) if prev else base_threshold
    prev_empathy = float(prev.get("a2a_empathy_gain", empathy_gain_base)) if prev else empathy_gain_base
    target_threshold = max(min_threshold, min(max_threshold, base_threshold - 0.2 * forgive_rate))
    target_empathy = min(max_empathy, empathy_gain_base + 0.3 * forgive_rate)
    smoothed_threshold = _smooth(prev_threshold, target_threshold, ema_alpha)
    smoothed_empathy = _smooth(prev_empathy, target_empathy, ema_alpha)
    delta_threshold, delta_empathy = _apply_budgets(
        smoothed_threshold - prev_threshold,
        smoothed_empathy - prev_empathy,
        l1_budget,
        l_inf_budget,
    )
    new_threshold = max(min_threshold, min(max_threshold, prev_threshold + delta_threshold))
    new_empathy_gain = max(0.0, min(max_empathy, prev_empathy + delta_empathy))
    value_shift = {
        "delta_policy_threshold": new_threshold - prev_threshold,
        "delta_empathy_gain": new_empathy_gain - prev_empathy,
    }
    if events_detail:
        for detail in events_detail:
            record = {
                "ts_ms": int(time.time() * 1000),
                "nightly_id": nightly_id,
                "harmed_values": detail.get("harmed_values") or [],
                "severity": detail.get("severity"),
                "replay_fidelity": detail.get("replay_fidelity"),
                "target": detail.get("target"),
                "care_target": bool(detail.get("care_target")),
                "value_shift": value_shift,
            }
            append_jsonl(str(VALUE_INFLUENCE_LOG), record)
    payload = {
        "ts_ms": int(time.time() * 1000),
        "nightly_id": nightly_id,
        "policy_feedback_threshold": new_threshold,
        "a2a_empathy_gain": new_empathy_gain,
        "forgive_rate": forgive_rate,
        "ema_alpha": ema_alpha,
        "l1_budget": l1_budget,
        "l_inf_budget": l_inf_budget,
        "used_threshold_target": target_threshold,
        "used_empathy_target": target_empathy,
        "value_shift": value_shift,
    }
    append_jsonl(str(POLICY_LOG), payload)
    return payload


__all__ = [
    "PainEvent",
    "log_pain_event",
    "evaluate_and_forgive",
    "policy_update_from_forgiveness",
    "adaptive_forgive_threshold",
    "PAIN_LOG",
    "FORGIVE_LOG",
    "STATS_LOG",
    "POLICY_LOG",
    "VALUE_INFLUENCE_LOG",
]
