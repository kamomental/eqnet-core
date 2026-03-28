# -*- coding: utf-8 -*-
"""Nightly consolidation orchestration for ATRi."""

from __future__ import annotations

import argparse
import glob
import datetime as dt
import json
import time
import math
import os
import hashlib
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import yaml

try:
    from . import terrain_compact
except Exception:  # pragma: no cover - optional dependency
    terrain_compact = None

try:  # pragma: no cover - matplotlib may be optional in some deployments
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # fallback to headless/no-op
    matplotlib = None
    plt = None

from emot_terrain_lab.memory.go_sc import extract_features, weighted_score
from emot_terrain_lab.memory.inner_os_working_memory_bridge import (
    derive_working_memory_replay_bias,
)
from emot_terrain_lab.utils.fastpath_config import (
    FASTPATH_OVERRIDE_PATH,
    fail_safe_settings,
    load_fastpath_defaults,
    load_fastpath_overrides,
)
from telemetry import plot_ignition as ignition_plots
from telemetry import plot_memory_graph
from telemetry import plot_resonance
from telemetry import plot_culture_resonance
from telemetry import plot_culture_trend
from ops import resonance_metrics
from telemetry import plot_affective_map
from calendar import monthrange

MEMORY_REF_LOG_PATH = Path("logs/memory_ref.jsonl")
INNER_OS_SLEEP_SCHEMA = "inner_os_sleep_consolidation_snapshot/v1"
INNER_OS_WORKING_MEMORY_SCHEMA = "inner_os_working_memory_snapshot/v1"

from emot_terrain_lab.ops.care_canary import select_canary_ids
from emot_terrain_lab.ops.monthly_highlights import generate_value_influence_highlights
from emot_terrain_lab.ops.pain_loop import VALUE_INFLUENCE_LOG, evaluate_and_forgive, policy_update_from_forgiveness
from eqnet.logs.moment_log import iter_moment_entries_for_day
from eqnet.logs.moment_log import iter_moment_entries
from eqnet.memory.store import MemoryStore
from inner_os.group_thread_registry import (
    build_group_thread_key,
    summarize_group_thread_registry_snapshot,
)
from inner_os.discussion_thread_registry import (
    summarize_discussion_thread_registry_snapshot,
    update_discussion_thread_registry_snapshot,
)
from inner_os.memory_core import MemoryCore
from inner_os.daily_carry_summary import DailyCarrySummaryBuilder
from inner_os.identity_arc import IdentityArcSummaryBuilder
from inner_os.identity_memory import IdentityArcRegistry
from inner_os.group_relation_arc import GroupRelationArcSummaryBuilder
from inner_os.relation_arc import RelationArcSummaryBuilder
from inner_os.relation_memory import RelationArcRegistry

def _dump_events_if_requested(events):
    out_path = os.getenv("EQNET_DUMP_REPLAY_EVENTS_JSONL", "").strip()
    if not out_path:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _softmax_probs(weights: List[float]) -> List[float]:
    if not weights:
        return []
    max_w = max(weights)
    exps = [math.exp(w - max_w) for w in weights]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [val / total for val in exps]


def _working_memory_event_text(event: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key in ("text", "dialogue", "summary", "label", "tag"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    meta = event.get("meta")
    if isinstance(meta, Mapping):
        for key in ("text", "dialogue", "summary", "label", "tag", "anchor", "memory_anchor"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return " ".join(parts)


def _working_memory_alignment_score(
    event: Mapping[str, Any],
    bias: Optional[Mapping[str, Any]],
) -> float:
    if not isinstance(bias, Mapping):
        return 0.0
    terms = bias.get("terms")
    if not isinstance(terms, list) or not terms:
        return 0.0
    text = _working_memory_event_text(event).lower()
    if not text:
        return 0.0
    hits = sum(1 for term in terms if isinstance(term, str) and term and term in text)
    if hits <= 0:
        return 0.0
    return float(hits / max(1, len(terms)))


def _working_memory_event_identifier(event: Mapping[str, Any]) -> str:
    for key in ("id", "trace_id", "episode_id"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = event.get("meta")
    if isinstance(meta, Mapping):
        for key in ("trace_id", "episode_id", "memory_anchor", "anchor", "id"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""

def _evaluate_monument_floor_test(
    events: List[Dict[str, Any]],
    cfg: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    floor = _safe_float(cfg.get("monument_connection_floor"), None)
    if floor is None or floor <= 0:
        return {"status": "skipped", "reason": "floor_disabled"}
    memory_dir = cfg.get("memory_dir") if isinstance(cfg, dict) else None
    monument_episode_ids, monument_err = _load_monument_episode_ids(memory_dir)
    if monument_err:
        return {"status": "skipped", "reason": "monument_load_error", "error": monument_err}
    if not monument_episode_ids:
        return {"status": "skipped", "reason": "no_monuments"}
    weights: List[float] = []
    is_monument: List[bool] = []
    for event in events:
        episode_id = str(event.get("episode_id", ""))
        is_monument.append(bool(episode_id and episode_id in monument_episode_ids))
        w = _safe_float(event.get("weight"), 1.0)
        if w is None:
            w = 1.0
        weights.append(float(w))
    if not weights:
        return {"status": "skipped", "reason": "no_events"}
    base_probs = _softmax_probs(weights)
    n_m = sum(1 for flag in is_monument if flag)
    if n_m == 0:
        return {"status": "skipped", "reason": "no_monument_events"}
    floor_total = float(floor) * n_m
    if floor_total >= 1.0:
        return {"status": "skipped", "reason": "floor_too_high", "floor_total": floor_total}
    base_total = sum(base_probs)
    if base_total <= 0:
        return {"status": "skipped", "reason": "invalid_base_probs"}
    remaining = 1.0 - floor_total
    after_probs: List[float] = []
    for prob, flag in zip(base_probs, is_monument):
        if flag:
            after_probs.append(float(floor) + remaining * (prob / base_total))
        else:
            after_probs.append(remaining * (prob / base_total))
    base_mon = [p for p, flag in zip(base_probs, is_monument) if flag]
    base_non = [p for p, flag in zip(base_probs, is_monument) if not flag]
    after_mon = [p for p, flag in zip(after_probs, is_monument) if flag]
    after_non = [p for p, flag in zip(after_probs, is_monument) if not flag]
    diff_mon: List[float] = []
    top_candidates: List[Tuple[float, str]] = []
    for event, base_p, after_p, flag in zip(events, base_probs, after_probs, is_monument):
        if not flag:
            continue
        diff = float(after_p - base_p)
        diff_mon.append(diff)
        identifier = str(event.get("episode_id") or event.get("trace_id") or "")
        top_candidates.append((diff, identifier))
    top_k = int(cfg.get("monument_floor_top_k", 5)) if isinstance(cfg, Mapping) else 5
    top_candidates.sort(key=lambda item: item[0], reverse=True)
    top_diff = [
        {"id": item[1], "delta_p": float(item[0])}
        for item in top_candidates[: max(0, top_k)]
        if item[1]
    ]
    diff_stats = _basic_stats(diff_mon)
    diff_min = min(diff_mon) if diff_mon else 0.0
    diff_max = max(diff_mon) if diff_mon else 0.0
    return {
        "status": "applied",
        "floor": float(floor),
        "monument_events": float(n_m),
        "base_monument_mean": _basic_stats(base_mon).get("mean", 0.0),
        "base_non_monument_mean": _basic_stats(base_non).get("mean", 0.0),
        "after_monument_mean": _basic_stats(after_mon).get("mean", 0.0),
        "after_non_monument_mean": _basic_stats(after_non).get("mean", 0.0),
        "delta_p_min": float(diff_min),
        "delta_p_mean": float(diff_stats.get("mean", 0.0)),
        "delta_p_max": float(diff_max),
        "delta_p_top": top_diff,
    }

def _basic_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * 0.95)
    return {
        "count": float(len(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "p95": float(ordered[idx]),
        "max": float(ordered[-1]),
    }

def _extract_seed_ids(event: Mapping[str, Any]) -> List[str]:
    meta = event.get("meta") or {}
    seeds: List[str] = []
    for container in (meta, meta.get("receipt") or {}):
        replay_unified = container.get("replay_unified") or {}
        for seed in replay_unified.get("seeds") or []:
            if isinstance(seed, dict):
                trace_id = seed.get("trace_id")
                if trace_id:
                    seeds.append(str(trace_id))
    return seeds

def _collect_recall_counts(events: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        for trace_id in _extract_seed_ids(event):
            counts[trace_id] = counts.get(trace_id, 0) + 1
    return counts

def _load_monument_episode_ids(memory_dir: Optional[str]) -> Tuple[set[str], Optional[str]]:
    if not memory_dir:
        return set(), None
    try:
        store = MemoryStore(Path(memory_dir))
        episodes, monuments = store.load_all()
        episode_ids = {ep_id for mon in monuments for ep_id in mon.episodes}
        return episode_ids, None
    except Exception as exc:
        return set(), str(exc)

def _extract_affect_value(entry: object, field_specs: Sequence[str], agg_mode: str) -> Optional[float]:
    values: List[float] = []
    for spec in field_specs:
        if not spec:
            continue
        if spec.startswith("qualia_vec:"):
            try:
                idx = int(spec.split(":", 1)[1])
            except (TypeError, ValueError):
                continue
            qualia = getattr(entry, "qualia_vec", None)
            if not qualia or idx < 0 or idx >= len(qualia):
                continue
            val = _safe_float(qualia[idx])
            if val is not None:
                values.append(float(val))
            continue
        if "." in spec:
            head, tail = spec.split(".", 1)
            container = getattr(entry, head, None)
            if isinstance(container, dict):
                val = _safe_float(container.get(tail))
                if val is not None:
                    values.append(float(val))
            continue
        val = _safe_float(getattr(entry, spec, None))
        if val is not None:
            values.append(float(val))
    if not values:
        return None
    if agg_mode == "first_nonzero":
        for val in values:
            if abs(val) > 0.0:
                return val
        return values[0]
    if agg_mode == "mean_abs":
        return sum(abs(v) for v in values) / len(values)
    return max(abs(v) for v in values)

def _compute_affect_load(moment_log_path: Optional[str], day: dt.date, affect_fields: Sequence[str], agg_mode: str) -> Tuple[float, int]:
    if not moment_log_path:
        return 0.0, 0
    values: List[float] = []
    for entry in iter_moment_entries_for_day(moment_log_path, day):
        stress = _extract_affect_value(entry, affect_fields, agg_mode)
        if stress is None:
            continue
        values.append(abs(float(stress)))
    if not values:
        return 0.0, 0
    return float(sum(values) / len(values)), len(values)


def _find_latest_moment_day(moment_log_path: Optional[str]) -> Optional[dt.date]:
    if not moment_log_path:
        return None
    latest: Optional[dt.date] = None
    for entry in iter_moment_entries(moment_log_path):
        try:
            stamp = dt.datetime.fromtimestamp(entry.ts, tz=dt.timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            continue
        if latest is None or stamp > latest:
            latest = stamp
    return latest

def _resolve_affect_day(
    *,
    mode: str,
    moment_log_path: Optional[str],
    now_ts: float,
) -> dt.date:
    if mode == "latest_available":
        latest = _find_latest_moment_day(moment_log_path)
        if latest is not None:
            return latest
    return dt.datetime.utcfromtimestamp(now_ts).date()
def _apply_nightly_forgetting(
    events: List[Dict[str, Any]],
    cfg: Mapping[str, Any],
    *,
    moment_log_path: Optional[str],
    memory_dir: Optional[str],
    now_ts: float,
    forgetfulness_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    enabled = bool(cfg.get("enable", False))
    if not enabled:
        return None, {"status": "disabled"}

    recall_window_days = _safe_float(cfg.get("recall_window_days"), None)
    recall_k = _safe_float(cfg.get("recall_k"), 0.0) or 0.0
    recall_weight = _safe_float(cfg.get("recall_weight"), 0.0) or 0.0
    affect_weight = _safe_float(cfg.get("affect_weight"), 0.0) or 0.0
    interference_weight = _safe_float(cfg.get("interference_weight"), 0.0) or 0.0
    interference_k = _safe_float(cfg.get("interference_k"), 0.0) or 0.0
    reconsolidation_rate = _safe_float(cfg.get("reconsolidation_rate"), 0.0) or 0.0
    base_delta = _safe_float(cfg.get("base_delta"), 0.0) or 0.0
    max_delta_w = _safe_float(cfg.get("max_delta_w"), 0.0) or 0.0
    min_w = _safe_float(cfg.get("min_w"), 0.0) or 0.0
    max_w = _safe_float(cfg.get("max_w"), 1.0) or 1.0
    monument_w_lock = bool(cfg.get("monument_w_lock", True))
    monument_floor = _safe_float(cfg.get("monument_connection_floor"), None)
    consent_floor = _safe_float(cfg.get("consent_floor"), min_w)
    consent_tags = [str(tag) for tag in (cfg.get("consent_override_tags") or [])]

    input_sources = [str(src) for src in (cfg.get("input_sources") or [])]
    source_weights = cfg.get("source_weights") or {}
    source_paths = cfg.get("source_paths") or {}
    sources_used: List[str] = []
    if "hub" in input_sources:
        sources_used.append("hub")

    recall_counts = _collect_recall_counts(events)
    day_mode = str(cfg.get("affect_day_mode", "today"))
    day = _resolve_affect_day(mode=day_mode, moment_log_path=moment_log_path, now_ts=now_ts)
    affect_load = 0.0
    affect_samples = 0
    affect_day_used: Optional[str] = None
    affect_fields = cfg.get("affect_fields") or []
    affect_agg = str(cfg.get("affect_agg", "max_abs"))
    if "heart_os_session" in input_sources:
        path = source_paths.get("heart_os_session") or moment_log_path
        affect_load, affect_samples = _compute_affect_load(path, day, affect_fields, affect_agg)
        if affect_samples == 0 and day_mode == "latest_available":
            day = _resolve_affect_day(mode="latest_available", moment_log_path=path, now_ts=now_ts)
            affect_load, affect_samples = _compute_affect_load(path, day, affect_fields, affect_agg)
        if affect_samples:
            sources_used.append("heart_os_session")
            affect_day_used = day.isoformat()

    monument_episode_ids, monument_err = _load_monument_episode_ids(memory_dir)

    item_log_cfg = cfg.get("audit", {}) or {}
    item_log_enabled = bool(item_log_cfg.get("enable_item_log", False))
    item_log_path = item_log_cfg.get("item_log_path")
    item_handle = None
    if item_log_enabled and item_log_path:
        item_path = Path(str(item_log_path))
        item_path.parent.mkdir(parents=True, exist_ok=True)
        item_handle = item_path.open("a", encoding="utf-8")

    deltas: List[float] = []
    abs_deltas: List[float] = []
    delta_pos = 0
    delta_neg = 0
    delta_nonzero_count = 0
    evidence_nonzero_counts = {"recall": 0, "affect": 0, "interference": 0}
    floors_applied = 0
    consent_applied = 0
    locked_applied = 0

    metrics_enabled = False
    metrics_mode = "observe"
    metrics_epsilon = 0.0
    sign_balance_mode = "ratio"
    thresholds = {}
    if isinstance(forgetfulness_cfg, dict):
        metrics_enabled = bool(forgetfulness_cfg.get("enable", False))
        metrics_mode = str(forgetfulness_cfg.get("mode", "observe"))
        metrics_epsilon = _safe_float(forgetfulness_cfg.get("epsilon"), 0.0) or 0.0
        sign_balance_mode = str(forgetfulness_cfg.get("sign_balance_mode", "ratio"))
        threshold_candidate = forgetfulness_cfg.get("thresholds")
        if isinstance(threshold_candidate, dict):
            thresholds = threshold_candidate

    updated: List[Dict[str, Any]] = []
    for event in events:
        trace_id = str(event.get("trace_id", ""))
        episode_id = str(event.get("episode_id", ""))
        memory_kind = str(event.get("memory_kind", ""))
        w_before = _safe_float(event.get("weight"), 1.0) or 1.0

        tags = event.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        consent_override = any(tag in consent_tags for tag in tags)
        is_monument = memory_kind.lower() == "monument"
        lock_weight = monument_w_lock and is_monument

        factors = {"recall": 0.0, "affect": 0.0, "interference": 0.0}
        guard = {"consent_override": consent_override, "monument_lock": lock_weight}
        w_after = w_before
        delta = 0.0

        if consent_override:
            w_after = min(w_before, float(consent_floor))
            consent_applied += 1
        elif lock_weight:
            locked_applied += 1
        else:
            recall_count = recall_counts.get(trace_id, 0)
            recall_score = 0.0
            if recall_k > 0:
                recall_score = min(1.0, recall_k * recall_count)
            if recall_window_days is not None:
                age_days = (now_ts - float(event.get("timestamp", now_ts))) / 86400.0
                if age_days > float(recall_window_days):
                    recall_score = 0.0
            factors["recall"] = recall_score

            emotion_mod = _safe_float(event.get("emotion_modulation"), 0.0) or 0.0
            affect_score = abs(emotion_mod)
            if affect_load and "heart_os_session" in sources_used:
                affect_score += float(affect_load)
            factors["affect"] = min(1.0, affect_score)

            interference_score = 0.0
            meta = event.get("meta") or {}
            interference = meta.get("interference") if isinstance(meta, dict) else None
            if isinstance(interference, dict):
                interference_score = _safe_float(interference.get("similarity"), 0.0) or 0.0
            factors["interference"] = min(1.0, interference_k * interference_score)

            delta = base_delta
            delta += recall_weight * factors["recall"]
            delta -= affect_weight * factors["affect"] * reconsolidation_rate
            delta -= interference_weight * factors["interference"]
            if max_delta_w > 0:
                delta = _clamp(delta, -max_delta_w, max_delta_w)
            w_after = _clamp(w_before + delta, min_w, max_w)

        if monument_floor is not None and episode_id and episode_id in monument_episode_ids:
            if w_after < float(monument_floor):
                w_after = float(monument_floor)
                floors_applied += 1
                guard["monument_floor"] = True

        delta = w_after - w_before
        deltas.append(float(delta))
        abs_delta = abs(delta)
        abs_deltas.append(float(abs_delta))
        if abs_delta > metrics_epsilon:
            delta_nonzero_count += 1
        if delta > metrics_epsilon:
            delta_pos += 1
        elif delta < -metrics_epsilon:
            delta_neg += 1

        if factors["recall"] > metrics_epsilon:
            evidence_nonzero_counts["recall"] += 1
        if factors["affect"] > metrics_epsilon:
            evidence_nonzero_counts["affect"] += 1
        if factors["interference"] > metrics_epsilon:
            evidence_nonzero_counts["interference"] += 1
        event["weight"] = float(w_after)

        if item_handle is not None:
            item = {
                "trace_id": trace_id,
                "episode_id": episode_id,
                "memory_kind": memory_kind,
                "w_before": float(w_before),
                "w_after": float(w_after),
                "delta": float(delta),
                "factors": factors,
                "guard": guard,
                "sources": sources_used or ["hub"],
            }
            item_handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        updated.append(event)

    if item_handle is not None:
        item_handle.close()

    source_counts: Dict[str, int] = {}
    if "hub" in sources_used:
        source_counts["hub"] = int(len(events))
    if "heart_os_session" in sources_used:
        source_counts["heart_os_session"] = int(affect_samples)

    summary = {
        "status": "applied",
        "sources_used": sources_used or ["hub"],
        "source_weights": dict(source_weights),
        "source_counts": source_counts,
        "affect_load": float(affect_load),
        "affect_samples": int(affect_samples),
        "affect_day_mode": day_mode,
        "affect_day_used": affect_day_used,
        "monument_episode_count": float(len(monument_episode_ids)),
        "monument_load_error": monument_err,
        "delta_stats": _basic_stats(deltas),
        "floors_applied": float(floors_applied),
        "consent_overrides": float(consent_applied),
        "monument_locks": float(locked_applied),
        "item_log_path": str(item_log_path) if item_log_enabled and item_log_path else None,
    }
    if metrics_enabled:
        abs_sum = float(sum(abs_deltas))
        abs_count = len(abs_deltas)
        abs_max = float(max(abs_deltas)) if abs_deltas else 0.0
        sign_total = delta_pos + delta_neg
        if sign_total > 0:
            if sign_balance_mode == "signed":
                sign_balance = (delta_pos - delta_neg) / float(sign_total)
            else:
                sign_balance = delta_pos / float(sign_total)
        else:
            sign_balance = 0.0

        reasons: List[str] = []
        if metrics_mode == "alert":
            max_threshold = _safe_float(thresholds.get("delta_abs_max"), None)
            nonzero_min = _safe_float(thresholds.get("delta_nonzero_count_min"), None)
            sign_abs = _safe_float(thresholds.get("delta_sign_balance_abs"), None)
            floors_max = _safe_float(thresholds.get("floors_applied_max"), None)
            consent_max = _safe_float(thresholds.get("consent_overrides_max"), None)
            locks_max = _safe_float(thresholds.get("monument_locks_max"), None)

            if max_threshold is not None and abs_max > max_threshold:
                reasons.append("drift_spike")
            if nonzero_min is not None and delta_nonzero_count < nonzero_min:
                reasons.append("input_drop")
            if sign_abs is not None and abs(sign_balance) > sign_abs:
                reasons.append("suppression_bias")
            if floors_max is not None and floors_applied > floors_max:
                reasons.append("guard_overactive")
            if consent_max is not None and consent_applied > consent_max:
                reasons.append("guard_overactive")
            if locks_max is not None and locked_applied > locks_max:
                reasons.append("guard_overactive")

        summary["forgetfulness_health"] = {
            "mode": metrics_mode,
            "delta_abs_max": abs_max,
            "delta_abs_mean": _safe_mean(abs_sum, abs_count),
            "delta_nonzero_count": int(delta_nonzero_count),
            "delta_sign_balance": float(sign_balance),
            "delta_sign_balance_mode": sign_balance_mode,
            "evidence_nonzero_counts": evidence_nonzero_counts,
            "guard_counts": {
                "floors_applied": float(floors_applied),
                "consent_overrides": float(consent_applied),
                "monument_locks": float(locked_applied),
            },
            "reasons": reasons,
        }
    return updated, summary


def _defrag_private_id(value: str, mode: str) -> str:
    if mode == "raw":
        return value
    if mode == "hash":
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return "***"


def _defrag_observe(events: List[Dict[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    observe_cfg = cfg.get("observe") if isinstance(cfg, dict) else None
    if not isinstance(observe_cfg, dict):
        observe_cfg = {}

    duplicate_threshold = observe_cfg.get("duplicate_similarity_threshold")
    if duplicate_threshold is None:
        duplicate_threshold = cfg.get("cluster_similarity_threshold") if isinstance(cfg, dict) else None
    duplicate_threshold = _safe_float(duplicate_threshold, None)

    conflict_threshold = observe_cfg.get("conflict_interference_threshold")
    if conflict_threshold is None:
        conflict_threshold = cfg.get("conflict_isolation_strength") if isinstance(cfg, dict) else None
    conflict_threshold = _safe_float(conflict_threshold, None)

    cooccur_min = observe_cfg.get("cooccur_min")
    if cooccur_min is not None:
        cooccur_min = int(cooccur_min)
    top_k = observe_cfg.get("top_k")
    if top_k is not None:
        top_k = int(top_k)
    max_pairs = observe_cfg.get("max_pairs")
    if max_pairs is not None:
        max_pairs = int(max_pairs)
    rollup_budget = observe_cfg.get("rollup_trigger_budget")
    if rollup_budget is None:
        rollup_budget = cfg.get("rollup_trigger_budget") if isinstance(cfg, dict) else None
    if rollup_budget is not None:
        rollup_budget = int(rollup_budget)

    id_privacy = str(observe_cfg.get("id_privacy", "mask"))
    similarity_bands = observe_cfg.get("similarity_bands")
    if similarity_bands is not None and not isinstance(similarity_bands, list):
        similarity_bands = None

    duplicate_ids: set[str] = set()
    conflict_ids: set[str] = set()
    similarity_values: List[float] = []
    similarity_rows: List[Tuple[str, float]] = []

    tags_index: Dict[str, set[str]] = defaultdict(set)
    l1_l2_count = 0

    for event in events:
        episode_id = str(event.get("episode_id", ""))
        memory_kind = str(event.get("memory_kind", "")).lower()
        if "l1" in memory_kind or "l2" in memory_kind:
            l1_l2_count += 1

        tags = event.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        for tag in tags:
            if tag:
                tags_index[str(tag)].add(episode_id)

        meta = event.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        interference = meta.get("interference")
        if not isinstance(interference, dict):
            continue
        action = str(interference.get("action", "")).lower()
        sim = _safe_float(interference.get("similarity"), None)
        if sim is not None:
            similarity_values.append(float(sim))
            similarity_rows.append((episode_id, float(sim)))
            if duplicate_threshold is not None and sim >= float(duplicate_threshold) and action != "mask":
                duplicate_ids.add(episode_id)
            if conflict_threshold is not None and sim >= float(conflict_threshold):
                conflict_ids.add(episode_id)
        if action == "mask":
            conflict_ids.add(episode_id)

    reference_candidate_count = 0
    if cooccur_min is not None and cooccur_min > 1 and max_pairs is not None and max_pairs > 0:
        pairs: set[Tuple[str, str]] = set()
        for _, ids in tags_index.items():
            if len(ids) < cooccur_min:
                continue
            id_list = list(ids)
            for i in range(len(id_list)):
                for j in range(i + 1, len(id_list)):
                    a = id_list[i]
                    b = id_list[j]
                    if a == b:
                        continue
                    pair = (a, b) if a < b else (b, a)
                    pairs.add(pair)
                    if len(pairs) >= max_pairs:
                        break
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        reference_candidate_count = len(pairs)

    rollup_candidate_count = 0
    if rollup_budget is not None and rollup_budget >= 0:
        rollup_candidate_count = max(0, l1_l2_count - rollup_budget)

    top_pairs: List[Dict[str, Any]] = []
    if top_k is not None and top_k > 0 and similarity_rows:
        for episode_id, sim in sorted(similarity_rows, key=lambda x: x[1], reverse=True)[:top_k]:
            top_pairs.append(
                {"id": _defrag_private_id(episode_id, id_privacy), "similarity": float(sim)}
            )

    clusters_by_band: Dict[str, int] = {}
    if similarity_bands:
        bands = sorted(float(b) for b in similarity_bands)
        if len(bands) >= 2:
            for idx in range(len(bands) - 1):
                clusters_by_band[f"{bands[idx]}-{bands[idx+1]}"] = 0
            clusters_by_band[f">={bands[-1]}"] = 0
            for sim in similarity_values:
                placed = False
                for idx in range(len(bands) - 1):
                    if bands[idx] <= sim < bands[idx + 1]:
                        clusters_by_band[f"{bands[idx]}-{bands[idx+1]}"] += 1
                        placed = True
                        break
                if not placed and sim >= bands[-1]:
                    clusters_by_band[f">={bands[-1]}"] += 1

    observe_report: Dict[str, Any] = {
        "duplicate_cluster_count": len(duplicate_ids),
        "conflict_cluster_count": len(conflict_ids),
        "reference_candidate_count": int(reference_candidate_count),
        "rollup_candidate_count": int(rollup_candidate_count),
        "id_privacy": id_privacy,
        "top_pairs": top_pairs,
    }
    if clusters_by_band:
        observe_report["clusters_by_band"] = clusters_by_band

    return {
        "status": "observed",
        "mode": "observe",
        "observe": observe_report,
    }


def _dump_report(report: Dict[str, Any], *, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"{ts}.json"
    _ensure_dir(path)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run(hub, cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Perform nightly housekeeping using a live hub instance."""
    cfg = dict(cfg or {})
    report: Dict[str, Any] = {"ts": time.time()}
    working_memory_bias_payload: Optional[Dict[str, Any]] = None

    go_sc_cfg = cfg.get("go_sc", {}) or {}
    nightly_cfg = cfg.get("nightly", {}) or {}
    fastpath_cfg = load_fastpath_defaults()
    runtime_fastpath = cfg.get("fastpath") or {}
    if isinstance(runtime_fastpath, dict):
        fastpath_cfg.update(runtime_fastpath)
    override_fastpath = load_fastpath_overrides()
    if override_fastpath:
        fastpath_cfg.update(override_fastpath)

    artifact_dir = Path(cfg.get("nightly_artifacts_dir", "reports/nightly_artifacts"))
    cfg_dict = _load_runtime_cfg_dict(Path("config/runtime.yaml"))
    working_memory_snapshot_path, working_memory_snapshot_payload, working_memory_snapshot_warning = _resolve_inner_os_working_memory_snapshot(cfg_dict)
    if working_memory_snapshot_warning:
        report.setdefault("warnings", []).append(working_memory_snapshot_warning)
    if working_memory_snapshot_path and isinstance(working_memory_snapshot_payload, dict):
        report["inner_os_working_memory_snapshot_path"] = working_memory_snapshot_path
        snapshot = working_memory_snapshot_payload.get("snapshot") or {}
        if isinstance(snapshot, dict):
            if snapshot.get("current_focus"):
                report["inner_os_working_memory_focus"] = snapshot["current_focus"]
            if snapshot.get("promotion_readiness") is not None:
                report["inner_os_working_memory_readiness"] = snapshot["promotion_readiness"]
            theme_focus = str(snapshot.get("long_term_theme_focus") or "").strip()
            theme_anchor = str(snapshot.get("long_term_theme_anchor") or "").strip()
            theme_summary = str(snapshot.get("long_term_theme_summary") or "").strip()
            if theme_focus or theme_anchor or theme_summary:
                report["inner_os_long_term_theme_summary"] = {
                    "focus": theme_focus,
                    "anchor": theme_anchor,
                    "kind": str(snapshot.get("long_term_theme_kind") or "").strip(),
                    "summary": theme_summary,
                    "strength": float(snapshot.get("long_term_theme_strength") or 0.0),
                }
        working_memory_bias_payload = derive_working_memory_replay_bias(working_memory_snapshot_payload)
    if working_memory_bias_payload:
        report["inner_os_working_memory_replay_bias"] = {
            "focus": str(working_memory_bias_payload.get("current_focus") or ""),
            "anchor": str(working_memory_bias_payload.get("focus_anchor") or ""),
            "strength": float(working_memory_bias_payload.get("strength") or 0.0),
        }
    memory_class_summary = _summarize_inner_os_memory_class(cfg_dict)
    if memory_class_summary:
        report["inner_os_memory_class_summary"] = memory_class_summary
    agenda_summary = _summarize_inner_os_agenda_trace(cfg_dict)
    if agenda_summary:
        report["inner_os_agenda_summary"] = agenda_summary
    commitment_summary = _summarize_inner_os_commitment_trace(cfg_dict)
    if commitment_summary:
        report["inner_os_commitment_summary"] = commitment_summary
    insight_summary = _summarize_inner_os_insight_trace(cfg_dict)
    if insight_summary:
        report["inner_os_insight_summary"] = insight_summary
    partner_relation_registry_summary = _summarize_inner_os_partner_relation_registry(cfg_dict)
    if partner_relation_registry_summary:
        report["inner_os_partner_relation_registry_summary"] = partner_relation_registry_summary
    group_thread_registry_summary = _summarize_inner_os_group_thread_registry(cfg_dict)
    if group_thread_registry_summary:
        report["inner_os_group_thread_registry_summary"] = group_thread_registry_summary
    discussion_thread_registry_summary = _summarize_inner_os_discussion_thread_registry(cfg_dict)
    if discussion_thread_registry_summary:
        report["inner_os_discussion_thread_registry_summary"] = discussion_thread_registry_summary
    partner_relation_summary = _summarize_inner_os_partner_relation(
        cfg_dict,
        registry_summary=partner_relation_registry_summary,
    )
    if partner_relation_summary:
        report["inner_os_partner_relation_summary"] = partner_relation_summary
    relation_arc_summary = RelationArcSummaryBuilder().build(report).to_dict()
    if relation_arc_summary.get("arc_kind") or relation_arc_summary.get("summary"):
        report["inner_os_relation_arc_summary"] = relation_arc_summary
    group_relation_arc_summary = GroupRelationArcSummaryBuilder().build(report).to_dict()
    if group_relation_arc_summary.get("arc_kind") or group_relation_arc_summary.get("summary"):
        report["inner_os_group_relation_arc_summary"] = group_relation_arc_summary
    identity_arc_summary = IdentityArcSummaryBuilder().build(report).to_dict()
    if identity_arc_summary.get("arc_kind") or identity_arc_summary.get("summary"):
        report["inner_os_identity_arc_summary"] = identity_arc_summary
    relation_arc_registry_summary = _summarize_inner_os_relation_arc_registry(cfg_dict)
    if relation_arc_registry_summary:
        report["inner_os_relation_arc_registry_summary"] = relation_arc_registry_summary
    identity_arc_registry_summary = _summarize_inner_os_identity_arc_registry(cfg_dict)
    if identity_arc_registry_summary:
        report["inner_os_identity_arc_registry_summary"] = identity_arc_registry_summary

    def _hub_tau_now() -> float:
        tk = getattr(hub, "timekeeper", None)
        if tk is None:
            return 0.0
        tau_now = getattr(tk, "tau_now", None)
        if callable(tau_now):
            try:
                return float(tau_now())
            except Exception:
                pass
        return float(getattr(tk, "tau", 0.0))

    # Memory TTL / GC
    events: List[Dict[str, Any]] = []
    if hasattr(hub, "memory_ttl") and hasattr(hub, "replay_memory"):
        try:
            events = hub.replay_memory.load_all()
            _dump_events_if_requested(events)
            if go_sc_cfg.get("enabled", False):
                baseline_ttl_tau = float(getattr(getattr(hub.memory_ttl, "cfg", None), "ttl_tau_default", 24.0))
                go_report = _apply_go_sc_gate(
                    events,
                    go_sc_cfg,
                    _hub_tau_now(),
                    baseline_ttl_tau,
                    nightly_cfg,
                    fastpath_cfg,
                    working_memory_bias_payload,
                )
                report["go_sc"] = go_report
                if go_report.get("fastpath"):
                    report["fastpath"] = go_report["fastpath"]
                if go_report.get("inner_replay"):
                    report["inner_replay"] = go_report["inner_replay"]
                if "nightly_metrics" in go_report:
                    report["nightly_metrics"] = go_report["nightly_metrics"]
                    wm_bias_report = (go_report.get("nightly_metrics") or {}).get("working_memory_replay_bias")
                    if isinstance(wm_bias_report, dict):
                        report["inner_os_working_memory_replay_bias"] = wm_bias_report
            kept, stats = hub.memory_ttl.gc(events)
            if kept is not None:
                events = kept
            report["memory_gc"] = stats

            forgetting_cfg = cfg.get("forgetting") if isinstance(cfg, dict) else None
            updated_events: Optional[List[Dict[str, Any]]] = None
            if isinstance(forgetting_cfg, dict):
                forgetfulness_cfg = None
                if isinstance(nightly_cfg, dict):
                    forgetfulness_cfg = nightly_cfg.get("forgetfulness_metrics")
                moment_log_path = None
                moment_candidate = cfg.get("moment_log_path") if isinstance(cfg, dict) else None
                if isinstance(moment_candidate, str) and moment_candidate.strip():
                    moment_log_path = moment_candidate
                updated_events, forget_report = _apply_nightly_forgetting(
                    events,
                    forgetting_cfg,
                    moment_log_path=moment_log_path,
                    memory_dir=forgetting_cfg.get("memory_dir") if isinstance(forgetting_cfg, dict) else None,
                    now_ts=time.time(),
                    forgetfulness_cfg=forgetfulness_cfg,
                )
                report["forgetting"] = forget_report

            defrag_cfg = cfg.get("defrag") if isinstance(cfg, dict) else None
            if isinstance(defrag_cfg, dict):
                if defrag_cfg.get("enable", False):
                    defrag_mode = str(defrag_cfg.get("mode", "observe"))
                    if defrag_mode == "observe":
                        report["defrag"] = _defrag_observe(events, defrag_cfg)
                    else:
                        report["defrag"] = {"status": "no_op", "mode": defrag_mode, "reason": "placeholder"}
                else:
                    report["defrag"] = {"status": "disabled"}

            if isinstance(forgetting_cfg, dict) and forgetting_cfg.get("monument_floor_test", False):
                report["monument_floor_test"] = _evaluate_monument_floor_test(events, forgetting_cfg)

            if updated_events is not None:
                events = updated_events
                hub.replay_memory.rewrite(events)
            elif kept is not None:
                hub.replay_memory.rewrite(events)
        except Exception as exc:
            report["memory_gc"] = {"error": str(exc)}

    # Value weight persistence (reuse current weights)
    if hasattr(hub, "value_weights"):
        report["value_weights"] = dict(hub.value_weights)

    # SelfModel coherence snapshot
    if hasattr(hub, "self_model"):
        try:
            report["coherence"] = {"current": float(hub.self_model.coherence())}
        except Exception:
            report["coherence"] = {"current": None}

    # Drift stub (placeholder until dedicated service is wired)
    drift = {"D": 0.0, "S": 0.0, "B": 0.0, "decision": "continue_safe_CPT", "alerts": []}
    report["drift"] = drift

    # Optional terrain compaction
    terrain_cfg = cfg.get("terrain_compact", {})
    if terrain_cfg.get("enabled", False):
        comp_report = _run_terrain_compact(terrain_cfg)
        report["terrain_compact"] = comp_report

    # Persist nightly report
    report_dir = Path(cfg.get("report_dir", "reports/nightly"))
    report_path = _dump_report(report, report_dir=report_dir)
    report["report_path"] = str(report_path)

    scatter_path, scatter_summary = _save_uhat_veto_scatter(events, artifact_dir)
    if scatter_path:
        report.setdefault("artifacts", {})["inner_replay_scatter"] = scatter_path
    if scatter_summary:
        report["inner_replay_summary"] = scatter_summary
    grid_result = _grid_search_beta_tau(events)
    if grid_result:
        report["beta_tau_grid"] = grid_result

    # Optional override: heartiness start value for next session
    overrides = {}
    if drift["decision"] == "switch_to_clean_CPT":
        overrides.setdefault("autopilot", {})["heartiness_start"] = 0.3
    else:
        overrides.setdefault("autopilot", {})["heartiness_start"] = 0.4
    if overrides:
        override_path = Path("config/overrides/autopilot.yaml")
        _ensure_dir(override_path)
        override_path.write_text(
            yaml.safe_dump(overrides, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        report["overrides_path"] = str(override_path)

    telemetry_log = Path(cfg.get("telemetry_log", "logs/telemetry_events.jsonl"))
    plots_dir = Path(cfg.get("plots_dir", "reports/plots"))
    md_path = Path(cfg.get("markdown_path", "reports/nightly.md"))
    culture_feedback_enabled = bool(
        cfg.get("culture_feedback_enabled", False)
        or nightly_cfg.get("enable_culture_feedback", False)
    )
    _generate_telemetry_section(
        report,
        telemetry_log,
        plots_dir,
        md_path,
        culture_feedback_enabled=culture_feedback_enabled,
    )
    json_path = _write_json_summary(report, out_dir="reports")
    print(f"[nightly] JSON summary -> {json_path}")

    return report


def _run_terrain_compact(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if terrain_compact is None:
        return {"error": "terrain_compact module not available (missing duckdb?)"}
    output = Path(cfg.get("output", "dataset/terrain_parquet"))
    input_glob = cfg.get("input")
    if not input_glob:
        return {"error": "terrain_compact.input is required"}
    ts_col = cfg.get("timestamp_column", "ts")
    date_col = cfg.get("date_column", "date")
    compression = cfg.get("compression", "ZSTD")
    threads = int(cfg.get("threads", 0))
    database = cfg.get("database", ":memory:")
    args = [
        "--input",
        str(input_glob),
        "--output",
        str(output),
        "--timestamp-column",
        str(ts_col),
        "--date-column",
        str(date_col),
        "--compression",
        str(compression),
        "--database",
        str(database),
    ]
    if threads:
        args.extend(["--threads", str(threads)])
    try:
        terrain_compact.convert(args)
        return {"status": "ok", "output": str(output)}
    except SystemExit as exc:  # convert() may call SystemExit on dependency errors
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


def _apply_go_sc_gate(
    events: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    tau_now: float,
    baseline_ttl_tau: float,
    nightly_cfg: Dict[str, Any],
    fastpath_cfg: Optional[Dict[str, Any]] = None,
    working_memory_bias: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    total_events = len(events)
    if total_events == 0:
        return {"status": "skipped", "reason": "no_events"}
    fastpath_cfg = dict(fastpath_cfg or {})
    fail_safe_cfg = fail_safe_settings(fastpath_cfg)

    ttl_budget_cfg = nightly_cfg.get("ttl_budget", {}) or {}
    hygiene_cfg = nightly_cfg.get("hygiene", {}) or {}
    phase_metrics_cfg = nightly_cfg.get("phase", {}) or {}
    read_only_block = bool(ttl_budget_cfg.get("block_if_readonly", False) and nightly_cfg.get("read_only"))
    if read_only_block:
        return {"status": "skipped", "reason": "read_only_night"}

    gate_cfg = cfg.get("gate", {}) or {}
    weights_cfg = (cfg.get("score", {}) or {}).get("weights") or cfg.get("weights") or {}
    scale_range = gate_cfg.get("ttl_scale_range", [])
    scale_hi = float(
        gate_cfg.get(
            "ttl_scale_high",
            scale_range[1] if isinstance(scale_range, (list, tuple)) and len(scale_range) >= 2 else gate_cfg.get("selected_ttl_scale", 1.3),
        )
    )
    scale_lo = float(
        gate_cfg.get(
            "ttl_scale_low",
            scale_range[0] if isinstance(scale_range, (list, tuple)) and scale_range else gate_cfg.get("remainder_ttl_scale", 0.8),
        )
    )
    winner_pct = gate_cfg.get("winner_min_percentile")
    if winner_pct is None:
        top_percent = gate_cfg.get("top_percent")
        if top_percent is not None:
            try:
                winner_pct = 1.0 - float(top_percent)
            except Exception:
                winner_pct = None
    winner_pct = float(winner_pct if winner_pct is not None else 0.5)
    precedence_cfg = cfg.get("precedence", {}) or {}
    rescue_cfg = precedence_cfg.get("rescue", precedence_cfg.get("go_sc_rescue", {})) or {}
    rescue_enabled = bool(rescue_cfg.get("enabled", False))
    rescue_pct = float(rescue_cfg.get("min_percentile", rescue_cfg.get("min_percentile_required", 0.99)))
    rescue_rarity = float(rescue_cfg.get("min_rarity", 0.9))
    rescue_scale = float(rescue_cfg.get("ttl_scale_on_rescue", rescue_cfg.get("rescue_ttl_scale", scale_hi)))
    interference_overrides = bool(precedence_cfg.get("interference_overrides_go_sc", True))

    junk_threshold = hygiene_cfg.get("exclude_if_junk_prob_ge")
    if junk_threshold is not None:
        junk_threshold = float(junk_threshold)
    max_total_hours = ttl_budget_cfg.get("max_total_hours", float("inf"))
    try:
        max_total_hours = float(max_total_hours)
    except Exception:
        max_total_hours = float("inf")

    resolution_counts = {"mask_wins": 0, "rescued": 0, "go_sc_only": 0}
    hygiene_filtered = 0
    masked_total = 0
    rescued_total = 0
    mask_windows: List[float] = []
    reverse_ratios: List[float] = []
    total_extension = 0.0
    total_compression = 0.0
    wm_strength = _safe_float((working_memory_bias or {}).get("strength"), 0.0) or 0.0
    wm_boost_cap = min(0.12, wm_strength * 0.12)
    wm_matches = 0
    wm_boosts: List[float] = []
    wm_match_details: List[Tuple[float, float, str]] = []

    # Ensure go_score/percentile populated for legacy traces.
    scored: List[Tuple[float, int]] = []
    for idx, event in enumerate(events):
        meta = event.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            event["meta"] = meta
        meta.setdefault("ttl_scale", float(meta.get("ttl_scale", 1.0)))
        meta.setdefault("resolution_reason", "go_sc_only")
        score = meta.get("go_score")
        if score is None:
            features = extract_features(event)
            score = weighted_score(weights_cfg, features) if weights_cfg else 0.0
            meta["go_score"] = score
        score = float(score)
        wm_alignment = _working_memory_alignment_score(event, working_memory_bias)
        wm_boost = wm_alignment * wm_boost_cap if wm_boost_cap > 0.0 else 0.0
        if wm_boost > 0.0:
            wm_matches += 1
            wm_boosts.append(float(wm_boost))
            wm_match_details.append((float(wm_boost), float(wm_alignment), _working_memory_event_identifier(event)))
            meta["working_memory_replay_bias"] = {
                "alignment": round(float(wm_alignment), 4),
                "boost": round(float(wm_boost), 6),
                "focus": str((working_memory_bias or {}).get("current_focus") or ""),
                "anchor": str((working_memory_bias or {}).get("focus_anchor") or ""),
            }
        scored.append((score + wm_boost, idx))

    sorted_scores = sorted(score for score, _ in scored)
    denom = max(1, len(sorted_scores) - 1)
    for score, idx in scored:
        meta = events[idx]["meta"]
        if meta.get("go_percentile") is not None:
            continue
        if len(sorted_scores) < 2:
            percentile = 0.5
        else:
            pos = bisect_left(sorted_scores, score)
            percentile = pos / denom if denom else 0.5
        meta["go_percentile"] = max(0.0, min(1.0, float(percentile)))

    baseline_ttl_tau = max(1e-6, float(baseline_ttl_tau))

    for event in events:
        meta = event.get("meta") or {}
        reverse_ratio = meta.get("reverse_ratio")
        if reverse_ratio is not None:
            try:
                reverse_ratios.append(float(reverse_ratio))
            except Exception:
                pass
        scale = float(meta.get("ttl_scale", 1.0))
        reason = "go_sc_only"
        go_pct = float(meta.get("go_percentile", 0.0))
        rarity = float(meta.get("rarity", meta.get("salience", 0.0) or 0.0))
        masked = bool((meta.get("interference") or {}).get("action") == "mask")
        if masked:
            masked_total += 1
            mask_info = meta.get("interference") or {}
            mask_until = mask_info.get("mask_until_tau")
            mask_start = mask_info.get("tau")
            if mask_until is not None and mask_start is not None:
                try:
                    mask_windows.append(float(mask_until) - float(mask_start))
                except Exception:
                    pass
        safety_block = bool(meta.get("safety_block"))
        junk_prob = float(meta.get("junk_prob", 0.0))
        hygiene_blocked = safety_block or (
            junk_threshold is not None and junk_prob >= junk_threshold
        )
        if hygiene_blocked:
            hygiene_filtered += 1
            scale = min(scale, 1.0)
            reason = "hygiene_filtered"
        elif masked and interference_overrides:
            if rescue_enabled and go_pct >= rescue_pct and rarity >= rescue_rarity:
                scale = rescue_scale
                reason = "rescued_by_go_sc"
                rescued_total += 1
                resolution_counts["rescued"] += 1
            else:
                scale = 1.0
                reason = "mask_wins"
                resolution_counts["mask_wins"] += 1
        else:
            scale = scale_hi if go_pct >= winner_pct else scale_lo
            resolution_counts["go_sc_only"] += 1

        if scale > 1.0:
            delta = baseline_ttl_tau * (scale - 1.0)
            if total_extension + delta > max_total_hours:
                remaining = max_total_hours - total_extension
                if remaining <= 0.0:
                    scale = 1.0
                else:
                    scale = 1.0 + remaining / baseline_ttl_tau
                reason = "ttl_budget_exhausted"
                delta = baseline_ttl_tau * (scale - 1.0)
            total_extension += max(delta, 0.0)
        elif scale < 1.0:
            delta = baseline_ttl_tau * (scale - 1.0)
            total_compression += max(-delta, 0.0)

        meta["ttl_scale"] = float(scale)
        meta["resolution_reason"] = reason

    median_rr = median(reverse_ratios) if reverse_ratios else None
    ema_rr = None
    if reverse_ratios:
        alpha = float(phase_metrics_cfg.get("ema_alpha_for_report", 0.3))
        ema_rr = reverse_ratios[0]
        for value in reverse_ratios[1:]:
            ema_rr = alpha * value + (1.0 - alpha) * ema_rr

    masked_pct = masked_total / max(1, total_events)
    avg_window_tau = (
        sum(mask_windows) / len(mask_windows) if mask_windows else None
    )
    nightly_metrics = {
        "phase_shift": {"median_reverse_ratio": median_rr, "ema_reverse_ratio": ema_rr, "tau_now": tau_now},
        "ttl_budget_audit": {
            ">1_total_hours": total_extension,
            "<1_total_hours": total_compression,
        },
        "interference_impact": {
            "masked_total": masked_total,
            "rescued_total": rescued_total,
            "masked_pct": masked_pct,
            "remask_rate": resolution_counts["mask_wins"] / max(1, masked_total) if masked_total else 0.0,
            "avg_window_tau": avg_window_tau,
        },
        "rescue_rate": rescued_total / max(1, masked_total) if masked_total else 0.0,
    }
    if wm_boosts:
        wm_match_details.sort(key=lambda item: item[0], reverse=True)
        nightly_metrics["working_memory_replay_bias"] = {
            "focus": str((working_memory_bias or {}).get("current_focus") or ""),
            "anchor": str((working_memory_bias or {}).get("focus_anchor") or ""),
            "strength": round(float(wm_strength), 4),
            "matched_events": int(wm_matches),
            "boost_mean": round(float(sum(wm_boosts) / len(wm_boosts)), 6),
            "boost_max": round(float(max(wm_boosts)), 6),
            "top_matches": [
                {
                    "id": item[2],
                    "alignment": round(item[1], 4),
                    "boost": round(item[0], 6),
                }
                for item in wm_match_details[:3]
                if item[2]
            ],
        }

    ttl_report = {
        "extension_total": total_extension,
        "compression_total": total_compression,
        "baseline_ttl_tau": baseline_ttl_tau,
        "max_total_hours": max_total_hours if math.isfinite(max_total_hours) else None,
    }
    if math.isfinite(max_total_hours):
        ttl_report["remaining_hours"] = max(0.0, max_total_hours - total_extension)

    fast_metrics = _summarize_fastpath_metrics(events)
    profile_counts = fast_metrics.get("profiles") or {}
    fast_report = {k: v for k, v in fast_metrics.items() if k != "profiles"}
    fast_report["coverage_rate"] = fast_report["coverage"] / max(total_events, 1)
    fast_report["override_rate"] = fast_report["override"] / max(fast_report["predicate_true"], 1)
    fast_report["mode"] = fastpath_cfg.get("enforce_actions", "record_only")
    profile_rates: Dict[str, Dict[str, Any]] = {}
    for name, stats in profile_counts.items():
        profile_rates[name] = {
            **stats,
            "coverage_rate": stats["coverage"] / max(total_events, 1),
            "override_rate": stats["override"] / max(stats["predicate_true"], 1),
        }
    fast_report["profiles"] = profile_rates
    fail_safe_meta = _maybe_trigger_fastpath_fail_safe(fast_report, fail_safe_cfg)
    if fail_safe_meta:
        fast_report["fail_safe"] = fail_safe_meta
    inner_replay = _summarize_inner_replay_metrics(events)
    qualia_report = _summarize_qualia_metrics(events)

    return {
        "status": "ok",
        "total": total_events,
        "resolution": resolution_counts,
        "hygiene": {"filtered": hygiene_filtered},
        "rescue_rate": nightly_metrics["rescue_rate"],
        "ttl_budget": ttl_report,
        "nightly_metrics": nightly_metrics,
        "fastpath": fast_report,
        "inner_replay": inner_replay,
        "qualia": qualia_report,
    }


def _write_fastpath_override(mode: str) -> bool:
    current_override = load_fastpath_overrides()
    if current_override.get("enforce_actions") == mode:
        return False
    payload = {"fastpath": {"enforce_actions": mode}}
    _ensure_dir(FASTPATH_OVERRIDE_PATH)
    FASTPATH_OVERRIDE_PATH.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return True


def _maybe_trigger_fastpath_fail_safe(
    fast_report: Dict[str, Any], fail_safe_cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    override_rate = float(fast_report.get("override_rate", 0.0) or 0.0)
    threshold = float(fail_safe_cfg.get("override_rate_threshold", 0.2))
    if override_rate <= threshold:
        return None
    fallback_mode = str(fail_safe_cfg.get("fallback_mode", "soft_hint"))
    override_written = _write_fastpath_override(fallback_mode)
    lookback_days = int(fail_safe_cfg.get("lookback_days", 1))
    return {
        "activated": True,
        "fallback_mode": fallback_mode,
        "override_rate": override_rate,
        "threshold": threshold,
        "override_written": override_written,
        "lookback_days": lookback_days,
    }


def _summarize_fastpath_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    coverage = 0
    predicate_true = 0
    override = 0
    profiles: Dict[str, Dict[str, int]] = defaultdict(lambda: {"coverage": 0, "predicate_true": 0, "override": 0})
    for event in events:
        meta = event.get("meta") or {}
        receipt = meta.get("receipt") or {}
        fast_section = receipt.get("fastpath") or {}
        final_reason = meta.get("resolution_reason") or (receipt.get("go_sc") or {}).get("resolution_reason")
        for profile_name, data in fast_section.items():
            if not data.get("final_ok"):
                continue
            coverage += 1
            profiles[profile_name]["coverage"] += 1
            preds = data.get("predicates") or {}
            if preds.get("fast_rescue"):
                predicate_true += 1
                profiles[profile_name]["predicate_true"] += 1
                if final_reason not in {"rescued_by_go_sc", "mask_wins"}:
                    override += 1
                    profiles[profile_name]["override"] += 1
    profile_dict = {name: dict(stats) for name, stats in profiles.items()}
    return {
        "coverage": coverage,
        "predicate_true": predicate_true,
        "override": override,
        "profiles": profile_dict,
    }


def _summarize_inner_replay_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    execute = 0
    cancel = 0
    veto_sum = 0.0
    uhat_sum = 0.0
    smax_sum = 0.0
    tprep_sum = 0.0
    cfg_counts: Dict[str, int] = {}
    for event in events:
        receipt = (event.get("meta") or {}).get("receipt") or {}
        ir = receipt.get("inner_replay")
        if not ir:
            continue
        total += 1
        veto = float((ir.get("veto") or {}).get("score", 0.0))
        decision = (ir.get("veto") or {}).get("decision", "execute")
        if decision == "execute":
            execute += 1
        else:
            cancel += 1
        uhat_sum += float(ir.get("u_hat", 0.0))
        prep = ir.get("prep") or {}
        smax_sum += float(prep.get("s_max", 0.0))
        tprep_sum += float(prep.get("t_prep", 0.0))
        veto_sum += veto
        meta = ir.get("meta") or {}
        cfg_b2 = str(meta.get("cfg_b2", "na"))
        cfg_counts[cfg_b2] = cfg_counts.get(cfg_b2, 0) + 1
    if total == 0:
        return {"count": 0}
    cfg_top = sorted(cfg_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {
        "count": total,
        "execute_rate": _safe_mean(execute, total),
        "cancel_rate": _safe_mean(cancel, total),
        "veto_mean": _safe_mean(veto_sum, total),
        "u_hat_mean": _safe_mean(uhat_sum, total),
        "s_max_mean": _safe_mean(smax_sum, total),
        "t_prep_mean": _safe_mean(tprep_sum, total),
        "cfg_top": cfg_top,
    }


def _summarize_qualia_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    u_sum = 0.0
    m_sum = 0.0
    load_sum = 0.0
    gate_samples = 0
    gate_open_sum = 0.0
    unconscious = 0
    for event in events:
        receipt = (event.get("meta") or {}).get("receipt") or {}
        qualia_bucket = receipt.get("qualia") or {}
        if not qualia_bucket:
            continue
        total += 1
        try:
            u_sum += float(qualia_bucket.get("u_t", 0.0) or 0.0)
            m_sum += float(qualia_bucket.get("m_t", 0.0) or 0.0)
            load_sum += float(qualia_bucket.get("load", qualia_bucket.get("load_t", 0.0)) or 0.0)
        except (TypeError, ValueError):
            pass
        if "a_t" in qualia_bucket:
            gate_samples += 1
            try:
                gate_open_sum += float(qualia_bucket.get("a_t", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        try:
            unconscious += int(qualia_bucket.get("unconscious_success", 0))
        except (TypeError, ValueError):
            pass
    if total == 0:
        return {"count": 0}
    return {
        "count": total,
        "u_mean": _safe_mean(u_sum, total),
        "m_mean": _safe_mean(m_sum, total),
        "load_mean": _safe_mean(load_sum, total),
        "gate_open_rate": _safe_mean(gate_open_sum, gate_samples) if gate_samples else None,
        "unconscious_success_rate": _safe_mean(unconscious, total),
    }


def _safe_mean(total: float, count: int) -> float:
    return total / count if count else 0.0


def _collect_inner_replay_points(events: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[int]]:
    xs: List[float] = []
    ys: List[float] = []
    cs: List[int] = []
    for event in events:
        receipt = (event.get("meta") or {}).get("receipt") or {}
        ir = receipt.get("inner_replay")
        if not ir:
            continue
        veto = ir.get("veto") or {}
        u_hat = ir.get("u_hat")
        veto_score = veto.get("score")
        if u_hat is None or veto_score is None:
            continue
        try:
            u = float(u_hat)
            v = float(veto_score)
        except (TypeError, ValueError):
            continue
        if math.isnan(u) or math.isnan(v):
            continue
        xs.append(u)
        ys.append(v)
        cs.append(1 if (veto.get("decision", "execute") == "execute") else 0)
    return xs, ys, cs


def _save_uhat_veto_scatter(
    events: List[Dict[str, Any]], out_dir: Path, fname: str = "inner_replay_scatter.png"
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if plt is None:
        return None, None
    xs, ys, cs = _collect_inner_replay_points(events)
    if len(xs) < 10:
        return None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    ex_x = [x for x, c in zip(xs, cs) if c == 1]
    ex_y = [y for y, c in zip(ys, cs) if c == 1]
    ca_x = [x for x, c in zip(xs, cs) if c == 0]
    ca_y = [y for y, c in zip(ys, cs) if c == 0]
    plt.figure(figsize=(6, 5), dpi=120)
    plt.scatter(ex_x, ex_y, s=12, alpha=0.6, label="execute")
    plt.scatter(ca_x, ca_y, s=12, alpha=0.6, marker="x", label="cancel")
    plt.xlabel("u_hat")
    plt.ylabel("veto_score")
    plt.legend(loc="best")
    plt.tight_layout()
    path = out_dir / fname
    plt.savefig(path)
    plt.close()
    summary = _describe_scatter(xs, ys, cs)
    return str(path), summary


def _describe_scatter(xs: List[float], ys: List[float], cs: List[int]) -> Dict[str, Any]:
    total = len(xs)
    if not total:
        return {}
    u_med = sorted(xs)[total // 2]
    v_med = sorted(ys)[total // 2]
    br = tl = mid = 0
    for u, v, c in zip(xs, ys, cs):
        if u >= u_med and v <= v_med:
            br += 1
        elif u <= u_med and v >= v_med:
            tl += 1
        else:
            mid += 1
    br_ratio = br / total
    tl_ratio = tl / total
    if br_ratio >= 0.5:
        verdict = "�E���ɓ_�������F�悢��� (execute ���f��)�B"
        action = "���܂̐ݒ���ێ����U�z�}���Ď��B"
    elif tl_ratio >= 0.4:
        verdict = "����ɓ_�������F�u���[�L�����߁B"
        action = "�� �� �� ������������Ă������B"
    else:
        verdict = "�����ɓ_���U��΂�����C���B"
        action = "�q�X�e���V�X�����L���邩��� ��?�� �����K�� A/B�B"
    return {
        "verdict": verdict,
        "action": action,
        "stats": {
            "bottom_right_ratio": br_ratio,
            "top_left_ratio": tl_ratio,
            "middle_ratio": mid / total,
        },
    }


def _grid_search_beta_tau(
    events: List[Dict[str, Any]],
    betas: Tuple[float, ...] = (0.8, 1.0, 1.2),
    taus: Tuple[float, ...] = (-0.1, 0.0, 0.1),
) -> Optional[Dict[str, Any]]:
    points: List[Tuple[float, float, float]] = []
    for event in events:
        receipt = (event.get("meta") or {}).get("receipt") or {}
        ir = receipt.get("inner_replay")
        if not ir:
            continue
        veto = ir.get("veto") or {}
        u_hat = ir.get("u_hat")
        veto_score = veto.get("score")
        if u_hat is None or veto_score is None:
            continue
        try:
            u = float(u_hat)
            v = float(veto_score)
        except (TypeError, ValueError):
            continue
        if math.isnan(u) or math.isnan(v):
            continue
        meta = event.get("meta") or {}
        regret = float((meta.get("outcome") or {}).get("regret", 0.0))
        points.append((u, v, regret))
    if len(points) < 50:
        return {"grid": [], "best": None}
    grid: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for beta in betas:
        for tau in taus:
            execute = 0
            cancel = 0
            regret_sum = 0.0
            for u, v, regret in points:
                decision_execute = (u - beta * v) >= tau
                if decision_execute:
                    execute += 1
                    regret_sum += regret
                else:
                    cancel += 1
            total = execute + cancel
            execute_rate = execute / total if total else 0.0
            regret_mean = regret_sum / max(1, execute)
            row = {
                "beta": beta,
                "tau": tau,
                "execute_rate": execute_rate,
                "regret_mean": regret_mean,
            }
            grid.append(row)
            score = -(regret_mean) - 0.1 * max(0.0, 0.4 - execute_rate)
            if best is None or score > best["score"]:
                best = {
                    "beta": beta,
                    "tau": tau,
                    "score": score,
                    "execute_rate": execute_rate,
                    "regret_mean": regret_mean,
                }
    return {"grid": grid, "best": best}


def _summarize_field_telemetry(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    S: List[float] = []
    H: List[float] = []
    rho: List[float] = []
    ignition: List[float] = []
    valence: List[float] = []
    arousal: List[float] = []
    delta_m_vals: List[float] = []
    jerk_vals: List[float] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "field.metrics":
                continue
            data = row.get("data") or {}
            try:
                S.append(float(data.get("S", np.nan)))
                H.append(float(data.get("H", np.nan)))
                rho.append(float(data.get("rho", np.nan)))
                ignition.append(float(data.get("Ignition", np.nan)))
                if "delta_m" in data:
                    delta_m_vals.append(float(data.get("delta_m", np.nan)))
                if "jerk" in data:
                    jerk_vals.append(float(data.get("jerk", np.nan)))
                if "valence" in data:
                    valence.append(float(data.get("valence", np.nan)))
                if "arousal" in data:
                    arousal.append(float(data.get("arousal", np.nan)))
            except Exception:
                continue
    except Exception:
        return None
    if not S:
        return None
    S_arr = np.array(S, dtype=float)
    H_arr = np.array(H, dtype=float)
    rho_arr = np.array(rho, dtype=float)
    I_arr = np.array(ignition, dtype=float)
    V_arr = np.array(valence, dtype=float) if valence else np.full(1, np.nan)
    A_arr = np.array(arousal, dtype=float) if arousal else np.full(1, np.nan)
    delta_arr = np.array(delta_m_vals, dtype=float) if delta_m_vals else np.full(1, np.nan)
    jerk_arr = np.array(jerk_vals, dtype=float) if jerk_vals else np.full(1, np.nan)
    rho_I_corr = _safe_corr(rho_arr, I_arr)
    S_I_corr = _safe_corr(S_arr, I_arr)
    H_I_corr = _safe_corr(H_arr, I_arr)
    valence_mean = _mean_clean(V_arr)
    arousal_mean = _mean_clean(A_arr)
    valence_I_corr = _safe_corr(V_arr, I_arr)
    arousal_I_corr = _safe_corr(A_arr, I_arr)
    return {
        "S_mean": _mean_clean(S_arr),
        "H_mean": _mean_clean(H_arr),
        "rho_mean": _mean_clean(rho_arr),
        "Ignition_mean": _mean_clean(I_arr),
        "rho_I_corr": rho_I_corr,
        "S_I_corr": S_I_corr,
        "H_I_corr": H_I_corr,
        "valence_mean": valence_mean,
        "arousal_mean": arousal_mean,
        "valence_I_corr": valence_I_corr,
        "arousal_I_corr": arousal_I_corr,
        "delta_m_p95": _nan_percentile(delta_arr, 95.0),
        "jerk_p95": _nan_percentile(jerk_arr, 95.0),
    }


def _extract_field_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    delta_vals: List[float] = []
    jerk_vals: List[float] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "field.metrics":
                continue
            data = row.get("data") or {}
            try:
                if "delta_m" in data:
                    delta_vals.append(float(data.get("delta_m", np.nan)))
                if "jerk" in data:
                    jerk_vals.append(float(data.get("jerk", np.nan)))
            except Exception:
                continue
    except Exception:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return np.array(delta_vals, dtype=float), np.array(jerk_vals, dtype=float)


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        as_float = float(value)
    except Exception:
        return None
    if not math.isfinite(as_float):
        return None
    return as_float


def _render_assoc_plots(
    delta_series: np.ndarray,
    jerk_series: np.ndarray,
    entropy_series: np.ndarray,
    max_abs_series: np.ndarray,
    plots_dir: Path,
) -> Dict[str, Path]:
    if plt is None:
        return {}
    plots_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, Path] = {}
    delta_clean = delta_series[np.isfinite(delta_series)] if delta_series.size else delta_series
    jerk_clean = jerk_series[np.isfinite(jerk_series)] if jerk_series.size else jerk_series
    entropy_clean = entropy_series[np.isfinite(entropy_series)] if entropy_series.size else entropy_series
    max_abs_clean = max_abs_series[np.isfinite(max_abs_series)] if max_abs_series.size else max_abs_series
    if delta_clean.size:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(delta_clean, label="delta_m", color="#1f77b4", linewidth=1.2)
        if jerk_clean.size:
            ax.plot(jerk_clean, label="jerk", color="#ff7f0e", linewidth=1.0, alpha=0.9)
        ax.set_title("��m / jerk (recent)")
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        fig.tight_layout()
        out_path = plots_dir / "assoc_delta_jerk.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        if out_path.exists():
            created["assoc_delta_jerk"] = out_path
    if entropy_clean.size:
        fig, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(entropy_clean, color="#2ca02c", linewidth=1.1, label="attn_entropy")
        ax1.set_ylabel("entropy")
        ax1.grid(alpha=0.2)
        ax2 = ax1.twinx()
        if max_abs_clean.size:
            ax2.plot(max_abs_clean, color="#d62728", linewidth=1.0, label="max|score|")
        ax2.set_ylabel("max|score|")
        ax1.set_xlabel("record")
        ax1.set_title("Assoc attention health")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            ax1.legend(lines + lines2, labels + labels2, loc="upper right")
        fig.tight_layout()
        out_path = plots_dir / "assoc_attention_health.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        if out_path.exists():
            created["assoc_attention_health"] = out_path
    return created


def _summarize_assoc_kernel(
    canary_path: Path,
    plots_dir: Path,
    delta_series: np.ndarray,
    jerk_series: np.ndarray,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Path]]:
    if not canary_path.exists():
        return None, {}
    entropy_vals: List[float] = []
    max_abs_vals: List[float] = []
    guard_counts: Dict[str, int] = defaultdict(int)
    icl1_vals: List[float] = []
    icl3_vals: List[float] = []
    records = 0
    try:
        for line in canary_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            records += 1
            entropy = rec.get("attn_entropy")
            if isinstance(entropy, (int, float)) and math.isfinite(float(entropy)):
                entropy_vals.append(float(entropy))
            max_abs = rec.get("max_score_abs")
            if isinstance(max_abs, (int, float)) and math.isfinite(float(max_abs)):
                max_abs_vals.append(float(max_abs))
            guard_action = rec.get("guard_action")
            if guard_action:
                guard_counts[str(guard_action)] += 1
            icl1 = rec.get("icl_at1")
            if isinstance(icl1, (int, float)) and math.isfinite(float(icl1)):
                icl1_vals.append(float(icl1))
            icl3 = rec.get("icl_at3")
            if isinstance(icl3, (int, float)) and math.isfinite(float(icl3)):
                icl3_vals.append(float(icl3))
    except Exception:
        return None, {}
    if records == 0:
        return None, {}

    entropy_arr = np.array(entropy_vals, dtype=float) if entropy_vals else np.empty(0, dtype=float)
    max_abs_arr = np.array(max_abs_vals, dtype=float) if max_abs_vals else np.empty(0, dtype=float)
    summary = {
        "schema": "assoc.v1",
        "records": records,
        "attn_entropy_mean": _finite_or_none(_mean_clean(entropy_arr) if entropy_arr.size else None),
        "max_score_p99": _finite_or_none(_nan_percentile(max_abs_arr, 99.0) if max_abs_arr.size else None),
        "dm_p95": _finite_or_none(
            _nan_percentile(delta_series, 95.0) if isinstance(delta_series, np.ndarray) and delta_series.size else None
        ),
        "jerk_p95": _finite_or_none(
            _nan_percentile(jerk_series, 95.0) if isinstance(jerk_series, np.ndarray) and jerk_series.size else None
        ),
        "icl_at1": _finite_or_none(np.mean(icl1_vals) if icl1_vals else None),
        "icl_at3": _finite_or_none(np.mean(icl3_vals) if icl3_vals else None),
        "guard_actions": {k: int(v) for k, v in guard_counts.items()},
    }
    plots = _render_assoc_plots(delta_series, jerk_series, entropy_arr, max_abs_arr, plots_dir)
    return summary, plots


def _summarize_vision_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    events = 0
    total_detections = 0
    counts_by_kind: Dict[str, int] = defaultdict(int)
    valences: List[float] = []
    arousals: List[float] = []
    dominances: List[float] = []
    fps_values: List[float] = []
    pose_keys = ("yaw_mean", "pitch_mean", "roll_mean", "score_mean")
    pose_sum: Dict[str, float] = defaultdict(float)
    pose_count: Dict[str, int] = defaultdict(int)
    ts_last: Optional[float] = None
    last_event: Optional[Dict[str, Any]] = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "vision.metrics":
                continue
            data = row.get("data") or {}
            events += 1
            ts = data.get("ts_ms") or data.get("ts") or row.get("ts")
            try:
                ts_last = float(ts)
            except (TypeError, ValueError):
                ts_last = ts_last
            total = data.get("detections_total")
            try:
                total_detections += int(total)
            except (TypeError, ValueError):
                detections = data.get("detections")
                if isinstance(detections, list):
                    total_detections += len(detections)
            by_kind = data.get("counts_by_kind") or data.get("detections_by_class") or {}
            if isinstance(by_kind, dict):
                for kind, count in by_kind.items():
                    try:
                        counts_by_kind[str(kind)] += int(count)
                    except (TypeError, ValueError):
                        continue
            val = data.get("valence")
            if isinstance(val, (int, float)) and math.isfinite(val):
                valences.append(float(val))
            aro = data.get("arousal")
            if isinstance(aro, (int, float)) and math.isfinite(aro):
                arousals.append(float(aro))
            dom = data.get("dominance")
            if isinstance(dom, (int, float)) and math.isfinite(dom):
                dominances.append(float(dom))
            features = data.get("features")
            if isinstance(features, dict):
                fps = features.get("fps")
                if isinstance(fps, (int, float)) and math.isfinite(fps):
                    fps_values.append(float(fps))
            pose = data.get("pose_summary") or data.get("pose")
            if isinstance(pose, dict):
                for key in pose_keys:
                    val_pose = pose.get(key)
                    if isinstance(val_pose, (int, float)) and math.isfinite(val_pose):
                        pose_sum[key] += float(val_pose)
                        pose_count[key] += 1
            last_event = {
                "ts_ms": ts_last,
                "detections_total": total,
                "counts_by_kind": by_kind,
                "valence": val,
                "arousal": aro,
            }
    except Exception:
        return None
    if events == 0:
        return None

    def _mean(values: List[float]) -> Optional[float]:
        return float(sum(values) / len(values)) if values else None

    pose_mean: Dict[str, float] = {}
    for key in pose_keys:
        if pose_count.get(key):
            pose_mean[key] = pose_sum[key] / pose_count[key]

    return {
        "events": events,
        "detections_total": total_detections,
        "counts_by_kind": dict(counts_by_kind),
        "mean_valence": _mean(valences),
        "mean_arousal": _mean(arousals),
        "mean_dominance": _mean(dominances),
        "pose_mean": pose_mean or None,
        "fps_mean": _mean(fps_values),
        "ts_last": ts_last,
        "last_event": last_event,
    }


def _render_vision_plots(snapshot: Mapping[str, Any], plots_dir: Path) -> Dict[str, Path]:
    if plt is None or not isinstance(snapshot, Mapping):
        return {}
    plots_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, Path] = {}
    counts = snapshot.get("counts_by_kind")
    if isinstance(counts, Mapping) and counts:
        items = sorted(
            ((str(k), float(v)) for k, v in counts.items() if isinstance(v, (int, float))),
            key=lambda kv: kv[1],
            reverse=True,
        )[:12]
        if items:
            labels, values = zip(*items)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(labels, values, color="#1f77b4")
            ax.set_ylabel("count")
            ax.set_title("Vision detection counts (top)")
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            out_path = plots_dir / "vision_counts.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            if out_path.exists():
                created["vision_counts"] = out_path
    pose_mean = snapshot.get("pose_mean")
    if isinstance(pose_mean, Mapping):
        pose_items = [
            (key, float(val))
            for key, val in pose_mean.items()
            if isinstance(val, (int, float)) and math.isfinite(float(val))
        ]
        if pose_items:
            labels, values = zip(*pose_items)
            fig, ax = plt.subplots(figsize=(4, 2.8))
            ax.barh(labels, values, color="#ff7f0e")
            ax.set_xlabel("mean")
            ax.set_title("Pose summary (mean)")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            out_path = plots_dir / "vision_pose.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            if out_path.exists():
                created["vision_pose"] = out_path
    return created


def _render_vision_narrative(snapshot: Mapping[str, Any], *, top_k: int = 3) -> List[str]:
    if not isinstance(snapshot, Mapping):
        return []
    counts = snapshot.get("counts_by_kind")
    if not isinstance(counts, Mapping) or not counts:
        return []
    try:
        events = float(snapshot.get("events", 0.0))
    except (TypeError, ValueError):
        events = 0.0
    detections_total = snapshot.get("detections_total")
    try:
        detections_total_f = float(detections_total)
    except (TypeError, ValueError):
        detections_total_f = 0.0
    denom = events if events > 0 else detections_total_f if detections_total_f > 0 else 1.0
    ranked = sorted(
        ((str(k), float(v)) for k, v in counts.items() if isinstance(v, (int, float))),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not ranked:
        return []
    lines = ["", "**Vision quick notes**"]
    for label, count in ranked[: max(1, top_k)]:
        rate = count / denom if denom else 0.0
        lines.append(f"- {label}: count={count:.1f}, share={rate:.2%}")
    lines.append("")
    return lines


def _mean_clean(arr: np.ndarray) -> float:
    mask = np.isfinite(arr)
    if not mask.any():
        return float("nan")
    return float(np.mean(arr[mask]))


def _nan_percentile(arr: np.ndarray, percentile: float) -> float:
    mask = np.isfinite(arr)
    if not mask.any():
        return float("nan")
    try:
        return float(np.percentile(arr[mask], percentile))
    except Exception:
        return float("nan")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    try:
        return float(np.corrcoef(a[mask], b[mask])[0, 1])
    except Exception:
        return float("nan")


def _compute_alerts(
    stats: Dict[str, Any],
    cfg_alerts: Mapping[str, Any],
    resonance: Optional[Dict[str, Any]] = None,
    culture_stats: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> tuple[List[str], List[Dict[str, Any]]]:
    alerts: List[str] = []
    details: List[Dict[str, Any]] = []
    thresholds = dict(cfg_alerts or {})

    max_abs_valence_mean = float(thresholds.get("max_abs_valence_mean", 0.6))
    valence_mean = stats.get("valence_mean") if stats else None
    if valence_mean is not None and math.isfinite(valence_mean):
        if abs(valence_mean) > max_abs_valence_mean:
            alerts.append(f"valence_mean_out_of_range:{valence_mean:.3f}")
            details.append(
                {
                    "kind": "valence_mean_out_of_range",
                    "value": float(valence_mean),
                    "threshold": max_abs_valence_mean,
                }
            )

    min_corr_rho = float(thresholds.get("min_corr_rho_I", 0.2))
    rho_corr = stats.get("rho_I_corr") if stats else None
    if rho_corr is not None and math.isfinite(rho_corr) and rho_corr < min_corr_rho:
        alerts.append(f"low_rho_I_correlation:{rho_corr:.3f}")
        details.append(
            {
                "kind": "low_rho_I_correlation",
                "value": float(rho_corr),
                "threshold": min_corr_rho,
            }
        )

    min_corr_arousal = float(thresholds.get("min_corr_arousal_I", 0.1))
    arousal_corr = stats.get("arousal_I_corr") if stats else None
    if (
        arousal_corr is not None
        and math.isfinite(arousal_corr)
        and arousal_corr < min_corr_arousal
    ):
        alerts.append(f"low_arousal_I_correlation:{arousal_corr:.3f}")
        details.append(
            {
                "kind": "low_arousal_I_correlation",
                "value": float(arousal_corr),
                "threshold": min_corr_arousal,
            }
        )

    if resonance:
        min_corr_rho_rho = float(thresholds.get("min_corr_rho_rho", 0.2))
        max_allowed_lag = float(thresholds.get("max_allowed_lag", 8.0))
        min_samples = float(thresholds.get("min_resonance_samples", 0))
        for pair in resonance.get("pairs", []):
            corr = pair.get("rho_corr")
            if corr is not None and math.isfinite(corr) and corr < min_corr_rho_rho:
                alerts.append(f"low_rho_rho_correlation:{corr:.3f}")
                details.append(
                    {
                        "kind": "low_rho_rho_correlation",
                        "value": float(corr),
                        "threshold": min_corr_rho_rho,
                        "pair": pair.get("agents"),
                    }
                )
            lag = pair.get("rho_cross_corr_lag_refined")
            if lag is None or not math.isfinite(lag):
                lag = pair.get("rho_cross_corr_lag")
            if lag is not None and math.isfinite(lag) and abs(lag) > max_allowed_lag:
                alerts.append(f"excessive_resonance_lag:{lag:.2f}")
                details.append(
                    {
                        "kind": "excessive_resonance_lag",
                        "value": float(lag),
                        "threshold": max_allowed_lag,
                        "pair": pair.get("agents"),
                    }
                )
            n_eff = pair.get("n_eff")
            if n_eff is not None and math.isfinite(n_eff) and n_eff < min_samples:
                alerts.append(f"low_resonance_samples:{n_eff:.0f}")
                details.append(
                    {
                        "kind": "low_resonance_samples",
                        "value": float(n_eff),
                        "threshold": min_samples,
                        "pair": pair.get("agents"),
                    }
                )
    if culture_stats:
        min_culture_samples = float(thresholds.get("min_culture_samples", 0))
        max_abs_culture_valence = float(
            thresholds.get("max_abs_culture_valence_mean", thresholds.get("max_abs_valence_mean", 0.6))
        )
        min_culture_rho = float(thresholds.get("min_culture_rho_mean", thresholds.get("min_corr_rho_I", 0.2)))
        for tag, stats_tag in culture_stats.items():
            if not isinstance(stats_tag, Mapping):
                continue
            count = stats_tag.get("count")
            try:
                count_val = float(count)
            except Exception:
                count_val = 0.0
            if count_val < min_culture_samples:
                continue
            valence_mean = stats_tag.get("mean_valence")
            if valence_mean is not None and math.isfinite(valence_mean) and abs(valence_mean) > max_abs_culture_valence:
                alerts.append(f"culture.high_abs_valence:{tag}")
                details.append(
                    {
                        "kind": "culture.high_abs_valence",
                        "tag": tag,
                        "value": float(valence_mean),
                        "threshold": max_abs_culture_valence,
                    }
                )
            rho_mean = stats_tag.get("mean_rho")
            if rho_mean is not None and math.isfinite(rho_mean) and rho_mean < min_culture_rho:
                alerts.append(f"culture.low_rho:{tag}")
                details.append(
                    {
                        "kind": "culture.low_rho",
                        "tag": tag,
                        "value": float(rho_mean),
                        "threshold": min_culture_rho,
                    }
                )
    return alerts, details


def _extract_run_seed(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "run.seed":
                data = row.get("data") or {}
                seed = data.get("seed")
                if seed is not None:
                    try:
                        return int(seed)
                    except Exception:
                        return None
    except Exception:
        return None
    return None


def _resolve_resonance_logs(specs: Iterable[str]) -> List[Tuple[str, Path]]:
    resolved: List[Tuple[str, Path]] = []
    for entry in specs or []:
        matches = glob.glob(entry) if isinstance(entry, str) else []
        paths = matches or [entry]
        for item in paths:
            if isinstance(item, (list, tuple)):
                continue
            token = str(item)
            if "=" in token:
                label, path_str = token.split("=", 1)
                label = label.strip()
                path = Path(path_str.strip())
            else:
                path = Path(token)
                label = path.stem
            resolved.append((label or path.stem, path))
    return resolved


def _summarize_memory_reference(log_path: Path | None) -> Optional[Dict[str, Any]]:
    if not log_path or not log_path.exists():
        return None
    asked = 0
    high_fidelity = 0
    repair_used = 0
    repair_success = 0
    disputed = 0
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asked += 1
                fidelity = float(row.get("fidelity", 0.0) or 0.0)
                mode = str(row.get("mode", "recall"))
                if fidelity >= 0.65:
                    high_fidelity += 1
                if mode == "mend":
                    repair_used += 1
                    if bool(row.get("repair_success")):
                        repair_success += 1
                if row.get("disputed"):
                    disputed += 1
    except Exception:
        return None
    if asked == 0:
        return None
    summary: Dict[str, Any] = {
        "asked": asked,
        "high_fidelity": high_fidelity,
        "repair_used": repair_used,
        "repair_success": repair_success,
        "disputed": disputed,
    }
    summary["repair_success_rate"] = (
        repair_success / max(1, repair_used) if repair_used > 0 else None
    )
    return summary


def _summarize_culture_stats(log_path: Path | None) -> Optional[Dict[str, Dict[str, float]]]:
    if not log_path or not log_path.exists():
        return None
    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"valence": [], "arousal": [], "rho": [], "politeness": [], "intimacy": []}
    )

    def _to_float(value: object) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            mood = row.get("mood") if isinstance(row.get("mood"), dict) else None
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else None
            tag = row.get("culture_tag") or row.get("tag") or "unknown"
            valence_val = row.get("valence")
            if valence_val is None and mood:
                valence_val = mood.get("valence")
            arousal_val = row.get("arousal")
            if arousal_val is None and mood:
                arousal_val = mood.get("arousal")
            valence = _to_float(valence_val)
            arousal = _to_float(arousal_val)
            if valence is None or arousal is None:
                continue
            rho_val = row.get("rho")
            if rho_val is None and metrics:
                rho_val = metrics.get("rho") or metrics.get("R")
            rho = _to_float(rho_val)
            if rho is None:
                rho = 0.0
            buckets[tag]["valence"].append(valence)
            buckets[tag]["arousal"].append(arousal)
            buckets[tag]["rho"].append(rho)
            politeness_val = row.get("politeness")
            if politeness_val is None and metrics:
                politeness_val = metrics.get("politeness")
            politeness = _to_float(politeness_val)
            if politeness is not None:
                buckets[tag]["politeness"].append(politeness)
            intimacy_val = row.get("intimacy")
            if intimacy_val is None and metrics:
                intimacy_val = metrics.get("intimacy")
            intimacy = _to_float(intimacy_val)
            if intimacy is not None:
                buckets[tag]["intimacy"].append(intimacy)
    except Exception:
        return None

    summary: Dict[str, Dict[str, float]] = {}
    for tag, data in buckets.items():
        if not data["valence"]:
            continue
        entry: Dict[str, float] = {
            "count": float(len(data["valence"])),
            "mean_valence": mean(data["valence"]),
            "mean_arousal": mean(data["arousal"]),
            "mean_rho": mean(data["rho"]),
        }
        if data["politeness"]:
            entry["mean_politeness"] = mean(data["politeness"])
        if data["intimacy"]:
            entry["mean_intimacy"] = mean(data["intimacy"])
        summary[tag] = entry
    return summary or None




def _generate_telemetry_section(
    report: Dict[str, Any],
    telemetry_log: Path,
    plots_dir: Path,
    md_path: Path,
    *,
    culture_feedback_enabled: bool = False,
) -> None:
    field_state = _summarize_field_telemetry(telemetry_log)
    if field_state:
        report["field_state"] = field_state
    delta_series, jerk_series = _extract_field_series(telemetry_log)
    assoc_summary, assoc_plots = _summarize_assoc_kernel(
        Path("logs/canary/assoc_kernel.jsonl"),
        plots_dir,
        delta_series,
        jerk_series,
    )
    if assoc_summary:
        report["assoc_kernel"] = assoc_summary
        summary_path = Path("reports/nightly/assoc_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(assoc_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts_assoc = report.setdefault("artifacts", {}).setdefault("assoc_kernel", {})
        artifacts_assoc["summary"] = str(summary_path)
        report["assoc_summary_path"] = str(summary_path)
        report["plots"] = report.get("plots", {})
        for key, plot_path in assoc_plots.items():
            artifacts_assoc[key] = str(plot_path)
            report["plots"][key] = str(plot_path)

    vision_summary = _summarize_vision_metrics(telemetry_log)
    if vision_summary:
        report["vision_snapshot"] = vision_summary
    report["config_snapshot"] = cfg_dict
    sleep_snapshot_path, sleep_snapshot_payload, sleep_snapshot_warning = _resolve_inner_os_sleep_snapshot(cfg_dict)
    if sleep_snapshot_warning:
        report.setdefault("warnings", []).append(sleep_snapshot_warning)
    if sleep_snapshot_path and isinstance(sleep_snapshot_payload, dict):
        report["inner_os_sleep_snapshot_path"] = sleep_snapshot_path
        snapshot = sleep_snapshot_payload.get("snapshot") or {}
        if isinstance(snapshot, dict):
            if snapshot.get("mode"):
                report["inner_os_sleep_mode"] = snapshot["mode"]
            if snapshot.get("memory_class_focus"):
                report["inner_os_sleep_memory_class_focus"] = str(snapshot["memory_class_focus"])
            if snapshot.get("agenda_focus"):
                report["inner_os_sleep_agenda_focus"] = str(snapshot["agenda_focus"])
            if snapshot.get("agenda_bias") is not None:
                report["inner_os_sleep_agenda_bias"] = float(snapshot["agenda_bias"] or 0.0)
            if snapshot.get("agenda_reason"):
                report["inner_os_sleep_agenda_reason"] = str(snapshot["agenda_reason"])
            if snapshot.get("agenda_window_focus"):
                report["inner_os_sleep_agenda_window_focus"] = str(snapshot["agenda_window_focus"])
            if snapshot.get("agenda_window_bias") is not None:
                report["inner_os_sleep_agenda_window_bias"] = float(snapshot["agenda_window_bias"] or 0.0)
            if snapshot.get("agenda_window_reason"):
                report["inner_os_sleep_agenda_window_reason"] = str(snapshot["agenda_window_reason"])
            if snapshot.get("agenda_window_carry_target"):
                report["inner_os_sleep_agenda_window_carry_target"] = str(snapshot["agenda_window_carry_target"])
            if snapshot.get("learning_mode_focus"):
                report["inner_os_sleep_learning_mode_focus"] = str(snapshot["learning_mode_focus"])
            if snapshot.get("learning_mode_carry_bias") is not None:
                report["inner_os_sleep_learning_mode_carry_bias"] = float(snapshot["learning_mode_carry_bias"] or 0.0)
            if snapshot.get("social_experiment_focus"):
                report["inner_os_sleep_social_experiment_focus"] = str(snapshot["social_experiment_focus"])
            if snapshot.get("social_experiment_carry_bias") is not None:
                report["inner_os_sleep_social_experiment_carry_bias"] = float(snapshot["social_experiment_carry_bias"] or 0.0)
            if snapshot.get("commitment_target_focus"):
                report["inner_os_sleep_commitment_target_focus"] = str(snapshot["commitment_target_focus"])
            if snapshot.get("commitment_state_focus"):
                report["inner_os_sleep_commitment_state_focus"] = str(snapshot["commitment_state_focus"])
            if snapshot.get("commitment_carry_bias") is not None:
                report["inner_os_sleep_commitment_carry_bias"] = float(snapshot["commitment_carry_bias"] or 0.0)
            if snapshot.get("commitment_followup_focus"):
                report["inner_os_sleep_commitment_followup_focus"] = str(snapshot["commitment_followup_focus"])
            if snapshot.get("commitment_mode_focus"):
                report["inner_os_sleep_commitment_mode_focus"] = str(snapshot["commitment_mode_focus"])
            if snapshot.get("commitment_carry_reason"):
                report["inner_os_sleep_commitment_carry_reason"] = str(snapshot["commitment_carry_reason"])
            if snapshot.get("terrain_reweighting_bias") is not None:
                report["inner_os_sleep_terrain_reweighting_bias"] = float(snapshot["terrain_reweighting_bias"] or 0.0)
            if snapshot.get("insight_class_focus"):
                report["inner_os_sleep_insight_class_focus"] = str(snapshot["insight_class_focus"])
            if snapshot.get("insight_reframing_bias") is not None:
                report["inner_os_sleep_insight_reframing_bias"] = float(snapshot["insight_reframing_bias"] or 0.0)
            if snapshot.get("association_reweighting_bias") is not None:
                report["inner_os_sleep_association_reweighting_bias"] = float(snapshot["association_reweighting_bias"] or 0.0)
            if snapshot.get("association_reweighting_focus"):
                report["inner_os_sleep_association_reweighting_focus"] = str(snapshot["association_reweighting_focus"])
            if snapshot.get("association_reweighting_reason"):
                report["inner_os_sleep_association_reweighting_reason"] = str(snapshot["association_reweighting_reason"])
            if snapshot.get("insight_terrain_shape_bias") is not None:
                report["inner_os_sleep_insight_terrain_shape_bias"] = float(snapshot["insight_terrain_shape_bias"] or 0.0)
            if snapshot.get("insight_terrain_shape_reason"):
                report["inner_os_sleep_insight_terrain_shape_reason"] = str(snapshot["insight_terrain_shape_reason"])
            if snapshot.get("insight_terrain_shape_target"):
                report["inner_os_sleep_insight_terrain_shape_target"] = str(snapshot["insight_terrain_shape_target"])
            if snapshot.get("insight_anchor_center") is not None:
                report["inner_os_sleep_insight_anchor_center"] = list(snapshot["insight_anchor_center"] or [])
            if snapshot.get("insight_anchor_dispersion") is not None:
                report["inner_os_sleep_insight_anchor_dispersion"] = float(snapshot["insight_anchor_dispersion"] or 0.0)
            if snapshot.get("temperament_focus"):
                report["inner_os_sleep_temperament_focus"] = str(snapshot["temperament_focus"])
            if snapshot.get("temperament_forward_bias") is not None:
                report["inner_os_sleep_temperament_forward_bias"] = float(snapshot["temperament_forward_bias"] or 0.0)
            if snapshot.get("temperament_guard_bias") is not None:
                report["inner_os_sleep_temperament_guard_bias"] = float(snapshot["temperament_guard_bias"] or 0.0)
            if snapshot.get("temperament_bond_bias") is not None:
                report["inner_os_sleep_temperament_bond_bias"] = float(snapshot["temperament_bond_bias"] or 0.0)
            if snapshot.get("temperament_recovery_bias") is not None:
                report["inner_os_sleep_temperament_recovery_bias"] = float(snapshot["temperament_recovery_bias"] or 0.0)
            if snapshot.get("homeostasis_budget_focus"):
                report["inner_os_sleep_homeostasis_budget_focus"] = str(snapshot["homeostasis_budget_focus"])
            if snapshot.get("homeostasis_budget_bias") is not None:
                report["inner_os_sleep_homeostasis_budget_bias"] = float(snapshot["homeostasis_budget_bias"] or 0.0)
            if snapshot.get("body_homeostasis_focus"):
                report["inner_os_sleep_body_homeostasis_focus"] = str(snapshot["body_homeostasis_focus"])
            if snapshot.get("body_homeostasis_carry_bias") is not None:
                report["inner_os_sleep_body_homeostasis_carry_bias"] = float(snapshot["body_homeostasis_carry_bias"] or 0.0)
            if snapshot.get("relational_continuity_focus"):
                report["inner_os_sleep_relational_continuity_focus"] = str(snapshot["relational_continuity_focus"])
            if snapshot.get("relational_continuity_carry_bias") is not None:
                report["inner_os_sleep_relational_continuity_carry_bias"] = float(snapshot["relational_continuity_carry_bias"] or 0.0)
            if snapshot.get("group_thread_focus"):
                report["inner_os_sleep_group_thread_focus"] = str(snapshot["group_thread_focus"])
            if snapshot.get("group_thread_carry_bias") is not None:
                report["inner_os_sleep_group_thread_carry_bias"] = float(snapshot["group_thread_carry_bias"] or 0.0)
            if snapshot.get("autobiographical_thread_mode"):
                report["inner_os_sleep_autobiographical_thread_mode"] = str(snapshot["autobiographical_thread_mode"])
            if snapshot.get("autobiographical_thread_anchor"):
                report["inner_os_sleep_autobiographical_thread_anchor"] = str(snapshot["autobiographical_thread_anchor"])
            if snapshot.get("autobiographical_thread_focus"):
                report["inner_os_sleep_autobiographical_thread_focus"] = str(snapshot["autobiographical_thread_focus"])
            if snapshot.get("autobiographical_thread_strength") is not None:
                report["inner_os_sleep_autobiographical_thread_strength"] = float(snapshot["autobiographical_thread_strength"] or 0.0)
            if snapshot.get("temporal_membrane_focus"):
                report["inner_os_sleep_temporal_membrane_focus"] = str(snapshot["temporal_membrane_focus"])
            if snapshot.get("temporal_timeline_bias") is not None:
                report["inner_os_sleep_temporal_timeline_bias"] = float(snapshot["temporal_timeline_bias"] or 0.0)
            if snapshot.get("temporal_reentry_bias") is not None:
                report["inner_os_sleep_temporal_reentry_bias"] = float(snapshot["temporal_reentry_bias"] or 0.0)
            if snapshot.get("temporal_supersession_bias") is not None:
                report["inner_os_sleep_temporal_supersession_bias"] = float(snapshot["temporal_supersession_bias"] or 0.0)
            if snapshot.get("temporal_continuity_bias") is not None:
                report["inner_os_sleep_temporal_continuity_bias"] = float(snapshot["temporal_continuity_bias"] or 0.0)
            if snapshot.get("temporal_relation_reentry_bias") is not None:
                report["inner_os_sleep_temporal_relation_reentry_bias"] = float(snapshot["temporal_relation_reentry_bias"] or 0.0)
            if snapshot.get("expressive_style_focus"):
                report["inner_os_sleep_expressive_style_focus"] = str(snapshot["expressive_style_focus"])
            if snapshot.get("expressive_style_carry_bias") is not None:
                report["inner_os_sleep_expressive_style_carry_bias"] = float(snapshot["expressive_style_carry_bias"] or 0.0)
            if snapshot.get("expressive_style_history_focus"):
                report["inner_os_sleep_expressive_style_history_focus"] = str(snapshot["expressive_style_history_focus"])
            if snapshot.get("expressive_style_history_bias") is not None:
                report["inner_os_sleep_expressive_style_history_bias"] = float(snapshot["expressive_style_history_bias"] or 0.0)
            if snapshot.get("banter_style_focus"):
                report["inner_os_sleep_banter_style_focus"] = str(snapshot["banter_style_focus"])
            if snapshot.get("lexical_variation_carry_bias") is not None:
                report["inner_os_sleep_lexical_variation_carry_bias"] = float(snapshot["lexical_variation_carry_bias"] or 0.0)
    daily_carry_summary = DailyCarrySummaryBuilder().build(report).to_dict()
    if daily_carry_summary.get("same_turn_focus") or daily_carry_summary.get("active_carry_channels"):
        report["inner_os_daily_carry_summary"] = daily_carry_summary
        temporal_alignment = dict(daily_carry_summary.get("temporal_alignment") or {})
        if temporal_alignment:
            report["inner_os_temporal_alignment"] = temporal_alignment
    nightly_id = report.setdefault("nightly_id", dt.datetime.utcnow().strftime("N-%Y%m%d-%H%M%S"))
    tuning = _derive_tuning_suggestion(field_state, cfg_dict)
    if tuning:
        report["tuning_suggestion"] = tuning

    emotion_cfg = cfg_dict.get("emotion", {}) if isinstance(cfg_dict, dict) else {}
    alerts_cfg = cfg_dict.get("alerts", {}) if isinstance(cfg_dict, dict) else {}
    resonance_cfg = cfg_dict.get("resonance", {}) if isinstance(cfg_dict, dict) else {}
    culture_cfg_snapshot = cfg_dict.get("culture", {}) if isinstance(cfg_dict, dict) else {}

    affective_log_path = Path(emotion_cfg.get("affective_log_path", "memory/affective_log.jsonl"))
    moment_log_candidate = cfg_dict.get("moment_log_path") if isinstance(cfg_dict, dict) else None
    if isinstance(moment_log_candidate, str) and moment_log_candidate.strip():
        culture_log_path = Path(moment_log_candidate)
    else:
        culture_log_path = affective_log_path
    run_seed = _extract_run_seed(telemetry_log)
    if run_seed is not None:
        report["run_seed"] = run_seed

    memory_ref_cfg = cfg_dict.get("memory_reference") if isinstance(cfg_dict, dict) else {}
    log_path_str = memory_ref_cfg.get("log_path") if isinstance(memory_ref_cfg, dict) else None
    memory_ref_path = Path(log_path_str) if log_path_str else MEMORY_REF_LOG_PATH
    memory_ref_stats = _summarize_memory_reference(memory_ref_path)
    if memory_ref_stats:
        report["memory_match"] = memory_ref_stats
        if memory_ref_stats.get("repair_success_rate") is not None:
            report["repair_success_rate"] = memory_ref_stats["repair_success_rate"]

    report["plots"] = report.get("plots", {})
    plot_info: Dict[str, Path] = {}
    plot_error: Optional[str] = None
    if assoc_plots:
        for key, path in assoc_plots.items():
            plot_info.setdefault(key, path)

    if vision_summary:
        vision_plot_paths = _render_vision_plots(vision_summary, plots_dir)
        if vision_plot_paths:
            vision_artifacts = report.setdefault("artifacts", {}).setdefault("vision_plots", {})
            for key, path in vision_plot_paths.items():
                report["plots"][key] = str(path)
                plot_info.setdefault(key, path)
                vision_artifacts[key] = str(path)

    expected_plot_paths = {
        "ignition_timeseries": plots_dir / "ignition_timeseries.png",
        "rho_vs_I_scatter": plots_dir / "rho_vs_I_scatter.png",
        "affective_map": plots_dir / "affective_map.png",
    }
    try:
        ignition_output = ignition_plots.render_plots(telemetry_log, plots_dir)
        normalized: Dict[str, str] = {}
        if "timeseries" in ignition_output:
            normalized["ignition_timeseries"] = str(ignition_output["timeseries"])
        if "scatter" in ignition_output:
            normalized["rho_vs_I_scatter"] = str(ignition_output["scatter"])
        report.setdefault("artifacts", {})["field_plots"] = normalized
        report["plots"].update(normalized)
    except Exception as exc:
        plot_error = str(exc)
        report.setdefault("warnings", []).append(f"plot_failed ({telemetry_log}): {plot_error}")
    finally:
        fallback = {k: str(v) for k, v in expected_plot_paths.items()}
        report["plots"].update({k: report["plots"].get(k, path) for k, path in fallback.items()})
        report.setdefault("artifacts", {}).setdefault("field_plots", {}).update(
            {k: report["plots"][k] for k in fallback}
        )

    mem_path = plots_dir / "memory_graph.png"
    try:
        plot_memory_graph.render_memory_graph(telemetry_log, mem_path)
    except Exception as exc:
        report.setdefault("warnings", []).append(f"memory_graph_failed ({telemetry_log}): {exc}")
    finally:
        report.setdefault("artifacts", {})["memory_graph"] = str(mem_path)
        report["plots"]["memory_graph"] = str(mem_path)
        plot_info.setdefault("memory_graph", mem_path)

    affective_plot = expected_plot_paths["affective_map"]
    affective_stats_path = plots_dir / "affective_stats.json"
    try:
        plot_affective_map.render_affective_map(
            affective_log_path,
            affective_plot,
            json_out_path=affective_stats_path,
        )
        if affective_stats_path.exists():
            report["affective_stats_path"] = str(affective_stats_path)
            try:
                report["affective_stats"] = json.loads(affective_stats_path.read_text(encoding="utf-8"))
            except Exception as exc:
                report.setdefault("warnings", []).append(
                    f"affective_stats_failed ({affective_stats_path}): {exc}"
                )
    except Exception as exc:
        report.setdefault("warnings", []).append(f"affective_map_failed ({affective_log_path}): {exc}")
    finally:
        report.setdefault("artifacts", {}).setdefault("field_plots", {})["affective_map"] = str(affective_plot)
        report["plots"]["affective_map"] = str(affective_plot)
        plot_info.setdefault("affective_map", affective_plot)
        if affective_stats_path.exists():
            report.setdefault("artifacts", {}).setdefault("stats", {})["affective"] = str(affective_stats_path)

    min_culture_samples = int(float(alerts_cfg.get("min_culture_samples", 0))) if isinstance(alerts_cfg, dict) else 0
    culture_stats = _summarize_culture_stats(culture_log_path)
    culture_for_plot: Dict[str, Dict[str, float]] = {}
    if culture_stats:
        report["culture_stats"] = culture_stats
        for tag, stats_dict in culture_stats.items():
            try:
                count_val = float(stats_dict.get("count", 0.0))
            except Exception:
                count_val = 0.0
            if count_val >= max(0, min_culture_samples):
                culture_for_plot[tag] = stats_dict
        history_path = Path("reports/culture_history.jsonl")
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a", encoding="utf-8") as handle:
                for tag, stats_dict in culture_stats.items():
                    record = {
                        "ts": float(report.get("ts", time.time())),
                        "tag": tag,
                        "mean_valence": stats_dict.get("mean_valence"),
                        "mean_rho": stats_dict.get("mean_rho"),
                        "count": stats_dict.get("count"),
                    }
                    json.dump(record, handle, ensure_ascii=False)
                    handle.write("\n")
            report["culture_history_path"] = str(history_path)
        except Exception as exc:
            report.setdefault("warnings", []).append(f"culture_history_failed: {exc}")

        if culture_for_plot:
            culture_plot = plots_dir / "culture_resonance.png"
            try:
                plot_culture_resonance.render_culture_resonance(culture_for_plot, culture_plot)
            except Exception as exc:
                report.setdefault("warnings", []).append(f"culture_resonance_failed: {exc}")
            else:
                if culture_plot.exists():
                    report.setdefault("plots", {})["culture_resonance"] = str(culture_plot)
                    plot_info.setdefault("culture_resonance", culture_plot)
                    report.setdefault("artifacts", {}).setdefault("field_plots", {})[
                        "culture_resonance"
                    ] = str(culture_plot)
        if report.get("culture_history_path"):
            trend_plot = plots_dir / "culture_trend.png"
            try:
                plot_culture_trend.render_culture_trend(
                    report["culture_history_path"],
                    str(trend_plot),
                    min_count=max(0, min_culture_samples),
                )
            except Exception as exc:
                report.setdefault("warnings", []).append(f"culture_trend_failed: {exc}")
            else:
                if trend_plot.exists():
                    report.setdefault("plots", {})["culture_trend"] = str(trend_plot)
                    report.setdefault("artifacts", {}).setdefault("field_plots", {})[
                        "culture_trend"
                    ] = str(trend_plot)

    resonance_metrics_result: Optional[Dict[str, Any]] = None
    resonance_summary: Optional[Dict[str, Any]] = None
    history_path = Path("reports/resonance_history.jsonl")
    resonance_logs = _resolve_resonance_logs(resonance_cfg.get("logs", []))
    if len(resonance_logs) >= 2:
        try:
            resonance_metrics_result = resonance_metrics.compute_resonance_metrics(
                resonance_logs,
                resample_ms=resonance_cfg.get("resample_ms"),
                zscore=bool(resonance_cfg.get("zscore", False)),
                detrend=bool(resonance_cfg.get("detrend", False)),
                window=resonance_cfg.get("window", "none"),
                alpha=float(resonance_cfg.get("alpha", 0.0)),
                beta=float(resonance_cfg.get("beta", 0.0)),
                return_series=True,
            )
            report["resonance"] = resonance_metrics_result
            created = plot_resonance.render_resonance_plots(resonance_metrics_result, plots_dir)
            if created:
                report.setdefault("plots", {}).setdefault("resonance", created)
                for idx, path in enumerate(created):
                    plot_info.setdefault(f"resonance_{idx}", Path(path))
            pairs = resonance_metrics_result.get("pairs") or []
            if pairs:
                best = max(
                    pairs,
                    key=lambda p: (
                        p.get("objective")
                        if p.get("objective") is not None
                        else (p.get("rho_corr") or float("-inf"))
                    ),
                )
                resonance_summary = {
                    "agents": best.get("agents"),
                    "corr": best.get("rho_corr"),
                    "lag": best.get("rho_cross_corr_lag_refined", best.get("rho_cross_corr_lag")),
                    "energy": best.get("energy"),
                    "objective": best.get("objective"),
                    "n_eff": best.get("n_eff"),
                    "partial_corr": best.get("partial_corr"),
                    "granger": best.get("granger"),
                }
                resonance_metrics_result["summary"] = resonance_summary
                resonance_metrics_result["partial_corr"] = resonance_summary.get("partial_corr")
                resonance_metrics_result["granger"] = resonance_summary.get("granger")
                try:
                    history_path.parent.mkdir(parents=True, exist_ok=True)
                    history_entry = {
                        "ts": report.get("ts", time.time()),
                        "k_res": resonance_cfg.get("k_res"),
                        **{k: v for k, v in resonance_summary.items() if k != "agents"},
                    }
                    with history_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(history_entry, ensure_ascii=False) + "\n")
                    report["resonance_history_path"] = str(history_path)
                    report.setdefault("artifacts", {}).setdefault("stats", {})[
                        "resonance_history"
                    ] = str(history_path)
                    objective_plot = plots_dir / "resonance_objective.png"
                    plot_resonance.plot_objective_history(history_path, objective_plot)
                    if objective_plot.exists():
                        report["plots"]["resonance_objective"] = str(objective_plot)
                        plot_info.setdefault("resonance_objective", objective_plot)
                    trace_path = Path("reports/resonance_bayes_trace.jsonl")
                    trace_plot = plots_dir / "resonance_bayes_trace.png"
                    try:
                        plot_resonance.plot_bayes_trace(trace_path, trace_plot)
                    except Exception as trace_exc:
                        report.setdefault("warnings", []).append(f"resonance_bayes_trace_failed: {trace_exc}")
                    else:
                        if trace_plot.exists():
                            if trace_path.exists():
                                report["resonance_bayes_trace_path"] = str(trace_path)
                            report.setdefault("plots", {})["resonance_bayes_trace"] = str(trace_plot)
                            plot_info.setdefault("resonance_bayes_trace", trace_plot)
                            report.setdefault("artifacts", {}).setdefault("field_plots", {})["resonance_bayes_trace"] = str(trace_plot)
                except Exception as exc:
                    report.setdefault("warnings", []).append(f"resonance_history_failed: {exc}")
        except Exception as exc:
            report.setdefault("warnings", []).append(f"resonance_failed: {exc}")

    default_tag = "default"
    baseline_politeness = 0.5
    baseline_intimacy = 0.5
    if isinstance(culture_cfg_snapshot, dict):
        default_tag = culture_cfg_snapshot.get("tag", default_tag)
        try:
            baseline_politeness = float(culture_cfg_snapshot.get("politeness", baseline_politeness))
        except (TypeError, ValueError):
            baseline_politeness = 0.5
        try:
            baseline_intimacy = float(culture_cfg_snapshot.get("intimacy", baseline_intimacy))
        except (TypeError, ValueError):
            baseline_intimacy = 0.5
    feedback_cfg = (
        culture_cfg_snapshot.get("feedback", {}) if isinstance(culture_cfg_snapshot, dict) else {}
    )
    vision_snapshot = report.get("vision_snapshot") if isinstance(report.get("vision_snapshot"), dict) else {}
    vision_counts = vision_snapshot.get("counts_by_kind") if isinstance(vision_snapshot, dict) else {}
    try:
        vision_events_total = float(vision_snapshot.get("events", 0.0))
    except (TypeError, ValueError):
        vision_events_total = 0.0
    try:
        vision_detections_total = float(vision_snapshot.get("detections_total", 0.0))
    except (TypeError, ValueError):
        vision_detections_total = 0.0
    vision_normalisers = {
        "events": vision_events_total,
        "detections": vision_detections_total,
    }
    if isinstance(feedback_cfg, dict):
        vision_coeffs = feedback_cfg.get("vision_coefficients") if isinstance(feedback_cfg.get("vision_coefficients"), dict) else {}
        if feedback_cfg.get("enabled") and culture_feedback_enabled:
            politeness_before = baseline_politeness
            intimacy_before = baseline_intimacy
            corr_high = float(feedback_cfg.get("corr_high", 0.65))
            corr_low = float(feedback_cfg.get("corr_low", 0.35))
            delta = float(feedback_cfg.get("delta", 0.02))
            clamp_min = float(feedback_cfg.get("clamp_min", 0.0))
            clamp_max = float(feedback_cfg.get("clamp_max", 1.0))
            corr_value: Optional[float] = None
            corr_source = None
            if resonance_summary and resonance_summary.get("corr") is not None:
                try:
                    corr_candidate = float(resonance_summary.get("corr"))
                except (TypeError, ValueError):
                    corr_candidate = None
                if corr_candidate is not None and math.isfinite(corr_candidate):
                    corr_value = corr_candidate
                    corr_source = "resonance_summary"
            if corr_value is None and culture_stats:
                tag_stats = culture_stats.get(default_tag)
                if tag_stats and tag_stats.get("mean_rho") is not None:
                    try:
                        corr_candidate = float(tag_stats.get("mean_rho"))
                    except (TypeError, ValueError):
                        corr_candidate = None
                    if corr_candidate is not None and math.isfinite(corr_candidate):
                        corr_value = corr_candidate
                        corr_source = f"culture_stats[{default_tag}]"
            reason = "insufficient_data"
            politeness_after = politeness_before
            intimacy_after = intimacy_before
            if corr_value is not None:
                if corr_value >= corr_high:
                    politeness_after = min(clamp_max, politeness_before + delta)
                    reason = "resonance_high"
                elif corr_value <= corr_low:
                    politeness_after = max(clamp_min, politeness_before - delta)
                    reason = "resonance_low"
                else:
                    reason = "resonance_mid"
            vision_pol_delta = 0.0
            vision_int_delta = 0.0
            vision_details: List[Dict[str, Any]] = []
            if isinstance(vision_counts, dict) and vision_coeffs:
                for kind, coeff in vision_coeffs.items():
                    if not isinstance(coeff, Mapping):
                        continue
                    try:
                        count_val = float(vision_counts.get(kind, 0.0))
                    except (TypeError, ValueError):
                        continue
                    if count_val <= 0.0:
                        continue
                    normalize = str(coeff.get("normalize", "events")).lower()
                    try:
                        scale = float(coeff.get("scale", 1.0))
                    except (TypeError, ValueError):
                        scale = 1.0
                    if normalize == "detections":
                        base = vision_detections_total if vision_detections_total > 0 else 1.0
                    elif normalize in ("none", "raw"):
                        base = 1.0
                    else:
                        base = vision_events_total if vision_events_total > 0 else 1.0
                        normalize = "events"
                    effective = (count_val / base) * scale
                    pol_unit = float(coeff.get("politeness_delta", 0.0))
                    pol_delta = pol_unit * effective
                    intim_unit = float(coeff.get("intimacy_delta", 0.0))
                    int_delta = intim_unit * effective
                    max_abs = coeff.get("max_abs")
                    try:
                        max_abs_val = abs(float(max_abs)) if max_abs is not None else None
                    except (TypeError, ValueError):
                        max_abs_val = None
                    if max_abs_val:
                        pol_delta = max(-max_abs_val, min(max_abs_val, pol_delta))
                        int_delta = max(-max_abs_val, min(max_abs_val, int_delta))
                    vision_pol_delta += pol_delta
                    vision_int_delta += int_delta
                    detail: Dict[str, Any] = {
                        "kind": kind,
                        "count": count_val,
                        "effective": effective,
                        "normalize": normalize,
                    }
                    if pol_unit:
                        detail["politeness_delta"] = pol_delta
                    if intim_unit:
                        detail["intimacy_delta"] = int_delta
                    vision_details.append(detail)
            if vision_pol_delta != 0.0:
                politeness_after = min(clamp_max, max(clamp_min, politeness_after + vision_pol_delta))
            if vision_int_delta != 0.0:
                intimacy_after = min(clamp_max, max(clamp_min, intimacy_after + vision_int_delta))
            applied_delta = politeness_after - politeness_before
            applied_int_delta = intimacy_after - intimacy_before
            report["policy_feedback"] = {
                "enabled": True,
                "politeness_before": politeness_before,
                "politeness_after": politeness_after,
                "delta": applied_delta,
                "intimacy_before": intimacy_before,
                "intimacy_after": intimacy_after,
                "intimacy_delta": applied_int_delta,
                "corr": corr_value,
                "corr_source": corr_source,
                "reason": reason,
                "vision_adjustment": {
                    "politeness_delta": vision_pol_delta,
                    "intimacy_delta": vision_int_delta,
                    "details": vision_details,
                    "counts_by_kind": dict(vision_counts or {}),
                    "normalisers": vision_normalisers,
                },
            }
        elif feedback_cfg.get("enabled"):
            report["policy_feedback"] = {
                "enabled": False,
                "reason": "disabled_by_cli",
                "politeness_before": baseline_politeness,
                "politeness_after": baseline_politeness,
                "delta": 0.0,
                "intimacy_before": baseline_intimacy,
                "intimacy_after": baseline_intimacy,
                "intimacy_delta": 0.0,
                "corr": None,
                "corr_source": None,
                "vision_adjustment": {
                    "politeness_delta": 0.0,
                    "intimacy_delta": 0.0,
                    "details": [],
                    "counts_by_kind": dict(vision_counts or {}),
                    "normalisers": vision_normalisers,
                },
            }
        elif feedback_cfg:
            report["policy_feedback"] = {
                "enabled": False,
                "reason": "disabled_in_config",
                "politeness_before": baseline_politeness,
                "politeness_after": baseline_politeness,
                "delta": 0.0,
                "intimacy_before": baseline_intimacy,
                "intimacy_after": baseline_intimacy,
                "intimacy_delta": 0.0,
                "corr": None,
                "corr_source": None,
                "vision_adjustment": {
                    "politeness_delta": 0.0,
                    "intimacy_delta": 0.0,
                    "details": [],
                    "counts_by_kind": dict(vision_counts or {}),
                    "normalisers": vision_normalisers,
                },
            }

    policy_feedback = report.get("policy_feedback")
    if isinstance(policy_feedback, dict):
        policy_feedback.setdefault("reference_tag", default_tag)
        ref_stats = (culture_stats or {}).get(default_tag, {}) if culture_stats else {}
        policy_feedback.setdefault("reference_rho", ref_stats.get("mean_rho"))
        policy_feedback.setdefault("reference_valence", ref_stats.get("mean_valence"))
        policy_feedback.setdefault("alerts", list(report.get("alerts", [])))
        policy_feedback.setdefault("intimacy_before", baseline_intimacy)
        policy_feedback.setdefault("intimacy_after", baseline_intimacy)
        policy_feedback.setdefault("intimacy_delta", 0.0)
        policy_feedback.setdefault(
            "vision_adjustment",
            {
                "politeness_delta": 0.0,
                "intimacy_delta": 0.0,
                "details": [],
                "counts_by_kind": dict(vision_counts or {}),
                "normalisers": vision_normalisers,
            },
        )
        history_entry = {
            "ts": report.get("ts"),
            "enabled": bool(policy_feedback.get("enabled")),
            "politeness_before": policy_feedback.get("politeness_before"),
            "politeness_after": policy_feedback.get("politeness_after"),
            "delta": policy_feedback.get("delta"),
            "intimacy_before": policy_feedback.get("intimacy_before"),
            "intimacy_after": policy_feedback.get("intimacy_after"),
            "intimacy_delta": policy_feedback.get("intimacy_delta"),
            "corr": policy_feedback.get("corr"),
            "reason": policy_feedback.get("reason"),
            "reference_tag": policy_feedback.get("reference_tag"),
            "reference_rho": policy_feedback.get("reference_rho"),
            "reference_valence": policy_feedback.get("reference_valence"),
            "alerts": policy_feedback.get("alerts", []),
            "vision_adjustment": policy_feedback.get("vision_adjustment"),
        }
        history_path = Path("reports/policy_feedback_history.jsonl")
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(history_entry, ensure_ascii=False) + "\n")
            report["policy_feedback_history_path"] = str(history_path)
        except Exception as exc:
            report.setdefault("warnings", []).append(f"policy_feedback_history_failed: {exc}")

    pain_cfg = cfg_dict.get("pain_loop", {}) if isinstance(cfg_dict, dict) else {}
    try:
        max_events_cfg = pain_cfg.get("max_events_per_nightly")
        try:
            max_events_int = int(max_events_cfg) if max_events_cfg is not None else None
        except (TypeError, ValueError):
            max_events_int = None
        care_targets_cfg = pain_cfg.get("care_targets", [])
        if isinstance(care_targets_cfg, str):
            care_targets_list = [care_targets_cfg]
        elif isinstance(care_targets_cfg, Iterable):
            care_targets_list = [str(x) for x in care_targets_cfg if isinstance(x, (str, bytes))]
        else:
            care_targets_list = []
        comfort_gain_base = float(pain_cfg.get("comfort_gain_base", 0.15))
        protection_bias = float(pain_cfg.get("protection_bias", 0.3))
        growth_reward = float(pain_cfg.get("growth_reward", 0.2))
        patience_budget = float(pain_cfg.get("patience_budget", 0.5))
        ema_alpha_cfg = float(pain_cfg.get("ema_alpha", 0.4))
        l1_budget_cfg = float(pain_cfg.get("l1_budget", 0.8))
        l_inf_budget_cfg = float(pain_cfg.get("l_inf_budget", 0.3))
        total, forgiven, pain_stats = evaluate_and_forgive(
            nightly_id,
            base_threshold=float(pain_cfg.get("forgive_threshold", 0.35)),
            adaptive=bool(pain_cfg.get("adaptive", True)),
            min_threshold=float(pain_cfg.get("min_threshold", 0.2)),
            max_threshold=float(pain_cfg.get("max_threshold", 0.6)),
            quantile=float(pain_cfg.get("quantile", 0.35)),
            min_samples=int(pain_cfg.get("min_samples", 50)),
            severity_window=int(pain_cfg.get("severity_window", 200)),
            rotate_daily=bool(pain_cfg.get("rotate_daily", False)),
            max_events=max_events_int,
            care_targets=care_targets_list,
            comfort_gain_base=comfort_gain_base,
            protection_bias=protection_bias,
            growth_reward=growth_reward,
            patience_budget=patience_budget,
            replay_eval=bool(pain_cfg.get("replay_eval", False)),
        )
        policy_update = policy_update_from_forgiveness(
            nightly_id,
            base_threshold=float(pain_cfg.get("policy_threshold_base", 0.5)),
            empathy_gain_base=float(pain_cfg.get("a2a_empathy_gain_base", 0.1)),
            ema_alpha=ema_alpha_cfg,
            l1_budget=l1_budget_cfg,
            l_inf_budget=l_inf_budget_cfg,
            min_threshold=float(pain_cfg.get("min_threshold", 0.2)),
            max_threshold=float(pain_cfg.get("max_threshold", 0.6)),
            max_empathy=float(pain_cfg.get("max_empathy", 0.5)),
            events_detail=pain_stats.get("events_detail"),
        )
        unforgiven_total = max(total - forgiven, 0)
        events_detail = pain_stats.get("events_detail") or []
        value_shift = policy_update.get("value_shift", {}) if isinstance(policy_update, Mapping) else {}
        influence_agg: Dict[Tuple[str, bool], Dict[str, Any]] = {}
        for detail in events_detail:
            harmed = detail.get("harmed_values") or ["unspecified"]
            care_flag = bool(detail.get("care_target"))
            severity = float(detail.get("severity") or 0.0)
            for value_name in harmed:
                key = (value_name, care_flag)
                bucket = influence_agg.setdefault(
                    key,
                    {
                        "value": value_name,
                        "care_target": care_flag,
                        "severity_sum": 0.0,
                        "count": 0,
                        "delta_empathy_gain": float(value_shift.get("delta_empathy_gain", 0.0)),
                        "delta_policy_threshold": float(value_shift.get("delta_policy_threshold", 0.0)),
                    },
                )
                bucket["severity_sum"] += severity
                bucket["count"] += 1
        influence_top: List[Dict[str, Any]] = []
        if influence_agg:
            ordered = sorted(influence_agg.values(), key=lambda item: item["severity_sum"], reverse=True)
            for entry in ordered[:5]:
                influence_top.append(
                    {
                        "value": entry["value"],
                        "care_target": entry["care_target"],
                        "severity_sum": entry["severity_sum"],
                        "count": entry["count"],
                        "delta_empathy_gain": entry["delta_empathy_gain"],
                        "delta_policy_threshold": entry["delta_policy_threshold"],
                    }
                )
        care_mode_cfg = pain_cfg.get("care_mode", {}) if isinstance(pain_cfg, Mapping) else {}
        canary_enabled = bool(care_mode_cfg.get("enabled", True))
        canary_ratio = float(care_mode_cfg.get("canary_ratio", 0.0))
        canary_seed = int(care_mode_cfg.get("canary_seed", 227))
        budgets_cfg = care_mode_cfg.get("budgets", {}) if isinstance(care_mode_cfg, Mapping) else {}
        canary_l1 = float(budgets_cfg.get("l1", l1_budget_cfg))
        canary_l_inf = float(budgets_cfg.get("l_inf", l_inf_budget_cfg))

        care_candidates = [
            str(detail.get("target"))
            for detail in events_detail
            if detail.get("care_target") and detail.get("target")
        ]
        care_candidates = sorted(set(candidate for candidate in care_candidates if candidate))
        if canary_enabled and care_candidates:
            selected_ids = select_canary_ids(care_candidates, canary_ratio, canary_seed)
        else:
            selected_ids = set()
        care_canary = {
            "enabled": canary_enabled,
            "ratio": canary_ratio,
            "seed": canary_seed,
            "candidates": len(care_candidates),
            "selected": len(selected_ids),
            "selected_ids": sorted(selected_ids),
            "budgets": {"l1": canary_l1, "l_inf": canary_l_inf},
        }

        care_stats_raw = pain_stats.get("care_stats")
        care_stats = dict(care_stats_raw) if isinstance(care_stats_raw, Mapping) else {}
        if care_stats:
            care_stats["targets"] = care_stats.get("targets", {})
            care_stats["canary"] = {
                "applied": len(selected_ids),
                "candidates": len(care_candidates),
                "ratio": canary_ratio,
            }

        pain_loop_info = {
            "total": total,
            "forgiven": forgiven,
            "unforgiven_total": unforgiven_total,
            "forgive_rate": pain_stats.get("forgive_rate"),
            "used_forgive_threshold": pain_stats.get("forgive_threshold"),
            "breakdown": pain_stats.get("breakdown", {}),
            "care_stats": care_stats,
            "policy_update": policy_update,
            "ema_alpha": ema_alpha_cfg,
            "l1_budget": l1_budget_cfg,
            "l_inf_budget": l_inf_budget_cfg,
            "comfort_gain_base": comfort_gain_base,
            "protection_bias": protection_bias,
            "growth_reward": growth_reward,
            "patience_budget": patience_budget,
            "value_influence_top": influence_top,
            "care_canary": care_canary,
        }

        today = dt.datetime.utcnow().date()
        last_day = monthrange(today.year, today.month)[1]
        monthly_limit = int(pain_cfg.get("monthly_highlight_limit", 5))
        if today.day == last_day:
            try:
                monthly = generate_value_influence_highlights(
                    month=today.strftime("%Y-%m"),
                    limit=monthly_limit,
                    log_path=VALUE_INFLUENCE_LOG,
                    output_dir="reports/monthly",
                    write_files=True,
                )
                pain_loop_info["monthly_highlights"] = monthly.get("items", [])
                pain_loop_info["monthly_highlights_paths"] = monthly.get("paths")
            except Exception as exc:
                report.setdefault("warnings", []).append(f"value_influence_highlights_failed: {exc}")

        report["pain_loop"] = pain_loop_info
    except Exception as exc:
        report.setdefault("warnings", []).append(f"pain_loop_failed: {exc}")

    alerts, alert_details = _compute_alerts(
        field_state or {},
        alerts_cfg,
        resonance_metrics_result,
        culture_stats if isinstance(culture_stats, Mapping) else None,
    )
    report["alerts"] = alerts
    if alert_details:
        report["alerts_detail"] = alert_details

    _write_markdown_summary(report, md_path, plot_info, telemetry_log, plot_error)
    report["markdown_path"] = str(md_path)


def _render_culture_alerts_table(report: Mapping[str, Any]) -> List[str]:
    raw_alerts = [
        a for a in report.get("alerts", []) if isinstance(a, str) and a.startswith("culture.")
    ]
    if not raw_alerts:
        return []
    rows: List[tuple[str, str]] = []
    for alert in raw_alerts:
        try:
            kind_part, remainder = alert.split(":", 1)
        except ValueError:
            kind_part, remainder = alert, ""
        tag = remainder.split(":", 1)[0] if remainder else "unknown"
        kind = kind_part.replace("culture.", "")
        rows.append((kind, tag or "unknown"))
    rows.sort()
    lines = ["", "## Culture Alerts (summary)", "| alert | tag |", "|---|---|"]
    for kind, tag in rows:
        lines.append(f"| {kind} | {tag} |")
    lines.append("")
    return lines


def _render_culture_narrative(
    stats: Mapping[str, Mapping[str, Any]],
    *,
    top_k: int = 3,
) -> List[str]:
    if not isinstance(stats, Mapping) or not stats:
        return []
    candidates: List[tuple[float, str, float, float, int]] = []
    for tag, values in stats.items():
        if not isinstance(values, Mapping):
            continue
        valence = values.get("mean_valence")
        rho_mean = values.get("mean_rho")
        count = values.get("count")
        try:
            valence_f = float(valence)
            rho_f = float(rho_mean) if rho_mean is not None else float("nan")
            count_f = int(round(float(count))) if count is not None else 0
        except Exception:
            continue
        candidates.append((abs(valence_f), tag, valence_f, rho_f, count_f))
    if not candidates:
        return []
    candidates.sort(reverse=True)
    lines = ["", "**Culture quick notes**"]
    for _, tag, valence_f, rho_f, count_f in candidates[: max(1, top_k)]:
        tendency = "�|�W���" if valence_f >= 0 else "�l�K���"
        lines.append(f"- {tag}: {tendency} (valence {valence_f:+.2f}), ��={rho_f:.2f}, n={count_f}")
    lines.append("")
    return lines


def _write_markdown_summary(
    report: Dict[str, Any],
    md_path: Path,
    plot_info: Dict[str, Path],
    telemetry_log: Path,
    plot_error: Optional[str],
) -> None:
    lines: List[str] = ["# Nightly Report", ""]
    lines.append(f"- Generated: {dt.datetime.utcnow().isoformat()}Z")
    lines.append(f"- Telemetry log: {telemetry_log}")
    cfg_text = _read_runtime_config_text(Path("config/runtime.yaml"))
    lines.append("")
    lines.append("## Runtime Config")
    lines.append("```yaml")
    lines.append(cfg_text if cfg_text.strip() else "# empty")
    lines.append("```")

    alerts_cfg = ((report.get("config_snapshot") or {}).get("alerts") or {}) if isinstance(report.get("config_snapshot"), dict) else {}
    if alerts_cfg:
        line = (
            "_Alert thresholds_: "
            f"|valence_mean| ? {alerts_cfg.get('max_abs_valence_mean', 0.6)}, "
            f"corr(rho,I) ? {alerts_cfg.get('min_corr_rho_I', 0.2)}, "
            f"corr(resonance) ? {alerts_cfg.get('min_corr_rho_rho', 0.2)}, "
            f"|lag| ? {alerts_cfg.get('max_allowed_lag', 8.0)}, "
            f"n_eff ? {alerts_cfg.get('min_resonance_samples', 0)}"
        )
        lines.append("")
        lines.append(line)

    warnings = list(report.get("warnings", []))
    if plot_error:
        warnings.append(f"plot_failed ({telemetry_log}): {plot_error}")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for warn in warnings:
            lines.append(f"- {warn}")

    suggestion = report.get("tuning_suggestion")
    if suggestion:
        lines.append("")
        lines.append("## Auto Tuning Proposal")
        lines.append(f"- Reason: {suggestion.get('reason', 'n/a')}")
        theta_on = suggestion.get("theta_on", {})
        theta_off = suggestion.get("theta_off", {})
        lines.append(
            f"- theta_on: current={theta_on.get('current')} -> suggested={theta_on.get('suggested')}"
        )
        if theta_off:
            lines.append(
                f"- theta_off: current={theta_off.get('current')} -> suggested={theta_off.get('suggested')}"
            )

    alerts = report.get("alerts", [])
    if alerts:
        lines.append("")
        lines.append("## Alerts")
        for alert in alerts:
            lines.append(f"- {alert}")
        lines.extend(_render_culture_alerts_table(report))

    field_state = report.get("field_state")
    if field_state:
        lines.append("")
        lines.append("## Field Metrics")
        lines.append(f"- S_mean: {_fmt_float(field_state.get('S_mean'))}")
        lines.append(f"- H_mean: {_fmt_float(field_state.get('H_mean'))}")
        lines.append(f"- rho_mean: {_fmt_float(field_state.get('rho_mean'))}")
        lines.append(f"- Ignition_mean: {_fmt_float(field_state.get('Ignition_mean'))}")
        lines.append(f"- valence_mean: {_fmt_float(field_state.get('valence_mean'))}")
        lines.append(f"- arousal_mean: {_fmt_float(field_state.get('arousal_mean'))}")
        lines.append(f"- corr(rho, I): {_fmt_float(field_state.get('rho_I_corr'))}")
        lines.append(f"- corr(S, I): {_fmt_float(field_state.get('S_I_corr'))}")
        lines.append(f"- corr(H, I): {_fmt_float(field_state.get('H_I_corr'))}")
        lines.append(f"- corr(valence, I): {_fmt_float(field_state.get('valence_I_corr'))}")
        lines.append(f"- corr(arousal, I): {_fmt_float(field_state.get('arousal_I_corr'))}")

    vision_summary = report.get("vision_snapshot")
    if isinstance(vision_summary, dict):
        lines.append("")
        lines.append("## Vision Snapshot")
        try:
            events_val = float(vision_summary.get("events", 0.0))
        except (TypeError, ValueError):
            events_val = 0.0
        try:
            detections_val = float(vision_summary.get("detections_total", 0.0))
        except (TypeError, ValueError):
            detections_val = 0.0
        lines.append(f"- events: {int(events_val)}")
        lines.append(f"- detections_total: {int(detections_val)}")
        if events_val > 0:
            lines.append(f"- detections_per_event: {_fmt_float(detections_val / max(events_val, 1.0))}")
        counts = vision_summary.get("counts_by_kind") or {}
        if counts:
            counts_str = ", ".join(f"{k}={v}" for k, v in counts.items())
            lines.append(f"- counts_by_kind: {counts_str}")
        mean_valence = vision_summary.get("mean_valence")
        mean_arousal = vision_summary.get("mean_arousal")
        mean_dominance = vision_summary.get("mean_dominance")
        if mean_valence is not None or mean_arousal is not None or mean_dominance is not None:
            lines.append(
                "- mean valence/arousal/dominance: {} / {} / {}".format(
                    _fmt_float(mean_valence),
                    _fmt_float(mean_arousal),
                    _fmt_float(mean_dominance),
                )
            )
        fps_mean = vision_summary.get("fps_mean")
        if fps_mean is not None:
            lines.append(f"- fps_mean: {_fmt_float(fps_mean)}")
        pose_mean = vision_summary.get("pose_mean")
        if isinstance(pose_mean, dict) and pose_mean:
            pose_str = ", ".join(f"{k}={_fmt_float(v)}" for k, v in pose_mean.items())
            lines.append(f"- pose_mean: {pose_str}")
        lines.extend(_render_vision_narrative(vision_summary))
    pain_loop = report.get("pain_loop")
    if isinstance(pain_loop, dict):
        lines.append("")
        lines.append("## Pain Loop")
        total = int(pain_loop.get("total", 0))
        forgiven = int(pain_loop.get("forgiven", 0))
        rate = _fmt_float(pain_loop.get("forgive_rate"))
        used_threshold = _fmt_float(pain_loop.get("used_forgive_threshold"))
        policy_update = pain_loop.get("policy_update", {})
        lines.append(
            f"- Pain->Forgive: {forgiven}/{total} (rate {rate}), used_threshold={used_threshold}"
        )
        lines.append(
            "- policy_threshold={} empathy_gain={} (EMA, L1={}, L_inf={})".format(
                _fmt_float(policy_update.get("policy_feedback_threshold")),
                _fmt_float(policy_update.get("a2a_empathy_gain")),
                _fmt_float(pain_loop.get("l1_budget", policy_update.get("l1_budget"))),
                _fmt_float(pain_loop.get("l_inf_budget", policy_update.get("l_inf_budget"))),
            )
        )
        care_stats = pain_loop.get("care_stats") or {}
        if care_stats:
            intv = int(care_stats.get("interventions", 0))
            watch = int(care_stats.get("watch_only", 0))
            targets = care_stats.get("targets") or {}
            top_targets: List[Tuple[str, Any]] = []
            for tgt, entry in targets.items():
                try:
                    total_care = int(entry.get("total", 0))
                except Exception:
                    total_care = 0
                if total_care > 0:
                    top_targets.append((tgt, total_care))
            top_targets.sort(key=lambda x: x[1], reverse=True)
            summary = ", ".join(f"{name}:{count}" for name, count in top_targets[:2]) if top_targets else "none"
            canary_stats = care_stats.get("canary") or {}
            canary_applied = canary_stats.get("applied", 0)
            canary_candidates = canary_stats.get("candidates", 0)
            lines.append(
                f"- Care: interventions {intv}, watch {watch}, top targets: {summary} "
                f"(canary {canary_applied}/{canary_candidates})"
            )
            lines.append(
                "- comfort_gain={} protection_bias={} (EMA, L1={}, L_inf={})".format(
                    _fmt_float(care_stats.get("comfort_gain_applied")),
                    _fmt_float(pain_loop.get("protection_bias")),
                    _fmt_float(pain_loop.get("l1_budget")),
                    _fmt_float(pain_loop.get("l_inf_budget")),
                )
            )
        influence_top = pain_loop.get("value_influence_top") or []
        if influence_top:
            rendered = ", ".join(
                f"{item.get('value','?')}@{'care' if item.get('care_target') else 'all'}"
                f"(dE={_fmt_float(item.get('delta_empathy_gain'))})"
                for item in influence_top[:3]
            )
            lines.append(f"- value_influence: {rendered}")
        monthly_highlights = pain_loop.get("monthly_highlights") or []
        if monthly_highlights:
            summary = ", ".join(
                f"{item.get('value','?')}@{'care' if item.get('care_target') else 'all'}"
                f"(sum={item.get('sum_delta_empathy_gain', 0.0):.4f})"
                for item in monthly_highlights[:3]
            )
            lines.append(f"- monthly_highlights: {summary}")
        breakdown = (pain_loop.get("breakdown") or {}).get("by_kind") or {}
        top_unforgiven: List[Tuple[str, int]] = []
        for kind, stats in breakdown.items():
            try:
                total_k = int(stats.get("total", 0))
                forgiven_k = int(stats.get("forgiven", 0))
            except Exception:
                continue
            delta = total_k - forgiven_k
            if delta > 0:
                top_unforgiven.append((kind, delta))
        if top_unforgiven:
            top_unforgiven.sort(key=lambda x: x[1], reverse=True)
            summary = ", ".join(f"{kind}�~{count}" for kind, count in top_unforgiven[:2])
            lines.append(f"- Unforgiven top: {summary}")
    affective_stats = report.get("affective_stats")
    if isinstance(affective_stats, dict):
        lines.append("")
        lines.append("## Affective Summary")
        for axis in ("valence", "arousal"):
            axis_stats = affective_stats.get(axis)
            if isinstance(axis_stats, dict):
                mean_val = _fmt_float(axis_stats.get("mean"))
                std_val = _fmt_float(axis_stats.get("std"))
                q25 = _fmt_float(axis_stats.get("q25"))
                q50 = _fmt_float(axis_stats.get("q50"))
                q75 = _fmt_float(axis_stats.get("q75"))
                lines.append(
                    f"- {axis}: mean={mean_val} std={std_val} q25={q25} q50={q50} q75={q75}"
                )
    culture_stats = report.get("culture_stats")
    if isinstance(culture_stats, dict) and culture_stats:
        min_culture_samples = int(float(alerts_cfg.get("min_culture_samples", 0))) if isinstance(alerts_cfg, dict) else 0
        lines.append("")
        lines.append("## Culture Resonance")
        sparse_lines: List[str] = []
        for tag, stats in culture_stats.items():
            if not isinstance(stats, dict):
                continue
            count_raw = stats.get("count")
            try:
                count_val = int(round(float(count_raw)))
            except Exception:
                count_val = 0
            if count_val < max(0, min_culture_samples):
                sparse_lines.append(f"- {tag}: count={count_val} (<{min_culture_samples}) insufficient samples")
                continue
            valence_mean = _fmt_float(stats.get("mean_valence"))
            arousal_mean = _fmt_float(stats.get("mean_arousal"))
            rho_mean = _fmt_float(stats.get("mean_rho"))
            pol = stats.get("mean_politeness")
            pol_str = _fmt_float(pol) if pol is not None else "n/a"
            intimacy = stats.get("mean_intimacy")
            intimacy_str = _fmt_float(intimacy) if intimacy is not None else "n/a"
            lines.append(
                f"- {tag}: count={count_val} valence={valence_mean} arousal={arousal_mean} "
                f"rho={rho_mean} politeness={pol_str} intimacy={intimacy_str}"
            )
        if sparse_lines:
            lines.append("")
            lines.append(f"### Culture Tags Below Sample Threshold (n < {min_culture_samples})")
            lines.extend(sparse_lines)
    culture_trend_plot = report.get("plots", {}).get("culture_trend")
    if culture_trend_plot:
        lines.append("")
        lines.append("### Culture Trend (multi-day)")
        lines.append(f"![Culture Trend]({_relpath(Path(culture_trend_plot), md_path.parent)})")
    lines.extend(_render_culture_narrative(culture_stats))

    policy_feedback = report.get("policy_feedback")
    if isinstance(policy_feedback, dict) and policy_feedback.get("enabled"):
        lines.append("")
        lines.append("## Policy Feedback (experimental)")
        lines.append(f"- reason: {policy_feedback.get('reason')}")
        corr_val = policy_feedback.get("corr")
        if corr_val is not None:
            source = policy_feedback.get("corr_source") or "n/a"
            lines.append(f"- corr reference: {_fmt_float(corr_val)} ({source})")
        lines.append(f"- politeness: {_fmt_float(policy_feedback.get('politeness_before'))} -> {_fmt_float(policy_feedback.get('politeness_after'))} (delta={_fmt_float(policy_feedback.get('delta'))})")
        lines.append(f"- intimacy: {_fmt_float(policy_feedback.get('intimacy_before'))} -> {_fmt_float(policy_feedback.get('intimacy_after'))} (delta={_fmt_float(policy_feedback.get('intimacy_delta'))})")
    resonance_section = (report.get("resonance") or {}).get("summary")
    if isinstance(resonance_section, dict):
        lines.append("")
        lines.append("## Resonance Summary")
        lines.append(f"- corr: {_fmt_float(resonance_section.get('corr'))}")
        lines.append(f"- lag: {_fmt_float(resonance_section.get('lag'))}")
        lines.append(f"- energy: {_fmt_float(resonance_section.get('energy'))}")
        lines.append(f"- objective: {_fmt_float(resonance_section.get('objective'))}")
        lines.append(f"- n_eff: {_fmt_float(resonance_section.get('n_eff'))}")
        lines.append(f"- partial_corr: {_fmt_float(resonance_section.get('partial_corr'))}")
        granger_section = resonance_section.get("granger")
        if isinstance(granger_section, dict):
            lines.append(
                f"- granger F(a->b)={_fmt_float(granger_section.get('a_to_b_f'))} "
                f"F(b->a)={_fmt_float(granger_section.get('b_to_a_f'))} "
                f"(lag={granger_section.get('lag')})"
            )

    if plot_info:
        titles = {
            "ignition_timeseries": "Ignition / S/H/rho Timeseries",
            "rho_vs_I_scatter": "rho vs Ignition Scatter",
            "affective_map": "Valence vs Arousal Scatter",
            "memory_graph": "Memory Graph",
            "resonance_objective": "Resonance Objective History",
            "resonance_bayes_trace": "Resonance Bayesian Trace",
            "culture_resonance": "Culture Resonance Summary",
            "culture_trend": "Culture Trend (multi-day)",
            "vision_counts": "Vision Detection Counts",
            "vision_pose": "Vision Pose Summary",
        }
        lines.append("")
        lines.append("## Visuals")
        for key, path in plot_info.items():
            label = titles.get(key, key)
            rel = _relpath(path, md_path.parent)
            lines.append(f"![{label}]({rel})")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def _fmt_float(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(val):
        return "n/a"
    return f"{val:.3f}"


def _relpath(path: Path, base: Path) -> str:
    try:
        rel = os.path.relpath(path, base)
    except ValueError:
        rel = str(path)
    return Path(rel).as_posix()


def _read_runtime_config_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return "# runtime config unavailable"


def _load_runtime_cfg_dict(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _resolve_inner_os_sleep_snapshot(
    cfg_dict: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    candidates: List[Path] = []
    nightly_cfg = cfg_dict.get("nightly", {}) if isinstance(cfg_dict, Mapping) else {}
    explicit = None
    if isinstance(nightly_cfg, Mapping):
        explicit = nightly_cfg.get("inner_os_sleep_snapshot_path")
    if not explicit and isinstance(cfg_dict, Mapping):
        explicit = cfg_dict.get("inner_os_sleep_snapshot_path")
    if isinstance(explicit, str) and explicit.strip():
        candidates.append(Path(explicit.strip()))

    state_dir = cfg_dict.get("state_dir") if isinstance(cfg_dict, Mapping) else None
    if isinstance(state_dir, str) and state_dir.strip():
        candidates.append(Path(state_dir.strip()) / "inner_os_sleep_snapshot.json")

    candidates.append(Path("data/state/inner_os_sleep_snapshot.json"))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, None, f"inner_os_sleep_snapshot_read_failed ({candidate}): {exc}"
        if not isinstance(payload, dict):
            return None, None, f"inner_os_sleep_snapshot_invalid_payload ({candidate})"
        if str(payload.get("schema") or "") != INNER_OS_SLEEP_SCHEMA:
            return None, None, f"inner_os_sleep_snapshot_schema_mismatch ({candidate})"
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            return None, None, f"inner_os_sleep_snapshot_missing_snapshot ({candidate})"
        return str(candidate), payload, None
    return None, None, None


def _resolve_inner_os_working_memory_snapshot(
    cfg_dict: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    candidates: List[Path] = []
    nightly_cfg = cfg_dict.get("nightly", {}) if isinstance(cfg_dict, Mapping) else {}
    explicit = None
    if isinstance(nightly_cfg, Mapping):
        explicit = nightly_cfg.get("inner_os_working_memory_snapshot_path")
    if not explicit and isinstance(cfg_dict, Mapping):
        explicit = cfg_dict.get("inner_os_working_memory_snapshot_path")
    if isinstance(explicit, str) and explicit.strip():
        candidates.append(Path(explicit.strip()))

    state_dir = cfg_dict.get("state_dir") if isinstance(cfg_dict, Mapping) else None
    if isinstance(state_dir, str) and state_dir.strip():
        candidates.append(Path(state_dir.strip()) / "inner_os_working_memory_snapshot.json")

    candidates.append(Path("data/state/inner_os_working_memory_snapshot.json"))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, None, f"inner_os_working_memory_snapshot_read_failed ({candidate}): {exc}"
        if not isinstance(payload, dict):
            return None, None, f"inner_os_working_memory_snapshot_invalid_payload ({candidate})"
        if str(payload.get("schema") or "") != INNER_OS_WORKING_MEMORY_SCHEMA:
            return None, None, f"inner_os_working_memory_snapshot_schema_mismatch ({candidate})"
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            return None, None, f"inner_os_working_memory_snapshot_missing_snapshot ({candidate})"
        return str(candidate), payload, None
    return None, None, None


def _resolve_inner_os_memory_path(cfg_dict: Mapping[str, Any]) -> Path:
    explicit = None
    if isinstance(cfg_dict, Mapping):
        explicit = cfg_dict.get("inner_os_memory_path")
        nightly_cfg = cfg_dict.get("nightly")
        if not explicit and isinstance(nightly_cfg, Mapping):
            explicit = nightly_cfg.get("inner_os_memory_path")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit.strip())
    return MemoryCore().path


def _resolve_diary_state_path(cfg_dict: Mapping[str, Any]) -> Path:
    nightly_cfg = cfg_dict.get("nightly", {}) if isinstance(cfg_dict, Mapping) else {}
    explicit = None
    if isinstance(nightly_cfg, Mapping):
        explicit = nightly_cfg.get("diary_state_path")
    if not explicit and isinstance(cfg_dict, Mapping):
        explicit = cfg_dict.get("diary_state_path")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit.strip())
    state_dir = cfg_dict.get("state_dir") if isinstance(cfg_dict, Mapping) else None
    if isinstance(state_dir, str) and state_dir.strip():
        return Path(state_dir.strip()) / "diary.json"
    return Path("data/state/diary.json")


def _summarize_inner_os_identity_arc_registry(
    cfg_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    path = _resolve_diary_state_path(cfg_dict)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    registry_payload = payload.get("identity_arc_registry")
    if not isinstance(registry_payload, Mapping):
        return {}
    summary = IdentityArcRegistry.from_dict(registry_payload).summary()
    if int(summary.get("total_arcs") or 0) <= 0:
        return {}
    return dict(summary)


def _summarize_inner_os_relation_arc_registry(
    cfg_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    path = _resolve_diary_state_path(cfg_dict)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    registry_payload = payload.get("relation_arc_registry")
    if not isinstance(registry_payload, Mapping):
        return {}
    summary = RelationArcRegistry.from_dict(registry_payload).summary()
    if int(summary.get("total_arcs") or 0) <= 0:
        return {}
    return dict(summary)


def _summarize_inner_os_partner_relation_registry(
    cfg_dict: Mapping[str, Any],
    *,
    now: Optional[dt.datetime] = None,
    lookback_hours: int = 120,
) -> Dict[str, Any]:
    path = _resolve_inner_os_memory_path(cfg_dict)
    if not path.exists():
        return {}
    current_time = now or dt.datetime.utcnow()
    cutoff = current_time - dt.timedelta(hours=max(1, int(lookback_hours)))
    grouped: Dict[str, Dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(payload.get("kind") or "") != "relationship_trace":
                    continue
                person_id = str(payload.get("related_person_id") or "").strip()
                if not person_id:
                    continue
                stamp = _coerce_inner_os_record_time(payload.get("timestamp"))
                if stamp is None or stamp < cutoff:
                    continue
                attachment = _safe_float(payload.get("attachment"), 0.0) or 0.0
                familiarity = _safe_float(payload.get("familiarity"), 0.0) or 0.0
                trust_memory = _safe_float(payload.get("trust_memory"), 0.0) or 0.0
                social_pull = _safe_float(payload.get("candidate_social_pull"), 0.0) or 0.0
                consolidation = _safe_float(payload.get("consolidation_priority"), 0.0) or 0.0
                recency_hours = max(0.0, (current_time - stamp).total_seconds() / 3600.0)
                recency_bonus = _clamp(1.0 - recency_hours / max(lookback_hours, 1), 0.0, 1.0)
                score = (
                    attachment * 0.3
                    + familiarity * 0.24
                    + trust_memory * 0.22
                    + social_pull * 0.14
                    + consolidation * 0.05
                    + recency_bonus * 0.05
                )
                bucket = grouped.setdefault(
                    person_id,
                    {
                        "count": 0,
                        "score_sum": 0.0,
                        "attachment_sum": 0.0,
                        "familiarity_sum": 0.0,
                        "trust_sum": 0.0,
                        "social_pull_sum": 0.0,
                        "consolidation_sum": 0.0,
                        "recency_bonus_sum": 0.0,
                        "best_score": -1.0,
                        "best_record": {},
                        "history": [],
                    },
                )
                bucket["count"] += 1
                bucket["score_sum"] += score
                bucket["attachment_sum"] += attachment
                bucket["familiarity_sum"] += familiarity
                bucket["trust_sum"] += trust_memory
                bucket["social_pull_sum"] += social_pull
                bucket["consolidation_sum"] += consolidation
                bucket["recency_bonus_sum"] += recency_bonus
                history = bucket["history"]
                history.append(
                    {
                        "observation": str(payload.get("summary") or payload.get("text") or "").strip()[:160],
                        "memory_anchor": str(payload.get("memory_anchor") or "").strip()[:160],
                        "timestamp": stamp.isoformat(),
                    }
                )
                if len(history) > 3:
                    del history[0 : len(history) - 3]
                if score > float(bucket["best_score"] or -1.0):
                    bucket["best_score"] = score
                    bucket["best_record"] = payload
    except OSError:
        return {}
    if not grouped:
        return {}

    persons: Dict[str, Any] = {}
    ranking: list[tuple[str, float]] = []
    for person_id, bucket in grouped.items():
        count = max(1, int(bucket.get("count") or 0))
        mean_score = float(bucket.get("score_sum") or 0.0) / count
        attachment = _clamp(float(bucket.get("attachment_sum") or 0.0) / count, 0.0, 1.0)
        familiarity = _clamp(float(bucket.get("familiarity_sum") or 0.0) / count, 0.0, 1.0)
        trust_memory = _clamp(float(bucket.get("trust_sum") or 0.0) / count, 0.0, 1.0)
        social_pull = _clamp(float(bucket.get("social_pull_sum") or 0.0) / count, 0.0, 1.0)
        consolidation = _clamp(float(bucket.get("consolidation_sum") or 0.0) / count, 0.0, 1.0)
        recency_bonus = _clamp(float(bucket.get("recency_bonus_sum") or 0.0) / count, 0.0, 1.0)
        repeat_bonus = _clamp(min(count, 4) / 4.0, 0.0, 1.0)
        best_record = dict(bucket.get("best_record") or {})
        strength = _clamp(
            mean_score * 0.76
            + repeat_bonus * 0.16
            + recency_bonus * 0.08,
            0.0,
            1.0,
        )
        continuity_score = _clamp(
            strength * 0.52
            + familiarity * 0.2
            + trust_memory * 0.16
            + repeat_bonus * 0.12,
            0.0,
            1.0,
        )
        social_grounding = _clamp(
            strength * 0.44
            + attachment * 0.16
            + trust_memory * 0.16
            + social_pull * 0.14
            + consolidation * 0.1,
            0.0,
            1.0,
        )
        confidence = _clamp(
            0.42 + strength * 0.34 + repeat_bonus * 0.14 + recency_bonus * 0.1,
            0.0,
            1.0,
        )
        persons[person_id] = {
            "person_id": person_id,
            "stable_traits": {
                "community_marker": 1.0 if str(best_record.get("community_id") or "").strip() else 0.0,
                "culture_marker": 1.0 if str(best_record.get("culture_id") or "").strip() else 0.0,
                "role_marker": 1.0 if str(best_record.get("social_role") or "").strip() else 0.0,
            },
            "adaptive_traits": {
                "attachment": round(attachment, 4),
                "familiarity": round(familiarity, 4),
                "trust_memory": round(trust_memory, 4),
                "continuity_score": round(continuity_score, 4),
                "social_grounding": round(social_grounding, 4),
            },
            "continuity_history": list(bucket.get("history") or []),
            "confidence": round(confidence, 4),
            "ambiguity_flag": False,
            "summary": str(best_record.get("summary") or best_record.get("text") or "").strip()[:160],
            "memory_anchor": str(best_record.get("memory_anchor") or "").strip()[:160],
            "social_role": str(best_record.get("social_role") or "").strip(),
            "social_interpretation": str(best_record.get("social_interpretation") or "").strip()[:160],
            "address_hint": str(best_record.get("address_hint") or "").strip(),
            "timing_hint": str(best_record.get("timing_hint") or "").strip(),
            "stance_hint": str(best_record.get("stance_hint") or "").strip(),
            "strength": round(strength, 4),
        }
        ranking.append((person_id, strength))
    ranking.sort(key=lambda item: (item[1], item[0]), reverse=True)
    top_person_ids = [person_id for person_id, _ in ranking[:4]]
    dominant_person_id = top_person_ids[0] if top_person_ids else ""
    dominant_strength = ranking[0][1] if ranking else 0.0
    return {
        "persons": persons,
        "top_person_ids": top_person_ids,
        "dominant_person_id": dominant_person_id,
        "total_people": len(persons),
        "uncertainty": round(_clamp(1.0 - dominant_strength * 0.72, 0.0, 1.0), 4),
    }


def _summarize_inner_os_group_thread_registry(
    cfg_dict: Mapping[str, Any],
    *,
    now: Optional[dt.datetime] = None,
    lookback_hours: int = 120,
) -> Dict[str, Any]:
    path = _resolve_inner_os_memory_path(cfg_dict)
    if not path.exists():
        return {}
    current_time = now or dt.datetime.utcnow()
    cutoff = current_time - dt.timedelta(hours=max(1, int(lookback_hours)))
    grouped: Dict[str, Dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(payload.get("kind") or "") != "group_thread_trace":
                    continue
                stamp = _coerce_inner_os_record_time(payload.get("timestamp"))
                if stamp is None or stamp < cutoff:
                    continue
                top_person_ids = [
                    str(item).strip()
                    for item in list(payload.get("top_person_ids") or [])
                    if str(item).strip()
                ]
                dominant_person_id = str(payload.get("related_person_id") or "").strip()
                focus = str(payload.get("group_thread_focus") or "").strip()
                thread_id = str(payload.get("group_thread_id") or "").strip() or build_group_thread_key(
                    topology_state=focus,
                    top_person_ids=top_person_ids,
                    dominant_person_id=dominant_person_id,
                )
                if not thread_id:
                    continue
                continuity_score = _safe_float(payload.get("continuity_score"), 0.0) or 0.0
                social_grounding = _safe_float(payload.get("social_grounding"), 0.0) or 0.0
                threading_pressure = _safe_float(payload.get("threading_pressure"), 0.0) or 0.0
                visibility_pressure = _safe_float(payload.get("visibility_pressure"), 0.0) or 0.0
                hierarchy_pressure = _safe_float(payload.get("hierarchy_pressure"), 0.0) or 0.0
                total_people = max(int(payload.get("thread_total_people") or 0), len(top_person_ids))
                recency_hours = max(0.0, (current_time - stamp).total_seconds() / 3600.0)
                recency_bonus = _clamp(1.0 - recency_hours / max(lookback_hours, 1), 0.0, 1.0)
                score = _clamp(
                    continuity_score * 0.28
                    + social_grounding * 0.2
                    + threading_pressure * 0.18
                    + visibility_pressure * 0.12
                    + hierarchy_pressure * 0.1
                    + recency_bonus * 0.12,
                    0.0,
                    1.0,
                )
                bucket = grouped.setdefault(
                    thread_id,
                    {
                        "count": 0,
                        "score_sum": 0.0,
                        "continuity_sum": 0.0,
                        "grounding_sum": 0.0,
                        "threading_sum": 0.0,
                        "visibility_sum": 0.0,
                        "hierarchy_sum": 0.0,
                        "people_max": 0,
                        "top_people": [],
                        "best_score": -1.0,
                        "best_record": {},
                    },
                )
                bucket["count"] += 1
                bucket["score_sum"] += score
                bucket["continuity_sum"] += continuity_score
                bucket["grounding_sum"] += social_grounding
                bucket["threading_sum"] += threading_pressure
                bucket["visibility_sum"] += visibility_pressure
                bucket["hierarchy_sum"] += hierarchy_pressure
                bucket["people_max"] = max(int(bucket.get("people_max") or 0), total_people)
                if top_person_ids:
                    bucket["top_people"] = top_person_ids[:4]
                if score > float(bucket.get("best_score") or -1.0):
                    bucket["best_score"] = score
                    bucket["best_record"] = payload
    except OSError:
        return {}
    if not grouped:
        return {}

    threads: Dict[str, Any] = {}
    dominant_strength = 0.0
    for thread_id, bucket in grouped.items():
        count = max(1, int(bucket.get("count") or 0))
        best_record = dict(bucket.get("best_record") or {})
        strength = _clamp(
            float(bucket.get("score_sum") or 0.0) / count
            + min(count, 4) / 4.0 * 0.08,
            0.0,
            1.0,
        )
        dominant_strength = max(dominant_strength, strength)
        threads[thread_id] = {
            "thread_id": thread_id,
            "last_topology_state": str(best_record.get("group_thread_focus") or "").strip(),
            "dominant_person_id": str(best_record.get("related_person_id") or "").strip(),
            "top_person_ids": list(bucket.get("top_people") or []),
            "total_people": int(bucket.get("people_max") or 0),
            "continuity_score": round(float(bucket.get("continuity_sum") or 0.0) / count, 4),
            "social_grounding": round(float(bucket.get("grounding_sum") or 0.0) / count, 4),
            "threading_pressure": round(float(bucket.get("threading_sum") or 0.0) / count, 4),
            "visibility_pressure": round(float(bucket.get("visibility_sum") or 0.0) / count, 4),
            "hierarchy_pressure": round(float(bucket.get("hierarchy_sum") or 0.0) / count, 4),
            "community_id": str(best_record.get("community_id") or "").strip(),
            "culture_id": str(best_record.get("culture_id") or "").strip(),
            "social_role": str(best_record.get("social_role") or "").strip(),
            "count": count,
            "confidence": round(_clamp(0.24 + min(count, 5) * 0.12, 0.0, 1.0), 4),
        }
    summary = summarize_group_thread_registry_snapshot(
        {
            "threads": threads,
            "uncertainty": round(_clamp(1.0 - dominant_strength * 0.72, 0.0, 1.0), 4),
        }
    )
    return {
        "threads": threads,
        "uncertainty": summary.get("uncertainty", 1.0),
        **summary,
        "lookback_hours": int(lookback_hours),
    }


def _summarize_inner_os_discussion_thread_registry(
    cfg_dict: Mapping[str, Any],
    *,
    now: Optional[dt.datetime] = None,
    lookback_hours: int = 72,
) -> Dict[str, Any]:
    path = _resolve_inner_os_memory_path(cfg_dict)
    if not path.exists():
        return {}
    current_time = now or dt.datetime.utcnow()
    cutoff = current_time - dt.timedelta(hours=max(1, int(lookback_hours)))
    snapshot: Dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(payload.get("kind") or "") != "discussion_thread_trace":
                    continue
                stamp = _coerce_inner_os_record_time(payload.get("timestamp"))
                if stamp is None or stamp < cutoff:
                    continue
                snapshot = update_discussion_thread_registry_snapshot(
                    existing_snapshot=snapshot,
                    recent_dialogue_state={
                        "state": str(payload.get("recent_dialogue_state") or "").strip(),
                        "thread_carry": _safe_float(payload.get("recent_dialogue_thread_carry"), 0.0) or 0.0,
                        "reopen_pressure": _safe_float(payload.get("recent_dialogue_reopen_pressure"), 0.0) or 0.0,
                        "recent_anchor": str(
                            payload.get("issue_anchor")
                            or payload.get("discussion_thread_anchor")
                            or payload.get("memory_anchor")
                            or ""
                        ).strip(),
                    },
                    discussion_thread_state={
                        "state": str(payload.get("discussion_thread_state") or "").strip(),
                        "topic_anchor": str(payload.get("discussion_thread_anchor") or "").strip(),
                        "unresolved_pressure": _safe_float(payload.get("discussion_unresolved_pressure"), 0.0) or 0.0,
                        "revisit_readiness": _safe_float(payload.get("discussion_revisit_readiness"), 0.0) or 0.0,
                        "thread_visibility": _safe_float(payload.get("discussion_thread_visibility"), 0.0) or 0.0,
                    },
                    issue_state={
                        "state": str(payload.get("issue_state") or "").strip(),
                        "issue_anchor": str(payload.get("issue_anchor") or "").strip(),
                        "question_pressure": _safe_float(payload.get("issue_question_pressure"), 0.0) or 0.0,
                        "pause_readiness": _safe_float(payload.get("issue_pause_readiness"), 0.0) or 0.0,
                        "resolution_readiness": _safe_float(payload.get("issue_resolution_readiness"), 0.0) or 0.0,
                    },
                )
    except OSError:
        return {}
    if not snapshot:
        return {}
    summary = summarize_discussion_thread_registry_snapshot(snapshot)
    return {
        **snapshot,
        **summary,
        "lookback_hours": int(lookback_hours),
    }


def _summarize_inner_os_partner_relation(
    cfg_dict: Mapping[str, Any],
    *,
    now: Optional[dt.datetime] = None,
    lookback_hours: int = 120,
    registry_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    registry = dict(registry_summary or {})
    if not registry:
        registry = _summarize_inner_os_partner_relation_registry(
            cfg_dict,
            now=now,
            lookback_hours=lookback_hours,
        )
    dominant_person_id = str(registry.get("dominant_person_id") or "").strip()
    persons = dict(registry.get("persons") or {})
    dominant = dict(persons.get(dominant_person_id) or {})
    adaptive_traits = dict(dominant.get("adaptive_traits") or {})
    if not dominant_person_id or not dominant:
        return {}
    return {
        "person_id": dominant_person_id,
        "summary": str(dominant.get("summary") or "").strip()[:160],
        "memory_anchor": str(dominant.get("memory_anchor") or "").strip()[:160],
        "social_role": str(dominant.get("social_role") or "").strip(),
        "social_interpretation": str(dominant.get("social_interpretation") or "").strip()[:160],
        "address_hint": str(dominant.get("address_hint") or "").strip(),
        "timing_hint": str(dominant.get("timing_hint") or "").strip(),
        "stance_hint": str(dominant.get("stance_hint") or "").strip(),
        "attachment": round(_safe_float(adaptive_traits.get("attachment"), 0.0) or 0.0, 4),
        "familiarity": round(_safe_float(adaptive_traits.get("familiarity"), 0.0) or 0.0, 4),
        "trust_memory": round(_safe_float(adaptive_traits.get("trust_memory"), 0.0) or 0.0, 4),
        "strength": round(_safe_float(dominant.get("strength"), 0.0) or 0.0, 4),
    }


def _summarize_inner_os_memory_class(
    cfg_dict: Mapping[str, Any],
    *,
    now: Optional[dt.datetime] = None,
    lookback_hours: int = 120,
) -> Dict[str, Any]:
    path = _resolve_inner_os_memory_path(cfg_dict)
    if not path.exists():
        return {}
    current_time = now or dt.datetime.utcnow()
    cutoff = current_time - dt.timedelta(hours=max(1, int(lookback_hours)))
    counts: Dict[str, int] = defaultdict(int)
    weighted_counts: Dict[str, float] = defaultdict(float)
    dominant_reasons: Dict[tuple[str, str], float] = defaultdict(float)
    recent_records = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                memory_class = str(payload.get("memory_write_class") or "").strip()
                if not memory_class:
                    continue
                stamp = _coerce_inner_os_record_time(payload.get("timestamp"))
                if stamp is None or stamp < cutoff:
                    continue
                recent_records += 1
                counts[memory_class] += 1

                age_hours = max(0.0, (current_time - stamp).total_seconds() / 3600.0)
                recency_weight = _clamp(1.0 - age_hours / max(float(lookback_hours), 1.0), 0.2, 1.0)
                confidence = _safe_float(payload.get("confidence"), 0.0) or 0.0
                access_count = _safe_float(payload.get("access_count"), 0.0) or 0.0
                primed_weight = _safe_float(payload.get("primed_weight"), 0.0) or 0.0
                usage_weight = 1.0 + min(access_count, 5.0) * 0.05 + _clamp(primed_weight, 0.0, 1.0) * 0.15
                confidence_weight = 0.7 + _clamp(confidence, 0.0, 1.0) * 0.3
                score = recency_weight * usage_weight * confidence_weight
                weighted_counts[memory_class] += score

                reason = str(payload.get("memory_write_class_reason") or "").strip()
                if reason:
                    dominant_reasons[(memory_class, reason)] += score
    except OSError:
        return {}

    if not counts:
        return {}

    dominant_class = max(
        counts.keys(),
        key=lambda key: (weighted_counts.get(key, 0.0), counts.get(key, 0), key),
    )
    dominant_reason = ""
    if dominant_reasons:
        dominant_reason = max(
            (
                (reason, score)
                for (memory_class, reason), score in dominant_reasons.items()
                if memory_class == dominant_class
            ),
            default=("", 0.0),
            key=lambda item: item[1],
        )[0]
    return {
        "dominant_class": dominant_class,
        "dominant_reason": dominant_reason,
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "weighted_counts": {key: round(float(value), 4) for key, value in sorted(weighted_counts.items())},
        "recent_records": int(recent_records),
        "lookback_hours": int(max(1, lookback_hours)),
    }


def _summarize_inner_os_insight_trace(
    cfg_dict: Mapping[str, Any],
    *,
    now: Optional[dt.datetime] = None,
    lookback_hours: int = 120,
) -> Dict[str, Any]:
    path = _resolve_inner_os_memory_path(cfg_dict)
    if not path.exists():
        return {}
    current_time = now or dt.datetime.utcnow()
    cutoff = current_time - dt.timedelta(hours=max(1, int(lookback_hours)))
    class_counts: Dict[str, int] = defaultdict(int)
    weighted_class_counts: Dict[str, float] = defaultdict(float)
    link_counts: Dict[str, int] = defaultdict(int)
    reframed_topics: Dict[str, float] = defaultdict(float)
    records_for_shape: List[Dict[str, Any]] = []
    recent_records = 0

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(payload.get("kind") or "") != "insight_trace":
                    continue
                stamp = _coerce_inner_os_record_time(payload.get("timestamp"))
                if stamp is None or stamp < cutoff:
                    continue
                recent_records += 1
                insight_class = str(payload.get("insight_class") or "insight_trace").strip() or "insight_trace"
                class_counts[insight_class] += 1

                age_hours = max(0.0, (current_time - stamp).total_seconds() / 3600.0)
                recency_weight = _clamp(1.0 - age_hours / max(float(lookback_hours), 1.0), 0.2, 1.0)
                confidence = _safe_float(payload.get("confidence"), 0.0) or 0.0
                insight_score = _safe_float(payload.get("insight_score"), 0.0) or 0.0
                coherence_gain = _safe_float(payload.get("coherence_gain"), 0.0) or 0.0
                prediction_drop = _safe_float(payload.get("prediction_drop"), 0.0) or 0.0
                weight = recency_weight * (
                    0.38
                    + _clamp(confidence, 0.0, 1.0) * 0.22
                    + _clamp(insight_score, 0.0, 1.0) * 0.22
                    + _clamp(coherence_gain, 0.0, 1.0) * 0.12
                    + _clamp(prediction_drop, 0.0, 1.0) * 0.06
                )
                weighted_class_counts[insight_class] += weight

                link_key = str(payload.get("association_link_key") or "").strip()
                if link_key:
                    link_counts[link_key] += 1

                reframed_topic = str(payload.get("reframed_topic") or "").strip()
                if reframed_topic:
                    reframed_topics[reframed_topic] += weight
                anchor_center_raw = payload.get("anchor_center")
                anchor_center: List[float] = []
                if isinstance(anchor_center_raw, (list, tuple)):
                    for item in anchor_center_raw:
                        try:
                            anchor_center.append(float(item))
                        except (TypeError, ValueError):
                            continue
                records_for_shape.append(
                    {
                        "insight_class": insight_class,
                        "link_key": link_key,
                        "weight": weight,
                        "confidence": _clamp(confidence, 0.0, 1.0),
                        "anchor_center": anchor_center,
                        "anchor_dispersion": max(0.0, _safe_float(payload.get("anchor_dispersion"), 0.0) or 0.0),
                    }
                )
    except OSError:
        return {}

    if not class_counts:
        return {}

    dominant_insight_class = max(
        class_counts.keys(),
        key=lambda key: (weighted_class_counts.get(key, 0.0), class_counts.get(key, 0), key),
    )
    dominant_reframed_topic = ""
    if reframed_topics:
        dominant_reframed_topic = max(reframed_topics.items(), key=lambda item: item[1])[0]
    repeated_links = sum(max(0, count - 1) for count in link_counts.values())
    association_reweighting_bias = _clamp(
        repeated_links * 0.18
        + weighted_class_counts.get("new_link_hypothesis", 0.0) * 0.08
        + weighted_class_counts.get("insight_trace", 0.0) * 0.05
        + weighted_class_counts.get("reframed_relation", 0.0) * 0.06,
        0.0,
        1.0,
    )
    association_reweighting_focus = ""
    association_reweighting_reason = ""
    if association_reweighting_bias > 0.0:
        if repeated_links > 0:
            association_reweighting_focus = "repeated_links"
            association_reweighting_reason = "repeated_insight_trace"
        elif weighted_class_counts.get("new_link_hypothesis", 0.0) > 0.0:
            association_reweighting_focus = "hypothesis_links"
            association_reweighting_reason = "new_link_hypothesis"
        elif weighted_class_counts.get("reframed_relation", 0.0) > 0.0:
            association_reweighting_focus = "reframed_links"
            association_reweighting_reason = "reframed_relation"
    insight_reframing_bias = _clamp(
        weighted_class_counts.get("reframed_relation", 0.0) * 0.18
        + repeated_links * 0.08,
        0.0,
        1.0,
    )
    insight_shape_scores: Dict[str, float] = defaultdict(float)
    anchor_weight_total = 0.0
    anchor_center_acc: Optional[np.ndarray] = None
    anchor_dispersion_acc = 0.0
    repeated_trace_links = {
        key
        for key, count in link_counts.items()
        if count >= 2
    }
    for row in records_for_shape:
        insight_class = str(row.get("insight_class") or "").strip()
        link_key = str(row.get("link_key") or "").strip()
        weight = max(0.0, float(row.get("weight") or 0.0))
        if weight <= 0.0:
            continue
        if insight_class == "reframed_relation":
            shape_reason = "reframed_relation"
            shape_weight = weight * 0.58
        elif insight_class == "insight_trace" and link_key in repeated_trace_links:
            shape_reason = "repeated_insight_trace"
            shape_weight = weight * 0.36
        elif insight_class == "new_link_hypothesis":
            shape_reason = "new_link_hypothesis"
            shape_weight = weight * 0.08
        else:
            continue
        insight_shape_scores[shape_reason] += shape_weight
        anchor_center = row.get("anchor_center") or []
        if not anchor_center:
            continue
        center_vec = np.asarray(anchor_center, dtype=np.float32)
        if center_vec.ndim != 1 or center_vec.size == 0:
            continue
        if anchor_center_acc is None:
            anchor_center_acc = np.zeros_like(center_vec, dtype=np.float32)
        if anchor_center_acc.shape != center_vec.shape:
            continue
        anchor_center_acc += center_vec * shape_weight
        anchor_dispersion_acc += max(0.0, float(row.get("anchor_dispersion") or 0.0)) * shape_weight
        anchor_weight_total += shape_weight

    insight_terrain_shape_reason = ""
    insight_terrain_shape_bias = 0.0
    insight_terrain_shape_target = ""
    insight_anchor_center: List[float] = []
    insight_anchor_dispersion = 0.0
    if insight_shape_scores:
        insight_terrain_shape_reason = max(insight_shape_scores.items(), key=lambda item: item[1])[0]
        raw_shape = sum(insight_shape_scores.values())
        insight_terrain_shape_bias = _clamp(raw_shape * 0.32, 0.0, 1.0)
        if insight_terrain_shape_reason == "new_link_hypothesis":
            insight_terrain_shape_bias = min(insight_terrain_shape_bias, 0.08)
        insight_terrain_shape_target = _insight_shape_target_from_reason(insight_terrain_shape_reason)
        if anchor_weight_total > 0.0 and anchor_center_acc is not None:
            mean_center = anchor_center_acc / anchor_weight_total
            insight_anchor_center = [round(float(item), 4) for item in mean_center.tolist()]
            insight_anchor_dispersion = round(float(anchor_dispersion_acc / anchor_weight_total), 4)
    return {
        "dominant_insight_class": dominant_insight_class,
        "dominant_reframed_topic": dominant_reframed_topic,
        "insight_class_counts": {key: int(value) for key, value in sorted(class_counts.items())},
        "weighted_class_counts": {key: round(float(value), 4) for key, value in sorted(weighted_class_counts.items())},
        "insight_link_counts": {key: int(value) for key, value in sorted(link_counts.items())},
        "association_reweighting_bias": round(float(association_reweighting_bias), 4),
        "association_reweighting_focus": association_reweighting_focus,
        "association_reweighting_reason": association_reweighting_reason,
        "insight_reframing_bias": round(float(insight_reframing_bias), 4),
        "insight_terrain_shape_bias": round(float(insight_terrain_shape_bias), 4),
        "insight_terrain_shape_reason": insight_terrain_shape_reason,
        "insight_terrain_shape_target": insight_terrain_shape_target,
        "insight_anchor_center": insight_anchor_center,
        "insight_anchor_dispersion": insight_anchor_dispersion,
        "recent_records": int(recent_records),
        "lookback_hours": int(max(1, lookback_hours)),
    }


def _summarize_inner_os_commitment_trace(
    cfg_dict: Mapping[str, Any],
    *,
    now: Optional[dt.datetime] = None,
    lookback_hours: int = 120,
) -> Dict[str, Any]:
    path = _resolve_inner_os_memory_path(cfg_dict)
    if not path.exists():
        return {}
    current_time = now or dt.datetime.utcnow()
    cutoff = current_time - dt.timedelta(hours=max(1, int(lookback_hours)))
    state_counts: Dict[str, int] = defaultdict(int)
    target_counts: Dict[str, int] = defaultdict(int)
    weighted_target_counts: Dict[str, float] = defaultdict(float)
    dominant_reasons: Dict[tuple[str, str], float] = defaultdict(float)
    recent_records = 0

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                target = str(payload.get("commitment_target") or "").strip()
                state = str(payload.get("commitment_state") or "").strip()
                if not target or not state:
                    continue
                stamp = _coerce_inner_os_record_time(payload.get("timestamp"))
                if stamp is None or stamp < cutoff:
                    continue
                recent_records += 1
                state_counts[state] += 1
                target_counts[target] += 1

                age_hours = max(0.0, (current_time - stamp).total_seconds() / 3600.0)
                recency_weight = _clamp(1.0 - age_hours / max(float(lookback_hours), 1.0), 0.2, 1.0)
                score = _clamp(_safe_float(payload.get("commitment_score"), 0.0) or 0.0, 0.0, 1.0)
                margin = _clamp(_safe_float(payload.get("commitment_winner_margin"), 0.0) or 0.0, 0.0, 1.0)
                accepted_cost = _clamp(_safe_float(payload.get("commitment_accepted_cost"), 0.0) or 0.0, 0.0, 1.0)
                state_gain = 1.0
                if state == "commit":
                    state_gain = 1.14
                elif state == "settle":
                    state_gain = 0.92
                elif state == "waver":
                    state_gain = 0.56
                weight = recency_weight * state_gain * (
                    0.34
                    + score * 0.28
                    + margin * 0.22
                    + accepted_cost * 0.16
                )
                weighted_target_counts[target] += weight

                reason = str(
                    payload.get("memory_write_class_reason")
                    or payload.get("memory_write_class")
                    or payload.get("summary")
                    or ""
                ).strip()
                if reason:
                    dominant_reasons[(target, reason)] += weight
    except OSError:
        return {}

    if not target_counts:
        return {}

    dominant_target = max(
        target_counts.keys(),
        key=lambda key: (weighted_target_counts.get(key, 0.0), target_counts.get(key, 0), key),
    )
    dominant_state = max(
        state_counts.keys(),
        key=lambda key: (state_counts.get(key, 0), key == "commit", key == "settle", key),
    )
    dominant_reason = ""
    if dominant_reasons:
        dominant_reason = max(
            (
                (reason, score)
                for (target, reason), score in dominant_reasons.items()
                if target == dominant_target
            ),
            default=("", 0.0),
            key=lambda item: item[1],
        )[0]
    repeated_target = max(0, int(target_counts.get(dominant_target, 0)) - 1)
    state_bonus = 0.0
    if dominant_state == "commit":
        state_bonus = 0.18
    elif dominant_state == "settle":
        state_bonus = 0.08
    commitment_carry_bias = _clamp(
        weighted_target_counts.get(dominant_target, 0.0) * 0.2
        + repeated_target * 0.08
        + state_bonus,
        0.0,
        1.0,
    )
    commitment_followup_focus = _commitment_followup_focus_for_target(dominant_target)
    commitment_mode_focus = _commitment_mode_focus_for_target(dominant_target)

    return {
        "dominant_target": dominant_target,
        "dominant_state": dominant_state,
        "dominant_reason": dominant_reason,
        "target_counts": {key: int(value) for key, value in sorted(target_counts.items())},
        "state_counts": {key: int(value) for key, value in sorted(state_counts.items())},
        "weighted_target_counts": {key: round(float(value), 4) for key, value in sorted(weighted_target_counts.items())},
        "commitment_carry_bias": round(float(commitment_carry_bias), 4),
        "commitment_followup_focus": commitment_followup_focus,
        "commitment_mode_focus": commitment_mode_focus,
        "recent_records": int(recent_records),
        "lookback_hours": int(max(1, lookback_hours)),
    }


def _summarize_inner_os_agenda_trace(
    cfg_dict: Mapping[str, Any],
    *,
    now: Optional[dt.datetime] = None,
    lookback_hours: int = 120,
) -> Dict[str, Any]:
    path = _resolve_inner_os_memory_path(cfg_dict)
    if not path.exists():
        return {}
    current_time = now or dt.datetime.utcnow()
    cutoff = current_time - dt.timedelta(hours=max(1, int(lookback_hours)))
    state_counts: Dict[str, int] = defaultdict(int)
    weighted_state_counts: Dict[str, float] = defaultdict(float)
    dominant_reasons: Dict[tuple[str, str], float] = defaultdict(float)
    recent_records = 0

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state = str(payload.get("agenda_state") or "").strip()
                if not state:
                    continue
                kind = str(payload.get("kind") or "").strip()
                if kind and kind != "agenda_trace":
                    continue
                stamp = _coerce_inner_os_record_time(payload.get("timestamp"))
                if stamp is None or stamp < cutoff:
                    continue
                recent_records += 1
                state_counts[state] += 1

                age_hours = max(0.0, (current_time - stamp).total_seconds() / 3600.0)
                recency_weight = _clamp(1.0 - age_hours / max(float(lookback_hours), 1.0), 0.2, 1.0)
                score = _clamp(_safe_float(payload.get("agenda_score"), 0.0) or 0.0, 0.0, 1.0)
                margin = _clamp(_safe_float(payload.get("agenda_winner_margin"), 0.0) or 0.0, 0.0, 1.0)
                state_gain = 1.0
                if state == "step_forward":
                    state_gain = 1.12
                elif state == "repair":
                    state_gain = 1.06
                elif state == "revisit":
                    state_gain = 0.92
                elif state == "hold":
                    state_gain = 0.88
                weight = recency_weight * state_gain * (
                    0.34
                    + score * 0.38
                    + margin * 0.28
                )
                weighted_state_counts[state] += weight

                reason = str(payload.get("agenda_reason") or payload.get("summary") or "").strip()
                if reason:
                    dominant_reasons[(state, reason)] += weight
    except OSError:
        return {}

    if not state_counts:
        return {}

    dominant_agenda = max(
        state_counts.keys(),
        key=lambda key: (weighted_state_counts.get(key, 0.0), state_counts.get(key, 0), key),
    )
    dominant_reason = ""
    if dominant_reasons:
        dominant_reason = max(
            (
                (reason, score)
                for (state, reason), score in dominant_reasons.items()
                if state == dominant_agenda
            ),
            default=("", 0.0),
            key=lambda item: item[1],
        )[0]
    repeated_agenda = max(0, int(state_counts.get(dominant_agenda, 0)) - 1)
    focus_bonus = 0.0
    if dominant_agenda == "step_forward":
        focus_bonus = 0.12
    elif dominant_agenda == "repair":
        focus_bonus = 0.1
    elif dominant_agenda == "revisit":
        focus_bonus = 0.08
    elif dominant_agenda == "hold":
        focus_bonus = 0.06
    agenda_carry_bias = _clamp(
        weighted_state_counts.get(dominant_agenda, 0.0) * 0.18
        + repeated_agenda * 0.07
        + focus_bonus,
        0.0,
        1.0,
    )

    return {
        "dominant_agenda": dominant_agenda,
        "dominant_reason": dominant_reason,
        "state_counts": {key: int(value) for key, value in sorted(state_counts.items())},
        "weighted_state_counts": {key: round(float(value), 4) for key, value in sorted(weighted_state_counts.items())},
        "agenda_carry_bias": round(float(agenda_carry_bias), 4),
        "recent_records": int(recent_records),
        "lookback_hours": int(max(1, lookback_hours)),
    }


def _commitment_followup_focus_for_target(target: str) -> str:
    normalized_target = str(target or "").strip()
    if normalized_target == "step_forward":
        return "offer_next_step"
    if normalized_target in {"repair", "bond_protect"}:
        return "reopen_softly"
    if normalized_target in {"hold", "stabilize"}:
        return "hold"
    return ""


def _commitment_mode_focus_for_target(target: str) -> str:
    normalized_target = str(target or "").strip()
    if normalized_target == "step_forward":
        return "monitor"
    if normalized_target in {"repair", "bond_protect"}:
        return "repair"
    if normalized_target == "stabilize":
        return "stabilize"
    if normalized_target == "hold":
        return "contain"
    return ""


def _coerce_inner_os_record_time(value: Any) -> Optional[dt.datetime]:
    if isinstance(value, (int, float)):
        try:
            return dt.datetime.utcfromtimestamp(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _derive_tuning_suggestion(field_state: Optional[Dict[str, float]], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not field_state or not cfg:
        return None
    ignition_cfg = cfg.get("ignition", {})
    theta_on = float(ignition_cfg.get("theta_on", 0.62))
    theta_off = float(ignition_cfg.get("theta_off", 0.48))
    corr = field_state.get("rho_I_corr")
    mean_I = field_state.get("Ignition_mean")
    suggestion: Optional[Dict[str, Any]] = None
    if corr is not None and math.isfinite(corr) and corr < 0.2:
        new_on = round(max(0.4, min(0.9, theta_on - 0.02)), 3)
        new_off = round(max(0.2, min(new_on - 0.05, theta_off - 0.02)), 3)
        suggestion = {
            "reason": f"rho/I correlation low ({corr:.3f})",
            "theta_on": {"current": theta_on, "suggested": new_on},
            "theta_off": {"current": theta_off, "suggested": new_off},
        }
    elif mean_I is not None and math.isfinite(mean_I) and mean_I > 0.8:
        new_on = round(min(0.95, theta_on + 0.01), 3)
        new_off = round(min(new_on - 0.02, theta_off + 0.01), 3)
        suggestion = {
            "reason": f"Ignition mean high ({mean_I:.3f})",
            "theta_on": {"current": theta_on, "suggested": new_on},
            "theta_off": {"current": theta_off, "suggested": new_off},
        }
    elif mean_I is not None and math.isfinite(mean_I) and mean_I < 0.3:
        new_on = round(max(theta_on - 0.01, theta_off + 0.05), 3)
        suggestion = {
            "reason": f"Ignition mean low ({mean_I:.3f})",
            "theta_on": {"current": theta_on, "suggested": new_on},
            "theta_off": {"current": theta_off, "suggested": theta_off},
        }
    return suggestion


def _insight_shape_target_from_reason(reason: str) -> str:
    normalized_reason = str(reason or "").strip()
    if normalized_reason == "reframed_relation":
        return "soft_relation"
    if normalized_reason == "repeated_insight_trace":
        return "trace_basin"
    if normalized_reason == "new_link_hypothesis":
        return "hypothesis_hold"
    return ""


def _write_json_summary(report: Dict[str, Any], out_dir: str = "reports") -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    field_state = report.get("field_state") or {}
    stats = {
        "mean": {
            "S": field_state.get("S_mean"),
            "H": field_state.get("H_mean"),
            "rho": field_state.get("rho_mean"),
            "I": field_state.get("Ignition_mean"),
            "valence": field_state.get("valence_mean"),
            "arousal": field_state.get("arousal_mean"),
        },
        "corr": {
            "rho_I": field_state.get("rho_I_corr"),
            "S_I": field_state.get("S_I_corr"),
            "H_I": field_state.get("H_I_corr"),
            "valence_I": field_state.get("valence_I_corr"),
            "arousal_I": field_state.get("arousal_I_corr"),
        },
    }
    plots = {k: str(v) for k, v in report.get("plots", {}).items()}
    payload = {
        "schema": "nightly.v1",
        "ts": int(time.time()),
        "nightly_id": report.get("nightly_id"),
        "config_snapshot": report.get("config_snapshot", {}),
        "stats": stats,
        "tuning_suggestion": report.get("tuning_suggestion"),
        "warnings": report.get("warnings", []),
        "alerts": report.get("alerts", []),
        "alerts_detail": report.get("alerts_detail", []),
        "run_seed": report.get("run_seed"),
        "plots": plots,
        "affective_stats": report.get("affective_stats"),
        "resonance": report.get("resonance"),
        "resonance_history_path": report.get("resonance_history_path"),
    }
    if report.get("culture_stats"):
        payload["culture_stats"] = report["culture_stats"]
    if report.get("culture_history_path"):
        payload["culture_history_path"] = report["culture_history_path"]
    if report.get("vision_snapshot"):
        payload["vision_snapshot"] = report["vision_snapshot"]
    if report.get("memory_match"):
        payload["memory_match"] = report["memory_match"]
    if report.get("repair_success_rate") is not None:
        payload["repair_success_rate"] = report["repair_success_rate"]
    if report.get("policy_feedback"):
        payload["policy_feedback"] = report["policy_feedback"]
    if report.get("policy_feedback_history_path"):
        payload["policy_feedback_history_path"] = report["policy_feedback_history_path"]
    if report.get("forgetting"):
        payload["forgetting"] = report["forgetting"]
    if report.get("defrag"):
        payload["defrag"] = report["defrag"]
    if report.get("monument_floor_test") is not None:
        payload["monument_floor_test"] = report["monument_floor_test"]
    if report.get("pain_loop"):
        payload["pain_loop"] = report["pain_loop"]
    if report.get("resonance_bayes_trace_path"):
        payload["resonance_bayes_trace_path"] = report["resonance_bayes_trace_path"]
    if report.get("assoc_kernel"):
        payload["assoc_kernel"] = report["assoc_kernel"]
    if report.get("assoc_summary_path"):
        payload["assoc_summary_path"] = report["assoc_summary_path"]
    if report.get("inner_os_sleep_snapshot_path"):
        payload["inner_os_sleep_snapshot_path"] = report["inner_os_sleep_snapshot_path"]
    if report.get("inner_os_sleep_mode"):
        payload["inner_os_sleep_mode"] = report["inner_os_sleep_mode"]
    if report.get("inner_os_working_memory_snapshot_path"):
        payload["inner_os_working_memory_snapshot_path"] = report["inner_os_working_memory_snapshot_path"]
    if report.get("inner_os_working_memory_focus"):
        payload["inner_os_working_memory_focus"] = report["inner_os_working_memory_focus"]
    if report.get("inner_os_working_memory_readiness") is not None:
        payload["inner_os_working_memory_readiness"] = report["inner_os_working_memory_readiness"]
    if report.get("inner_os_working_memory_replay_bias"):
        payload["inner_os_working_memory_replay_bias"] = report["inner_os_working_memory_replay_bias"]
    if report.get("inner_os_long_term_theme_summary"):
        payload["inner_os_long_term_theme_summary"] = report["inner_os_long_term_theme_summary"]
    if report.get("inner_os_relation_arc_summary"):
        payload["inner_os_relation_arc_summary"] = report["inner_os_relation_arc_summary"]
    if report.get("inner_os_group_relation_arc_summary"):
        payload["inner_os_group_relation_arc_summary"] = report["inner_os_group_relation_arc_summary"]
    if report.get("inner_os_identity_arc_summary"):
        payload["inner_os_identity_arc_summary"] = report["inner_os_identity_arc_summary"]
    if report.get("inner_os_relation_arc_registry_summary"):
        payload["inner_os_relation_arc_registry_summary"] = report["inner_os_relation_arc_registry_summary"]
    if report.get("inner_os_identity_arc_registry_summary"):
        payload["inner_os_identity_arc_registry_summary"] = report["inner_os_identity_arc_registry_summary"]
    if report.get("inner_os_memory_class_summary"):
        payload["inner_os_memory_class_summary"] = report["inner_os_memory_class_summary"]
    if report.get("inner_os_agenda_summary"):
        payload["inner_os_agenda_summary"] = report["inner_os_agenda_summary"]
    if report.get("inner_os_commitment_summary"):
        payload["inner_os_commitment_summary"] = report["inner_os_commitment_summary"]
    if report.get("inner_os_insight_summary"):
        payload["inner_os_insight_summary"] = report["inner_os_insight_summary"]
    if report.get("inner_os_partner_relation_registry_summary"):
        payload["inner_os_partner_relation_registry_summary"] = report["inner_os_partner_relation_registry_summary"]
    if report.get("inner_os_group_thread_registry_summary"):
        payload["inner_os_group_thread_registry_summary"] = report["inner_os_group_thread_registry_summary"]
    if report.get("inner_os_discussion_thread_registry_summary"):
        payload["inner_os_discussion_thread_registry_summary"] = report["inner_os_discussion_thread_registry_summary"]
    if report.get("inner_os_partner_relation_summary"):
        payload["inner_os_partner_relation_summary"] = report["inner_os_partner_relation_summary"]
    if report.get("inner_os_daily_carry_summary"):
        payload["inner_os_daily_carry_summary"] = report["inner_os_daily_carry_summary"]
    if report.get("inner_os_temporal_alignment"):
        payload["inner_os_temporal_alignment"] = report["inner_os_temporal_alignment"]
    if report.get("inner_os_sleep_memory_class_focus"):
        payload["inner_os_sleep_memory_class_focus"] = report["inner_os_sleep_memory_class_focus"]
    if report.get("inner_os_sleep_agenda_focus"):
        payload["inner_os_sleep_agenda_focus"] = report["inner_os_sleep_agenda_focus"]
    if report.get("inner_os_sleep_agenda_bias") is not None:
        payload["inner_os_sleep_agenda_bias"] = report["inner_os_sleep_agenda_bias"]
    if report.get("inner_os_sleep_agenda_reason"):
        payload["inner_os_sleep_agenda_reason"] = report["inner_os_sleep_agenda_reason"]
    if report.get("inner_os_sleep_agenda_window_focus"):
        payload["inner_os_sleep_agenda_window_focus"] = report["inner_os_sleep_agenda_window_focus"]
    if report.get("inner_os_sleep_agenda_window_bias") is not None:
        payload["inner_os_sleep_agenda_window_bias"] = report["inner_os_sleep_agenda_window_bias"]
    if report.get("inner_os_sleep_agenda_window_reason"):
        payload["inner_os_sleep_agenda_window_reason"] = report["inner_os_sleep_agenda_window_reason"]
    if report.get("inner_os_sleep_agenda_window_carry_target"):
        payload["inner_os_sleep_agenda_window_carry_target"] = report["inner_os_sleep_agenda_window_carry_target"]
    if report.get("inner_os_sleep_learning_mode_focus"):
        payload["inner_os_sleep_learning_mode_focus"] = report["inner_os_sleep_learning_mode_focus"]
    if report.get("inner_os_sleep_learning_mode_carry_bias") is not None:
        payload["inner_os_sleep_learning_mode_carry_bias"] = report["inner_os_sleep_learning_mode_carry_bias"]
    if report.get("inner_os_sleep_social_experiment_focus"):
        payload["inner_os_sleep_social_experiment_focus"] = report["inner_os_sleep_social_experiment_focus"]
    if report.get("inner_os_sleep_social_experiment_carry_bias") is not None:
        payload["inner_os_sleep_social_experiment_carry_bias"] = report["inner_os_sleep_social_experiment_carry_bias"]
    if report.get("inner_os_sleep_commitment_target_focus"):
        payload["inner_os_sleep_commitment_target_focus"] = report["inner_os_sleep_commitment_target_focus"]
    if report.get("inner_os_sleep_commitment_state_focus"):
        payload["inner_os_sleep_commitment_state_focus"] = report["inner_os_sleep_commitment_state_focus"]
    if report.get("inner_os_sleep_commitment_carry_bias") is not None:
        payload["inner_os_sleep_commitment_carry_bias"] = report["inner_os_sleep_commitment_carry_bias"]
    if report.get("inner_os_sleep_commitment_followup_focus"):
        payload["inner_os_sleep_commitment_followup_focus"] = report["inner_os_sleep_commitment_followup_focus"]
    if report.get("inner_os_sleep_commitment_mode_focus"):
        payload["inner_os_sleep_commitment_mode_focus"] = report["inner_os_sleep_commitment_mode_focus"]
    if report.get("inner_os_sleep_commitment_carry_reason"):
        payload["inner_os_sleep_commitment_carry_reason"] = report["inner_os_sleep_commitment_carry_reason"]
    if report.get("inner_os_sleep_terrain_reweighting_bias") is not None:
        payload["inner_os_sleep_terrain_reweighting_bias"] = report["inner_os_sleep_terrain_reweighting_bias"]
    if report.get("inner_os_sleep_insight_class_focus"):
        payload["inner_os_sleep_insight_class_focus"] = report["inner_os_sleep_insight_class_focus"]
    if report.get("inner_os_sleep_insight_reframing_bias") is not None:
        payload["inner_os_sleep_insight_reframing_bias"] = report["inner_os_sleep_insight_reframing_bias"]
    if report.get("inner_os_sleep_association_reweighting_bias") is not None:
        payload["inner_os_sleep_association_reweighting_bias"] = report["inner_os_sleep_association_reweighting_bias"]
    if report.get("inner_os_sleep_association_reweighting_focus"):
        payload["inner_os_sleep_association_reweighting_focus"] = report["inner_os_sleep_association_reweighting_focus"]
    if report.get("inner_os_sleep_association_reweighting_reason"):
        payload["inner_os_sleep_association_reweighting_reason"] = report["inner_os_sleep_association_reweighting_reason"]
    if report.get("inner_os_sleep_insight_terrain_shape_bias") is not None:
        payload["inner_os_sleep_insight_terrain_shape_bias"] = report["inner_os_sleep_insight_terrain_shape_bias"]
    if report.get("inner_os_sleep_insight_terrain_shape_reason"):
        payload["inner_os_sleep_insight_terrain_shape_reason"] = report["inner_os_sleep_insight_terrain_shape_reason"]
    if report.get("inner_os_sleep_insight_terrain_shape_target"):
        payload["inner_os_sleep_insight_terrain_shape_target"] = report["inner_os_sleep_insight_terrain_shape_target"]
    if report.get("inner_os_sleep_insight_anchor_center") is not None:
        payload["inner_os_sleep_insight_anchor_center"] = report["inner_os_sleep_insight_anchor_center"]
    if report.get("inner_os_sleep_insight_anchor_dispersion") is not None:
        payload["inner_os_sleep_insight_anchor_dispersion"] = report["inner_os_sleep_insight_anchor_dispersion"]
    if report.get("inner_os_sleep_temperament_focus"):
        payload["inner_os_sleep_temperament_focus"] = report["inner_os_sleep_temperament_focus"]
    if report.get("inner_os_sleep_temperament_forward_bias") is not None:
        payload["inner_os_sleep_temperament_forward_bias"] = report["inner_os_sleep_temperament_forward_bias"]
    if report.get("inner_os_sleep_temperament_guard_bias") is not None:
        payload["inner_os_sleep_temperament_guard_bias"] = report["inner_os_sleep_temperament_guard_bias"]
    if report.get("inner_os_sleep_temperament_bond_bias") is not None:
        payload["inner_os_sleep_temperament_bond_bias"] = report["inner_os_sleep_temperament_bond_bias"]
    if report.get("inner_os_sleep_temperament_recovery_bias") is not None:
        payload["inner_os_sleep_temperament_recovery_bias"] = report["inner_os_sleep_temperament_recovery_bias"]
    if report.get("inner_os_sleep_homeostasis_budget_focus"):
        payload["inner_os_sleep_homeostasis_budget_focus"] = report["inner_os_sleep_homeostasis_budget_focus"]
    if report.get("inner_os_sleep_homeostasis_budget_bias") is not None:
        payload["inner_os_sleep_homeostasis_budget_bias"] = report["inner_os_sleep_homeostasis_budget_bias"]
    if report.get("inner_os_sleep_body_homeostasis_focus"):
        payload["inner_os_sleep_body_homeostasis_focus"] = report["inner_os_sleep_body_homeostasis_focus"]
    if report.get("inner_os_sleep_body_homeostasis_carry_bias") is not None:
        payload["inner_os_sleep_body_homeostasis_carry_bias"] = report["inner_os_sleep_body_homeostasis_carry_bias"]
    if report.get("inner_os_sleep_relational_continuity_focus"):
        payload["inner_os_sleep_relational_continuity_focus"] = report["inner_os_sleep_relational_continuity_focus"]
    if report.get("inner_os_sleep_relational_continuity_carry_bias") is not None:
        payload["inner_os_sleep_relational_continuity_carry_bias"] = report["inner_os_sleep_relational_continuity_carry_bias"]
    if report.get("inner_os_sleep_group_thread_focus"):
        payload["inner_os_sleep_group_thread_focus"] = report["inner_os_sleep_group_thread_focus"]
    if report.get("inner_os_sleep_group_thread_carry_bias") is not None:
        payload["inner_os_sleep_group_thread_carry_bias"] = report["inner_os_sleep_group_thread_carry_bias"]
    if report.get("inner_os_sleep_autobiographical_thread_mode"):
        payload["inner_os_sleep_autobiographical_thread_mode"] = report["inner_os_sleep_autobiographical_thread_mode"]
    if report.get("inner_os_sleep_autobiographical_thread_anchor"):
        payload["inner_os_sleep_autobiographical_thread_anchor"] = report["inner_os_sleep_autobiographical_thread_anchor"]
    if report.get("inner_os_sleep_autobiographical_thread_focus"):
        payload["inner_os_sleep_autobiographical_thread_focus"] = report["inner_os_sleep_autobiographical_thread_focus"]
    if report.get("inner_os_sleep_autobiographical_thread_strength") is not None:
        payload["inner_os_sleep_autobiographical_thread_strength"] = report["inner_os_sleep_autobiographical_thread_strength"]
    if report.get("inner_os_sleep_temporal_membrane_focus"):
        payload["inner_os_sleep_temporal_membrane_focus"] = report["inner_os_sleep_temporal_membrane_focus"]
    if report.get("inner_os_sleep_temporal_timeline_bias") is not None:
        payload["inner_os_sleep_temporal_timeline_bias"] = report["inner_os_sleep_temporal_timeline_bias"]
    if report.get("inner_os_sleep_temporal_reentry_bias") is not None:
        payload["inner_os_sleep_temporal_reentry_bias"] = report["inner_os_sleep_temporal_reentry_bias"]
    if report.get("inner_os_sleep_temporal_supersession_bias") is not None:
        payload["inner_os_sleep_temporal_supersession_bias"] = report["inner_os_sleep_temporal_supersession_bias"]
    if report.get("inner_os_sleep_temporal_continuity_bias") is not None:
        payload["inner_os_sleep_temporal_continuity_bias"] = report["inner_os_sleep_temporal_continuity_bias"]
    if report.get("inner_os_sleep_temporal_relation_reentry_bias") is not None:
        payload["inner_os_sleep_temporal_relation_reentry_bias"] = report["inner_os_sleep_temporal_relation_reentry_bias"]
    if report.get("inner_os_sleep_expressive_style_focus"):
        payload["inner_os_sleep_expressive_style_focus"] = report["inner_os_sleep_expressive_style_focus"]
    if report.get("inner_os_sleep_expressive_style_carry_bias") is not None:
        payload["inner_os_sleep_expressive_style_carry_bias"] = report["inner_os_sleep_expressive_style_carry_bias"]
    if report.get("inner_os_sleep_expressive_style_history_focus"):
        payload["inner_os_sleep_expressive_style_history_focus"] = report["inner_os_sleep_expressive_style_history_focus"]
    if report.get("inner_os_sleep_expressive_style_history_bias") is not None:
        payload["inner_os_sleep_expressive_style_history_bias"] = report["inner_os_sleep_expressive_style_history_bias"]
    if report.get("inner_os_sleep_banter_style_focus"):
        payload["inner_os_sleep_banter_style_focus"] = report["inner_os_sleep_banter_style_focus"]
    if report.get("inner_os_sleep_lexical_variation_carry_bias") is not None:
        payload["inner_os_sleep_lexical_variation_carry_bias"] = report["inner_os_sleep_lexical_variation_carry_bias"]
    json_path = out_path / "nightly.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate nightly summary from telemetry")
    parser.add_argument("--telemetry_log", default="logs/telemetry_events.jsonl")
    parser.add_argument("--plots_dir", default="reports/plots")
    parser.add_argument("--markdown_path", default="reports/nightly.md")
    parser.add_argument(
        "--enable-culture-feedback",
        action="store_true",
        help="Allow policy feedback adjustments when culture feedback config is enabled.",
    )
    args = parser.parse_args()
    report: Dict[str, Any] = {"ts": time.time()}
    _generate_telemetry_section(
        report,
        Path(args.telemetry_log),
        Path(args.plots_dir),
        Path(args.markdown_path),
        culture_feedback_enabled=bool(args.enable_culture_feedback),
    )
    json_path = _write_json_summary(report, out_dir="reports")
    if report.get("alerts"):
        print("[nightly] alerts:", ", ".join(report["alerts"]))
    else:
        print("[nightly] alerts: none")
    print(f"[nightly] JSON summary -> {json_path}")
    print(f"Nightly summary written to {args.markdown_path}")


__all__ = ["run", "_apply_go_sc_gate", "_summarize_fastpath_metrics", "_summarize_inner_replay_metrics", "_summarize_qualia_metrics"]


if __name__ == "__main__":
    main()

