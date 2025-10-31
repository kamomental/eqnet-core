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

from emot_terrain_lab.ops.care_canary import select_canary_ids
from emot_terrain_lab.ops.monthly_highlights import generate_value_influence_highlights
from emot_terrain_lab.ops.pain_loop import VALUE_INFLUENCE_LOG, evaluate_and_forgive, policy_update_from_forgiveness

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
            if go_sc_cfg.get("enabled", False):
                baseline_ttl_tau = float(getattr(getattr(hub.memory_ttl, "cfg", None), "ttl_tau_default", 24.0))
                go_report = _apply_go_sc_gate(
                    events,
                    go_sc_cfg,
                    _hub_tau_now(),
                    baseline_ttl_tau,
                    nightly_cfg,
                    fastpath_cfg,
                )
                report["go_sc"] = go_report
                if go_report.get("fastpath"):
                    report["fastpath"] = go_report["fastpath"]
                if go_report.get("inner_replay"):
                    report["inner_replay"] = go_report["inner_replay"]
                if "nightly_metrics" in go_report:
                    report["nightly_metrics"] = go_report["nightly_metrics"]
            kept, stats = hub.memory_ttl.gc(events)
            if kept is not None:
                hub.replay_memory.rewrite(kept)
            report["memory_gc"] = stats
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
        scored.append((float(score), idx))

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
        verdict = "右下に点が多い：よい状態 (execute が素直)。"
        action = "いまの設定を維持しつつ散布図を監視。"
    elif tl_ratio >= 0.4:
        verdict = "左上に点が多い：ブレーキが強め。"
        action = "β や τ を少し下げる案を検討。"
    else:
        verdict = "中央に点が散らばり迷い気味。"
        action = "ヒステリシス幅を広げるか提案 β–τ を小規模 A/B。"
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
        ax.set_title("Δm / jerk (recent)")
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
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = row.get("culture_tag") or "unknown"
            buckets[tag]["valence"].append(float(row.get("valence", 0.0)))
            buckets[tag]["arousal"].append(float(row.get("arousal", 0.0)))
            buckets[tag]["rho"].append(float(row.get("rho", 0.0)))
            if row.get("politeness") is not None:
                buckets[tag]["politeness"].append(float(row.get("politeness")))
            if row.get("intimacy") is not None:
                buckets[tag]["intimacy"].append(float(row.get("intimacy")))
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
    cfg_dict = _load_runtime_cfg_dict(Path("config/runtime.yaml"))
    report["config_snapshot"] = cfg_dict
    nightly_id = report.setdefault("nightly_id", dt.datetime.utcnow().strftime("N-%Y%m%d-%H%M%S"))
    tuning = _derive_tuning_suggestion(field_state, cfg_dict)
    if tuning:
        report["tuning_suggestion"] = tuning

    emotion_cfg = cfg_dict.get("emotion", {}) if isinstance(cfg_dict, dict) else {}
    alerts_cfg = cfg_dict.get("alerts", {}) if isinstance(cfg_dict, dict) else {}
    resonance_cfg = cfg_dict.get("resonance", {}) if isinstance(cfg_dict, dict) else {}
    culture_cfg_snapshot = cfg_dict.get("culture", {}) if isinstance(cfg_dict, dict) else {}

    affective_log_path = Path(emotion_cfg.get("affective_log_path", "memory/affective_log.jsonl"))
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
    culture_stats = _summarize_culture_stats(affective_log_path)
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
        tendency = "ポジ寄り" if valence_f >= 0 else "ネガ寄り"
        lines.append(f"- {tag}: {tendency} (valence {valence_f:+.2f}), ρ={rho_f:.2f}, n={count_f}")
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
            f"|valence_mean| ≤ {alerts_cfg.get('max_abs_valence_mean', 0.6)}, "
            f"corr(rho,I) ≥ {alerts_cfg.get('min_corr_rho_I', 0.2)}, "
            f"corr(resonance) ≥ {alerts_cfg.get('min_corr_rho_rho', 0.2)}, "
            f"|lag| ≤ {alerts_cfg.get('max_allowed_lag', 8.0)}, "
            f"n_eff ≥ {alerts_cfg.get('min_resonance_samples', 0)}"
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
            summary = ", ".join(f"{kind}×{count}" for kind, count in top_unforgiven[:2])
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
    if report.get("pain_loop"):
        payload["pain_loop"] = report["pain_loop"]
    if report.get("resonance_bayes_trace_path"):
        payload["resonance_bayes_trace_path"] = report["resonance_bayes_trace_path"]
    if report.get("assoc_kernel"):
        payload["assoc_kernel"] = report["assoc_kernel"]
    if report.get("assoc_summary_path"):
        payload["assoc_summary_path"] = report["assoc_summary_path"]
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


__all__ = ["run", "_apply_go_sc_gate", "_summarize_fastpath_metrics", "_summarize_inner_replay_metrics"]


if __name__ == "__main__":
    main()
