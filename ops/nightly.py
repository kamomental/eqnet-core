# -*- coding: utf-8 -*-
"""Nightly consolidation orchestration for ATRi."""

from __future__ import annotations

import json
import time
import math
import os
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

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


__all__ = ["run", "_apply_go_sc_gate", "_summarize_fastpath_metrics", "_summarize_inner_replay_metrics"]
