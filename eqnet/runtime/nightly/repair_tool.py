from __future__ import annotations

import hashlib
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


def run_repair_cycle(config: Any) -> Optional[Dict[str, Any]]:
    forgetting_cfg = _forgetting_policy(config)
    if not forgetting_cfg:
        return None
    if not bool(forgetting_cfg.get("enable", False)):
        repair_ops: list[dict[str, Any]] = []
        return {
            "repair_tool_version": "repair_tool_v1",
            "repair_status": "disabled",
            "repaired_events_count": 0,
            "scanned_events_count": 0,
            "rewrite_applied": False,
            "repair_plan_id": _repair_plan_id([], forgetting_cfg),
            "replay_token": _replay_token(forgetting_cfg),
            "repair_ops": repair_ops,
            "repair_ops_digest": _repair_ops_digest(repair_ops),
            "forgetting": {"status": "disabled"},
        }

    replay_path = _resolve_replay_memory_path(config, forgetting_cfg)
    events = _load_replay_events(replay_path)
    if not events:
        repair_ops: list[dict[str, Any]] = []
        return {
            "repair_tool_version": "repair_tool_v1",
            "repair_status": "skipped",
            "reason": "replay_memory_empty",
            "repaired_events_count": 0,
            "scanned_events_count": 0,
            "rewrite_applied": False,
            "replay_memory_path": str(replay_path),
            "repair_plan_id": _repair_plan_id(events, forgetting_cfg),
            "replay_token": _replay_token(forgetting_cfg),
            "repair_ops": repair_ops,
            "repair_ops_digest": _repair_ops_digest(repair_ops),
            "forgetting": {
                "status": "skipped",
                "reason": "replay_memory_empty",
                "replay_memory_path": str(replay_path),
            },
        }

    updated, forgetting_report = _apply_forgetting_reweight(events, forgetting_cfg, now_ts=time.time())
    changed_count = int(forgetting_report.get("changed_count") or 0)
    rewrite_applied = False
    if changed_count > 0:
        _rewrite_replay_events_atomic(replay_path, updated)
        rewrite_applied = True
    forgetting_report["rewrite_applied"] = rewrite_applied
    forgetting_report["replay_memory_path"] = str(replay_path)
    repair_ops = list(forgetting_report.get("repair_ops") or [])
    repair_ops_digest = _repair_ops_digest(repair_ops)
    plan_id = _repair_plan_id(updated, forgetting_cfg)
    replay_token = _replay_token(forgetting_cfg)
    return {
        "repair_tool_version": "repair_tool_v1",
        "repair_status": "applied",
        "repaired_events_count": changed_count,
        "scanned_events_count": int(forgetting_report.get("event_count") or len(events)),
        "rewrite_applied": rewrite_applied,
        "replay_memory_path": str(replay_path),
        "repair_plan_id": plan_id,
        "replay_token": replay_token,
        "repair_ops": repair_ops,
        "repair_ops_digest": repair_ops_digest,
        "forgetting": forgetting_report,
    }


def _forgetting_policy(config: Any) -> Dict[str, Any]:
    runtime_policy = getattr(config, "runtime_policy", None)
    if not isinstance(runtime_policy, Mapping):
        return {}
    forgetting = runtime_policy.get("forgetting")
    if not isinstance(forgetting, Mapping):
        return {}
    return dict(forgetting)


def _resolve_replay_memory_path(config: Any, forgetting_cfg: Mapping[str, Any]) -> Path:
    explicit = forgetting_cfg.get("replay_memory_path")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit)
    state_dir = Path(getattr(config, "state_dir", Path("state")))
    return state_dir / "replay_memory.jsonl"


def _load_replay_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _rewrite_replay_events_atomic(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f"{path.name}.tmp-{int(time.time() * 1000)}-{os.getpid()}"
    tmp_path = path.parent / tmp_name
    with tmp_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp_path, path)


def _extract_seed_ids(event: Mapping[str, Any]) -> list[str]:
    seeds: list[str] = []
    meta = event.get("meta")
    if isinstance(meta, Mapping):
        replay = meta.get("replay")
        if isinstance(replay, Mapping):
            seeds_payload = replay.get("seeds")
            if isinstance(seeds_payload, list):
                for seed in seeds_payload:
                    if not isinstance(seed, Mapping):
                        continue
                    trace_id = seed.get("trace_id")
                    if trace_id:
                        seeds.append(str(trace_id))
    return seeds


def _collect_recall_counts(events: list[dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        for trace_id in _extract_seed_ids(event):
            counts[trace_id] = counts.get(trace_id, 0) + 1
    return counts


def _load_monument_episode_ids(memory_dir: Optional[str]) -> Tuple[set[str], Optional[str]]:
    if not memory_dir:
        return set(), "memory_dir_missing"
    try:
        from eqnet.memory.store import MemoryStore

        store = MemoryStore(Path(memory_dir))
        _, monuments = store.load_all()
        episode_ids = {str(ep_id) for mon in monuments for ep_id in (mon.episodes or []) if ep_id}
        return episode_ids, None
    except Exception as exc:  # pragma: no cover - defensive
        return set(), str(exc)


def _apply_forgetting_reweight(
    events: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    *,
    now_ts: float,
) -> Tuple[list[dict[str, Any]], Dict[str, Any]]:
    recall_k = _safe_float(cfg.get("recall_k"), 0.0)
    recall_weight = _safe_float(cfg.get("recall_weight"), 0.0)
    affect_weight = _safe_float(cfg.get("affect_weight"), 0.0)
    interference_weight = _safe_float(cfg.get("interference_weight"), 0.0)
    interference_k = _safe_float(cfg.get("interference_k"), 0.0)
    reconsolidation_rate = _safe_float(cfg.get("reconsolidation_rate"), 0.0)
    base_delta = _safe_float(cfg.get("base_delta"), 0.0)
    max_delta_w = _safe_float(cfg.get("max_delta_w"), 0.0)
    min_w = _safe_float(cfg.get("min_w"), 0.0)
    max_w = _safe_float(cfg.get("max_w"), 1.0)
    monument_w_lock = bool(cfg.get("monument_w_lock", True))
    monument_floor = cfg.get("monument_connection_floor")
    monument_floor = _safe_float(monument_floor, -1.0) if monument_floor is not None else None
    consent_floor = _safe_float(cfg.get("consent_floor"), min_w)
    consent_tags = {str(tag) for tag in (cfg.get("consent_override_tags") or [])}
    memory_dir_raw = cfg.get("memory_dir")
    memory_dir = str(memory_dir_raw) if isinstance(memory_dir_raw, str) and memory_dir_raw.strip() else None

    recall_counts = _collect_recall_counts(events)
    monument_episode_ids, monument_err = _load_monument_episode_ids(memory_dir)

    locked_applied = 0
    floors_applied = 0
    consent_applied = 0
    changed = 0
    repair_ops: list[dict[str, Any]] = []
    max_repair_ops = max(0, int(_safe_float(cfg.get("max_repair_ops"), 256)))

    for event in events:
        trace_id = str(event.get("trace_id", ""))
        episode_id = str(event.get("episode_id", ""))
        memory_kind = str(event.get("memory_kind", ""))
        w_before = _safe_float(event.get("weight"), 1.0)

        tags_raw = event.get("tags")
        tags: list[str]
        if isinstance(tags_raw, list):
            tags = [str(tag) for tag in tags_raw]
        elif tags_raw is None:
            tags = []
        else:
            tags = [str(tags_raw)]

        consent_override = any(tag in consent_tags for tag in tags)
        is_monument = memory_kind.lower() == "monument"
        lock_weight = monument_w_lock and is_monument

        if consent_override:
            w_after = min(w_before, consent_floor)
            consent_applied += 1
        elif lock_weight:
            w_after = w_before
            locked_applied += 1
        else:
            recall_count = recall_counts.get(trace_id, 0)
            recall_score = min(1.0, recall_k * float(recall_count)) if recall_k > 0 else 0.0

            emotion_mod = abs(_safe_float(event.get("emotion_modulation"), 0.0))
            affect_score = min(1.0, emotion_mod)

            interference_score = 0.0
            meta = event.get("meta")
            if isinstance(meta, Mapping):
                interference = meta.get("interference")
                if isinstance(interference, Mapping):
                    interference_score = _safe_float(interference.get("similarity"), 0.0)
            interference_score = min(1.0, interference_k * interference_score)

            delta = base_delta
            delta += recall_weight * recall_score
            delta -= affect_weight * affect_score * reconsolidation_rate
            delta -= interference_weight * interference_score
            if max_delta_w > 0:
                delta = _clamp(delta, -max_delta_w, max_delta_w)
            w_after = _clamp(w_before + delta, min_w, max_w)

        if monument_floor is not None and monument_floor >= 0 and episode_id and episode_id in monument_episode_ids:
            if w_after < monument_floor:
                w_after = monument_floor
                floors_applied += 1

        # Keep timestamps monotonic-safe if weight is missing/corrupt.
        _ = now_ts
        if abs(w_after - w_before) > 1e-12:
            changed += 1
            if len(repair_ops) < max_repair_ops:
                repair_ops.append(
                    {
                        "op": "reweight",
                        "target_hash": _target_hash(event),
                        "delta_weight": round(float(w_after - w_before), 6),
                    }
                )
        event["weight"] = float(w_after)

    report = {
        "status": "applied",
        "changed_count": int(changed),
        "event_count": int(len(events)),
        "monument_episode_count": int(len(monument_episode_ids)),
        "monument_load_error": monument_err,
        "guards": {
            "monument_locks": int(locked_applied),
            "monument_floors": int(floors_applied),
            "consent_overrides": int(consent_applied),
        },
        "repair_ops": repair_ops,
    }
    return events, report


def _target_hash(event: Mapping[str, Any]) -> str:
    payload = {
        "trace_id": str(event.get("trace_id") or ""),
        "episode_id": str(event.get("episode_id") or ""),
        "memory_kind": str(event.get("memory_kind") or ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _repair_plan_id(events: list[dict[str, Any]], cfg: Mapping[str, Any]) -> str:
    summary = {
        "event_count": len(events),
        "cfg_token": _replay_token(cfg),
    }
    raw = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _replay_token(cfg: Mapping[str, Any]) -> str:
    keys = [
        "recall_k",
        "recall_weight",
        "affect_weight",
        "interference_weight",
        "interference_k",
        "reconsolidation_rate",
        "base_delta",
        "max_delta_w",
        "min_w",
        "max_w",
        "monument_w_lock",
        "monument_connection_floor",
        "consent_floor",
    ]
    token_src = {k: cfg.get(k) for k in keys}
    raw = json.dumps(token_src, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _repair_ops_digest(ops: list[dict[str, Any]]) -> str:
    normalized = []
    for op in ops:
        normalized.append(
            {
                "op": str(op.get("op") or ""),
                "target_hash": str(op.get("target_hash") or ""),
                "delta_weight": float(_safe_float(op.get("delta_weight"), 0.0)),
            }
        )
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


__all__ = ["run_repair_cycle"]
