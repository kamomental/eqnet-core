from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from inner_os.memory_core import MemoryCore
from inner_os.schemas import INNER_OS_WORKING_MEMORY_SNAPSHOT_SCHEMA


PROMOTION_READINESS_THRESHOLD = 0.45
PROMOTION_RECENCY_HOURS = 48
STOP_TERMS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "still",
    "why",
    "how",
    "what",
    "when",
    "where",
    "which",
}


def build_inner_os_working_memory_snapshot(
    *,
    memory_path: str | Path | None = None,
    day: date | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    target_day = day
    current_time = now or datetime.utcnow()
    if target_day is None:
        target_day = current_time.date()

    records = list(
        _iter_working_memory_records(
            memory_path=memory_path,
            day=target_day,
            now=current_time,
        )
    )
    latest = records[-1] if records else {}

    focus_counts = Counter(str(item.get("current_focus") or "ambient") for item in records)
    anchor_counts = Counter(str(item.get("focus_anchor") or "").strip() for item in records if str(item.get("focus_anchor") or "").strip())
    loop_counts = Counter()
    autobiographical_mode_counts = Counter()
    autobiographical_anchor_counts = Counter()
    autobiographical_focus_counts = Counter()
    pressures = []
    pending_values = []
    carryover_values = []
    unresolved_values = []
    autobiographical_strengths = []
    for item in records:
        for loop in item.get("open_loops") or []:
            loop_counts[str(loop)] += 1
        autobiographical_mode = str(item.get("autobiographical_thread_mode") or "").strip()
        autobiographical_anchor = str(item.get("autobiographical_thread_anchor") or "").strip()
        autobiographical_focus = str(item.get("autobiographical_thread_focus") or "").strip()
        if autobiographical_mode:
            autobiographical_mode_counts[autobiographical_mode] += 1
        if autobiographical_anchor:
            autobiographical_anchor_counts[autobiographical_anchor] += 1
        if autobiographical_focus:
            autobiographical_focus_counts[autobiographical_focus] += 1
        pressures.append(_safe_float(item.get("memory_pressure"), 0.0))
        pending_values.append(_safe_float(item.get("pending_meaning"), 0.0))
        carryover_values.append(_safe_float(item.get("carryover_load"), 0.0))
        unresolved_values.append(_safe_float(item.get("unresolved_count"), 0.0))
        autobiographical_strengths.append(
            _safe_float(item.get("autobiographical_thread_strength"), 0.0)
        )

    dominant_focus = _counter_top(focus_counts, default="ambient")
    dominant_anchor = _counter_top(anchor_counts, default=str(latest.get("focus_anchor") or "").strip())
    trace_count = len(records)
    mean_memory_pressure = _mean(pressures)
    peak_memory_pressure = max(pressures, default=0.0)
    mean_pending_meaning = _mean(pending_values)
    mean_carryover_load = _mean(carryover_values)
    mean_unresolved = _mean(unresolved_values)
    mean_autobiographical_thread_strength = _mean(autobiographical_strengths)
    focus_repetition = _clamp01((focus_counts.get(dominant_focus, 0) / max(trace_count, 1)) if trace_count else 0.0)
    anchor_repetition = _clamp01((anchor_counts.get(dominant_anchor, 0) / max(trace_count, 1)) if dominant_anchor and trace_count else 0.0)
    dominant_autobiographical_thread_mode = _counter_top(
        autobiographical_mode_counts,
        default=str(latest.get("autobiographical_thread_mode") or "").strip(),
    )
    dominant_autobiographical_thread_anchor = _counter_top(
        autobiographical_anchor_counts,
        default=str(latest.get("autobiographical_thread_anchor") or "").strip(),
    )
    dominant_autobiographical_thread_focus = _counter_top(
        autobiographical_focus_counts,
        default=str(latest.get("autobiographical_thread_focus") or "").strip(),
    )
    promotion_readiness = _clamp01(
        mean_memory_pressure * 0.26
        + peak_memory_pressure * 0.18
        + mean_pending_meaning * 0.2
        + mean_carryover_load * 0.18
        + _clamp01(mean_unresolved / 3.0) * 0.1
        + focus_repetition * 0.05
        + anchor_repetition * 0.03
        + mean_autobiographical_thread_strength * 0.08
    )
    autobiographical_pressure = _clamp01(
        mean_pending_meaning * 0.3
        + mean_carryover_load * 0.22
        + focus_repetition * 0.14
        + anchor_repetition * 0.14
        + _clamp01(mean_unresolved / 3.0) * 0.1
        + mean_autobiographical_thread_strength * 0.1
    )

    return {
        "schema": INNER_OS_WORKING_MEMORY_SNAPSHOT_SCHEMA,
        "snapshot": {
            "available": bool(records),
            "current_focus": dominant_focus,
            "focus_anchor": dominant_anchor,
            "focus_text": str(latest.get("focus_text") or latest.get("text") or "").strip(),
            "source_trace_count": trace_count,
            "unresolved_count": int(round(mean_unresolved)) if records else 0,
            "pending_meaning": round(mean_pending_meaning, 4),
            "carryover_load": round(mean_carryover_load, 4),
            "mean_memory_pressure": round(mean_memory_pressure, 4),
            "peak_memory_pressure": round(peak_memory_pressure, 4),
            "promotion_readiness": round(promotion_readiness, 4),
            "autobiographical_pressure": round(autobiographical_pressure, 4),
            "autobiographical_thread_mode": dominant_autobiographical_thread_mode,
            "autobiographical_thread_anchor": dominant_autobiographical_thread_anchor,
            "autobiographical_thread_focus": dominant_autobiographical_thread_focus,
            "autobiographical_thread_strength": round(mean_autobiographical_thread_strength, 4),
            "dominant_open_loops": [item for item, _ in loop_counts.most_common(3)],
            "culture_id": latest.get("culture_id"),
            "community_id": latest.get("community_id"),
            "social_role": latest.get("social_role"),
            "long_term_theme_focus": str(latest.get("long_term_theme_focus") or "").strip(),
            "long_term_theme_anchor": str(latest.get("long_term_theme_anchor") or "").strip(),
            "long_term_theme_strength": round(_safe_float(latest.get("long_term_theme_strength"), 0.0), 4),
            "long_term_theme_kind": str(latest.get("long_term_theme_kind") or "").strip(),
            "long_term_theme_summary": str(latest.get("long_term_theme_summary") or "").strip()[:160],
            "window_day": target_day.isoformat(),
            "last_timestamp": _timestamp_to_iso(latest.get("timestamp")),
        },
        "derived_inputs": {
            "memory_path": str(Path(memory_path) if memory_path else MemoryCore().path),
            "focus_counts": dict(focus_counts),
            "anchor_counts": dict(anchor_counts),
            "open_loop_counts": dict(loop_counts),
            "autobiographical_mode_counts": dict(autobiographical_mode_counts),
        },
    }


def write_inner_os_working_memory_snapshot(
    *,
    out_path: str | Path,
    memory_path: str | Path | None = None,
    day: date | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload = build_inner_os_working_memory_snapshot(
        memory_path=memory_path,
        day=day,
        now=now,
    )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def derive_working_memory_replay_bias(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, Mapping):
        return {}
    if not snapshot.get("available"):
        return {}
    current_focus = str(snapshot.get("current_focus") or "").strip()
    focus_anchor = str(snapshot.get("focus_anchor") or "").strip()
    focus_text = str(snapshot.get("focus_text") or "").strip()
    open_loops = [str(item).strip() for item in snapshot.get("dominant_open_loops") or [] if str(item).strip()]
    terms = _terms(" ".join(filter(None, [current_focus, focus_anchor, focus_text, *open_loops])))
    if not terms:
        return {}
    readiness = _safe_float(snapshot.get("promotion_readiness"), 0.0)
    autobiographical_pressure = _safe_float(snapshot.get("autobiographical_pressure"), 0.0)
    strength = _clamp01(readiness * 0.65 + autobiographical_pressure * 0.35)
    if strength <= 0.0:
        return {}
    return {
        "current_focus": current_focus,
        "focus_anchor": focus_anchor,
        "focus_text": focus_text,
        "open_loops": open_loops,
        "promotion_readiness": round(readiness, 4),
        "autobiographical_pressure": round(autobiographical_pressure, 4),
        "strength": round(strength, 4),
        "terms": sorted(terms),
    }


def derive_reconstructed_replay_carryover(
    *,
    memory_path: str | Path | None = None,
    now: datetime | None = None,
    hours: int = 72,
) -> dict[str, Any]:
    current_time = now or datetime.utcnow()
    cutoff = current_time - timedelta(hours=max(1, int(hours)))
    path = Path(memory_path) if memory_path else MemoryCore().path
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
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
                if str(payload.get("kind") or "") != "reconstructed":
                    continue
                ts = _coerce_datetime(payload.get("timestamp"))
                if ts is None or ts < cutoff:
                    continue
                reinforcement = _safe_float(payload.get("working_memory_replay_reinforcement"), 0.0)
                if reinforcement <= 0.0:
                    continue
                latest = payload
    except OSError:
        return {}
    if not latest:
        return {}
    focus = str(latest.get("working_memory_replay_focus") or "").strip()
    anchor = str(latest.get("working_memory_replay_anchor") or latest.get("memory_anchor") or "").strip()
    strength = _clamp01(_safe_float(latest.get("working_memory_replay_strength"), 0.0))
    reinforcement = _clamp01(_safe_float(latest.get("working_memory_replay_reinforcement"), 0.0))
    alignment = _clamp01(_safe_float(latest.get("working_memory_replay_alignment"), 0.0))
    long_term_theme_summary = str(latest.get("long_term_theme_summary") or "").strip()
    long_term_theme_alignment = _clamp01(_safe_float(latest.get("long_term_theme_alignment"), 0.0))
    long_term_theme_reinforcement = _clamp01(_safe_float(latest.get("long_term_theme_reinforcement"), 0.0))
    if not focus and not anchor:
        return {}
    return {
        "focus": focus,
        "anchor": anchor,
        "strength": round(_clamp01(strength * 0.55 + reinforcement * 0.45), 4),
        "reinforcement": round(reinforcement, 4),
        "alignment": round(alignment, 4),
        "long_term_theme_summary": long_term_theme_summary[:160],
        "long_term_theme_alignment": round(long_term_theme_alignment, 4),
        "long_term_theme_reinforcement": round(long_term_theme_reinforcement, 4),
        "source_episode_id": latest.get("source_episode_id"),
        "kind": "reconstructed_replay_carryover",
    }


def derive_working_memory_seed_from_signature(
    signature_summary: Mapping[str, Any] | None,
    replay_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    signature = dict(signature_summary) if isinstance(signature_summary, Mapping) else {}
    replay = dict(replay_summary) if isinstance(replay_summary, Mapping) else {}
    dominant_focus = str(signature.get("dominant_focus") or "").strip()
    dominant_anchor = str(signature.get("dominant_anchor") or "").strip()
    recurrence_weight = _safe_float(signature.get("recurrence_weight"), 0.0)
    promotion_readiness = _safe_float(signature.get("promotion_readiness_mean"), 0.0)
    autobiographical_pressure = _safe_float(signature.get("autobiographical_pressure_mean"), 0.0)
    replay_strength = _safe_float(replay.get("strength"), 0.0)
    conscious_strength = _safe_float(replay.get("conscious_memory_strength"), 0.0)
    conscious_overlap = _safe_float(replay.get("conscious_memory_overlap"), 0.0)
    replay_focus = str(replay.get("focus") or "").strip()
    replay_anchor = str(replay.get("anchor") or "").strip()
    long_term_theme = (
        dict(signature.get("long_term_theme"))
        if isinstance(signature.get("long_term_theme"), Mapping)
        else {}
    )
    focus = replay_focus or dominant_focus
    anchor = replay_anchor or dominant_anchor
    if not focus and not anchor:
        return {}
    conscious_reinforcement = conscious_strength * conscious_overlap
    semantic_seed_strength = _clamp01(
        recurrence_weight * 0.18
        + promotion_readiness * 0.26
        + autobiographical_pressure * 0.26
        + replay_strength * 0.24
        + conscious_reinforcement * 0.06
    )
    seed = {
        "semantic_seed_focus": focus,
        "semantic_seed_anchor": anchor,
        "semantic_seed_strength": round(semantic_seed_strength, 4),
        "semantic_seed_recurrence": round(recurrence_weight, 4),
        "semantic_seed_autobiographical_pressure": round(autobiographical_pressure, 4),
    }
    theme_focus = str(long_term_theme.get("focus") or focus).strip()
    theme_anchor = str(long_term_theme.get("anchor") or anchor).strip()
    theme_strength = _clamp01(_safe_float(long_term_theme.get("strength"), semantic_seed_strength))
    if theme_focus or theme_anchor:
        seed.update(
            {
                "long_term_theme_focus": theme_focus,
                "long_term_theme_anchor": theme_anchor,
                "long_term_theme_strength": round(theme_strength, 4),
                "long_term_theme_kind": str(long_term_theme.get("kind") or theme_focus or "ambient").strip(),
                "long_term_theme_summary": str(long_term_theme.get("summary") or "").strip()[:160],
            }
        )
    return seed


def merge_working_memory_snapshot_with_seed(
    snapshot: Mapping[str, Any] | None,
    semantic_seed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base = dict(snapshot) if isinstance(snapshot, Mapping) else {}
    seed = dict(semantic_seed) if isinstance(semantic_seed, Mapping) else {}
    if not base:
        return seed
    if not seed:
        return base
    merged = dict(base)
    seed_strength = _safe_float(seed.get("semantic_seed_strength"), 0.0)
    if not merged.get("current_focus") and seed.get("semantic_seed_focus"):
        merged["current_focus"] = seed.get("semantic_seed_focus")
    if not merged.get("focus_anchor") and seed.get("semantic_seed_anchor"):
        merged["focus_anchor"] = seed.get("semantic_seed_anchor")
    merged["carryover_load"] = round(
        _clamp01(_safe_float(merged.get("carryover_load"), 0.0) + seed_strength * 0.12),
        4,
    )
    merged["pending_meaning"] = round(
        _clamp01(_safe_float(merged.get("pending_meaning"), 0.0) + seed_strength * 0.08),
        4,
    )
    merged["promotion_readiness"] = round(
        _clamp01(_safe_float(merged.get("promotion_readiness"), 0.0) + seed_strength * 0.06),
        4,
    )
    merged.update(seed)
    return merged


def merge_replay_carryover_summaries(
    primary: Mapping[str, Any] | None,
    secondary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    left = dict(primary) if isinstance(primary, Mapping) else {}
    right = dict(secondary) if isinstance(secondary, Mapping) else {}
    if not left:
        return right
    if not right:
        return left
    left_focus = str(left.get("focus") or left.get("current_focus") or "").strip()
    left_anchor = str(left.get("anchor") or left.get("focus_anchor") or "").strip()
    right_focus = str(right.get("focus") or right.get("current_focus") or "").strip()
    right_anchor = str(right.get("anchor") or right.get("focus_anchor") or "").strip()
    left_strength = _clamp01(_safe_float(left.get("strength"), 0.0))
    right_strength = _clamp01(_safe_float(right.get("strength"), 0.0))
    if left_focus == right_focus and left_anchor == right_anchor and (left_focus or left_anchor):
        merged = dict(left)
        merged["strength"] = round(_clamp01(max(left_strength, right_strength) + min(left_strength, right_strength) * 0.25), 4)
        if "matched_events" in left or "matched_events" in right:
            merged["matched_events"] = int(left.get("matched_events") or 0) + int(right.get("matched_events") or 0)
        if "reinforcement" in right:
            merged["reinforcement"] = right.get("reinforcement")
        if "alignment" in right:
            merged["alignment"] = right.get("alignment")
        if "long_term_theme_summary" in right:
            merged["long_term_theme_summary"] = right.get("long_term_theme_summary")
        if "long_term_theme_alignment" in right:
            merged["long_term_theme_alignment"] = right.get("long_term_theme_alignment")
        if "long_term_theme_reinforcement" in right:
            merged["long_term_theme_reinforcement"] = right.get("long_term_theme_reinforcement")
        return merged
    return right if right_strength > left_strength else left


def merge_conscious_working_memory_seed(
    replay_summary: Mapping[str, Any] | None,
    conscious_seed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base = dict(replay_summary) if isinstance(replay_summary, Mapping) else {}
    seed = dict(conscious_seed) if isinstance(conscious_seed, Mapping) else {}
    focus = str(seed.get("focus") or "").strip()
    anchor = str(seed.get("anchor") or "").strip()
    strength = _clamp01(_safe_float(seed.get("strength"), 0.0))
    if not focus and not anchor:
        return base
    if not base:
        return {
            "focus": focus,
            "anchor": anchor,
            "strength": round(strength, 4),
            "source": "conscious_memory",
        }
    merged = dict(base)
    base_focus = str(merged.get("focus") or "").strip()
    base_anchor = str(merged.get("anchor") or "").strip()
    if not merged.get("focus") and focus:
        merged["focus"] = focus
    if not merged.get("anchor") and anchor:
        merged["anchor"] = anchor
    focus_match = bool(base_focus and focus and base_focus == focus)
    anchor_match = bool(base_anchor and anchor and base_anchor == anchor)
    overlap = 1.0 if (focus_match or anchor_match) else 0.0
    boosted_strength = _safe_float(merged.get("strength"), 0.0)
    if overlap > 0.0:
        boosted_strength = _clamp01(boosted_strength + strength * (0.18 if focus_match and anchor_match else 0.1))
    merged["strength"] = round(
        _clamp01(max(boosted_strength, strength * 0.8)),
        4,
    )
    merged["conscious_memory_strength"] = round(strength, 4)
    merged["conscious_memory_overlap"] = round(overlap, 4)
    return merged


def prioritize_weekly_abstraction_episodes(
    *,
    episodes: Iterable[Mapping[str, Any]],
    replay_summary: Mapping[str, Any] | None,
    limit: int = 8,
    lookback: int = 16,
) -> list[Dict[str, Any]]:
    ordered = [dict(item) for item in episodes]
    if not ordered:
        return []
    limit = max(1, int(limit))
    lookback = max(limit, int(lookback))
    window = ordered[-lookback:]
    if not isinstance(replay_summary, Mapping):
        return window[-limit:]

    focus = str(replay_summary.get("focus") or "").strip().lower()
    anchor = str(replay_summary.get("anchor") or "").strip().lower()
    strength = _clamp01(_safe_float(replay_summary.get("strength"), 0.0))
    related_person_id = str(replay_summary.get("related_person_id") or "").strip()
    relation_seed_strength = _clamp01(_safe_float(replay_summary.get("relation_seed_strength"), 0.0))
    if strength <= 0.0 or (not focus and not anchor):
        if not related_person_id or relation_seed_strength <= 0.0:
            return window[-limit:]

    ranked: list[tuple[float, int, Dict[str, Any]]] = []
    total = max(1, len(window) - 1)
    for idx, episode in enumerate(window):
        promotion = episode.get("working_memory_promotion")
        if not isinstance(promotion, Mapping):
            promotion = {}
        dominant_focus = str(promotion.get("dominant_focus") or promotion.get("current_focus") or "").strip().lower()
        dominant_anchor = str(promotion.get("dominant_anchor") or promotion.get("focus_anchor") or "").strip().lower()
        focus_match = 1.0 if focus and dominant_focus and dominant_focus == focus else 0.0
        anchor_match = 1.0 if anchor and dominant_anchor and dominant_anchor == anchor else 0.0
        episode_person_id = str(
            promotion.get("related_person_id")
            or episode.get("related_person_id")
            or ""
        ).strip()
        person_match = 1.0 if related_person_id and episode_person_id and episode_person_id == related_person_id else 0.0
        recency = idx / total if total else 1.0
        score = (
            recency * 0.35
            + strength * (focus_match * 0.75 + anchor_match * 0.55)
            + relation_seed_strength * person_match * 0.42
        )
        ranked.append((score, idx, episode))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = sorted(ranked[:limit], key=lambda item: item[1])
    return [item[2] for item in selected]


def select_working_memory_promotion_candidates(
    *,
    experiences: Iterable[Mapping[str, Any]],
    memory_path: str | Path | None = None,
    day: date | None = None,
    now: datetime | None = None,
    limit: int = 4,
    semantic_seed: Mapping[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    current_time = now or datetime.utcnow()
    payload = build_inner_os_working_memory_snapshot(
        memory_path=memory_path,
        day=day or current_time.date(),
        now=current_time,
    )
    snapshot = payload.get("snapshot") if isinstance(payload, dict) else {}
    if not isinstance(snapshot, Mapping):
        return []
    if not snapshot.get("available"):
        return []
    readiness = _safe_float(snapshot.get("promotion_readiness"), 0.0)
    semantic_seed_strength = _safe_float((semantic_seed or {}).get("semantic_seed_strength"), 0.0)
    related_person_id = str((semantic_seed or {}).get("related_person_id") or "").strip()
    relation_seed_strength = _clamp01(_safe_float((semantic_seed or {}).get("relation_seed_strength"), 0.0))
    if readiness < PROMOTION_READINESS_THRESHOLD:
        return []

    focus_terms = _terms(
        " ".join(
            filter(
                None,
                [
                    str(snapshot.get("focus_anchor") or "").strip(),
                    str(snapshot.get("focus_text") or "").strip(),
                    " ".join(str(item) for item in snapshot.get("dominant_open_loops") or []),
                ],
            )
        )
    )
    if not focus_terms:
        return []

    cutoff = current_time - timedelta(hours=PROMOTION_RECENCY_HOURS)
    ranked: list[tuple[float, Dict[str, Any]]] = []
    for experience in experiences:
        item = dict(experience)
        ts = _coerce_datetime(item.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        dialogue = str(item.get("dialogue") or "").strip()
        if not dialogue:
            continue
        match_score = _score_terms(focus_terms, dialogue)
        if match_score <= 0.0:
            continue
        emotion_intensity = _safe_float(item.get("emotion_intensity"), 0.0)
        item_person_id = str(item.get("related_person_id") or "").strip()
        person_match = 1.0 if related_person_id and item_person_id and item_person_id == related_person_id else 0.0
        score = (
            match_score * 0.55
            + readiness * 0.25
            + _clamp01(emotion_intensity) * 0.15
            + semantic_seed_strength * 0.05
            + relation_seed_strength * person_match * 0.18
            + (0.05 if ts.date() == current_time.date() else 0.0)
        )
        context = dict(item.get("context") or {})
        context["working_memory_promotion"] = {
            "current_focus": str(snapshot.get("current_focus") or ""),
            "focus_anchor": str(snapshot.get("focus_anchor") or ""),
            "promotion_readiness": round(readiness, 4),
            "autobiographical_pressure": round(_safe_float(snapshot.get("autobiographical_pressure"), 0.0), 4),
            "pending_meaning": round(_safe_float(snapshot.get("pending_meaning"), 0.0), 4),
            "carryover_load": round(_safe_float(snapshot.get("carryover_load"), 0.0), 4),
            "semantic_seed_strength": round(semantic_seed_strength, 4),
            "related_person_id": related_person_id or item_person_id,
            "relation_seed_strength": round(relation_seed_strength, 4),
        }
        item["context"] = context
        ranked.append((score, item))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked[: max(1, int(limit))]]


def _iter_working_memory_records(
    *,
    memory_path: str | Path | None,
    day: date,
    now: datetime,
) -> Iterable[Dict[str, Any]]:
    path = Path(memory_path) if memory_path else MemoryCore().path
    if not path.exists():
        return []
    cutoff = now - timedelta(hours=24)
    records: list[Dict[str, Any]] = []
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
                if str(payload.get("kind") or "") != "working_memory_trace":
                    continue
                ts = _coerce_datetime(payload.get("timestamp"))
                if ts is None:
                    continue
                if ts.date() != day and ts < cutoff:
                    continue
                records.append(payload)
    except OSError:
        return []
    return records


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(value))
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _counter_top(counter: Counter[str], *, default: str) -> str:
    if not counter:
        return default
    return counter.most_common(1)[0][0]


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _timestamp_to_iso(value: Any) -> str | None:
    ts = _coerce_datetime(value)
    if ts is None:
        return None
    return ts.isoformat()


def _terms(text: str) -> set[str]:
    parts = [part.strip().lower() for part in str(text or "").replace("\n", " ").split(" ") if part.strip()]
    return {part for part in parts if len(part) >= 3 and part not in STOP_TERMS}


def _score_terms(query_terms: set[str], haystack: str) -> float:
    if not query_terms:
        return 0.0
    lowered = str(haystack or "").lower()
    matches = sum(1 for term in query_terms if term in lowered)
    return float(matches / max(len(query_terms), 1))
