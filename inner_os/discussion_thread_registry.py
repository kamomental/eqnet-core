from __future__ import annotations

from typing import Any, Mapping

from .anchor_normalization import normalize_anchor_hint


DISCUSSION_THREAD_REGISTRY_LIMIT = 4
DISCUSSION_THREAD_EMA_ALPHA = 0.42


def build_discussion_thread_key(
    *,
    anchor: str = "",
    discussion_state: str = "",
    issue_state: str = "",
) -> str:
    base_anchor = str(anchor or "").strip()
    if base_anchor:
        compact = "_".join(base_anchor.lower().split())
        return compact[:72]
    discussion = str(discussion_state or "").strip() or "ambient"
    issue = str(issue_state or "").strip() or "ambient"
    return f"{discussion}:{issue}"


def summarize_discussion_thread_registry_snapshot(
    snapshot: Mapping[str, Any] | None,
    *,
    limit: int = DISCUSSION_THREAD_REGISTRY_LIMIT,
) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        return _empty_summary()
    threads = snapshot.get("threads")
    if not isinstance(threads, Mapping):
        return {
            **_empty_summary(),
            "uncertainty": _clamp01(snapshot.get("uncertainty")),
        }
    scored = sorted(
        (
            (str(thread_id), _thread_node_score(payload))
            for thread_id, payload in threads.items()
            if isinstance(payload, Mapping) and str(thread_id).strip()
        ),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    top_thread_ids = [thread_id for thread_id, _ in scored[: max(0, limit)]]
    dominant_thread_id = top_thread_ids[0] if top_thread_ids else ""
    dominant_payload = dict(threads.get(dominant_thread_id) or {})
    return {
        "dominant_thread_id": dominant_thread_id,
        "dominant_anchor": str(dominant_payload.get("anchor") or "").strip(),
        "dominant_issue_state": str(dominant_payload.get("last_issue_state") or "").strip(),
        "top_thread_ids": top_thread_ids,
        "total_threads": len(scored),
        "uncertainty": _clamp01(snapshot.get("uncertainty")),
        "thread_scores": {
            thread_id: round(score, 4)
            for thread_id, score in scored[: max(0, limit)]
        },
    }


def update_discussion_thread_registry_snapshot(
    *,
    existing_snapshot: Mapping[str, Any] | None,
    recent_dialogue_state: Mapping[str, Any] | None,
    discussion_thread_state: Mapping[str, Any] | None,
    issue_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    recent = dict(recent_dialogue_state or {})
    discussion = dict(discussion_thread_state or {})
    issue = dict(issue_state or {})

    anchor = str(
        issue.get("issue_anchor")
        or discussion.get("topic_anchor")
        or recent.get("recent_anchor")
        or ""
    ).strip()
    anchor = normalize_anchor_hint(anchor, limit=48) or anchor
    discussion_state = str(discussion.get("state") or "").strip()
    issue_state_name = str(issue.get("state") or "").strip()
    if not anchor and not discussion_state and not issue_state_name:
        return dict(existing_snapshot or {})

    thread_id = build_discussion_thread_key(
        anchor=anchor,
        discussion_state=discussion_state,
        issue_state=issue_state_name,
    )
    existing_threads = (
        dict(existing_snapshot.get("threads") or {})
        if isinstance(existing_snapshot, Mapping)
        else {}
    )
    prior = dict(existing_threads.get(thread_id) or {})
    next_count = int(prior.get("count") or 0) + 1

    serialized = {
        **existing_threads,
        thread_id: {
            "thread_id": thread_id,
            "anchor": anchor or str(prior.get("anchor") or "").strip(),
            "last_recent_dialogue_state": str(recent.get("state") or prior.get("last_recent_dialogue_state") or "").strip(),
            "last_discussion_state": discussion_state or str(prior.get("last_discussion_state") or "").strip(),
            "last_issue_state": issue_state_name or str(prior.get("last_issue_state") or "").strip(),
            "thread_carry": round(_ema(prior.get("thread_carry"), recent.get("thread_carry")), 4),
            "reopen_pressure": round(_ema(prior.get("reopen_pressure"), recent.get("reopen_pressure")), 4),
            "unresolved_pressure": round(_ema(prior.get("unresolved_pressure"), discussion.get("unresolved_pressure")), 4),
            "revisit_readiness": round(_ema(prior.get("revisit_readiness"), discussion.get("revisit_readiness")), 4),
            "thread_visibility": round(_ema(prior.get("thread_visibility"), discussion.get("thread_visibility")), 4),
            "question_pressure": round(_ema(prior.get("question_pressure"), issue.get("question_pressure")), 4),
            "pause_readiness": round(_ema(prior.get("pause_readiness"), issue.get("pause_readiness")), 4),
            "resolution_readiness": round(_ema(prior.get("resolution_readiness"), issue.get("resolution_readiness")), 4),
            "count": next_count,
            "confidence": round(_clamp01(0.24 + min(next_count, 5) * 0.12), 4),
        },
    }
    summary = summarize_discussion_thread_registry_snapshot(
        {
            "threads": serialized,
            "uncertainty": _registry_uncertainty(serialized),
        }
    )
    return {
        "threads": serialized,
        "uncertainty": summary.get("uncertainty", 1.0),
        **summary,
    }


def _thread_node_score(payload: Mapping[str, Any] | None) -> float:
    source = dict(payload or {})
    return _clamp01(
        _clamp01(source.get("thread_carry")) * 0.16
        + _clamp01(source.get("reopen_pressure")) * 0.12
        + _clamp01(source.get("unresolved_pressure")) * 0.2
        + _clamp01(source.get("revisit_readiness")) * 0.14
        + _clamp01(source.get("thread_visibility")) * 0.16
        + _clamp01(source.get("question_pressure")) * 0.08
        + _clamp01(source.get("pause_readiness")) * 0.08
        + _clamp01(source.get("resolution_readiness")) * 0.06
        + _clamp01(min(int(source.get("count") or 0), 4) / 4.0) * 0.1
    )


def _registry_uncertainty(threads: Mapping[str, Any]) -> float:
    if not threads:
        return 1.0
    strongest = max(
        (
            _thread_node_score(payload)
            for payload in threads.values()
            if isinstance(payload, Mapping)
        ),
        default=0.0,
    )
    return round(_clamp01(1.0 - strongest * 0.72), 4)


def _ema(previous: Any, current: Any, *, alpha: float = DISCUSSION_THREAD_EMA_ALPHA) -> float:
    current_value = _clamp01(current)
    previous_value = _clamp01(previous)
    if previous_value <= 0.0:
        return current_value
    return _clamp01(previous_value * (1.0 - alpha) + current_value * alpha)


def _empty_summary() -> dict[str, Any]:
    return {
        "dominant_thread_id": "",
        "dominant_anchor": "",
        "dominant_issue_state": "",
        "top_thread_ids": [],
        "total_threads": 0,
        "uncertainty": 1.0,
        "thread_scores": {},
    }


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return float(numeric)
