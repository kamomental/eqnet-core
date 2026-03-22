from __future__ import annotations

from typing import Any, Mapping, Sequence


GROUP_THREAD_REGISTRY_LIMIT = 4
GROUP_THREAD_EMA_ALPHA = 0.42


def build_group_thread_key(
    *,
    thread_hint: str = "",
    topology_state: str = "",
    top_person_ids: Sequence[str] | None = None,
    dominant_person_id: str = "",
) -> str:
    explicit = str(thread_hint or "").strip()
    if explicit:
        return explicit
    topology = str(topology_state or "").strip() or "ambient"
    people = [
        str(item).strip()
        for item in top_person_ids or []
        if str(item).strip()
    ]
    if not people:
        dominant = str(dominant_person_id or "").strip()
        if dominant:
            people = [dominant]
    if not people:
        return f"{topology}:ambient"
    return f"{topology}:{'|'.join(sorted(dict.fromkeys(people))[:GROUP_THREAD_REGISTRY_LIMIT])}"


def summarize_group_thread_registry_snapshot(
    snapshot: Mapping[str, Any] | None,
    *,
    limit: int = GROUP_THREAD_REGISTRY_LIMIT,
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
    return {
        "dominant_thread_id": dominant_thread_id,
        "top_thread_ids": top_thread_ids,
        "total_threads": len(scored),
        "uncertainty": _clamp01(snapshot.get("uncertainty")),
        "thread_scores": {
            thread_id: round(score, 4)
            for thread_id, score in scored[: max(0, limit)]
        },
    }


def update_group_thread_registry_snapshot(
    *,
    existing_snapshot: Mapping[str, Any] | None,
    thread_hint: str = "",
    topology_state: Mapping[str, Any] | None,
    dominant_person_id: str = "",
    top_person_ids: Sequence[str] | None,
    total_people: int,
    continuity_score: float,
    social_grounding: float,
    community_id: str = "",
    culture_id: str = "",
    social_role: str = "",
) -> dict[str, Any]:
    topology = dict(topology_state or {})
    topology_name = str(topology.get("state") or "").strip() or "ambient"
    thread_id = build_group_thread_key(
        thread_hint=thread_hint,
        topology_state=topology_name,
        top_person_ids=top_person_ids,
        dominant_person_id=dominant_person_id,
    )
    if not thread_id:
        return dict(existing_snapshot or {})

    existing_threads = (
        dict(existing_snapshot.get("threads") or {})
        if isinstance(existing_snapshot, Mapping)
        else {}
    )
    prior = dict(existing_threads.get(thread_id) or {})
    next_count = int(prior.get("count") or 0) + 1
    next_top_people = [
        str(item).strip()
        for item in top_person_ids or []
        if str(item).strip()
    ][:GROUP_THREAD_REGISTRY_LIMIT]
    serialized = {
        **existing_threads,
        thread_id: {
            "thread_id": thread_id,
            "last_topology_state": topology_name,
            "dominant_person_id": str(dominant_person_id or "").strip(),
            "top_person_ids": next_top_people,
            "total_people": max(int(total_people or 0), len(next_top_people)),
            "continuity_score": round(
                _ema(prior.get("continuity_score"), continuity_score),
                4,
            ),
            "social_grounding": round(
                _ema(prior.get("social_grounding"), social_grounding),
                4,
            ),
            "visibility_pressure": round(
                _ema(prior.get("visibility_pressure"), topology.get("visibility_pressure")),
                4,
            ),
            "threading_pressure": round(
                _ema(prior.get("threading_pressure"), topology.get("threading_pressure")),
                4,
            ),
            "hierarchy_pressure": round(
                _ema(prior.get("hierarchy_pressure"), topology.get("hierarchy_pressure")),
                4,
            ),
            "community_id": str(community_id or "").strip(),
            "culture_id": str(culture_id or "").strip(),
            "social_role": str(social_role or "").strip(),
            "count": next_count,
            "confidence": round(_clamp01(0.26 + min(next_count, 5) * 0.12), 4),
        },
    }
    summary = summarize_group_thread_registry_snapshot(
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
    continuity_score = _clamp01(source.get("continuity_score"))
    social_grounding = _clamp01(source.get("social_grounding"))
    threading_pressure = _clamp01(source.get("threading_pressure"))
    visibility_pressure = _clamp01(source.get("visibility_pressure"))
    hierarchy_pressure = _clamp01(source.get("hierarchy_pressure"))
    total_people = int(source.get("total_people") or 0)
    count = int(source.get("count") or 0)
    repeat_bonus = _clamp01(min(count, 4) / 4.0)
    people_bonus = _clamp01(min(total_people, 4) / 4.0)
    return _clamp01(
        continuity_score * 0.28
        + social_grounding * 0.2
        + threading_pressure * 0.18
        + visibility_pressure * 0.12
        + hierarchy_pressure * 0.1
        + repeat_bonus * 0.08
        + people_bonus * 0.04
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


def _ema(previous: Any, current: Any, *, alpha: float = GROUP_THREAD_EMA_ALPHA) -> float:
    current_value = _clamp01(current)
    previous_value = _clamp01(previous)
    if previous_value <= 0.0:
        return current_value
    return _clamp01(previous_value * (1.0 - alpha) + current_value * alpha)


def _empty_summary() -> dict[str, Any]:
    return {
        "dominant_thread_id": "",
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
