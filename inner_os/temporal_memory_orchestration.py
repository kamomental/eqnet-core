from __future__ import annotations

from typing import Any, Mapping

from .memory_evidence_bundle import (
    MemoryEvidenceItem,
    ReentryContext,
    TemporalConstraint,
    build_memory_evidence_bundle,
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return float(numeric)


def build_temporal_memory_evidence_bundle(
    *,
    cue_text: str,
    current_state: Mapping[str, Any] | None = None,
    world_snapshot: Mapping[str, Any] | None = None,
    recall_payload: Mapping[str, Any] | None = None,
    retrieval_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = dict(current_state or {})
    world = dict(world_snapshot or {})
    recall = dict(recall_payload or {})
    retrieval = dict(retrieval_summary or {})

    focus = (
        _text(recall.get("summary"))
        or _text(state.get("current_focus"))
        or _text(recall.get("long_term_theme_focus"))
        or _text(recall.get("semantic_seed_focus"))
        or _text(recall.get("memory_anchor"))
    )

    facts_current: list[MemoryEvidenceItem] = []
    if _text(recall.get("summary")):
        facts_current.append(
            MemoryEvidenceItem(
                evidence_id=_text(recall.get("source_episode_id")) or "recall:current",
                kind=_text(recall.get("record_kind")) or "memory",
                summary=_text(recall.get("summary")),
                temporal_status="current",
                anchor=_text(recall.get("memory_anchor")),
                source_session_id=_text(recall.get("source_episode_id")),
                source_record_id=_text(recall.get("source_episode_id")),
                related_person_id=_text(recall.get("related_person_id"))
                or _text(recall.get("counterpart_person_id")),
                group_thread_id=_text(state.get("group_thread_id"))
                or _text(state.get("group_thread_focus")),
                weight=max(
                    _float01(recall.get("primed_weight")),
                    _float01(recall.get("consolidation_priority")),
                    0.4,
                ),
                tags=tuple(
                    item
                    for item in (
                        _text(recall.get("record_provenance")),
                        _text(recall.get("policy_hint")),
                        _text(recall.get("long_term_theme_kind")),
                    )
                    if item
                ),
            )
        )

    facts_superseded = [
        MemoryEvidenceItem.from_mapping(item)
        for item in retrieval.get("superseded_facts") or ()
        if isinstance(item, Mapping)
        and _text(item.get("evidence_id"))
        and _text(item.get("summary"))
    ]

    timeline_events: list[MemoryEvidenceItem] = []
    if _text(recall.get("source_episode_id")) and _text(recall.get("summary")):
        timeline_events.append(
            MemoryEvidenceItem(
                evidence_id=f"timeline:{_text(recall.get('source_episode_id'))}",
                kind="timeline_event",
                summary=_text(recall.get("summary")),
                temporal_status="timeline",
                anchor=_text(recall.get("memory_anchor")),
                source_session_id=_text(recall.get("source_episode_id")),
                related_person_id=_text(recall.get("related_person_id"))
                or _text(recall.get("counterpart_person_id")),
                group_thread_id=_text(state.get("group_thread_id"))
                or _text(state.get("group_thread_focus")),
                weight=max(_float01(recall.get("replay_signature_strength")), 0.32),
            )
        )
    timeline_events.extend(
        MemoryEvidenceItem.from_mapping(item)
        for item in retrieval.get("timeline_events") or ()
        if isinstance(item, Mapping)
        and _text(item.get("evidence_id"))
        and _text(item.get("summary"))
    )

    temporal_constraints: list[TemporalConstraint] = []
    if _text(recall.get("replay_signature_focus")):
        temporal_constraints.append(
            TemporalConstraint(
                kind="replay_signature",
                summary=(
                    f"prefer replay-aligned material around "
                    f"{_text(recall.get('replay_signature_focus'))}"
                ),
                focus=_text(recall.get("replay_signature_focus")),
                priority=max(_float01(recall.get("replay_signature_strength")), 0.24),
                source="working_memory_replay",
            )
        )
    if _text(recall.get("semantic_seed_focus")):
        temporal_constraints.append(
            TemporalConstraint(
                kind="semantic_seed",
                summary=(
                    f"preserve semantic continuity around "
                    f"{_text(recall.get('semantic_seed_focus'))}"
                ),
                focus=_text(recall.get("semantic_seed_focus")),
                priority=max(_float01(recall.get("semantic_seed_strength")), 0.22),
                source="semantic_seed",
            )
        )
    if _text(recall.get("long_term_theme_focus")):
        temporal_constraints.append(
            TemporalConstraint(
                kind="long_term_theme",
                summary=(
                    f"keep the long-term theme "
                    f"{_text(recall.get('long_term_theme_focus'))} coherent"
                ),
                focus=_text(recall.get("long_term_theme_focus")),
                priority=max(_float01(recall.get("long_term_theme_strength")), 0.28),
                source="long_term_theme",
            )
        )

    reentry_window = (
        _text(state.get("agenda_window_state"))
        or _text(state.get("agenda_window_focus"))
        or _text(recall.get("agenda_window_state"))
    )
    reentry_priority = max(
        _float01(state.get("agenda_window_bias")),
        _float01(state.get("relation_seed_strength")),
        _float01(recall.get("relation_seed_strength")),
    )
    reentry_contexts: list[ReentryContext] = []
    if reentry_window:
        reentry_contexts.append(
            ReentryContext(
                window=reentry_window,
                summary=_text(state.get("agenda_window_reason"))
                or _text(recall.get("relation_seed_summary"))
                or _text(recall.get("long_term_theme_summary"))
                or focus,
                related_person_id=_text(recall.get("related_person_id"))
                or _text(recall.get("counterpart_person_id")),
                group_thread_id=_text(state.get("group_thread_id"))
                or _text(state.get("group_thread_focus")),
                culture_id=_text(world.get("culture_id"))
                or _text(recall.get("culture_id")),
                community_id=_text(world.get("community_id"))
                or _text(recall.get("community_id")),
                priority=max(reentry_priority, 0.22),
            )
        )

    source_refs = tuple(
        item
        for item in (
            _text(recall.get("source_episode_id")),
            *[
                _text(hit.get("id"))
                for hit in retrieval.get("hits") or ()
                if isinstance(hit, Mapping) and _text(hit.get("id"))
            ],
        )
        if item
    )

    ambiguity_notes: list[str] = []
    if not facts_current:
        ambiguity_notes.append("no_current_fact_selected")
    if _text(recall.get("reinterpretation_mode")) == "grounding_deferral":
        ambiguity_notes.append("grounding_deferral_active")
    if reentry_window and reentry_priority <= 0.0:
        ambiguity_notes.append("reentry_window_has_low_priority")

    bundle = build_memory_evidence_bundle(
        cue_text=cue_text,
        focus=focus,
        facts_current=facts_current,
        facts_superseded=facts_superseded,
        timeline_events=timeline_events,
        temporal_constraints=temporal_constraints,
        reentry_contexts=reentry_contexts,
        source_refs=source_refs,
        ambiguity_notes=ambiguity_notes,
    )
    return bundle.to_dict()
