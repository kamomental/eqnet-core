from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from .memory_evidence_bundle import MemoryEvidenceBundle


WINDOW_REENTRY_WEIGHTS: dict[str, float] = {
    "now": 0.92,
    "next_private_window": 0.76,
    "next_same_group_window": 0.82,
    "next_same_culture_window": 0.72,
    "opportunistic_reentry": 0.58,
    "long_hold": 0.24,
}


@dataclass(frozen=True)
class QualiaMembraneTemporalBias:
    """時間整流済み記憶がクオリア膜へ与える弱い変形量。"""

    timeline_coherence: float = 0.0
    reentry_pull: float = 0.0
    supersession_pressure: float = 0.0
    continuity_pressure: float = 0.0
    relation_reentry_pull: float = 0.0
    dominant_mode: str = "ambient"
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_qualia_membrane_temporal_bias(
    *,
    memory_evidence_bundle: Mapping[str, Any] | MemoryEvidenceBundle,
    current_state: Mapping[str, Any] | None = None,
    world_snapshot: Mapping[str, Any] | None = None,
) -> QualiaMembraneTemporalBias:
    bundle = (
        memory_evidence_bundle
        if isinstance(memory_evidence_bundle, MemoryEvidenceBundle)
        else MemoryEvidenceBundle.from_mapping(memory_evidence_bundle)
    )
    state = dict(current_state or {})
    world = dict(world_snapshot or {})

    current_weight = _mean_weight(bundle.facts_current)
    timeline_weight = _mean_weight(bundle.timeline_events)
    superseded_weight = _mean_weight(bundle.facts_superseded)
    ambiguity_pressure = _clamp01(len(bundle.ambiguity_notes) * 0.12)

    replay_priority = _constraint_priority(bundle, "replay_signature")
    semantic_priority = _constraint_priority(bundle, "semantic_seed")
    theme_priority = _constraint_priority(bundle, "long_term_theme")
    continuity_pressure = _clamp01(
        theme_priority * 0.44
        + semantic_priority * 0.32
        + replay_priority * 0.24
    )

    reentry_pull = _derive_reentry_pull(bundle, state=state, world=world)
    relation_reentry_pull = _derive_relation_reentry_pull(bundle, state=state)
    supersession_pressure = _clamp01(
        superseded_weight * 0.72
        + ambiguity_pressure * 0.18
        + (0.1 if bundle.facts_superseded and not bundle.facts_current else 0.0)
    )
    timeline_coherence = _clamp01(
        current_weight * 0.32
        + timeline_weight * 0.28
        + continuity_pressure * 0.22
        + reentry_pull * 0.1
        + relation_reentry_pull * 0.08
        - supersession_pressure * 0.18
        - ambiguity_pressure * 0.1
    )

    mode_scores = {
        "cohere": timeline_coherence * 0.8 + continuity_pressure * 0.2,
        "reentry": reentry_pull * 0.7 + relation_reentry_pull * 0.3,
        "supersede": supersession_pressure,
        "ambient": 0.05,
    }
    dominant_mode = max(mode_scores.items(), key=lambda item: item[1])[0]

    cues: list[str] = [f"temporal_membrane_{dominant_mode}"]
    if continuity_pressure >= 0.38:
        cues.append("temporal_membrane_continuity")
    if reentry_pull >= 0.42:
        cues.append("temporal_membrane_reentry")
    if relation_reentry_pull >= 0.4:
        cues.append("temporal_membrane_relation")
    if supersession_pressure >= 0.34:
        cues.append("temporal_membrane_supersession")
    if ambiguity_pressure >= 0.24:
        cues.append("temporal_membrane_ambiguity")

    return QualiaMembraneTemporalBias(
        timeline_coherence=round(timeline_coherence, 4),
        reentry_pull=round(reentry_pull, 4),
        supersession_pressure=round(supersession_pressure, 4),
        continuity_pressure=round(continuity_pressure, 4),
        relation_reentry_pull=round(relation_reentry_pull, 4),
        dominant_mode=dominant_mode,
        cues=tuple(cues),
    )


def _derive_reentry_pull(
    bundle: MemoryEvidenceBundle,
    *,
    state: Mapping[str, Any],
    world: Mapping[str, Any],
) -> float:
    best = 0.0
    for context in bundle.reentry_contexts:
        weight = WINDOW_REENTRY_WEIGHTS.get(context.window, 0.48)
        culture_match = 1.0 if context.culture_id and context.culture_id == _text(world.get("culture_id")) else 0.0
        community_match = (
            1.0 if context.community_id and context.community_id == _text(world.get("community_id")) else 0.0
        )
        score = _clamp01(
            context.priority * weight
            + culture_match * 0.06
            + community_match * 0.08
        )
        best = max(best, score)
    state_bias = _float01(state.get("agenda_window_bias"))
    return _clamp01(best + state_bias * 0.08)


def _derive_relation_reentry_pull(
    bundle: MemoryEvidenceBundle,
    *,
    state: Mapping[str, Any],
) -> float:
    best = 0.0
    for context in bundle.reentry_contexts:
        relation_marker = 1.0 if context.related_person_id or context.group_thread_id else 0.0
        if relation_marker <= 0.0:
            continue
        best = max(best, _clamp01(context.priority * 0.72 + relation_marker * 0.12))

    related_state_bonus = 0.0
    if _text(state.get("related_person_id")):
        related_state_bonus += 0.08
    if _text(state.get("group_thread_id")) or _text(state.get("group_thread_focus")):
        related_state_bonus += 0.08

    return _clamp01(best + related_state_bonus)


def _constraint_priority(bundle: MemoryEvidenceBundle, kind: str) -> float:
    priorities = [item.priority for item in bundle.temporal_constraints if item.kind == kind]
    if not priorities:
        return 0.0
    return _clamp01(sum(priorities) / len(priorities))


def _mean_weight(items: tuple[Any, ...]) -> float:
    if not items:
        return 0.0
    weights = [float(getattr(item, "weight", 0.0) or 0.0) for item in items]
    if not weights:
        return 0.0
    return _clamp01(sum(weights) / len(weights))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return _clamp01(numeric)


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))
