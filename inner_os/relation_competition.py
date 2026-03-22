from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ActiveRelationEntry:
    person_id: str
    score: float
    attachment: float
    familiarity: float
    trust_memory: float
    continuity_score: float
    social_grounding: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "person_id": self.person_id,
            "score": round(self.score, 4),
            "attachment": round(self.attachment, 4),
            "familiarity": round(self.familiarity, 4),
            "trust_memory": round(self.trust_memory, 4),
            "continuity_score": round(self.continuity_score, 4),
            "social_grounding": round(self.social_grounding, 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class RelationCompetitionState:
    state: str
    dominant_person_id: str
    top_person_ids: list[str]
    total_people: int
    competition_level: float
    dominant_score: float
    winner_margin: float
    state_scores: dict[str, float]
    person_scores: dict[str, float]
    dominant_inputs: list[str]
    entries: list[ActiveRelationEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "dominant_person_id": self.dominant_person_id,
            "top_person_ids": list(self.top_person_ids),
            "total_people": int(self.total_people),
            "competition_level": round(self.competition_level, 4),
            "dominant_score": round(self.dominant_score, 4),
            "winner_margin": round(self.winner_margin, 4),
            "state_scores": {
                key: round(value, 4)
                for key, value in self.state_scores.items()
            },
            "person_scores": {
                key: round(value, 4)
                for key, value in self.person_scores.items()
            },
            "dominant_inputs": list(self.dominant_inputs),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def collect_related_person_ids(
    *values: Any,
    registry_snapshot: Mapping[str, Any] | None = None,
    limit: int = 4,
) -> list[str]:
    ordered: list[str] = []

    def _push(value: Any) -> None:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _push(item)
            return
        text = str(value or "").strip()
        if text and text not in ordered:
            ordered.append(text)

    for value in values:
        _push(value)

    top_person_ids = []
    if isinstance(registry_snapshot, Mapping):
        top_person_ids = [
            str(item).strip()
            for item in list(registry_snapshot.get("top_person_ids") or [])
            if str(item).strip()
        ]
    for person_id in top_person_ids:
        _push(person_id)

    if limit > 0:
        return ordered[:limit]
    return ordered


def summarize_person_registry_snapshot(
    snapshot: Mapping[str, Any] | None,
    *,
    limit: int = 4,
) -> dict[str, object]:
    if not isinstance(snapshot, Mapping):
        return {
            "dominant_person_id": "",
            "top_person_ids": [],
            "total_people": 0,
            "uncertainty": 1.0,
            "person_scores": {},
        }
    persons = snapshot.get("persons")
    if not isinstance(persons, Mapping):
        return {
            "dominant_person_id": "",
            "top_person_ids": [],
            "total_people": 0,
            "uncertainty": _clamp01(snapshot.get("uncertainty")),
            "person_scores": {},
        }
    scored = sorted(
        (
            (str(person_id), _person_node_score(payload))
            for person_id, payload in persons.items()
            if isinstance(payload, Mapping) and str(person_id).strip()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_person_ids = [person_id for person_id, _ in scored[: max(0, limit)]]
    dominant_person_id = top_person_ids[0] if top_person_ids else ""
    return {
        "dominant_person_id": dominant_person_id,
        "top_person_ids": top_person_ids,
        "total_people": len(scored),
        "uncertainty": _clamp01(snapshot.get("uncertainty")),
        "person_scores": {
            person_id: round(score, 4)
            for person_id, score in scored[: max(0, limit)]
        },
    }


def derive_relation_competition_state(
    *,
    self_state: Mapping[str, Any] | None,
    related_person_ids: Sequence[str] | None,
    dominant_hint_person_id: str = "",
    registry_snapshot: Mapping[str, Any] | None = None,
    limit: int = 4,
) -> RelationCompetitionState:
    self_payload = dict(self_state or {})
    registry = dict(registry_snapshot or self_payload.get("person_registry_snapshot") or {})
    registry_summary = summarize_person_registry_snapshot(registry, limit=limit)
    candidate_ids = collect_related_person_ids(
        dominant_hint_person_id,
        self_payload.get("related_person_id"),
        related_person_ids or [],
        registry_summary.get("top_person_ids") or [],
        registry_snapshot=registry,
        limit=limit,
    )
    persons = registry.get("persons")
    person_payloads = persons if isinstance(persons, Mapping) else {}
    primary_person_id = str(dominant_hint_person_id or self_payload.get("related_person_id") or "").strip()

    entries: list[ActiveRelationEntry] = []
    for person_id in candidate_ids:
        payload = person_payloads.get(person_id) if isinstance(person_payloads, Mapping) else {}
        payload = payload if isinstance(payload, Mapping) else {}
        adaptive = payload.get("adaptive_traits")
        adaptive_traits = adaptive if isinstance(adaptive, Mapping) else {}
        is_primary = bool(primary_person_id and person_id == primary_person_id)
        attachment = _max01(
            adaptive_traits.get("attachment"),
            self_payload.get("attachment") if is_primary else 0.0,
        )
        familiarity = _max01(
            adaptive_traits.get("familiarity"),
            self_payload.get("familiarity") if is_primary else 0.0,
        )
        trust_memory = _max01(
            adaptive_traits.get("trust_memory"),
            self_payload.get("trust_memory") if is_primary else 0.0,
        )
        continuity_score = _max01(
            adaptive_traits.get("continuity_score"),
            self_payload.get("continuity_score") if is_primary else 0.0,
        )
        social_grounding = _max01(
            adaptive_traits.get("social_grounding"),
            self_payload.get("social_grounding") if is_primary else 0.0,
        )
        source = "registry"
        if is_primary:
            source = "primary"
        elif person_id in list(registry_summary.get("top_person_ids") or []):
            source = "registry_top"
        score = _clamp01(
            attachment * 0.28
            + familiarity * 0.16
            + trust_memory * 0.22
            + continuity_score * 0.2
            + social_grounding * 0.1
            + (0.08 if is_primary else 0.0)
            + (0.04 if source == "registry_top" else 0.0)
            + (0.05 if any((attachment, familiarity, trust_memory, continuity_score, social_grounding)) else 0.0)
        )
        entries.append(
            ActiveRelationEntry(
                person_id=person_id,
                score=score,
                attachment=attachment,
                familiarity=familiarity,
                trust_memory=trust_memory,
                continuity_score=continuity_score,
                social_grounding=social_grounding,
                source=source,
            )
        )

    entries.sort(key=lambda item: item.score, reverse=True)
    total_people = len(entries)
    dominant_person_id = entries[0].person_id if entries else ""
    dominant_score = entries[0].score if entries else 0.0
    runner_up_score = entries[1].score if len(entries) > 1 else 0.0
    winner_margin = _clamp01(dominant_score - runner_up_score)
    competition_level = 0.0
    if len(entries) > 1:
        competition_level = _clamp01(
            runner_up_score * 0.52
            + min(len(entries) - 1, 3) * 0.08
            + (1.0 - winner_margin) * 0.18
        )

    state_scores = {
        "ambient": _clamp01((1.0 - dominant_score) if not entries else 0.0),
        "single_anchor": _clamp01(
            dominant_score * 0.66
            + winner_margin * 0.34
            + (0.18 if total_people == 1 else 0.0)
            - competition_level * 0.22
        ),
        "dominant_thread": _clamp01(
            dominant_score * 0.56
            + winner_margin * 0.28
            + (0.1 if total_people > 1 else 0.0)
            - competition_level * 0.12
        ),
        "competing_threads": _clamp01(
            competition_level * 0.62
            + runner_up_score * 0.22
            + (1.0 - winner_margin) * 0.16
        ),
    }
    if total_people == 0:
        state = "ambient"
    elif total_people == 1:
        state = "single_anchor"
    else:
        state = "competing_threads" if state_scores["competing_threads"] >= state_scores["dominant_thread"] else "dominant_thread"

    dominant_inputs = _compact(
        [
            "primary_relation_hint" if primary_person_id and dominant_person_id == primary_person_id else "",
            "multiple_people" if total_people > 1 else "",
            "tight_margin" if total_people > 1 and winner_margin <= 0.14 else "",
            "registry_top_person" if dominant_person_id and dominant_person_id in list(registry_summary.get("top_person_ids") or []) else "",
            "strong_continuity" if entries and entries[0].continuity_score >= 0.56 else "",
            "strong_trust_memory" if entries and entries[0].trust_memory >= 0.56 else "",
        ]
    )
    return RelationCompetitionState(
        state=state,
        dominant_person_id=dominant_person_id,
        top_person_ids=[entry.person_id for entry in entries],
        total_people=total_people,
        competition_level=competition_level,
        dominant_score=dominant_score,
        winner_margin=winner_margin,
        state_scores=state_scores,
        person_scores={entry.person_id: entry.score for entry in entries},
        dominant_inputs=dominant_inputs,
        entries=entries,
    )


def _person_node_score(payload: Mapping[str, Any]) -> float:
    adaptive = payload.get("adaptive_traits")
    traits = adaptive if isinstance(adaptive, Mapping) else {}
    return _clamp01(
        _float01(traits.get("attachment")) * 0.28
        + _float01(traits.get("familiarity")) * 0.16
        + _float01(traits.get("trust_memory")) * 0.22
        + _float01(traits.get("continuity_score")) * 0.2
        + _float01(traits.get("social_grounding")) * 0.1
        + _float01(payload.get("confidence")) * 0.04
    )


def _compact(values: Sequence[str]) -> list[str]:
    return [value for value in values if value]


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _float01(value: Any) -> float:
    return _clamp01(value)


def _max01(*values: Any) -> float:
    return max((_clamp01(value) for value in values), default=0.0)
