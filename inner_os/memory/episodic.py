from __future__ import annotations

from dataclasses import dataclass, field

from ..access.models import ForegroundState


@dataclass
class EpisodicRecord:
    episode_id: str
    summary: str
    uncertainty: float
    related_person_id: str = ""
    tags: list[str] = field(default_factory=list)
    salience: float = 0.0
    fixation_reasons: list[str] = field(default_factory=list)


def build_episodic_candidates(
    foreground_state: ForegroundState,
    *,
    uncertainty: float,
    episode_prefix: str = "fg",
) -> list[EpisodicRecord]:
    records: list[EpisodicRecord] = []
    continuity_persons = {
        hint.split(":", 1)[1]: hint.split(":", 1)[1]
        for hint in foreground_state.continuity_focus
        if hint.startswith("person:") and ":" in hint
    }
    default_person_id = next(iter(continuity_persons.values()), "")
    for index, entity_id in enumerate(foreground_state.memory_candidates):
        reasons = list(foreground_state.memory_reasons.get(entity_id, []))
        salience = float(foreground_state.reportability_scores.get(entity_id, 0.0))
        related_person_id = continuity_persons.get(entity_id, "")
        if not related_person_id and any(reason in {"social", "affiliation", "continuity"} for reason in reasons):
            related_person_id = default_person_id
        records.append(
            EpisodicRecord(
                episode_id=f"{episode_prefix}:{index}:{entity_id}",
                summary=entity_id,
                uncertainty=uncertainty,
                related_person_id=related_person_id,
                tags=sorted(set(reasons + foreground_state.selection_reasons)),
                salience=salience,
                fixation_reasons=reasons,
            )
        )
    return records
