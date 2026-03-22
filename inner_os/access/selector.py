from __future__ import annotations

from ..self_model.models import PersonRegistry, SelfState
from ..value_system.emotional_dft import AccessState as DFTAccessState
from ..value_system.emotional_dft import subjective_intensity
from ..value_system.models import ValueState
from ..world_model.models import WorldState
from .models import AccessCandidate, AttentionState, ForegroundState


def _entity_salience(
    entity_id: str,
    *,
    world_state: WorldState,
    self_state: SelfState,
    value_state: ValueState,
    attention_state: AttentionState,
    person_registry: PersonRegistry | None,
) -> AccessCandidate:
    reasons: list[str] = []
    salience = 0.15
    linked_person_id = ""
    for relation in world_state.social_relation_graph.get(entity_id, []):
        text = str(relation or "").strip()
        if text.startswith("person:"):
            linked_person_id = text.split(":", 1)[1].strip()
            break

    if value_state.danger_score > max(0.2, attention_state.uncertainty_tolerance):
        salience += 0.35
        reasons.append("danger")
    if entity_id in world_state.task_states:
        salience += 0.2
        reasons.append("task")
    if self_state.fatigue > 0.5:
        salience += 0.1
        reasons.append("stabilize-self")
    registry_person_id = entity_id if person_registry and entity_id in person_registry.persons else linked_person_id
    if person_registry and registry_person_id in person_registry.persons:
        node = person_registry.persons[registry_person_id]
        salience += attention_state.continuity_bias * max(0.0, node.confidence) * 0.35
        reasons.append("continuity")
        adaptive = node.adaptive_traits if isinstance(node.adaptive_traits, dict) else {}
        stable = node.stable_traits if isinstance(node.stable_traits, dict) else {}
        relationship_pull = min(
            1.0,
            max(
                0.0,
                float(adaptive.get("attachment", 0.0)) * 0.34
                + float(adaptive.get("familiarity", 0.0)) * 0.22
                + float(adaptive.get("trust_memory", 0.0)) * 0.22
                + float(adaptive.get("continuity_score", 0.0)) * 0.12
                + float(adaptive.get("social_grounding", 0.0)) * 0.1,
            ),
        )
        cultural_alignment = min(
            1.0,
            max(
                0.0,
                float(stable.get("community_marker", 0.0)) * attention_state.community_bias * 0.46
                + float(stable.get("culture_marker", 0.0)) * attention_state.culture_bias * 0.34
                + float(stable.get("role_marker", 0.0)) * (0.18 + attention_state.affiliation_bias * 0.12),
            ),
        )
        timing_gate = min(
            1.0,
            max(
                0.15,
                0.42
                + attention_state.affiliation_bias * 0.28
                + value_state.trust_preservation_score * 0.16
                - attention_state.caution_bias * 0.22
                - self_state.social_tension * 0.12,
            ),
        )
        relationship_pull = min(
            1.0,
            relationship_pull * (0.72 + cultural_alignment * 0.18 + timing_gate * 0.22),
        )
        if relationship_pull > 0.0:
            salience += attention_state.continuity_bias * relationship_pull * 0.22
            salience += attention_state.partner_style_relief * 0.08
            salience -= attention_state.partner_style_caution * 0.06
            salience += attention_state.relational_future_pull * 0.05
            salience += attention_state.shared_world_pull * 0.06
            salience += attention_state.relational_care * 0.04
            salience -= attention_state.relational_reverence * 0.02
            reasons.append("partner-trace")
            if cultural_alignment > 0.12:
                reasons.append("community-trace")
            if timing_gate < 0.45:
                salience -= (0.45 - timing_gate) * (0.16 + attention_state.caution_bias * 0.12)
                reasons.append("partner-delayed")
            if attention_state.partner_style_relief > 0.02:
                reasons.append("partner-style-relief")
            if attention_state.partner_style_caution > 0.02:
                reasons.append("partner-style-caution")
            if attention_state.relational_future_pull > 0.04:
                reasons.append("future-pull")
            if attention_state.shared_world_pull > 0.04:
                reasons.append("shared-world")
            if attention_state.relational_care > 0.04:
                reasons.append("care-trace")
            if attention_state.relational_reverence > 0.04:
                reasons.append("reverence-trace")
        continuity_hint = f"person:{registry_person_id}"
    else:
        continuity_hint = ""

    if value_state.terrain_snapshot is not None:
        access_signal = subjective_intensity(
            terrain=value_state.terrain_snapshot,
            access=DFTAccessState(
                attention=attention_state.salience_bias,
                reportability=1.0 - min(1.0, world_state.uncertainty),
                access_uncertainty=world_state.uncertainty,
                interface_curvature=value_state.terrain_snapshot.max_curvature,
            ),
            hypothesis="access_mode",
        )
        salience += 0.2 * min(1.0, access_signal)
        if access_signal > 0.18:
            reasons.append("terrain-access")

    return AccessCandidate(
        entity_id=entity_id,
        salience=round(min(1.0, salience), 4),
        reasons=reasons,
        continuity_hint=continuity_hint,
    )


def _reportability_score(
    candidate: AccessCandidate,
    *,
    world_state: WorldState,
    value_state: ValueState,
) -> float:
    score = candidate.salience
    if "danger" in candidate.reasons:
        score += 0.15
    if "continuity" in candidate.reasons:
        score += 0.1
    if "partner-trace" in candidate.reasons:
        score += 0.08
    if "partner-style-relief" in candidate.reasons:
        score += 0.03
    if "community-trace" in candidate.reasons:
        score += 0.05
    if "partner-delayed" in candidate.reasons:
        score -= 0.04
    if "partner-style-caution" in candidate.reasons:
        score -= 0.03
    if "future-pull" in candidate.reasons:
        score += 0.03
    if "shared-world" in candidate.reasons:
        score += 0.03
    if "reverence-trace" in candidate.reasons:
        score -= 0.02
    score -= min(0.25, world_state.uncertainty * 0.2)
    score += min(0.15, value_state.trust_preservation_score * 0.1)
    return round(max(0.0, min(1.0, score)), 4)


def _memory_candidate(
    candidate: AccessCandidate,
    *,
    reportability_score: float,
    value_state: ValueState,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if "continuity" in candidate.reasons:
        reasons.append("continuity")
    if "partner-trace" in candidate.reasons:
        reasons.append("social")
    if "partner-style-relief" in candidate.reasons:
        reasons.append("social_style")
    if "community-trace" in candidate.reasons:
        reasons.append("community")
    if "future-pull" in candidate.reasons:
        reasons.append("future_pull")
    if "care-trace" in candidate.reasons:
        reasons.append("care")
    if "danger" in candidate.reasons:
        reasons.append("danger")
    if value_state.terrain_energy > 0.3:
        reasons.append("terrain_energy")
    if reportability_score > 0.45:
        reasons.append("reportable")
    return (len(reasons) >= 2, reasons)


def select_foreground(
    world_state: WorldState,
    self_state: SelfState,
    value_state: ValueState,
    attention_state: AttentionState,
    person_registry: PersonRegistry | None = None,
) -> ForegroundState:
    candidates = [
        _entity_salience(
            entity_id,
            world_state=world_state,
            self_state=self_state,
            value_state=value_state,
            attention_state=attention_state,
            person_registry=person_registry,
        )
        for entity_id in world_state.object_states
    ]
    ranked = sorted(candidates, key=lambda item: item.salience, reverse=True)
    top = ranked[:3]

    risks = ["danger"] if value_state.danger_score > max(0.2, attention_state.uncertainty_tolerance) else []
    notes = [f"uncertainty:{world_state.uncertainty:.2f}"] if world_state.uncertainty > 0.2 else []
    continuity_focus = [item.continuity_hint for item in top if item.continuity_hint]
    selection_reasons = sorted({reason for item in top for reason in item.reasons})
    if value_state.terrain_snapshot is not None and value_state.terrain_energy > 0.2:
        notes = list(notes) + [f"terrain_energy:{value_state.terrain_energy:.2f}"]
    reportability_scores = {
        item.entity_id: _reportability_score(
            item,
            world_state=world_state,
            value_state=value_state,
        )
        for item in top
    }
    memory_candidates: list[str] = []
    memory_reasons: dict[str, list[str]] = {}
    for item in top:
        should_fix, reasons = _memory_candidate(
            item,
            reportability_score=reportability_scores[item.entity_id],
            value_state=value_state,
        )
        if should_fix:
            memory_candidates.append(item.entity_id)
            memory_reasons[item.entity_id] = reasons

    return ForegroundState(
        salient_entities=[item.entity_id for item in top],
        current_risks=risks,
        active_goals=["stabilize"] if self_state.fatigue > 0.5 else ["observe"],
        affective_summary=value_state.value_axes,
        reportable_facts=[item.entity_id for item in top],
        uncertainty_notes=notes,
        candidates=top,
        selection_reasons=selection_reasons,
        continuity_focus=continuity_focus,
        reportability_scores=reportability_scores,
        memory_candidates=memory_candidates,
        memory_reasons=memory_reasons,
    )
