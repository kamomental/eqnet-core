from __future__ import annotations

from dataclasses import dataclass, field

from ..action_posture import derive_action_posture
from ..actuation_plan import derive_actuation_plan
from ..access.models import ForegroundState
from ..interaction import compose_nonverbal_profile, derive_relational_mood, summarize_situation_state
from ..partner_style import relation_episode_naming_from_stance, resolve_partner_utterance_stance
from ..policy_packet import derive_interaction_policy_packet
from .episodic import EpisodicRecord, build_episodic_candidates
from .semantic import SemanticPattern, derive_semantic_hints


@dataclass
class MemoryContext:
    episodic_candidates: list[EpisodicRecord] = field(default_factory=list)
    semantic_hints: list[SemanticPattern] = field(default_factory=list)
    continuity_threads: list[str] = field(default_factory=list)
    retention_summary: list[str] = field(default_factory=list)
    related_person_ids: list[str] = field(default_factory=list)
    relation_bias_strength: float = 0.0
    partner_semantic_summary: str = ""
    partner_address_hint: str = ""
    partner_timing_hint: str = ""
    partner_stance_hint: str = ""


def build_memory_context(
    foreground_state: ForegroundState,
    *,
    uncertainty: float,
    episode_prefix: str = "fg",
    grounding_context: dict[str, str] | None = None,
) -> MemoryContext:
    episodic = build_episodic_candidates(
        foreground_state,
        uncertainty=uncertainty,
        episode_prefix=episode_prefix,
    )
    semantic = derive_semantic_hints(episodic)
    if not semantic and episodic:
        strongest = max(episodic, key=lambda record: record.salience)
        semantic = derive_semantic_hints([strongest], min_salience=0.0)
    retention_summary = [
        f"{record.summary}:{','.join(record.fixation_reasons)}"
        for record in episodic
    ]
    related_person_ids = [
        record.related_person_id
        for record in episodic
        if record.related_person_id
    ]
    unique_person_ids = list(dict.fromkeys(related_person_ids))
    relation_weight_hits = sum(
        1
        for record in episodic
        if record.related_person_id and any(reason in {"social", "affiliation", "continuity"} for reason in record.fixation_reasons)
    )
    return MemoryContext(
        episodic_candidates=episodic,
        semantic_hints=semantic,
        continuity_threads=list(foreground_state.continuity_focus),
        retention_summary=retention_summary,
        related_person_ids=unique_person_ids,
        relation_bias_strength=round(min(1.0, relation_weight_hits * 0.28), 4),
        partner_semantic_summary=next(
            (hint.label for hint in semantic if hint.label.startswith("relation:")),
            "",
        ),
        partner_address_hint=str((grounding_context or {}).get("address_hint") or ""),
        partner_timing_hint=str((grounding_context or {}).get("timing_hint") or ""),
        partner_stance_hint=str((grounding_context or {}).get("stance_hint") or ""),
    )


def build_memory_appends(memory_context: MemoryContext) -> list[dict[str, object]]:
    appends: list[dict[str, object]] = []
    partner_social_interpretation = ":".join(
        part
        for part in (
            memory_context.partner_stance_hint,
            memory_context.partner_address_hint,
            memory_context.partner_timing_hint,
        )
        if part
    )
    utterance_stance = resolve_partner_utterance_stance(
        relation_bias_strength=memory_context.relation_bias_strength,
        related_person_ids=memory_context.related_person_ids,
        partner_address_hint=memory_context.partner_address_hint,
        partner_timing_hint=memory_context.partner_timing_hint,
        partner_stance_hint=memory_context.partner_stance_hint,
    )
    relation_episode_naming = relation_episode_naming_from_stance(
        utterance_stance,
        social_interpretation=partner_social_interpretation,
    )
    situation_state = summarize_situation_state(
        affective_summary={},
        current_risks=[],
        active_goals=[],
        selection_reasons=[],
        relation_bias_strength=memory_context.relation_bias_strength,
    )
    relational_mood = derive_relational_mood(
        affective_summary={},
        situation_state=situation_state,
        partner_address_hint=memory_context.partner_address_hint,
        partner_timing_hint=memory_context.partner_timing_hint,
        partner_stance_hint=memory_context.partner_stance_hint,
    )
    nonverbal_profile = compose_nonverbal_profile(
        utterance_stance=utterance_stance,
        affective_summary={},
        situation_state=situation_state,
        relational_mood=relational_mood,
        partner_address_hint=memory_context.partner_address_hint,
        partner_timing_hint=memory_context.partner_timing_hint,
        partner_stance_hint=memory_context.partner_stance_hint,
    )
    interaction_policy = derive_interaction_policy_packet(
        dialogue_act="check_in" if memory_context.related_person_ids else "report",
        current_focus="person" if memory_context.related_person_ids else "ambient",
        current_risks=[],
        reportable_facts=[record.summary for record in memory_context.episodic_candidates],
        relation_bias_strength=memory_context.relation_bias_strength,
        related_person_ids=memory_context.related_person_ids,
        partner_address_hint=memory_context.partner_address_hint,
        partner_timing_hint=memory_context.partner_timing_hint,
        partner_stance_hint=memory_context.partner_stance_hint,
        partner_social_interpretation=partner_social_interpretation,
        orchestration={
            "orchestration_mode": "advance" if relational_mood.future_pull >= 0.48 else "attune",
            "dominant_driver": "future" if relational_mood.future_pull >= 0.48 else "shared_attention",
            "contact_readiness": memory_context.relation_bias_strength,
            "coherence_score": situation_state.shared_attention,
            "human_presence_signal": situation_state.shared_attention,
            "distance_strategy": "holding_space",
            "repair_bias": situation_state.repair_window_open,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": nonverbal_profile.gaze_mode,
        },
        live_regulation=type(
            "MemoryPolicyRegulation",
            (),
            {
                "repair_window_open": situation_state.repair_window_open,
                "strained_pause": 0.0,
                "future_loop_pull": relational_mood.future_pull,
                "fantasy_loop_pull": 0.0,
                "distance_expectation": "holding_space",
            },
        )(),
    )
    action_posture = derive_action_posture(interaction_policy)
    actuation_plan = derive_actuation_plan(interaction_policy, action_posture)
    focus_now = str(interaction_policy.get("focus_now") or "").strip()
    leave_closed_for_now = [
        str(item).strip()
        for item in interaction_policy.get("leave_closed_for_now") or []
        if str(item).strip()
    ]
    response_action_now = dict(interaction_policy.get("response_action_now") or {})
    wanted_effect_on_other = [
        dict(item)
        for item in interaction_policy.get("wanted_effect_on_other") or []
        if isinstance(item, dict)
    ]
    for record in memory_context.episodic_candidates:
        relation_priority_bonus = (
            memory_context.relation_bias_strength * 0.12
            if record.related_person_id and any(reason in {"social", "affiliation", "continuity"} for reason in record.fixation_reasons)
            else 0.0
        )
        appends.append(
            {
                "kind": "observed_real",
                "summary": record.summary,
                "text": record.summary,
                "memory_anchor": record.summary,
                "source_episode_id": record.episode_id,
                "related_person_id": record.related_person_id or None,
                "confidence": round(max(0.0, min(1.0, 1.0 - record.uncertainty)), 4),
                "policy_hint": ",".join(record.fixation_reasons),
                "continuity_score": 1.0 if "continuity" in record.fixation_reasons else 0.0,
                "consolidation_priority": round(min(1.0, record.salience + relation_priority_bonus), 4),
                "social_interpretation": partner_social_interpretation,
                "address_hint": memory_context.partner_address_hint,
                "timing_hint": memory_context.partner_timing_hint,
                "stance_hint": memory_context.partner_stance_hint,
                "relation_episode_naming": relation_episode_naming,
                "utterance_stance": utterance_stance,
                "interaction_policy_mode": interaction_policy["response_strategy"],
                "interaction_policy_dialogue_act": interaction_policy["dialogue_act"],
                "interaction_focus_now": focus_now,
                "interaction_leave_closed_for_now": leave_closed_for_now,
                "interaction_response_action_now": response_action_now,
                "interaction_wanted_effect_on_other": wanted_effect_on_other,
                "action_posture_mode": action_posture["engagement_mode"],
                "action_posture_goal": action_posture["outcome_goal"],
                "action_posture_boundary": action_posture["boundary_mode"],
                "actuation_execution_mode": actuation_plan["execution_mode"],
                "actuation_primary_action": actuation_plan["primary_action"],
                "actuation_reply_permission": actuation_plan["reply_permission"],
                "nonverbal_signature": ":".join(
                    (
                        nonverbal_profile.gaze_mode,
                        nonverbal_profile.pause_mode,
                        nonverbal_profile.proximity_mode,
                        nonverbal_profile.silence_mode,
                    )
                ),
                "situation_phase": situation_state.current_phase,
                "relational_mood_signature": ":".join(
                    (
                        f"future={relational_mood.future_pull:.2f}",
                        f"reverence={relational_mood.reverence:.2f}",
                        f"innocence={relational_mood.innocence:.2f}",
                        f"care={relational_mood.care:.2f}",
                        f"shared={relational_mood.shared_world_pull:.2f}",
                    )
                ),
            }
        )
    for hint in memory_context.semantic_hints:
        appends.append(
            {
                "kind": "reconstructed",
                "summary": hint.label,
                "text": hint.label,
                "memory_anchor": hint.label,
                "source_episode_id": hint.supporting_episode_ids[0] if hint.supporting_episode_ids else None,
                "confidence": hint.recurrence_weight,
                "policy_hint": "semantic_hint",
                "meaning_shift": hint.recurrence_weight,
                "reinterpretation_summary": hint.label,
                "continuity_score": 1.0 if hint.label.startswith("continuity:") else 0.0,
                "interaction_policy_mode": interaction_policy["response_strategy"],
                "interaction_focus_now": focus_now,
                "interaction_leave_closed_for_now": leave_closed_for_now,
                "interaction_response_action_now": response_action_now,
                "interaction_wanted_effect_on_other": wanted_effect_on_other,
                "action_posture_mode": action_posture["engagement_mode"],
                "actuation_execution_mode": actuation_plan["execution_mode"],
            }
        )
    return appends
