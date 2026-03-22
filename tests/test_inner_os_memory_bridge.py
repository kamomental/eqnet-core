from inner_os.access.models import ForegroundState
from inner_os.memory import build_episodic_candidates, build_memory_appends, build_memory_context, derive_semantic_hints


def test_build_episodic_candidates_uses_memory_candidates_and_reasons() -> None:
    foreground = ForegroundState(
        salient_entities=["user", "lamp"],
        reportable_facts=["user", "lamp"],
        selection_reasons=["continuity", "terrain-access"],
        continuity_focus=["person:user"],
        reportability_scores={"user": 0.82, "lamp": 0.31},
        memory_candidates=["user"],
        memory_reasons={"user": ["continuity", "reportable"]},
    )
    records = build_episodic_candidates(foreground, uncertainty=0.2, episode_prefix="turn")
    assert len(records) == 1
    assert records[0].episode_id == "turn:0:user"
    assert records[0].summary == "user"
    assert records[0].salience == 0.82
    assert "continuity" in records[0].fixation_reasons
    assert "terrain-access" in records[0].tags
    assert records[0].related_person_id == "user"


def test_memory_context_builds_semantic_hints_from_fixation_candidates() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user"],
        selection_reasons=["continuity", "terrain-access"],
        continuity_focus=["person:user"],
        reportability_scores={"user": 0.82},
        memory_candidates=["user"],
        memory_reasons={"user": ["continuity", "reportable", "terrain_energy"]},
    )
    records = build_episodic_candidates(foreground, uncertainty=0.2, episode_prefix="turn")
    semantic = derive_semantic_hints(records)
    memory_context = build_memory_context(foreground, uncertainty=0.2, episode_prefix="turn")
    assert semantic
    assert semantic[0].label == "relation:user:user"
    assert semantic[0].recurrence_weight > 0.82
    assert memory_context.semantic_hints[0].label == "relation:user:user"
    assert memory_context.continuity_threads == ["person:user"]
    assert memory_context.related_person_ids == ["user"]
    assert memory_context.relation_bias_strength > 0.0


def test_memory_context_can_be_exported_to_append_records() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user"],
        selection_reasons=["continuity", "terrain-access"],
        continuity_focus=["person:user"],
        reportability_scores={"user": 0.82},
        memory_candidates=["user"],
        memory_reasons={"user": ["continuity", "reportable", "terrain_energy"]},
    )
    memory_context = build_memory_context(foreground, uncertainty=0.2, episode_prefix="turn")
    appends = build_memory_appends(memory_context)
    assert len(appends) == 2
    assert appends[0]["kind"] == "observed_real"
    assert appends[0]["related_person_id"] == "user"
    assert appends[0]["consolidation_priority"] > 0.82
    assert appends[0]["social_interpretation"] == ""
    assert appends[1]["kind"] == "reconstructed"
    assert appends[1]["summary"] == "relation:user:user"


def test_memory_context_can_export_partner_grounding_hints_into_append_records() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user"],
        selection_reasons=["continuity", "terrain-access"],
        continuity_focus=["person:user"],
        reportability_scores={"user": 0.82},
        memory_candidates=["user"],
        memory_reasons={"user": ["social", "continuity", "reportable"]},
    )
    memory_context = build_memory_context(
        foreground,
        uncertainty=0.2,
        episode_prefix="turn",
        grounding_context={
            "address_hint": "companion",
            "timing_hint": "open",
            "stance_hint": "familiar",
        },
    )
    appends = build_memory_appends(memory_context)
    assert appends[0]["social_interpretation"] == "familiar:companion:open"
    assert appends[0]["relation_episode_naming"] == "warm_reconnection"
    assert appends[0]["utterance_stance"] == "warm_check_in"
    assert appends[0]["interaction_policy_mode"] in {"attune_then_extend", "shared_world_next_step"}
    assert appends[0]["interaction_policy_dialogue_act"] == "check_in"
    assert appends[0]["interaction_focus_now"]
    assert isinstance(appends[0]["interaction_leave_closed_for_now"], list)
    assert isinstance(appends[0]["interaction_response_action_now"], dict)
    assert appends[0]["interaction_response_action_now"]["primary_operation"]
    assert isinstance(appends[0]["interaction_wanted_effect_on_other"], list)
    assert appends[0]["interaction_wanted_effect_on_other"]
    assert appends[0]["action_posture_mode"] in {"attune", "co_move"}
    assert appends[0]["action_posture_goal"] in {"increase_safe_contact", "shared_progress"}
    assert appends[0]["action_posture_boundary"] in {"permeable", "forward_open"}
    assert appends[0]["actuation_execution_mode"] in {"attuned_contact", "shared_progression"}
    assert appends[0]["actuation_primary_action"] in {"hold_presence", "co_move"}
    assert appends[0]["nonverbal_signature"].startswith("shared_attention_hold:")
    assert appends[0]["situation_phase"] == "check_in"
    assert "future=" in appends[0]["relational_mood_signature"]
    assert "shared=" in appends[0]["relational_mood_signature"]
    assert appends[1]["interaction_focus_now"] == appends[0]["interaction_focus_now"]
    assert appends[1]["interaction_response_action_now"]["primary_operation"] == appends[0]["interaction_response_action_now"]["primary_operation"]
