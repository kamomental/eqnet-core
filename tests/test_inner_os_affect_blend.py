from inner_os.access.models import ForegroundState
from inner_os.affect_blend import derive_affect_blend_state
from inner_os.expression import DialogueContext, render_response
from inner_os.interaction.models import LiveInteractionRegulation, RelationalMood, SituationState
from inner_os.scene_state import derive_scene_state


def test_affect_blend_accepts_temporal_membrane_bias_as_weak_prior() -> None:
    scene_state = derive_scene_state(
        place_mode="relational_private",
        privacy_level=0.72,
        social_topology="one_to_one",
        task_phase="ongoing",
        temporal_phase="ongoing",
        norm_pressure=0.18,
        safety_margin=0.82,
        environmental_load=0.16,
    )
    base = derive_affect_blend_state(
        affective_summary={"arousal": 0.18, "social_tension": 0.12},
        relational_mood=RelationalMood(
            future_pull=0.24,
            reverence=0.12,
            innocence=0.14,
            care=0.42,
            shared_world_pull=0.3,
            confidence_signal=0.46,
        ),
        live_regulation=LiveInteractionRegulation(
            future_loop_pull=0.18,
            strained_pause=0.08,
            repair_window_open=False,
        ),
        situation_state=SituationState(
            scene_mode="co_present",
            shared_attention=0.42,
            social_pressure=0.18,
            continuity_weight=0.48,
            current_phase="ongoing",
        ),
        scene_state=scene_state,
        stress=0.12,
        recovery_need=0.1,
        safety_bias=0.08,
        relation_bias_strength=0.36,
    )
    temporal = derive_affect_blend_state(
        affective_summary={"arousal": 0.18, "social_tension": 0.12},
        relational_mood=RelationalMood(
            future_pull=0.24,
            reverence=0.12,
            innocence=0.14,
            care=0.42,
            shared_world_pull=0.3,
            confidence_signal=0.46,
        ),
        live_regulation=LiveInteractionRegulation(
            future_loop_pull=0.18,
            strained_pause=0.08,
            repair_window_open=False,
        ),
        situation_state=SituationState(
            scene_mode="co_present",
            shared_attention=0.42,
            social_pressure=0.18,
            continuity_weight=0.48,
            current_phase="ongoing",
        ),
        scene_state=scene_state,
        stress=0.12,
        recovery_need=0.1,
        safety_bias=0.08,
        relation_bias_strength=0.36,
        temporal_membrane_bias={
            "timeline_coherence": 0.62,
            "reentry_pull": 0.74,
            "supersession_pressure": 0.0,
            "continuity_pressure": 0.68,
            "relation_reentry_pull": 0.72,
            "dominant_mode": "reentry",
        },
    )

    assert temporal.care > base.care
    assert temporal.future_pull > base.future_pull
    assert temporal.shared_world_pull > base.shared_world_pull
    assert temporal.confidence >= base.confidence
    assert "blend_temporal_reentry" in temporal.cues
    assert "blend_temporal_continuity" in temporal.cues


def test_response_planner_uses_temporal_membrane_bias_in_affect_blend_payload() -> None:
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["we can pick this up again"],
        memory_candidates=["user"],
        reportability_scores={"user": 0.78},
        affective_summary={
            "arousal": 0.16,
            "social_tension": 0.12,
            "stress": 0.1,
            "recovery_need": 0.08,
            "safety_bias": 0.06,
        },
    )

    base_plan = render_response(
        foreground,
        DialogueContext(user_text="また次に話そう"),
    )
    temporal_plan = render_response(
        foreground,
        DialogueContext(
            user_text="また次に話そう",
            expression_hints={
                "qualia_membrane_temporal": {
                    "timeline_coherence": 0.58,
                    "reentry_pull": 0.76,
                    "supersession_pressure": 0.0,
                    "continuity_pressure": 0.66,
                    "relation_reentry_pull": 0.7,
                    "dominant_mode": "reentry",
                }
            },
        ),
    )

    assert temporal_plan.llm_payload["affect_blend_state"]["future_pull"] >= base_plan.llm_payload["affect_blend_state"]["future_pull"]
    assert temporal_plan.llm_payload["affect_blend_state"]["shared_world_pull"] >= base_plan.llm_payload["affect_blend_state"]["shared_world_pull"]
    assert "blend_temporal_reentry" in temporal_plan.llm_payload["affect_blend_state"]["cues"]
