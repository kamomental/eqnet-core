from inner_os.interaction.models import LiveInteractionRegulation, RelationalMood, SituationState
from inner_os.interaction_option_search import (
    compute_action_family_activations,
    generate_interaction_option_candidates,
)
from inner_os.scene_state import derive_scene_state


def test_scene_state_marks_repair_window_family() -> None:
    scene_state = derive_scene_state(
        place_mode="home_private",
        privacy_level=0.84,
        social_topology="one_to_one",
        task_phase="repair",
        temporal_phase="ongoing",
        norm_pressure=0.22,
        safety_margin=0.58,
        environmental_load=0.18,
        current_risks=(),
        active_goals=("repair",),
    )

    assert scene_state.scene_family == "repair_window"
    assert "private" in scene_state.scene_tags
    assert "goal:repair" in scene_state.scene_tags


def test_public_high_norm_scene_prefers_wait_over_co_move() -> None:
    scene_state = derive_scene_state(
        place_mode="public_shared",
        privacy_level=0.12,
        social_topology="group_present",
        task_phase="ongoing",
        temporal_phase="mid_interaction",
        norm_pressure=0.82,
        safety_margin=0.42,
        environmental_load=0.24,
        current_risks=(),
        active_goals=(),
    )
    situation_state = SituationState(
        scene_mode="co_present",
        repair_window_open=False,
        shared_attention=0.36,
        social_pressure=0.42,
        continuity_weight=0.34,
        current_phase="ongoing",
    )
    relational_mood = RelationalMood(
        future_pull=0.24,
        reverence=0.76,
        innocence=0.0,
        care=0.48,
        shared_world_pull=0.16,
        confidence_signal=0.44,
    )
    live_regulation = LiveInteractionRegulation(
        past_loop_pull=0.18,
        future_loop_pull=0.16,
        fantasy_loop_pull=0.08,
        shared_attention_active=0.22,
        strained_pause=0.34,
        repair_window_open=False,
        distance_expectation="holding_space",
    )

    activations = compute_action_family_activations(
        scene_state=scene_state,
        situation_state=situation_state,
        relational_mood=relational_mood,
        live_regulation=live_regulation,
    )

    assert activations[0].family_id == "wait"
    co_move_index = next(index for index, item in enumerate(activations) if item.family_id == "co_move")
    wait_index = next(index for index, item in enumerate(activations) if item.family_id == "wait")
    assert wait_index < co_move_index


def test_candidate_count_emerges_within_three_to_eight() -> None:
    scene_state = derive_scene_state(
        place_mode="shared_workspace",
        privacy_level=0.54,
        social_topology="one_to_one",
        task_phase="coordination",
        temporal_phase="ongoing",
        norm_pressure=0.34,
        safety_margin=0.62,
        environmental_load=0.16,
        current_risks=(),
        active_goals=("coordinate",),
    )
    situation_state = SituationState(
        scene_mode="co_present",
        repair_window_open=False,
        shared_attention=0.64,
        social_pressure=0.18,
        continuity_weight=0.48,
        current_phase="ongoing",
    )
    relational_mood = RelationalMood(
        future_pull=0.66,
        reverence=0.08,
        innocence=0.14,
        care=0.52,
        shared_world_pull=0.72,
        confidence_signal=0.58,
    )
    live_regulation = LiveInteractionRegulation(
        past_loop_pull=0.16,
        future_loop_pull=0.52,
        fantasy_loop_pull=0.08,
        shared_attention_active=0.58,
        strained_pause=0.12,
        repair_window_open=False,
        distance_expectation="cooperative",
    )

    candidates = generate_interaction_option_candidates(
        scene_state=scene_state,
        situation_state=situation_state,
        relational_mood=relational_mood,
        live_regulation=live_regulation,
    )

    assert 3 <= len(candidates) <= 8
    assert candidates[0].family_id == "co_move"
    assert any(candidate.family_id == "attune" for candidate in candidates)
