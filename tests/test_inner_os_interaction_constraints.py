from inner_os.expression.content_policy import derive_content_sequence
from inner_os.expression.interaction_constraints import derive_interaction_constraints
from inner_os.expression.turn_delta import derive_turn_delta


def test_interaction_constraints_detect_thread_and_return_point_pressure() -> None:
    constraints = derive_interaction_constraints(
        {
            "response_strategy": "attune_then_extend",
            "grice_guard_state": {"state": "attune_without_repeating"},
            "agenda_window_state": {"state": "next_private_window"},
            "relational_continuity_state": {"state": "holding_thread"},
            "identity_arc_kind": "repairing_bond",
            "identity_arc_open_tension": "timing_sensitive_reentry",
            "social_topology_state": {"state": "one_to_one"},
        }
    ).to_dict()

    assert constraints["avoid_obvious_advice"] is True
    assert constraints["avoid_overclosure"] is True
    assert constraints["prefer_return_point"] is True
    assert constraints["keep_thread_visible"] is True
    assert constraints["prefer_acknowledge_before_extension"] is True


def test_turn_delta_prefers_thread_visibility_before_generic_return_point() -> None:
    constraints = derive_interaction_constraints(
        {
            "response_strategy": "attune_then_extend",
            "agenda_window_state": {"state": "next_private_window"},
            "relational_continuity_state": {"state": "holding_thread"},
            "identity_arc_kind": "repairing_bond",
            "identity_arc_open_tension": "timing_sensitive_reentry",
        }
    ).to_dict()
    delta = derive_turn_delta(
        {
            "response_strategy": "attune_then_extend",
            "agenda_window_state": {"state": "next_private_window"},
            "identity_arc_kind": "repairing_bond",
        },
        interaction_constraints=constraints,
    ).to_dict()

    assert delta["kind"] == "continuity_thread"
    assert delta["preferred_act"] == "keep_shared_thread_visible"
    assert delta["priority"] > 0.7


def test_turn_delta_prefers_anchor_reopen_before_generic_thread_visibility() -> None:
    constraints = derive_interaction_constraints(
        {
            "response_strategy": "attune_then_extend",
            "agenda_window_state": {"state": "next_private_window"},
            "relational_continuity_state": {"state": "holding_thread"},
            "identity_arc_kind": "repairing_bond",
            "identity_arc_open_tension": "timing_sensitive_reentry",
        }
    ).to_dict()
    delta = derive_turn_delta(
        {
            "response_strategy": "attune_then_extend",
            "agenda_window_state": {"state": "next_private_window"},
            "identity_arc_kind": "repairing_bond",
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.72,
                "reopen_pressure": 0.64,
                "recent_anchor": "港での約束",
            },
        },
        interaction_constraints=constraints,
    ).to_dict()

    assert delta["kind"] == "reopening_thread"
    assert delta["preferred_act"] == "reopen_from_anchor"
    assert delta["anchor_hint"] == "港での約束"


def test_turn_delta_can_prefer_reopening_thread_without_long_arc() -> None:
    delta = derive_turn_delta(
        {
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.72,
                "reopen_pressure": 0.64,
            },
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "reopening_thread"
    assert delta["preferred_act"] == "leave_return_point"
    assert delta["priority"] >= 0.6


def test_turn_delta_can_use_discussion_registry_anchor_for_reopening() -> None:
    delta = derive_turn_delta(
        {
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.72,
                "reopen_pressure": 0.64,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "total_threads": 1,
            },
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "reopening_thread"
    assert delta["preferred_act"] == "reopen_from_anchor"
    assert delta["anchor_hint"] == "repair anchor"


def test_turn_delta_prefers_recent_anchor_over_current_discussion_anchor() -> None:
    delta = derive_turn_delta(
        {
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.72,
                "reopen_pressure": 0.64,
                "recent_anchor": "港での約束",
            },
            "discussion_thread_state": {
                "state": "revisit_issue",
                "topic_anchor": "前に少し引っかかっていた話の続きを、いま少しだけ戻したいです。",
                "revisit_readiness": 0.68,
                "unresolved_pressure": 0.34,
            },
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "discussion_revisit"
    assert delta["preferred_act"] == "reopen_from_anchor"
    assert delta["anchor_hint"] == "港での約束"


def test_turn_delta_can_fall_back_to_autobiographical_anchor_for_reopening() -> None:
    delta = derive_turn_delta(
        {
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.72,
                "reopen_pressure": 0.64,
            },
            "autobiographical_thread_anchor": "harbor promise",
            "autobiographical_thread_strength": 0.58,
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "reopening_thread"
    assert delta["preferred_act"] == "reopen_from_anchor"
    assert delta["anchor_hint"] == "harbor promise"


def test_turn_delta_can_reopen_from_autobiographical_thread_without_recent_dialogue() -> None:
    delta = derive_turn_delta(
        {
            "autobiographical_thread_mode": "unfinished_thread",
            "autobiographical_thread_anchor": "harbor promise",
            "autobiographical_thread_focus": "unfinished promise",
            "autobiographical_thread_strength": 0.62,
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "autobiographical_reopen"
    assert delta["preferred_act"] == "reopen_from_anchor"
    assert delta["anchor_hint"] == "harbor promise"


def test_turn_delta_can_leave_return_point_from_lingering_autobiographical_thread() -> None:
    delta = derive_turn_delta(
        {
            "autobiographical_thread_mode": "lingering_thread",
            "autobiographical_thread_anchor": "harbor promise",
            "autobiographical_thread_focus": "unfinished promise",
            "autobiographical_thread_strength": 0.58,
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "autobiographical_return"
    assert delta["preferred_act"] == "leave_return_point_from_anchor"
    assert delta["anchor_hint"] == "harbor promise"


def test_content_sequence_adds_context_specific_delta_for_thread_visibility() -> None:
    sequence = derive_content_sequence(
        current_text="I can stay with what feels visible here.",
        interaction_policy={
            "response_strategy": "attune_then_extend",
            "dialogue_act": "check_in",
            "agenda_window_state": {"state": "next_private_window"},
            "relational_continuity_state": {"state": "holding_thread"},
            "identity_arc_kind": "repairing_bond",
            "identity_arc_open_tension": "timing_sensitive_reentry",
        },
        conscious_access={"intent": "check_in"},
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert "keep_shared_thread_visible" in acts
    assert "thread that is already here" in text


def test_content_sequence_can_follow_recent_dialogue_state_without_explicit_thread_constraint() -> None:
    sequence = derive_content_sequence(
        current_text="We can pick up from where that thread started to gather again.",
        interaction_policy={
            "dialogue_act": "check_in",
            "recent_dialogue_state": {
                "state": "continuing_thread",
                "thread_carry": 0.58,
                "overlap_score": 0.28,
            },
        },
        conscious_access={"intent": "check_in"},
    )

    acts = [step["act"] for step in sequence]
    assert "keep_shared_thread_visible" in acts


def test_content_sequence_can_follow_autobiographical_thread_anchor_without_recent_dialogue() -> None:
    sequence = derive_content_sequence(
        current_text="I can stay with what still needs care without pushing it all open at once.",
        interaction_policy={
            "autobiographical_thread_mode": "unfinished_thread",
            "autobiographical_thread_anchor": "harbor promise",
            "autobiographical_thread_focus": "unfinished promise",
            "autobiographical_thread_strength": 0.62,
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert "reopen_from_anchor" in acts
    assert "harbor promise" in text


def test_content_sequence_can_use_alternate_anchor_reopen_when_exact_text_is_recent() -> None:
    sequence = derive_content_sequence(
        current_text="前に少し引っかかっていた話の続きを、いま少しだけ戻したいです。",
        interaction_policy={
            "response_strategy": "respectful_wait",
            "discussion_thread_state": {
                "state": "revisit_issue",
                "topic_anchor": "港での約束",
                "revisit_readiness": 0.68,
                "unresolved_pressure": 0.22,
            },
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
        repetition_guard={
            "recent_text_signatures": [
                "前に触れていた「港での約束」のところから、いま話せる分だけ戻れば十分です。"
            ],
            "suppress_exact_text_reuse": True,
        },
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert "reopen_from_anchor_soft" in acts
    assert "港での約束" in text
    assert "前に触れていた「港での約束」のところから、いま話せる分だけ戻れば十分です。" not in text
    assert "前に出ていた「港での約束」のところを、いま話せるぶんだけ拾う感じで大丈夫です。" in text


def test_turn_delta_can_use_discussion_thread_revisit_state() -> None:
    delta = derive_turn_delta(
        {
            "discussion_thread_state": {
                "state": "revisit_issue",
                "revisit_readiness": 0.68,
                "unresolved_pressure": 0.42,
            },
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "discussion_revisit"
    assert delta["preferred_act"] == "leave_return_point"


def test_content_sequence_can_follow_discussion_thread_issue_without_long_arc() -> None:
    sequence = derive_content_sequence(
        current_text="I want to stay with the unresolved part a little longer before we settle it.",
        interaction_policy={
            "dialogue_act": "check_in",
            "discussion_thread_state": {
                "state": "active_issue",
                "unresolved_pressure": 0.46,
                "thread_visibility": 0.38,
            },
        },
        conscious_access={"intent": "check_in"},
    )

    acts = [step["act"] for step in sequence]
    assert "stay_with_present_need" in acts


def test_turn_delta_can_use_issue_pause_state() -> None:
    delta = derive_turn_delta(
        {
            "issue_state": {
                "state": "pausing_issue",
                "pause_readiness": 0.62,
                "question_pressure": 0.18,
            },
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "issue_pause"
    assert delta["preferred_act"] == "leave_return_point"


def test_turn_delta_can_leave_return_point_from_anchor_when_issue_is_paused() -> None:
    delta = derive_turn_delta(
        {
            "issue_state": {
                "state": "pausing_issue",
                "issue_anchor": "repair anchor",
                "pause_readiness": 0.62,
                "question_pressure": 0.18,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "total_threads": 1,
            },
            "agenda_window_state": {"state": "next_private_window"},
        },
        interaction_constraints={"prefer_return_point": True},
    ).to_dict()

    assert delta["kind"] == "issue_pause"
    assert delta["preferred_act"] == "leave_return_point_from_anchor"
    assert delta["anchor_hint"] == "repair anchor"


def test_content_sequence_can_follow_issue_exploration_state_without_long_arc() -> None:
    sequence = derive_content_sequence(
        current_text="I want to keep looking at the unresolved part before we settle it.",
        interaction_policy={
            "dialogue_act": "check_in",
            "issue_state": {
                "state": "exploring_issue",
                "question_pressure": 0.44,
                "pause_readiness": 0.12,
            },
        },
        conscious_access={"intent": "check_in"},
    )

    acts = [step["act"] for step in sequence]
    assert "stay_with_present_need" in acts


def test_turn_delta_can_use_green_kernel_guarded_reflection_hold() -> None:
    delta = derive_turn_delta(
        {
            "green_kernel_composition": {
                "field": {
                    "affective_charge": 0.58,
                    "guardedness": 0.41,
                    "reopening_pull": 0.22,
                }
            },
            "issue_state": {
                "state": "exploring_issue",
                "question_pressure": 0.18,
            },
        },
        interaction_constraints={"avoid_obvious_advice": True},
    ).to_dict()

    assert delta["kind"] == "green_reflection_hold"
    assert delta["preferred_act"] == "stay_with_present_need"


def test_content_sequence_can_add_return_point_when_timing_prefers_later_reentry() -> None:
    sequence = derive_content_sequence(
        current_text="I can stay with what is here without settling the meaning too fast.",
        interaction_policy={
            "response_strategy": "reflect_without_settling",
            "dialogue_act": "check_in",
            "agenda_window_state": {"state": "next_same_group_window"},
            "temporal_membrane_mode": "same_group_reentry",
            "social_topology_state": {"state": "threaded_group"},
        },
        conscious_access={"intent": "check_in"},
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert "leave_return_point" in acts
    assert "come back to the rest" in text


def test_content_sequence_can_surface_discussion_anchor_in_japanese_reopening() -> None:
    sequence = derive_content_sequence(
        current_text="We can pick up from where this thread paused.",
        interaction_policy={
            "dialogue_act": "check_in",
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.72,
                "reopen_pressure": 0.64,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_thread_id": "repair_anchor",
                "dominant_anchor": "repair anchor",
                "dominant_issue_state": "pausing_issue",
                "total_threads": 1,
            },
            "agenda_window_state": {"state": "next_private_window"},
        },
        conscious_access={"intent": "check_in"},
        locale="ja-JP",
    )

    acts = [step["act"] for step in sequence]
    text = " ".join(step["text"] for step in sequence)
    assert "reopen_from_anchor" in acts
    assert "repair anchor" in text
