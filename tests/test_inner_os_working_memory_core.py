from inner_os.working_memory_core import WorkingMemoryCore


def test_working_memory_prefers_meaning_focus_under_questions_and_tentative_recall() -> None:
    core = WorkingMemoryCore()
    snap = core.snapshot(
        user_input={"text": "what does this place mean to us?"},
        sensor_input={"person_count": 1, "body_stress_index": 0.24},
        current_state={"temporal_pressure": 0.22, "affiliation_bias": 0.54},
        relational_world={"place_memory_anchor": "harbor slope", "social_role": "companion"},
        previous_trace={"carryover_load": 0.18},
        recall_payload={"reinterpretation_mode": "grounding_deferral", "tentative_bias": 0.42},
    )
    assert snap.current_focus == "meaning"
    assert snap.unresolved_count >= 1
    assert snap.pending_meaning > 0.4
    assert snap.memory_pressure > 0.0


def test_working_memory_settles_after_answer_but_keeps_deferred_meaning() -> None:
    core = WorkingMemoryCore()
    snap = core.snapshot(
        user_input={"text": "why did that happen?"},
        sensor_input={"body_stress_index": 0.2},
        current_state={"temporal_pressure": 0.24},
        relational_world={"place_memory_anchor": "backstage room"},
        previous_trace={"carryover_load": 0.26},
        recall_payload={"reinterpretation_mode": "grounding_deferral", "tentative_bias": 0.36},
    )
    settled = core.settle_after_turn(
        snapshot=snap,
        reply_text="I want to stay close to what is visible first.",
        current_state={"temporal_pressure": 0.18},
        recall_payload={"reinterpretation_mode": "grounding_deferral"},
    )
    assert settled.pending_meaning <= snap.pending_meaning
    assert settled.pending_meaning >= 0.28
    assert settled.memory_pressure >= 0.0
    assert settled.current_focus in {"meaning", "place", "ambient"}


def test_working_memory_snapshot_absorbs_semantic_seed_into_carryover() -> None:
    core = WorkingMemoryCore()
    snap = core.snapshot(
        user_input={"text": ""},
        sensor_input={"body_stress_index": 0.16},
        current_state={
            "temporal_pressure": 0.18,
            "semantic_seed_focus": "harbor",
            "semantic_seed_anchor": "harbor slope",
            "semantic_seed_strength": 0.52,
            "semantic_seed_recurrence": 2.6,
        },
        relational_world={},
        previous_trace={"carryover_load": 0.14},
        recall_payload={},
    )
    assert snap.focus_anchor == "harbor slope"
    assert snap.semantic_seed_focus == "harbor"
    assert snap.semantic_seed_strength == 0.52
    assert snap.carryover_load > 0.14
    assert snap.pending_meaning > 0.0


def test_working_memory_keeps_long_term_theme_summary_in_trace() -> None:
    core = WorkingMemoryCore()
    snap = core.snapshot(
        user_input={"text": ""},
        sensor_input={"body_stress_index": 0.14},
        current_state={
            "temporal_pressure": 0.16,
            "long_term_theme_focus": "harbor",
            "long_term_theme_anchor": "harbor slope",
            "long_term_theme_kind": "place",
            "long_term_theme_summary": "quiet harbor slope memory",
            "long_term_theme_strength": 0.61,
        },
        relational_world={},
        previous_trace={"carryover_load": 0.12},
        recall_payload={},
    )
    trace = core.build_trace_record(snapshot=snap, current_state={}, relational_world={})
    assert snap.long_term_theme_summary == "quiet harbor slope memory"
    assert trace["long_term_theme_summary"] == "quiet harbor slope memory"
    assert trace["long_term_theme_kind"] == "place"


def test_working_memory_absorbs_conscious_residue_without_promoting_to_long_term_theme() -> None:
    core = WorkingMemoryCore()
    snap = core.snapshot(
        user_input={"text": ""},
        sensor_input={"body_stress_index": 0.14},
        current_state={
            "temporal_pressure": 0.16,
            "conscious_residue_focus": "harbor",
            "conscious_residue_anchor": "harbor slope",
            "conscious_residue_summary": "quiet harbor slope memory",
            "conscious_residue_strength": 0.44,
        },
        relational_world={},
        previous_trace={"carryover_load": 0.12},
        recall_payload={},
    )
    trace = core.build_trace_record(snapshot=snap, current_state={}, relational_world={})
    assert snap.conscious_residue_focus == "harbor"
    assert snap.conscious_residue_strength == 0.44
    assert snap.long_term_theme_strength == 0.0
    assert snap.carryover_load > 0.12
    assert trace["conscious_residue_summary"] == "quiet harbor slope memory"


def test_working_memory_snapshot_absorbs_autobiographical_thread_pressure() -> None:
    core = WorkingMemoryCore()
    snap = core.snapshot(
        user_input={"text": ""},
        sensor_input={"body_stress_index": 0.12},
        current_state={
            "temporal_pressure": 0.16,
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.58,
                "reopen_pressure": 0.52,
                "recent_anchor": "harbor promise",
            },
            "discussion_thread_state": {
                "state": "revisit_issue",
                "topic_anchor": "harbor promise",
                "unresolved_pressure": 0.44,
                "revisit_readiness": 0.62,
            },
            "issue_state": {
                "state": "pausing_issue",
                "issue_anchor": "harbor promise",
                "pause_readiness": 0.64,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_anchor": "harbor promise",
                "dominant_issue_state": "pausing_issue",
                "thread_scores": {"harbor_promise": 0.68},
                "total_threads": 1,
                "uncertainty": 0.2,
            },
            "residual_reflection_mode": "withheld",
            "residual_reflection_focus": "unfinished promise",
            "residual_reflection_strength": 0.42,
            "related_person_id": "user",
        },
        relational_world={},
        previous_trace={"carryover_load": 0.1},
        recall_payload={},
    )

    trace = core.build_trace_record(snapshot=snap, current_state={}, relational_world={})

    assert snap.autobiographical_thread_mode != "none"
    assert snap.autobiographical_thread_anchor == "harbor promise"
    assert snap.autobiographical_thread_strength > 0.0
    assert snap.focus_anchor == "harbor promise"
    assert trace["autobiographical_thread_mode"] == snap.autobiographical_thread_mode
