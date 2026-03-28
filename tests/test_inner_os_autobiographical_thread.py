from inner_os.autobiographical_thread import derive_autobiographical_thread_summary


def test_autobiographical_thread_summary_stays_empty_without_thread_pressure() -> None:
    summary = derive_autobiographical_thread_summary({})

    assert summary.mode == "none"
    assert summary.anchor == ""
    assert summary.strength == 0.0


def test_autobiographical_thread_summary_can_hold_unfinished_relational_thread() -> None:
    summary = derive_autobiographical_thread_summary(
        {
            "recent_dialogue_state": {
                "state": "reopening_thread",
                "thread_carry": 0.58,
                "reopen_pressure": 0.54,
                "recent_anchor": "harbor promise",
            },
            "discussion_thread_state": {
                "state": "revisit_issue",
                "topic_anchor": "harbor promise",
                "unresolved_pressure": 0.46,
                "revisit_readiness": 0.62,
            },
            "issue_state": {
                "state": "pausing_issue",
                "issue_anchor": "harbor promise",
                "pause_readiness": 0.68,
            },
            "discussion_thread_registry_snapshot": {
                "dominant_anchor": "harbor promise",
                "dominant_issue_state": "pausing_issue",
                "thread_scores": {"harbor_promise": 0.72},
                "total_threads": 1,
                "uncertainty": 0.22,
            },
            "residual_reflection_mode": "withheld",
            "residual_reflection_focus": "unfinished promise",
            "residual_reflection_strength": 0.44,
            "related_person_id": "user",
        }
    )

    assert summary.mode in {"unfinished_thread", "relational_lingering_thread"}
    assert summary.anchor == "harbor promise"
    assert summary.strength > 0.4
    assert "discussion_registry" in summary.reason_tokens
