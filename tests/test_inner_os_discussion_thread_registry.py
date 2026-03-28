from inner_os.discussion_thread_registry import (
    summarize_discussion_thread_registry_snapshot,
    update_discussion_thread_registry_snapshot,
)


def test_discussion_thread_registry_snapshot_updates_from_issue_states() -> None:
    snapshot = update_discussion_thread_registry_snapshot(
        existing_snapshot=None,
        recent_dialogue_state={
            "state": "reopening_thread",
            "thread_carry": 0.62,
            "reopen_pressure": 0.54,
            "recent_anchor": "さっきの引っかかり",
        },
        discussion_thread_state={
            "state": "revisit_issue",
            "topic_anchor": "さっきの引っかかり",
            "unresolved_pressure": 0.48,
            "revisit_readiness": 0.66,
            "thread_visibility": 0.58,
        },
        issue_state={
            "state": "pausing_issue",
            "issue_anchor": "さっきの引っかかり",
            "question_pressure": 0.22,
            "pause_readiness": 0.64,
            "resolution_readiness": 0.18,
        },
    )

    assert snapshot["dominant_anchor"] == "さっきの引っかかり"
    assert snapshot["dominant_issue_state"] == "pausing_issue"
    assert snapshot["total_threads"] == 1


def test_discussion_thread_registry_summary_handles_empty_snapshot() -> None:
    summary = summarize_discussion_thread_registry_snapshot({})
    assert summary["dominant_thread_id"] == ""
    assert summary["total_threads"] == 0
