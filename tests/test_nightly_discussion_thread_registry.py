from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ops import nightly


def test_summarize_inner_os_discussion_thread_registry_reads_recent_trace_records(
    tmp_path: Path,
) -> None:
    memory_path = tmp_path / "inner_os_memory.jsonl"
    now = dt.datetime.utcnow()
    rows = [
        {
            "kind": "discussion_thread_trace",
            "timestamp": (now - dt.timedelta(hours=2)).isoformat(),
            "memory_anchor": "repair anchor",
            "recent_dialogue_state": "reopening_thread",
            "recent_dialogue_thread_carry": 0.62,
            "recent_dialogue_reopen_pressure": 0.54,
            "discussion_thread_state": "revisit_issue",
            "discussion_thread_anchor": "repair anchor",
            "discussion_unresolved_pressure": 0.48,
            "discussion_revisit_readiness": 0.66,
            "discussion_thread_visibility": 0.58,
            "issue_state": "pausing_issue",
            "issue_anchor": "repair anchor",
            "issue_question_pressure": 0.22,
            "issue_pause_readiness": 0.64,
            "issue_resolution_readiness": 0.18,
        },
        {
            "kind": "discussion_thread_trace",
            "timestamp": (now - dt.timedelta(hours=1)).isoformat(),
            "memory_anchor": "repair anchor",
            "recent_dialogue_state": "continuing_thread",
            "recent_dialogue_thread_carry": 0.58,
            "recent_dialogue_reopen_pressure": 0.42,
            "discussion_thread_state": "revisit_issue",
            "discussion_thread_anchor": "repair anchor",
            "discussion_unresolved_pressure": 0.44,
            "discussion_revisit_readiness": 0.61,
            "discussion_thread_visibility": 0.63,
            "issue_state": "pausing_issue",
            "issue_anchor": "repair anchor",
            "issue_question_pressure": 0.18,
            "issue_pause_readiness": 0.67,
            "issue_resolution_readiness": 0.24,
        },
    ]
    memory_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = nightly._summarize_inner_os_discussion_thread_registry(
        {"inner_os_memory_path": str(memory_path)},
        now=now,
        lookback_hours=72,
    )

    assert summary["dominant_thread_id"] == "repair_anchor"
    assert summary["dominant_anchor"] == "repair anchor"
    assert summary["dominant_issue_state"] == "pausing_issue"
    assert summary["total_threads"] == 1
    assert summary["lookback_hours"] == 72
    assert summary["thread_scores"]["repair_anchor"] > 0.0
