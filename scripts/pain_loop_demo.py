#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quick CLI to seed a few pain events and run the forgiveness loop."""

from __future__ import annotations

from emot_terrain_lab.ops.pain_loop import (
    evaluate_and_forgive,
    log_pain_event,
    policy_update_from_forgiveness,
)


def seed_demo() -> None:
    log_pain_event(
        "isolation",
        -0.40,
        ["isolation"],
        {"emotional": 0.5, "metabolic": 0.1, "ethical": 0.2},
        {"scene": "greeting_missed"},
    )
    log_pain_event(
        "energy_depletion",
        -0.28,
        ["overload"],
        {"emotional": 0.2, "metabolic": 0.8, "ethical": 0.1},
        {"scene": "long_task"},
    )
    log_pain_event(
        "value_conflict",
        -0.52,
        ["rudeness"],
        {"emotional": 0.6, "metabolic": 0.1, "ethical": 0.7},
        {"scene": "queue_cut"},
    )


if __name__ == "__main__":
    seed_demo()
    nightly_id = "nightly-local"
    total, forgiven, stats = evaluate_and_forgive(nightly_id, base_threshold=0.35)
    update = policy_update_from_forgiveness(
        nightly_id,
        events_detail=stats.get("events_detail"),
    )
    print(
        "DONE:",
        {
            "total": total,
            "forgiven": forgiven,
            "forgive_rate": stats.get("forgive_rate"),
            "used_threshold": stats.get("forgive_threshold"),
            "policy_threshold": update.get("policy_feedback_threshold"),
            "empathy_gain": update.get("a2a_empathy_gain"),
            "avg_replay_fidelity": stats.get("avg_replay_fidelity"),
            "care_stats": stats.get("care_stats"),
        },
    )

