from inner_os.group_thread_registry import (
    build_group_thread_key,
    summarize_group_thread_registry_snapshot,
    update_group_thread_registry_snapshot,
)


def test_build_group_thread_key_prefers_hint_and_sorts_people() -> None:
    assert build_group_thread_key(thread_hint="public:harbor") == "public:harbor"
    assert (
        build_group_thread_key(
            topology_state="threaded_group",
            top_person_ids=["friend", "user"],
        )
        == "threaded_group:friend|user"
    )


def test_update_group_thread_registry_snapshot_tracks_thread_summary() -> None:
    snapshot = update_group_thread_registry_snapshot(
        existing_snapshot={},
        topology_state={
            "state": "threaded_group",
            "threading_pressure": 0.72,
            "visibility_pressure": 0.24,
            "hierarchy_pressure": 0.1,
        },
        dominant_person_id="user",
        top_person_ids=["user", "friend"],
        total_people=2,
        continuity_score=0.64,
        social_grounding=0.58,
        community_id="harbor_collective",
        culture_id="coastal",
        social_role="companion",
    )

    assert snapshot["dominant_thread_id"] == "threaded_group:friend|user"
    assert snapshot["total_threads"] == 1
    node = snapshot["threads"]["threaded_group:friend|user"]
    assert node["dominant_person_id"] == "user"
    assert node["total_people"] == 2
    assert node["continuity_score"] > 0.0
    assert node["threading_pressure"] > 0.0


def test_summarize_group_thread_registry_snapshot_returns_scored_threads() -> None:
    summary = summarize_group_thread_registry_snapshot(
        {
            "threads": {
                "threaded_group:user|friend": {
                    "continuity_score": 0.72,
                    "social_grounding": 0.63,
                    "threading_pressure": 0.68,
                    "visibility_pressure": 0.2,
                    "hierarchy_pressure": 0.1,
                    "total_people": 2,
                    "count": 3,
                },
                "public_visible:user": {
                    "continuity_score": 0.34,
                    "social_grounding": 0.28,
                    "threading_pressure": 0.12,
                    "visibility_pressure": 0.74,
                    "hierarchy_pressure": 0.2,
                    "total_people": 1,
                    "count": 1,
                },
            },
            "uncertainty": 0.18,
        }
    )

    assert summary["dominant_thread_id"] == "threaded_group:user|friend"
    assert summary["total_threads"] == 2
    assert summary["thread_scores"]["threaded_group:user|friend"] >= summary["thread_scores"]["public_visible:user"]
    assert summary["uncertainty"] == 0.18
