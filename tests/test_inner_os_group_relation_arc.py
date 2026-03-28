from __future__ import annotations

from inner_os.group_relation_arc import GroupRelationArcSummaryBuilder


def test_group_relation_arc_builder_prefers_same_group_reentry_for_group_thread() -> None:
    summary = GroupRelationArcSummaryBuilder().build(
        {
            "inner_os_relation_arc_summary": {
                "arc_kind": "repairing_relation",
                "phase": "shifting",
                "summary": "repair is moving through a shared group thread in small steps",
                "group_thread_id": "threaded_group:user|friend",
                "related_person_id": "user",
                "stability": 0.61,
            },
            "inner_os_sleep_agenda_window_focus": "next_same_group_window",
        }
    ).to_dict()

    assert summary["arc_kind"] == "repairing_relation"
    assert summary["group_thread_id"] == "threaded_group:user|friend"
    assert summary["boundary_mode"] == "same_group_reentry"
    assert summary["reentry_window_focus"] == "next_same_group_window"


def test_group_relation_arc_builder_uses_private_hold_for_public_boundary() -> None:
    summary = GroupRelationArcSummaryBuilder().build(
        {
            "inner_os_relation_arc_registry_summary": {
                "top_arcs": [
                    {
                        "arc_kind": "repairing_relation",
                        "phase": "shifting",
                        "summary": "repair is moving through a shared group thread in small steps",
                        "group_thread_id": "threaded_group:user|friend",
                        "topology_focus": "public_visible",
                        "related_person_id": "user",
                    }
                ]
            },
            "inner_os_group_thread_registry_summary": {
                "dominant_thread_id": "threaded_group:user|friend",
                "thread_scores": {"threaded_group:user|friend": 0.63},
            },
            "inner_os_sleep_agenda_window_focus": "next_private_window",
        }
    ).to_dict()

    assert summary["group_thread_id"] == "threaded_group:user|friend"
    assert summary["boundary_mode"] == "public_to_private_hold"
    assert summary["stability"] == 0.63
