import json
from datetime import datetime, timezone

from emot_terrain_lab.memory.inner_os_working_memory_bridge import (
    build_inner_os_working_memory_snapshot,
    derive_reconstructed_replay_carryover,
    derive_working_memory_seed_from_signature,
    derive_working_memory_replay_bias,
    merge_conscious_working_memory_seed,
    merge_working_memory_snapshot_with_seed,
    merge_replay_carryover_summaries,
    prioritize_weekly_abstraction_episodes,
    select_working_memory_promotion_candidates,
    write_inner_os_working_memory_snapshot,
)


def test_build_inner_os_working_memory_snapshot_summarizes_recent_traces(tmp_path) -> None:
    memory_path = tmp_path / "inner_os_memory.jsonl"
    now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    records = [
        {
            "kind": "working_memory_trace",
            "timestamp": now.timestamp() - 1800,
            "current_focus": "meaning",
            "focus_anchor": "harbor slope",
            "focus_text": "why did the harbor exchange feel fragile",
            "unresolved_count": 2,
            "pending_meaning": 0.62,
            "carryover_load": 0.58,
            "memory_pressure": 0.66,
            "open_loops": ["question", "deferred_meaning"],
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "autobiographical_thread_mode": "unfinished_thread",
            "autobiographical_thread_anchor": "harbor slope",
            "autobiographical_thread_focus": "unfinished promise",
            "autobiographical_thread_strength": 0.44,
        },
        {
            "kind": "working_memory_trace",
            "timestamp": now.timestamp() - 600,
            "current_focus": "meaning",
            "focus_anchor": "harbor slope",
            "focus_text": "still unsure how to read the harbor exchange",
            "unresolved_count": 1,
            "pending_meaning": 0.54,
            "carryover_load": 0.49,
            "memory_pressure": 0.61,
            "open_loops": ["deferred_meaning"],
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "long_term_theme_focus": "meaning",
            "long_term_theme_anchor": "harbor slope",
            "long_term_theme_strength": 0.63,
            "long_term_theme_kind": "meaning",
            "long_term_theme_summary": "fragile harbor promise",
            "autobiographical_thread_mode": "unfinished_thread",
            "autobiographical_thread_anchor": "harbor slope",
            "autobiographical_thread_focus": "unfinished promise",
            "autobiographical_thread_strength": 0.58,
        },
    ]
    memory_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    payload = build_inner_os_working_memory_snapshot(
        memory_path=memory_path,
        day=now.date(),
        now=now.replace(tzinfo=None),
    )

    assert payload["schema"] == "inner_os_working_memory_snapshot/v1"
    snapshot = payload["snapshot"]
    assert snapshot["available"] is True
    assert snapshot["current_focus"] == "meaning"
    assert snapshot["focus_anchor"] == "harbor slope"
    assert snapshot["source_trace_count"] == 2
    assert snapshot["promotion_readiness"] > 0.0
    assert snapshot["autobiographical_thread_mode"] == "unfinished_thread"
    assert snapshot["autobiographical_thread_anchor"] == "harbor slope"
    assert snapshot["autobiographical_thread_strength"] > 0.4
    assert "deferred_meaning" in snapshot["dominant_open_loops"]
    assert snapshot["long_term_theme_focus"] == "meaning"
    assert snapshot["long_term_theme_summary"] == "fragile harbor promise"


def test_write_inner_os_working_memory_snapshot_persists_json(tmp_path) -> None:
    memory_path = tmp_path / "inner_os_memory.jsonl"
    now = datetime(2026, 3, 15, 18, 0, 0)
    memory_path.write_text(
        json.dumps(
            {
                "kind": "working_memory_trace",
                "timestamp": now.timestamp(),
                "current_focus": "body",
                "focus_anchor": "quiet room",
                "focus_text": "body is still tense",
                "unresolved_count": 0,
                "pending_meaning": 0.18,
                "carryover_load": 0.28,
                "memory_pressure": 0.31,
                "open_loops": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "inner_os_working_memory_snapshot.json"

    payload = write_inner_os_working_memory_snapshot(
        out_path=out_path,
        memory_path=memory_path,
        day=now.date(),
        now=now,
    )

    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["schema"] == "inner_os_working_memory_snapshot/v1"
    assert payload["snapshot"]["current_focus"] == "body"


def test_select_working_memory_promotion_candidates_prefers_matching_recent_dialogue(tmp_path) -> None:
    memory_path = tmp_path / "inner_os_memory.jsonl"
    now = datetime(2026, 3, 15, 18, 0, 0)
    memory_path.write_text(
        json.dumps(
            {
                "kind": "working_memory_trace",
                "timestamp": now.timestamp(),
                "current_focus": "meaning",
                "focus_anchor": "harbor slope",
                "focus_text": "why the harbor promise felt fragile",
                "unresolved_count": 2,
                "pending_meaning": 0.71,
                "carryover_load": 0.66,
                "memory_pressure": 0.74,
                "open_loops": ["question", "deferred_meaning"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    experiences = [
        {
            "id": "match",
            "timestamp": now.isoformat(),
            "dialogue": "the harbor promise still feels fragile tonight",
            "emotion_intensity": 0.62,
        },
        {
            "id": "miss",
            "timestamp": now.isoformat(),
            "dialogue": "the garden was bright and calm",
            "emotion_intensity": 0.81,
        },
    ]

    selected = select_working_memory_promotion_candidates(
        experiences=experiences,
        memory_path=memory_path,
        day=now.date(),
        now=now,
    )

    assert len(selected) == 1
    assert selected[0]["id"] == "match"
    promotion = selected[0]["context"]["working_memory_promotion"]
    assert promotion["current_focus"] == "meaning"
    assert promotion["promotion_readiness"] > 0.0


def test_derive_working_memory_replay_bias_extracts_focus_terms(tmp_path) -> None:
    memory_path = tmp_path / "inner_os_memory.jsonl"
    now = datetime(2026, 3, 15, 21, 0, 0)
    memory_path.write_text(
        json.dumps(
            {
                "kind": "working_memory_trace",
                "timestamp": now.timestamp(),
                "current_focus": "meaning",
                "focus_anchor": "harbor slope",
                "focus_text": "why the harbor promise felt fragile",
                "unresolved_count": 2,
                "pending_meaning": 0.71,
                "carryover_load": 0.66,
                "memory_pressure": 0.74,
                "open_loops": ["question", "deferred_meaning"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = build_inner_os_working_memory_snapshot(
        memory_path=memory_path,
        day=now.date(),
        now=now,
    )

    bias = derive_working_memory_replay_bias(payload)

    assert bias["current_focus"] == "meaning"
    assert bias["focus_anchor"] == "harbor slope"
    assert bias["strength"] > 0.0
    assert "harbor" in bias["terms"]


def test_prioritize_weekly_abstraction_episodes_prefers_matching_working_memory_focus() -> None:
    episodes = [
        {
            "id": "ep-1",
            "timestamp": "2026-03-10T10:00:00",
            "working_memory_promotion": {"dominant_focus": "body", "dominant_anchor": "quiet room"},
        },
        {
            "id": "ep-2",
            "timestamp": "2026-03-11T10:00:00",
            "working_memory_promotion": {"dominant_focus": "meaning", "dominant_anchor": "harbor slope"},
        },
        {
            "id": "ep-3",
            "timestamp": "2026-03-12T10:00:00",
            "working_memory_promotion": {"dominant_focus": "meaning", "dominant_anchor": "harbor slope"},
        },
        {
            "id": "ep-4",
            "timestamp": "2026-03-13T10:00:00",
            "working_memory_promotion": {"dominant_focus": "other", "dominant_anchor": "garden"},
        },
    ]

    selected = prioritize_weekly_abstraction_episodes(
        episodes=episodes,
        replay_summary={"focus": "meaning", "anchor": "harbor slope", "strength": 0.8},
        limit=2,
        lookback=4,
    )

    assert [item["id"] for item in selected] == ["ep-2", "ep-3"]


def test_prioritize_weekly_abstraction_episodes_can_prefer_matching_partner() -> None:
    episodes = [
        {
            "id": "ep-1",
            "timestamp": "2026-03-10T10:00:00",
            "related_person_id": "other",
            "working_memory_promotion": {"dominant_focus": "ambient", "dominant_anchor": "garden", "related_person_id": "other"},
        },
        {
            "id": "ep-2",
            "timestamp": "2026-03-11T10:00:00",
            "related_person_id": "user",
            "working_memory_promotion": {"dominant_focus": "ambient", "dominant_anchor": "garden", "related_person_id": "user"},
        },
    ]

    selected = prioritize_weekly_abstraction_episodes(
        episodes=episodes,
        replay_summary={"related_person_id": "user", "relation_seed_strength": 0.8},
        limit=1,
        lookback=2,
    )

    assert [item["id"] for item in selected] == ["ep-2"]


def test_derive_reconstructed_replay_carryover_reads_latest_reconstructed_signal(tmp_path) -> None:
    memory_path = tmp_path / "inner_os_memory.jsonl"
    memory_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "reconstructed",
                        "timestamp": "2026-03-15T11:00:00",
                        "memory_anchor": "market street",
                        "working_memory_replay_focus": "market",
                        "working_memory_replay_anchor": "market street",
                        "working_memory_replay_strength": 0.4,
                        "working_memory_replay_alignment": 0.5,
                        "working_memory_replay_reinforcement": 0.2,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "kind": "reconstructed",
                        "timestamp": "2026-03-15T12:00:00",
                        "memory_anchor": "harbor slope",
                        "working_memory_replay_focus": "harbor",
                        "working_memory_replay_anchor": "harbor slope",
                        "working_memory_replay_strength": 0.8,
                        "working_memory_replay_alignment": 1.0,
                        "working_memory_replay_reinforcement": 0.72,
                        "long_term_theme_summary": "quiet harbor slope memory",
                        "long_term_theme_alignment": 0.5,
                        "long_term_theme_reinforcement": 0.32,
                        "source_episode_id": "ep-9",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    carryover = derive_reconstructed_replay_carryover(
        memory_path=memory_path,
        now=datetime(2026, 3, 15, 13, 0, 0),
    )
    assert carryover["focus"] == "harbor"
    assert carryover["anchor"] == "harbor slope"
    assert carryover["source_episode_id"] == "ep-9"
    assert carryover["strength"] > 0.7
    assert carryover["long_term_theme_summary"] == "quiet harbor slope memory"
    assert carryover["long_term_theme_reinforcement"] == 0.32


def test_merge_replay_carryover_summaries_combines_matching_focus_and_anchor() -> None:
    merged = merge_replay_carryover_summaries(
        {"focus": "harbor", "anchor": "harbor slope", "strength": 0.6, "matched_events": 2},
        {"focus": "harbor", "anchor": "harbor slope", "strength": 0.5, "reinforcement": 0.4, "alignment": 1.0},
    )
    assert merged["focus"] == "harbor"
    assert merged["anchor"] == "harbor slope"
    assert merged["strength"] > 0.6
    assert merged["matched_events"] == 2
    assert merged["reinforcement"] == 0.4


def test_derive_working_memory_seed_from_signature_combines_signature_and_replay() -> None:
    seed = derive_working_memory_seed_from_signature(
        {
            "dominant_focus": "meaning",
            "dominant_anchor": "harbor slope",
            "promotion_readiness_mean": 0.68,
            "autobiographical_pressure_mean": 0.52,
            "recurrence_weight": 2.6,
            "long_term_theme": {
                "focus": "meaning",
                "anchor": "harbor slope",
                "kind": "meaning",
                "strength": 0.63,
                "summary": "fragile harbor promise",
            },
        },
        {
            "focus": "harbor",
            "anchor": "harbor slope",
            "strength": 0.58,
        },
    )
    assert seed["semantic_seed_focus"] == "harbor"
    assert seed["semantic_seed_anchor"] == "harbor slope"
    assert seed["semantic_seed_strength"] > 0.0
    assert seed["long_term_theme_focus"] == "meaning"
    assert seed["long_term_theme_anchor"] == "harbor slope"
    assert seed["long_term_theme_kind"] == "meaning"
    assert seed["long_term_theme_strength"] == 0.63


def test_derive_working_memory_seed_from_signature_reflects_conscious_overlap() -> None:
    plain = derive_working_memory_seed_from_signature(
        {
            "dominant_focus": "meaning",
            "dominant_anchor": "harbor slope",
            "promotion_readiness_mean": 0.68,
            "autobiographical_pressure_mean": 0.52,
            "recurrence_weight": 2.6,
        },
        {
            "focus": "harbor",
            "anchor": "harbor slope",
            "strength": 0.58,
        },
    )
    reinforced = derive_working_memory_seed_from_signature(
        {
            "dominant_focus": "meaning",
            "dominant_anchor": "harbor slope",
            "promotion_readiness_mean": 0.68,
            "autobiographical_pressure_mean": 0.52,
            "recurrence_weight": 2.6,
        },
        {
            "focus": "harbor",
            "anchor": "harbor slope",
            "strength": 0.58,
            "conscious_memory_strength": 0.5,
            "conscious_memory_overlap": 1.0,
        },
    )
    assert reinforced["semantic_seed_strength"] > plain["semantic_seed_strength"]


def test_merge_working_memory_snapshot_with_seed_raises_carryover_and_readiness() -> None:
    merged = merge_working_memory_snapshot_with_seed(
        {
            "current_focus": "meaning",
            "focus_anchor": "harbor slope",
            "carryover_load": 0.4,
            "pending_meaning": 0.5,
            "promotion_readiness": 0.6,
        },
        {
            "semantic_seed_focus": "harbor",
            "semantic_seed_anchor": "harbor slope",
            "semantic_seed_strength": 0.5,
        },
    )
    assert merged["carryover_load"] > 0.4
    assert merged["pending_meaning"] > 0.5
    assert merged["promotion_readiness"] > 0.6


def test_select_working_memory_promotion_candidates_accepts_semantic_seed(tmp_path) -> None:
    memory_path = tmp_path / "inner_os_memory.jsonl"
    now = datetime(2026, 3, 15, 18, 0, 0)
    memory_path.write_text(
        json.dumps(
            {
                "kind": "working_memory_trace",
                "timestamp": now.timestamp(),
                "current_focus": "meaning",
                "focus_anchor": "harbor slope",
                "focus_text": "why the harbor promise felt fragile",
                "unresolved_count": 2,
                "pending_meaning": 0.71,
                "carryover_load": 0.66,
                "memory_pressure": 0.74,
                "open_loops": ["question", "deferred_meaning"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    experiences = [
        {
            "id": "match",
            "timestamp": now.isoformat(),
            "dialogue": "the harbor promise still feels fragile tonight",
            "emotion_intensity": 0.62,
        },
    ]
    selected = select_working_memory_promotion_candidates(
        experiences=experiences,
        memory_path=memory_path,
        day=now.date(),
        now=now,
        semantic_seed={"semantic_seed_strength": 0.42},
    )
    assert selected[0]["context"]["working_memory_promotion"]["semantic_seed_strength"] == 0.42


def test_select_working_memory_promotion_candidates_prefers_matching_partner_seed(tmp_path) -> None:
    memory_path = tmp_path / "inner_os_memory.jsonl"
    now = datetime(2026, 3, 15, 18, 0, 0)
    memory_path.write_text(
        json.dumps(
            {
                "kind": "working_memory_trace",
                "timestamp": now.timestamp(),
                "current_focus": "ambient",
                "focus_anchor": "harbor slope",
                "focus_text": "staying near the harbor companion thread",
                "unresolved_count": 1,
                "pending_meaning": 0.44,
                "carryover_load": 0.41,
                "memory_pressure": 0.58,
                "open_loops": ["check_in"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    experiences = [
        {
            "id": "other",
            "timestamp": now.isoformat(),
            "dialogue": "we walked by the harbor slope",
            "emotion_intensity": 0.52,
            "related_person_id": "other",
        },
        {
            "id": "user",
            "timestamp": now.isoformat(),
            "dialogue": "we walked by the harbor slope",
            "emotion_intensity": 0.52,
            "related_person_id": "user",
        },
    ]
    selected = select_working_memory_promotion_candidates(
        experiences=experiences,
        memory_path=memory_path,
        day=now.date(),
        now=now,
        semantic_seed={"related_person_id": "user", "relation_seed_strength": 0.76},
        limit=1,
    )
    assert selected[0]["id"] == "user"
    assert selected[0]["context"]["working_memory_promotion"]["related_person_id"] == "user"
    assert selected[0]["context"]["working_memory_promotion"]["relation_seed_strength"] == 0.76


def test_merge_conscious_working_memory_seed_prefers_existing_replay_summary_but_adds_seed() -> None:
    merged = merge_conscious_working_memory_seed(
        {"focus": "market", "anchor": "market street", "strength": 0.52},
        {"focus": "harbor slope", "anchor": "harbor slope", "strength": 0.4},
    )
    assert merged["focus"] == "market"
    assert merged["anchor"] == "market street"
    assert merged["strength"] >= 0.52
    assert merged["conscious_memory_strength"] == 0.4
    assert merged["conscious_memory_overlap"] == 0.0


def test_merge_conscious_working_memory_seed_reinforces_matching_carryover() -> None:
    merged = merge_conscious_working_memory_seed(
        {"focus": "harbor_slope", "anchor": "harbor_slope", "strength": 0.4},
        {"focus": "harbor_slope", "anchor": "harbor_slope", "strength": 0.5},
    )
    assert merged["focus"] == "harbor_slope"
    assert merged["anchor"] == "harbor_slope"
    assert merged["conscious_memory_overlap"] == 1.0
    assert merged["strength"] > 0.4
