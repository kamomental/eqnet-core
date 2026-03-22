import json
from pathlib import Path

from ops import nightly
from inner_os.daily_carry_summary import DailyCarrySummaryBuilder


def test_resolve_inner_os_sleep_snapshot_reads_default_state_path(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = state_dir / "inner_os_sleep_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema": "inner_os_sleep_consolidation_snapshot/v1",
                "snapshot": {"mode": "reconsolidate"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    path, payload, warning = nightly._resolve_inner_os_sleep_snapshot({})

    assert warning is None
    assert Path(path).resolve() == snapshot_path.resolve()
    assert isinstance(payload, dict)
    assert payload["snapshot"]["mode"] == "reconsolidate"


def test_resolve_inner_os_sleep_snapshot_warns_on_schema_mismatch(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = state_dir / "inner_os_sleep_snapshot.json"
    snapshot_path.write_text(
        json.dumps({"schema": "unexpected", "snapshot": {"mode": "settle"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    path, payload, warning = nightly._resolve_inner_os_sleep_snapshot({})

    assert path is None
    assert payload is None
    assert warning is not None
    assert "schema_mismatch" in warning


def test_summarize_inner_os_memory_class_reads_recent_counts(tmp_path, monkeypatch) -> None:
    memory_path = tmp_path / "logs" / "inner_os_memory.jsonl"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "kind": "relationship_trace",
            "memory_write_class": "bond_protection",
            "memory_write_class_reason": "bond_protection_pressure",
            "timestamp": "2026-03-21T09:00:00",
            "confidence": 0.9,
            "access_count": 2,
            "primed_weight": 0.4,
        },
        {
            "kind": "identity_trace",
            "memory_write_class": "bond_protection",
            "memory_write_class_reason": "bond_protection_pressure",
            "timestamp": "2026-03-21T10:00:00",
            "confidence": 0.7,
        },
        {
            "kind": "observed_real",
            "memory_write_class": "safe_repeat",
            "memory_write_class_reason": "safe_repeatable_contact",
            "timestamp": "2026-03-21T11:00:00",
            "confidence": 0.6,
        },
    ]
    memory_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    summary = nightly._summarize_inner_os_memory_class(
        {"inner_os_memory_path": str(memory_path)},
        now=nightly.dt.datetime.fromisoformat("2026-03-21T12:00:00"),
        lookback_hours=24,
    )

    assert summary["dominant_class"] == "bond_protection"
    assert summary["dominant_reason"] == "bond_protection_pressure"
    assert summary["counts"]["bond_protection"] == 2
    assert summary["counts"]["safe_repeat"] == 1
    assert summary["recent_records"] == 3


def test_summarize_inner_os_insight_trace_reads_recent_links(tmp_path, monkeypatch) -> None:
    memory_path = tmp_path / "logs" / "inner_os_memory.jsonl"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "kind": "insight_trace",
            "insight_class": "reframed_relation",
            "association_link_key": "bond:user|memory:shared_thread",
            "reframed_topic": "shared thread",
            "timestamp": "2026-03-21T09:00:00",
            "confidence": 0.9,
            "insight_score": 0.74,
            "coherence_gain": 0.62,
            "prediction_drop": 0.2,
            "anchor_center": [0.18, -0.04, 0.1],
            "anchor_dispersion": 0.28,
        },
        {
            "kind": "insight_trace",
            "insight_class": "reframed_relation",
            "association_link_key": "bond:user|memory:shared_thread",
            "reframed_topic": "shared thread",
            "timestamp": "2026-03-21T10:00:00",
            "confidence": 0.72,
            "insight_score": 0.66,
            "coherence_gain": 0.54,
            "prediction_drop": 0.18,
            "anchor_center": [0.14, -0.02, 0.08],
            "anchor_dispersion": 0.24,
        },
        {
            "kind": "insight_trace",
            "insight_class": "new_link_hypothesis",
            "association_link_key": "external:cue|memory:shared_thread",
            "reframed_topic": "new shared cue",
            "timestamp": "2026-03-21T11:00:00",
            "confidence": 0.64,
            "insight_score": 0.58,
            "coherence_gain": 0.48,
            "prediction_drop": 0.14,
            "anchor_center": [0.42, 0.08, -0.06],
            "anchor_dispersion": 0.36,
        },
    ]
    memory_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    summary = nightly._summarize_inner_os_insight_trace(
        {"inner_os_memory_path": str(memory_path)},
        now=nightly.dt.datetime.fromisoformat("2026-03-21T12:00:00"),
        lookback_hours=24,
    )

    assert summary["dominant_insight_class"] == "reframed_relation"
    assert summary["dominant_reframed_topic"] == "shared thread"
    assert summary["insight_class_counts"]["reframed_relation"] == 2
    assert summary["insight_link_counts"]["bond:user|memory:shared_thread"] == 2
    assert summary["association_reweighting_bias"] > 0.0
    assert summary["association_reweighting_focus"] == "repeated_links"
    assert summary["association_reweighting_reason"] == "repeated_insight_trace"
    assert summary["insight_reframing_bias"] > 0.0
    assert summary["insight_terrain_shape_bias"] > 0.0
    assert summary["insight_terrain_shape_reason"] == "reframed_relation"
    assert summary["insight_terrain_shape_target"] == "soft_relation"
    assert len(summary["insight_anchor_center"]) == 3
    assert summary["insight_anchor_dispersion"] > 0.0


def test_summarize_inner_os_commitment_trace_reads_recent_targets(tmp_path, monkeypatch) -> None:
    memory_path = tmp_path / "logs" / "inner_os_memory.jsonl"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "kind": "commitment_trace",
            "commitment_target": "repair",
            "commitment_state": "commit",
            "memory_write_class_reason": "repair_trace",
            "timestamp": "2026-03-21T09:00:00",
            "commitment_score": 0.74,
            "commitment_winner_margin": 0.22,
            "commitment_accepted_cost": 0.3,
        },
        {
            "kind": "commitment_trace",
            "commitment_target": "repair",
            "commitment_state": "settle",
            "memory_write_class_reason": "repair_trace",
            "timestamp": "2026-03-21T10:00:00",
            "commitment_score": 0.58,
            "commitment_winner_margin": 0.12,
            "commitment_accepted_cost": 0.18,
        },
        {
            "kind": "commitment_trace",
            "commitment_target": "hold",
            "commitment_state": "waver",
            "memory_write_class_reason": "body_risk",
            "timestamp": "2026-03-21T11:00:00",
            "commitment_score": 0.28,
            "commitment_winner_margin": 0.06,
            "commitment_accepted_cost": 0.08,
        },
    ]
    memory_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    summary = nightly._summarize_inner_os_commitment_trace(
        {"inner_os_memory_path": str(memory_path)},
        now=nightly.dt.datetime.fromisoformat("2026-03-21T12:00:00"),
        lookback_hours=24,
    )

    assert summary["dominant_target"] == "repair"
    assert summary["dominant_state"] == "commit"
    assert summary["dominant_reason"] == "repair_trace"
    assert summary["target_counts"]["repair"] == 2
    assert summary["state_counts"]["commit"] == 1
    assert summary["commitment_carry_bias"] > 0.0
    assert summary["commitment_followup_focus"] == "reopen_softly"
    assert summary["commitment_mode_focus"] == "repair"


def test_summarize_inner_os_agenda_trace_reads_recent_states(tmp_path, monkeypatch) -> None:
    memory_path = tmp_path / "logs" / "inner_os_memory.jsonl"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "kind": "agenda_trace",
            "agenda_state": "repair",
            "agenda_reason": "repair_window",
            "timestamp": "2026-03-21T09:00:00",
            "agenda_score": 0.68,
            "agenda_winner_margin": 0.18,
        },
        {
            "kind": "agenda_trace",
            "agenda_state": "repair",
            "agenda_reason": "repair_window",
            "timestamp": "2026-03-21T10:00:00",
            "agenda_score": 0.62,
            "agenda_winner_margin": 0.14,
        },
        {
            "kind": "agenda_trace",
            "agenda_state": "revisit",
            "agenda_reason": "reopen_softly",
            "timestamp": "2026-03-21T11:00:00",
            "agenda_score": 0.44,
            "agenda_winner_margin": 0.08,
        },
    ]
    memory_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    summary = nightly._summarize_inner_os_agenda_trace(
        {"inner_os_memory_path": str(memory_path)},
        now=nightly.dt.datetime.fromisoformat("2026-03-21T12:00:00"),
        lookback_hours=24,
    )

    assert summary["dominant_agenda"] == "repair"
    assert summary["dominant_reason"] == "repair_window"
    assert summary["state_counts"]["repair"] == 2
    assert summary["agenda_carry_bias"] > 0.0


def test_summarize_inner_os_group_thread_registry_reads_recent_threads(tmp_path, monkeypatch) -> None:
    memory_path = tmp_path / "logs" / "inner_os_memory.jsonl"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "kind": "group_thread_trace",
            "group_thread_id": "threaded_group:user|friend",
            "group_thread_focus": "threaded_group",
            "related_person_id": "user",
            "top_person_ids": ["user", "friend"],
            "thread_total_people": 2,
            "threading_pressure": 0.74,
            "visibility_pressure": 0.22,
            "hierarchy_pressure": 0.08,
            "continuity_score": 0.68,
            "social_grounding": 0.62,
            "timestamp": "2026-03-21T09:00:00",
            "confidence": 0.84,
            "community_id": "harbor_collective",
            "culture_id": "coastal",
            "social_role": "companion",
        },
        {
            "kind": "group_thread_trace",
            "group_thread_id": "threaded_group:user|friend",
            "group_thread_focus": "threaded_group",
            "related_person_id": "user",
            "top_person_ids": ["user", "friend"],
            "thread_total_people": 2,
            "threading_pressure": 0.69,
            "visibility_pressure": 0.25,
            "hierarchy_pressure": 0.1,
            "continuity_score": 0.64,
            "social_grounding": 0.58,
            "timestamp": "2026-03-21T10:00:00",
            "confidence": 0.78,
            "community_id": "harbor_collective",
            "culture_id": "coastal",
            "social_role": "companion",
        },
    ]
    memory_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    summary = nightly._summarize_inner_os_group_thread_registry(
        {"inner_os_memory_path": str(memory_path)},
        now=nightly.dt.datetime.fromisoformat("2026-03-21T12:00:00"),
        lookback_hours=24,
    )

    assert summary["dominant_thread_id"] == "threaded_group:user|friend"
    assert summary["total_threads"] == 1
    assert summary["threads"]["threaded_group:user|friend"]["last_topology_state"] == "threaded_group"
    assert summary["threads"]["threaded_group:user|friend"]["dominant_person_id"] == "user"


def test_daily_carry_summary_builder_reads_nightly_fragments() -> None:
    summary = DailyCarrySummaryBuilder().build(
        {
            "inner_os_memory_class_summary": {"dominant_class": "repair_trace", "dominant_reason": "repair_trace"},
            "inner_os_agenda_summary": {"dominant_agenda": "repair", "dominant_reason": "repair_window"},
            "inner_os_commitment_summary": {"dominant_target": "repair", "dominant_state": "commit"},
            "inner_os_insight_summary": {"dominant_insight_class": "reframed_relation"},
            "inner_os_partner_relation_registry_summary": {
                "dominant_person_id": "user",
                "top_person_ids": ["user", "friend"],
                "total_people": 2,
            },
            "inner_os_group_thread_registry_summary": {
                "dominant_thread_id": "threaded_group:user|friend",
                "top_thread_ids": ["threaded_group:user|friend"],
                "total_threads": 1,
            },
            "inner_os_sleep_memory_class_focus": "repair_trace",
            "inner_os_sleep_agenda_focus": "repair",
            "inner_os_sleep_agenda_reason": "repair_window",
            "inner_os_sleep_commitment_target_focus": "repair",
            "inner_os_sleep_commitment_state_focus": "commit",
            "inner_os_sleep_commitment_followup_focus": "reopen_softly",
            "inner_os_sleep_association_reweighting_focus": "repeated_links",
            "inner_os_sleep_association_reweighting_reason": "repeated_insight_trace",
            "inner_os_sleep_insight_class_focus": "reframed_relation",
            "inner_os_sleep_insight_terrain_shape_target": "soft_relation",
            "inner_os_sleep_insight_terrain_shape_reason": "reframed_relation",
            "inner_os_sleep_body_homeostasis_focus": "recovering",
            "inner_os_sleep_homeostasis_budget_focus": "recovering",
            "inner_os_sleep_relational_continuity_focus": "reopening",
            "inner_os_sleep_group_thread_focus": "threaded_group",
            "inner_os_sleep_expressive_style_focus": "warm_companion",
            "inner_os_sleep_expressive_style_history_focus": "warm_companion",
            "inner_os_sleep_banter_style_focus": "gentle_tease",
            "inner_os_sleep_agenda_bias": 0.26,
            "inner_os_sleep_commitment_carry_bias": 0.31,
            "inner_os_sleep_body_homeostasis_carry_bias": 0.14,
            "inner_os_sleep_homeostasis_budget_bias": 0.09,
            "inner_os_sleep_relational_continuity_carry_bias": 0.11,
            "inner_os_sleep_group_thread_carry_bias": 0.1,
            "inner_os_sleep_expressive_style_carry_bias": 0.1,
            "inner_os_sleep_expressive_style_history_bias": 0.08,
            "inner_os_sleep_lexical_variation_carry_bias": 0.11,
            "inner_os_sleep_association_reweighting_bias": 0.2,
            "inner_os_sleep_insight_reframing_bias": 0.18,
            "inner_os_sleep_insight_terrain_shape_bias": 0.16,
            "inner_os_sleep_temperament_focus": "forward",
            "inner_os_sleep_temperament_forward_bias": 0.1,
        }
    ).to_dict()

    assert summary["same_turn_focus"]["commitment_target"] == "repair"
    assert summary["same_turn_focus"]["agenda_state"] == "repair"
    assert summary["same_turn_focus"]["partner_registry_dominant_person"] == "user"
    assert summary["same_turn_focus"]["group_thread_dominant_thread"] == "threaded_group:user|friend"
    assert summary["same_turn_focus"]["group_thread_total_threads"] == 1
    assert summary["overnight_focus"]["association_focus"] == "repeated_links"
    assert summary["overnight_focus"]["agenda_focus"] == "repair"
    assert summary["overnight_focus"]["body_homeostasis_focus"] == "recovering"
    assert summary["overnight_focus"]["homeostasis_budget_focus"] == "recovering"
    assert summary["overnight_focus"]["relational_continuity_focus"] == "reopening"
    assert summary["overnight_focus"]["group_thread_focus"] == "threaded_group"
    assert summary["overnight_focus"]["expressive_style_focus"] == "warm_companion"
    assert summary["overnight_focus"]["expressive_style_history_focus"] == "warm_companion"
    assert summary["overnight_focus"]["banter_style_focus"] == "gentle_tease"
    assert summary["carry_alignment"]["commitment_carry_visible"] is True
    assert summary["carry_alignment"]["insight_carry_visible"] is True
    assert summary["carry_alignment"]["body_homeostasis_carry_visible"] is True
    assert summary["carry_alignment"]["homeostasis_budget_visible"] is True
    assert summary["carry_alignment"]["relational_continuity_carry_visible"] is True
    assert summary["carry_alignment"]["expressive_style_carry_visible"] is True
    assert summary["carry_alignment"]["expressive_style_history_visible"] is True
    assert summary["carry_alignment"]["banter_style_carry_visible"] is True
    assert summary["carry_alignment"]["partner_registry_visible"] is True
    assert summary["carry_alignment"]["group_thread_carry_visible"] is True
    assert summary["carry_alignment"]["group_thread_registry_visible"] is True
    assert summary["carry_strengths"]["agenda"] == 0.26
