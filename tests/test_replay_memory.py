import json
from pathlib import Path

from devlife.mind.replay_memory import MemoryKind, ReplayMemory, ReplayTrace


def test_store_and_read(tmp_path):
    path = Path(tmp_path) / "mem.jsonl"
    memory = ReplayMemory(path=path)
    trace = ReplayTrace(
        trace_id="t1",
        episode_id="e1",
        timestamp=0.0,
        source="internal",
        horizon=3,
        uncertainty=0.4,
        mood={"mood_v": 0.1, "mood_a": 0.3, "effort": 0.0, "uncertainty": 0.4},
        value={"success": 0.6, "effort": 0.1, "aesthetic": 0.2, "consistency": 0.05},
        controls={"before": {}, "after": {}, "_replay_V": 0.12},
        imagined={
            "valence_pred": 0.0,
            "harm_prob": 0.0,
            "comfort_gain": 0.2,
            "success_prob": 0.6,
            "self_consistency_err": 0.1,
        },
        meta={"task": "t", "domain": "language"},
    )
    memory.store(trace)
    assert path.exists()
    rows = [json.loads(line) for line in path.open()]
    assert rows[0]["source"] == "internal"
    assert rows[0]["horizon"] == 3


def test_store_routes_unknown_kind_to_think_log_and_normalizes_kind(tmp_path):
    legacy = Path(tmp_path) / "mem.jsonl"
    think = Path(tmp_path) / "think.jsonl"
    act = Path(tmp_path) / "act.jsonl"
    memory = ReplayMemory(path=legacy, think_log_path=think, act_log_path=act)
    trace = ReplayTrace(
        trace_id="t2",
        episode_id="e2",
        timestamp=0.0,
        source="internal",
        horizon=1,
        uncertainty=0.1,
        mood={},
        value={},
        controls={},
        imagined={"best_action": "imagine"},
        meta={},
        memory_kind="invalid_kind",
    )
    memory.store(trace)
    assert think.exists()
    think_rows = [json.loads(line) for line in think.open(encoding="utf-8")]
    assert think_rows[0]["memory_kind"] == MemoryKind.UNKNOWN.value
    assert think_rows[0]["meta"]["audit_event"] == "MEMORY_KIND_NORMALIZED"
    assert not act.exists()


def test_store_maps_legacy_episodic_to_experience_act_log(tmp_path):
    memory = ReplayMemory(
        path=Path(tmp_path) / "mem.jsonl",
        think_log_path=Path(tmp_path) / "think.jsonl",
        act_log_path=Path(tmp_path) / "act.jsonl",
    )
    trace = ReplayTrace(
        trace_id="t3",
        episode_id="e3",
        timestamp=0.0,
        source="internal",
        horizon=1,
        uncertainty=0.1,
        mood={},
        value={},
        controls={},
        imagined={},
        meta={},
        memory_kind="episodic",
    )
    memory.store(trace)
    rows = [json.loads(line) for line in memory.act_log_path.open(encoding="utf-8")]
    assert rows[0]["memory_kind"] == MemoryKind.EXPERIENCE.value


def test_promote_with_evidence_blocks_without_evidence_event_id(tmp_path):
    memory = ReplayMemory(
        path=Path(tmp_path) / "mem.jsonl",
        think_log_path=Path(tmp_path) / "think.jsonl",
        act_log_path=Path(tmp_path) / "act.jsonl",
    )
    trace = ReplayTrace(
        trace_id="t4",
        episode_id="e4",
        timestamp=0.0,
        source="internal",
        horizon=1,
        uncertainty=0.1,
        mood={},
        value={},
        controls={},
        imagined={},
        meta={},
        memory_kind=MemoryKind.IMAGERY.value,
    )
    memory.store(trace)
    out = memory.promote_with_evidence("t4", evidence_event_id=None)
    assert out["promoted"] is False
    assert out["reason"] == "missing_evidence_event_id"
    assert not memory.act_log_path.exists()
    think_rows = [json.loads(line) for line in memory.think_log_path.open(encoding="utf-8")]
    assert any(
        (row.get("meta") or {}).get("audit_event") == "PROMOTION_GUARD_BLOCKED"
        for row in think_rows
    )


def test_promote_with_evidence_promotes_to_experience(tmp_path):
    memory = ReplayMemory(
        path=Path(tmp_path) / "mem.jsonl",
        think_log_path=Path(tmp_path) / "think.jsonl",
        act_log_path=Path(tmp_path) / "act.jsonl",
    )
    trace = ReplayTrace(
        trace_id="t5",
        episode_id="e5",
        timestamp=0.0,
        source="internal",
        horizon=1,
        uncertainty=0.1,
        mood={},
        value={},
        controls={},
        imagined={},
        meta={},
        memory_kind=MemoryKind.HYPOTHESIS.value,
    )
    memory.store(trace)
    out = memory.promote_with_evidence("t5", evidence_event_id="obs-1")
    assert out["promoted"] is True
    rows = [json.loads(line) for line in memory.act_log_path.open(encoding="utf-8")]
    assert rows[0]["memory_kind"] == MemoryKind.EXPERIENCE.value
    meta = rows[0].get("meta") or {}
    assert meta.get("promotion_evidence_event_id") == "obs-1"
    assert meta.get("promoted_from_memory_kind") == MemoryKind.HYPOTHESIS.value
