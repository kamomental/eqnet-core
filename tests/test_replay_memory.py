import json
from pathlib import Path

from devlife.mind.replay_memory import ReplayMemory, ReplayTrace


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
