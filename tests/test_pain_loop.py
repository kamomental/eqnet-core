from emot_terrain_lab.ops.pain_loop import PAIN_DIR, evaluate_and_forgive, log_pain_event, policy_update_from_forgiveness
from emot_terrain_lab.utils.jsonl_io import read_jsonl


def _cleanup() -> None:
    if not PAIN_DIR.exists():
        return
    for file in PAIN_DIR.glob("*.jsonl"):
        file.unlink(missing_ok=True)


def test_pain_to_forgive_cycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _cleanup()
    event = log_pain_event(
        "isolation",
        -0.42,
        ["isolation"],
        {"emotional": 0.6, "metabolic": 0.1, "ethical": 0.3},
        {"scene": "x"},
    )
    assert event["delta_aff"] < 0
    duplicate = log_pain_event(
        "isolation",
        -0.42,
        ["isolation"],
        {"emotional": 0.6, "metabolic": 0.1, "ethical": 0.3},
        {"scene": "x"},
    )
    assert duplicate["idempotency_key"] == event["idempotency_key"]
    records = list(read_jsonl(str(PAIN_DIR / "pain_events.jsonl")))
    assert len(records) == 1
    total, forgiven, stats = evaluate_and_forgive("nid-1", base_threshold=0.5, adaptive=False)
    assert total == 1
    assert forgiven in (0, 1)
    assert stats["forgive_threshold"] <= 0.5
    assert "breakdown" in stats and "by_kind" in stats["breakdown"]
    update = policy_update_from_forgiveness("nid-1", base_threshold=0.5, empathy_gain_base=0.1)
    assert 0.2 <= update["policy_feedback_threshold"] <= 0.5
    assert 0.1 <= update["a2a_empathy_gain"] <= 0.5
    assert "forgive_rate" in update


def test_negative_required(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _cleanup()
    event = log_pain_event(
        "energy_depletion",
        -0.05,
        ["overload"],
        {"emotional": 0.1, "metabolic": 0.5, "ethical": 0.0},
        {"task": "y"},
    )
    assert event["delta_aff"] < 0

