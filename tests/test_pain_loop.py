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
    details = stats.get("events_detail") or []
    assert details, "events_detail should contain entries"
    detail = details[0]
    assert "reason_tag" in detail and "next_action" in detail
    update = policy_update_from_forgiveness(
        "nid-1",
        base_threshold=0.5,
        empathy_gain_base=0.1,
        events_detail=stats.get("events_detail"),
    )
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


def test_care_targets_stats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _cleanup()
    log_pain_event(
        "isolation",
        -0.30,
        ["isolation"],
        {"emotional": 0.4, "metabolic": 0.1, "ethical": 0.2},
        {"scene": "play", "target": "childlike"},
    )
    log_pain_event(
        "value_conflict",
        -0.55,
        ["rudeness"],
        {"emotional": 0.6, "metabolic": 0.2, "ethical": 0.6},
        {"scene": "protect", "target": "childlike"},
    )
    total, forgiven, stats = evaluate_and_forgive(
        "nid-care",
        base_threshold=0.5,
        adaptive=False,
        care_targets=["childlike"],
        comfort_gain_base=0.2,
        protection_bias=0.4,
        growth_reward=0.1,
        patience_budget=0.6,
        replay_eval=False,
    )
    assert total == 2
    assert "care_stats" in stats
    care_stats = stats["care_stats"]
    assert care_stats["targets"]["childlike"]["total"] == 2
    assert care_stats["patience_budget"] == 0.6
    # comfort/protection signals should be finite numbers
    assert isinstance(care_stats["comfort_gain_applied"], float)
    assert isinstance(care_stats["protection_signal"], float)
    assert isinstance(care_stats["growth_signal"], float)
    assert care_stats["interventions"] + care_stats["watch_only"] == 2

