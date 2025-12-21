from __future__ import annotations

from pathlib import Path

from emot_terrain_lab.hub.recall_engine import RecallEngine, RecallEngineConfig


def _sample_seed(trace_id: str, novelty: float) -> dict:
    return {
        "trace_id": trace_id,
        "meta": {"novelty": novelty, "anchor": "bakery", "constraint_weight": 0.3},
        "value": {"total": 0.45},
    }


def test_recall_engine_emits_activation_trace(tmp_path):
    cfg = RecallEngineConfig(anchor_gain=0.5, confirm_gain=0.3)
    log_path = Path(tmp_path) / "activation.jsonl"
    engine = RecallEngine(cfg, log_path=str(log_path), clock=lambda: 42.0)
    ctx_time = {
        "flags": ["anchor:bakery", "constraint:child"],
        "salient_entities": ["grandmother", "father"],
        "constraints": ["child tired"],
        "novelty": 0.35,
    }
    plan = {"mood": {"valence": 0.2, "arousal": 0.3, "social": 0.6}}
    seeds = [_sample_seed("s1", 0.8), _sample_seed("s2", 0.4)]
    replay_details = {"steps": 2, "reverse_ratio": 0.4, "policy_update": True}

    trace, frames = engine.ignite(
        ctx_time=ctx_time,
        plan=plan,
        seeds=seeds,
        best_choice={"a": "reflect", "U": 0.42, "summary": {"total": 0.4}},
        replay_details=replay_details,
        field_signals={},
    )

    assert trace is not None
    assert trace.anchor_hit == "bakery"
    assert trace.confidence_curve[-1].conf_internal >= trace.confidence_curve[0].conf_internal
    assert frames and frames[0].anchor == "bakery"
    assert log_path.exists()
