from __future__ import annotations

from pathlib import Path

from eqnet_core.models.activation_trace import (
    ActivationNode,
    ActivationTrace,
    ActivationTraceLogger,
    ConfidenceSample,
    ReplayEvent,
)
from eqnet_core.models.scene_frame import AffectSnapshot, SceneAgent, SceneFrame
from runtime.nightly_report import generate_recall_report


def test_activation_trace_roundtrip(tmp_path):
    log_path = Path(tmp_path) / "activation.jsonl"
    logger = ActivationTraceLogger(log_path)
    frame = SceneFrame(
        scene_id="scene-1",
        anchor="bakery",
        agents=[SceneAgent(name="self", role="observer", perspective="first_person", certainty=1.0)],
        affect_snapshots=[AffectSnapshot(label="father_glare", intensity=0.8, replay_source="replay")],
    )
    trace = ActivationTrace(
        trace_id="trace-1",
        timestamp=123.0,
        trigger_context={"flags": ["anchor:bakery"], "context_tags": ["commute", "memory"]},
        anchor_hit="bakery",
        activation_chain=[ActivationNode(node_id="seed-1", activation=0.6)],
        confidence_curve=[
            ConfidenceSample(step=0, conf_internal=0.4, conf_external=0.1),
            ConfidenceSample(step=1, conf_internal=0.8, conf_external=0.15),
        ],
        replay_events=[ReplayEvent(scene_id="scene-1", payload={"anchor": "bakery"})],
        notes="Anchor bakery lifted confidence.",
        metadata={"reverse_ratio": 0.4},
        scene_frames=[frame],
    )
    logger.write(trace)

    stored = list(logger.iter_traces())
    assert len(stored) == 1
    assert stored[0].anchor_hit == "bakery"

    report = generate_recall_report(log_path)
    payload = report.to_dict()
    assert payload["trace_count"] == 1
    assert payload["anchors"].get("bakery") == 1
    assert "dream_prompt" in payload and "bakery" in payload["dream_prompt"]
