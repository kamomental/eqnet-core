from __future__ import annotations

from pathlib import Path

from eqnet.memory.state_vector import TemporalStateVector
from rag.retriever import build_assoc_kwargs
from runtime.config import load_runtime_cfg


def test_load_runtime_cfg_reads_rag_assoc_block(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "rag:",
                "  assoc_score:",
                "    enabled: true",
                "    temporal_tau_sec: 1234",
                "    weights:",
                "      semantic: 1.0",
                "      temporal: 0.2",
                "      affective: 0.1",
                "      value: 0.1",
                "      open_loop: 0.05",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_runtime_cfg(cfg_path)
    assert cfg.rag.assoc_score.enabled is True
    assert cfg.rag.assoc_score.temporal_tau_sec == 1234
    assert cfg.rag.assoc_score.weights.temporal == 0.2


def test_build_assoc_kwargs_uses_temporal_state():
    cfg = load_runtime_cfg()
    cfg.rag.assoc_score.enabled = True
    cfg.rag.assoc_score.temporal_tau_sec = 500.0
    state = TemporalStateVector(
        timestamp_ms=2000,
        valence=0.3,
        arousal=-0.2,
        value_tags={"safety": 0.8},
        open_loops=0.4,
        event_scale=0.7,
    )
    kwargs = build_assoc_kwargs(runtime_cfg=cfg, temporal_state=state)
    assert "assoc_context" in kwargs
    assert kwargs["assoc_context"]["timestamp_ms"] == 2000
    assert kwargs["assoc_context"]["temporal_tau_sec"] == 500.0
    assert kwargs["assoc_context"]["normalize_weights"] is True
    assert kwargs["assoc_context"]["clamp_min"] == -5.0
    assert kwargs["assoc_context"]["clamp_max"] == 5.0
    assert abs(sum(kwargs["assoc_weights"].values()) - 1.0) < 1e-9
    assert kwargs["assoc_weights"]["semantic"] > kwargs["assoc_weights"]["temporal"]


def test_build_assoc_kwargs_clamps_negative_and_normalizes():
    cfg = load_runtime_cfg()
    cfg.rag.assoc_score.enabled = True
    cfg.rag.assoc_score.normalize_weights = True
    cfg.rag.assoc_score.weights.semantic = -1.0
    cfg.rag.assoc_score.weights.temporal = 2.0
    cfg.rag.assoc_score.weights.affective = 0.0
    cfg.rag.assoc_score.weights.value = 0.0
    cfg.rag.assoc_score.weights.open_loop = 0.0

    kwargs = build_assoc_kwargs(runtime_cfg=cfg, temporal_state={"timestamp_sec": 1})
    weights = kwargs["assoc_weights"]
    assert weights["semantic"] == 0.0
    assert weights["temporal"] == 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_build_assoc_kwargs_clamps_negative_without_normalize():
    cfg = load_runtime_cfg()
    cfg.rag.assoc_score.enabled = True
    cfg.rag.assoc_score.normalize_weights = False
    cfg.rag.assoc_score.weights.semantic = -3.0
    cfg.rag.assoc_score.weights.temporal = 2.0
    cfg.rag.assoc_score.weights.affective = -1.0
    kwargs = build_assoc_kwargs(runtime_cfg=cfg, temporal_state={"timestamp_sec": 1})
    weights = kwargs["assoc_weights"]
    assert weights["semantic"] == 0.0
    assert weights["affective"] == 0.0
    assert weights["temporal"] == 2.0
