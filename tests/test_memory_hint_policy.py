from __future__ import annotations

from emot_terrain_lab.memory.memory_hint import render_memory_hint
from runtime.config import MemoryHintPolicyCfg


def test_memory_hint_blocks_verbatim() -> None:
    cfg = MemoryHintPolicyCfg(enable=True, allow_verbatim=False, max_reply_chars=80)
    label = "2019 Kyoto trip"
    hint = render_memory_hint(
        label,
        0.8,
        locale="ja-JP",
        cfg=cfg,
        gate_ctx={"since_last_user_ms": 5000.0, "text_input": False},
    )
    assert hint is not None
    assert hint["shown"] is True
    assert label not in hint["text"]


def test_memory_hint_respects_max_reply_chars() -> None:
    cfg = MemoryHintPolicyCfg(enable=True, allow_verbatim=False, max_reply_chars=5)
    hint = render_memory_hint(
        "label",
        0.8,
        locale="ja-JP",
        cfg=cfg,
        gate_ctx={"since_last_user_ms": 5000.0, "text_input": False},
    )
    assert hint is not None
    assert hint["shown"] is True
    assert len(hint["text"]) <= 5


def test_memory_hint_ban_patterns() -> None:
    cfg = MemoryHintPolicyCfg(enable=True, allow_verbatim=False, ban_patterns=["."])
    hint = render_memory_hint(
        "label",
        0.8,
        locale="ja-JP",
        cfg=cfg,
        gate_ctx={"since_last_user_ms": 5000.0, "text_input": False},
    )
    assert hint is not None
    assert hint["blocked"] is True


def test_memory_hint_interrupt_cost_blocks() -> None:
    cfg = MemoryHintPolicyCfg(enable=True, allow_verbatim=False, min_silence_ms_for_hint=2000)
    hint = render_memory_hint(
        "label",
        0.8,
        locale="ja-JP",
        cfg=cfg,
        gate_ctx={"since_last_user_ms": 500.0, "text_input": False},
    )
    assert hint is not None
    assert hint["blocked"] is True
    assert hint["reason"] == "interrupt_cost"


def test_memory_hint_low_confidence_blocks() -> None:
    cfg = MemoryHintPolicyCfg(enable=True, allow_verbatim=False, pressure_threshold=0.3)
    hint = render_memory_hint(
        "label",
        0.8,
        locale="ja-JP",
        cfg=cfg,
        gate_ctx={"since_last_user_ms": 5000.0, "text_input": False},
        shadow_uncertainty=0.95,
        prev_pressure=0.2,
        dt_s=0.0,
    )
    assert hint is not None
    assert hint["blocked"] is True
    assert hint["reason"] in ("low_confidence", "pressure")


def test_memory_hint_turn_taking_blocks() -> None:
    cfg = MemoryHintPolicyCfg(enable=True, allow_verbatim=False, turn_taking_mode="last_speaker_not_ai")
    hint = render_memory_hint(
        "label",
        0.8,
        locale="ja-JP",
        cfg=cfg,
        gate_ctx={"since_last_user_ms": 5000.0, "text_input": False},
        last_speaker="ai",
    )
    assert hint is not None
    assert hint["blocked"] is True
    assert hint["reason"] == "turn_taking"


def test_memory_hint_minimal_when_suppressed() -> None:
    cfg = MemoryHintPolicyCfg(
        enable=True,
        allow_verbatim=False,
        style_when_suppressed="minimal",
        min_silence_ms_for_hint=2000,
    )
    hint = render_memory_hint(
        "label",
        0.8,
        locale="ja-JP",
        cfg=cfg,
        gate_ctx={"since_last_user_ms": 500.0, "text_input": False},
    )
    assert hint is not None
    assert hint["shown"] is True
    assert hint["key"] == "memory_hint.minimal"


def test_memory_hint_pressure_decay() -> None:
    cfg = MemoryHintPolicyCfg(enable=True, allow_verbatim=False, pressure_decay_per_s=0.9)
    hint = render_memory_hint(
        "label",
        0.8,
        locale="ja-JP",
        cfg=cfg,
        gate_ctx={"since_last_user_ms": 5000.0, "text_input": False},
        prev_pressure=1.0,
        dt_s=10.0,
    )
    assert hint is not None
    assert hint["pressure"] <= 1.0
