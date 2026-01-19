from __future__ import annotations

from dataclasses import dataclass

from emot_terrain_lab.hub.runtime import GateContext, EmotionalHubRuntime
from runtime.config import PresenceCfg


@dataclass
class _PresenceHarness:
    _presence_cfg: PresenceCfg
    _last_presence_ack_ts: float = 0.0

    _should_presence = EmotionalHubRuntime._should_presence


def _ctx(**overrides) -> GateContext:
    data = {
        "engaged": False,
        "face_motion": 0.0,
        "blink": 0.0,
        "voice_energy": 0.0,
        "delta_m": 0.0,
        "jerk": 0.0,
        "text_input": False,
        "since_last_user_ms": 5000.0,
        "force_listen": False,
    }
    data.update(overrides)
    return GateContext(**data)


def test_presence_respects_force_listen_and_text_input() -> None:
    cfg = PresenceCfg()
    harness = _PresenceHarness(cfg)
    now_ts = 100.0
    assert not harness._should_presence(_ctx(force_listen=True), 0.8, None, now_ts)
    assert not harness._should_presence(_ctx(text_input=True), 0.8, None, now_ts)


def test_presence_respects_min_silence() -> None:
    cfg = PresenceCfg(min_silence_ms=1200)
    harness = _PresenceHarness(cfg)
    now_ts = 100.0
    assert not harness._should_presence(_ctx(since_last_user_ms=200.0), 0.8, None, now_ts)


def test_presence_accepts_last_gate_suppress() -> None:
    cfg = PresenceCfg()
    harness = _PresenceHarness(cfg)
    now_ts = 100.0
    assert harness._should_presence(_ctx(), None, False, now_ts)


def test_presence_accepts_shadow_threshold() -> None:
    cfg = PresenceCfg(shadow_threshold=0.7)
    harness = _PresenceHarness(cfg)
    now_ts = 100.0
    assert harness._should_presence(_ctx(), 0.7, None, now_ts)


def test_presence_respects_ack_interval() -> None:
    cfg = PresenceCfg(max_ack_interval_s=180.0)
    harness = _PresenceHarness(cfg, _last_presence_ack_ts=50.0)
    now_ts = 100.0
    assert not harness._should_presence(_ctx(), 0.9, None, now_ts)
