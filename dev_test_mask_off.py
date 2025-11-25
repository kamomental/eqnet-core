# -*- coding: utf-8 -*-
"""Quick smoke test for the EQNet runtime with mask layer disabled."""

from __future__ import annotations

import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "emot_terrain_lab"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import emot_terrain_lab.hub.runtime as runtime_mod
from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig, MaskLayerConfig


def _force_talk_gate(ctx):
    return runtime_mod.TalkMode.TALK

runtime_mod._decide_talk_mode = _force_talk_gate


class _FakeStats:
    def __init__(self) -> None:
        self.last_video_metrics = {"motion": 0.5, "blink": 0.05}
        self.last_audio_metrics = {"rms": 0.6}
        self.video_latency_ms = 0.0
        self.audio_latency_ms = 0.0
        self.fusion_latency_ms = 0.0
        self.video_latency_p95_ms = 0.0
        self.audio_latency_p95_ms = 0.0
        self.fusion_latency_p95_ms = 0.0


def _force_talk_mode(runtime: EmotionalHubRuntime) -> None:
    runtime.set_engaged(True)
    runtime.perception.stats = lambda: _FakeStats()
    runtime._last_user_ts = time.time() - 1.0


def _run_once(label: str, cfg: RuntimeConfig, user_text: str) -> None:
    runtime = EmotionalHubRuntime(config=cfg)
    runtime.observe_text(user_text)
    time.sleep(0.5)
    _force_talk_mode(runtime)
    result = runtime.step(user_text=user_text)
    response = result.get("response")
    if response is not None:
        print(f"[{label}] response.text = {getattr(response, 'text', response)}")
    else:
        print(f"[{label}] response=None")
    print(f"[{label}] persona_meta = {runtime.last_persona_meta}")


def main() -> None:
    user_text = "テストです。簡単な一言だけ返してください。"
    cfg = RuntimeConfig(mask_layer=MaskLayerConfig(enabled=False))
    _run_once("MASK_OFF", cfg, user_text)


if __name__ == "__main__":
    main()
