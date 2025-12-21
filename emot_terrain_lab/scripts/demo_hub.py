# -*- coding: utf-8 -*-
"""
Tiny demo that wires the EQNet hub scaffolding together.

This does **not** perform real perception; it simply feeds random frames/chunks
through the bridge to verify the runtime plumbing and EQNet integration.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass  # Not available on some interpreters.

from hub import EmotionalHubRuntime, RuntimeConfig


def main() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    # Simulate a single frame + audio chunk.
    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    audio = (np.random.randn(1600) * 0.02).astype(np.float32)

    runtime.observe_video(frame)
    runtime.observe_audio(audio)

    result = runtime.step(user_text="hello")
    response = result["response"]
    controls = result["controls"]

    print("Affect:", result["affect"])
    print("Perception stats:", runtime.perception.stats())
    print("EQNet metrics:", result["metrics"])
    if response:
        print("Response:", response.text)
        print("Latency (ms):", round(response.latency_ms, 1))
    print("Controls:", controls)


if __name__ == "__main__":
    main()
