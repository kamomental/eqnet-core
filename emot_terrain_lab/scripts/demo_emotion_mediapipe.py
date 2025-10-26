# -*- coding: utf-8 -*-
"""
Real-time emotion demo using MediaPipe + EQNet.

Requirements:
    pip install opencv-python mediapipe matplotlib

Controls:
    - ESC to exit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import mediapipe  # noqa: F401
except ImportError:  # pragma: no cover - runtime check
    print("MediaPipe is required: pip install mediapipe", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from hub import EmotionalHubRuntime, RuntimeConfig, PerceptionConfig
from terrain.emotion import AXES, AXIS_BOUNDS


def normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def build_theta(count: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, count, endpoint=True)


def advanced_radar_values(emotion_vec: np.ndarray, qualia: Dict[str, Any]) -> List[float]:
    axes = ["sensory", "temporal", "spatial", "affective", "cognitive", "social", "meta"]
    vals = [normalize(float(emotion_vec[AXES.index(axis)]), *AXIS_BOUNDS[axis]) for axis in axes]
    magnitude = float(abs(qualia.get("magnitude", 0.0)))
    enthalpy = float(qualia.get("enthalpy", 0.0))
    vals.append(float(np.clip(np.tanh(magnitude), 0.0, 1.0)))  # qualia magnitude
    vals.append(float(np.clip(np.tanh(enthalpy / 5.0), 0.0, 1.0)))  # qualia enthalpy
    vals.append(vals[0])
    return vals


def basic_radar_values(valence: float, arousal: float) -> List[float]:
    v = float(np.clip(valence, -1.0, 1.0))
    a = float(np.clip(arousal, -1.0, 1.0))
    joy = max(0.0, (v + a) / 2.0)
    calm = max(0.0, v * (1.0 - max(a, 0.0)))
    sadness = max(0.0, (-v - a) / 2.0)
    anger = max(0.0, (-v + a) / 2.0)
    surprise = max(0.0, a)
    vals = [joy, calm, sadness, anger, surprise]
    vals.append(vals[0])
    return vals


def main() -> None:
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=True,
            perception=PerceptionConfig(
                video_backend="mediapipe",
            ),
        )
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.", file=sys.stderr)
        sys.exit(1)

    plt.ion()
    fig = plt.figure(figsize=(10, 5))
    ax_adv = fig.add_subplot(1, 2, 1, projection="polar")
    ax_basic = fig.add_subplot(1, 2, 2, projection="polar")

    theta_adv = build_theta(10)  # 7 axes + 2 qualia + closing point
    theta_basic = build_theta(6)  # 5 axes + closing point

    adv_line, = ax_adv.plot(theta_adv, np.zeros_like(theta_adv), color="#ff7f0e")
    ax_adv.set_ylim(0, 1)
    ax_adv.set_title("EQNet 7D + Qualia")
    ax_adv.set_xticklabels(
        ["Sensory", "Temporal", "Spatial", "Affective", "Cognitive", "Social", "Meta", "Qualia Mag", "Qualia Ent"]
    )

    basic_line, = ax_basic.plot(theta_basic, np.zeros_like(theta_basic), color="#1f77b4")
    ax_basic.set_ylim(0, 1)
    ax_basic.set_title("Basic Emotions")
    ax_basic.set_xticklabels(["Joy", "Calm", "Sadness", "Anger", "Surprise"])

    frame_count = 0
    last_update = time.time()
    step_interval = 5  # frames

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            runtime.observe_video(frame_rgb)

            if frame_count % step_interval == 0:
                result = runtime.step(user_text="")

                affect = result["affect"]
                metrics = result["metrics"]
                qualia = runtime.last_qualia

                adv_vals = advanced_radar_values(runtime.last_E, qualia)
                basic_vals = basic_radar_values(affect.valence, affect.arousal)

                adv_line.set_data(theta_adv, adv_vals)
                basic_line.set_data(theta_basic, basic_vals)
                fig.canvas.draw()
                fig.canvas.flush_events()

                overlay = frame_bgr.copy()
                cv2.putText(overlay, f"Valence: {affect.valence:+.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"Arousal: {affect.arousal:+.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"Entropy: {metrics.get('entropy', 0.0):.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(overlay, f"Qualia Mag: {qualia.get('magnitude', 0.0):.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow("EQNet Emotion Demo (ESC to exit)", overlay)
            else:
                cv2.imshow("EQNet Emotion Demo (ESC to exit)", frame_bgr)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Limit update rate to avoid busy loop
            now = time.time()
            if now - last_update < 1 / 60:
                time.sleep(max(0.0, 1 / 60 - (now - last_update)))
            last_update = time.time()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close(fig)


if __name__ == "__main__":
    main()
