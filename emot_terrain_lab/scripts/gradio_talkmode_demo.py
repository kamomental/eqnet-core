# -*- coding: utf-8 -*-
"""
リアルタイムで表情を読み取り、TalkMode 世界モデルの挙動を体験する Gradio デモ。

Requirements
------------
pip install gradio mediapipe opencv-python

使い方
------
python -m emot_terrain_lab.scripts.gradio_talkmode_demo
ブラウザが開いたらカメラへのアクセスを許可してください。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np

from emot_terrain_lab.hub.perception import AffectSample, PerceptionConfig
from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig

# --------------------------------------------------------------------------- #
# MediaPipe 初期化
# --------------------------------------------------------------------------- #

_MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def _extract_face_metrics(frame_rgb: np.ndarray) -> Tuple[bool, float, float]:
    """MediaPipe で唇の開きと瞬き量を推定。"""
    result = _MP_FACE_MESH.process(frame_rgb)
    if not result.multi_face_landmarks:
        return False, 0.0, 0.0
    landmarks = result.multi_face_landmarks[0].landmark
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    mouth_open = abs(upper_lip.y - lower_lip.y)
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    blink = abs(left_eye_top.y - left_eye_bottom.y)
    return True, mouth_open, blink


# --------------------------------------------------------------------------- #
# 状態管理
# --------------------------------------------------------------------------- #


@dataclass
class DemoState:
    prev_valence: float = 0.0
    prev_delta: float = 0.0
    voice_energy: float = 0.35  # 音声入力が無い場合は穏やかに固定

    def update_affect(self, valence: float) -> Tuple[float, float]:
        delta = abs(valence - self.prev_valence)
        jerk = abs(delta - self.prev_delta)
        self.prev_valence = valence
        self.prev_delta = delta
        return delta, jerk


_STATE = DemoState()


# --------------------------------------------------------------------------- #
# TalkMode ランタイムの初期化
# --------------------------------------------------------------------------- #

runtime_cfg = RuntimeConfig(
    perception=PerceptionConfig(
        video_backend="mediapipe",
        fusion_hz=15.0,
        batch_enabled=False,
    )
)
RUNTIME = EmotionalHubRuntime(runtime_cfg)


# --------------------------------------------------------------------------- #
# Ack 表現（日本語）
# --------------------------------------------------------------------------- #

ACK_TEXT = {
    "watch": "👀 ここで待っているよ。必要になったら合図してね。",
    "soothe": "🤝 表情が少し硬いかも。一緒に深呼吸して整えようか。",
    "ask": "❓ さっきより元気が少なめに見える。何かあった？",
    "talk": "💬 続きを整えよう。どこから話そうか。",
}


# --------------------------------------------------------------------------- #
# メイン処理
# --------------------------------------------------------------------------- #


def _annotate(frame_bgr: np.ndarray, talk_mode: str, ack: str, engaged: bool) -> np.ndarray:
    h, w, _ = frame_bgr.shape
    color_map = {
        "watch": (200, 200, 200),
        "soothe": (60, 180, 255),
        "ask": (80, 120, 255),
        "talk": (50, 205, 50),
    }
    color = color_map.get(talk_mode, (200, 200, 200))
    status = "engaged" if engaged else "idle"
    cv2.putText(frame_bgr, f"Mode: {talk_mode.upper()} ({status})", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    y = h - 60
    cv2.rectangle(frame_bgr, (20, y - 40), (w - 20, y + 10), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(frame_bgr, ack, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame_bgr


def process_frame(frame: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, float], str, Dict[str, float]]:
    if frame is None:
        empty = np.zeros((480, 640, 3), dtype=np.uint8)
        return empty, {"watch": 1.0}, "カメラ入力を待っています。", {}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected, mouth_open, blink = _extract_face_metrics(frame_rgb)
    engaged = bool(detected)

    smile = max(0.0, mouth_open - 0.015)
    valence = float(np.clip(smile * 16.0 - 0.2, -1.0, 1.0))
    arousal = float(np.clip((mouth_open + blink) * 10.0, 0.0, 1.0))
    delta_m, jerk = _STATE.update_affect(valence)

    ts = time.time()
    RUNTIME.perception._last_affect = AffectSample(valence=valence, arousal=arousal, confidence=0.9, timestamp=ts)
    RUNTIME.perception._last_timestamp = ts
    RUNTIME.perception._last_video_metrics = {"motion": float(mouth_open), "blink": float(blink)}
    RUNTIME.perception._last_audio_metrics = {"rms": _STATE.voice_energy}
    RUNTIME.set_engaged(engaged)

    resp = RUNTIME.step(user_text=None, fast_only=True)
    talk_mode = (resp.get("talk_mode") or "watch").lower()
    gate_ctx = resp.get("gate_context") or {}

    ack_text = ACK_TEXT.get(talk_mode, "")
    response_obj = resp.get("response")
    if response_obj is not None and getattr(response_obj, "text", ""):
        ack_text = getattr(response_obj, "text")
    elif not ack_text:
        ack_text = "ここにいるよ。"

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated = _annotate(frame_bgr, talk_mode, ack_text, engaged)
    frame_out = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    mode_label = {talk_mode: 1.0}
    debug_ctx = {
        "mouth_open": round(mouth_open, 4),
        "blink": round(blink, 4),
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "delta_m": round(delta_m, 3),
        "jerk": round(jerk, 3),
        **gate_ctx,
    }
    return frame_out, mode_label, ack_text, debug_ctx


def toggle_listening(flag: bool) -> str:
    RUNTIME.set_listening(bool(flag))
    return "見守り固定 (WATCH)" if flag else "自動切替 (AUTO)"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## EQNet TalkMode デモ : 表情から“間合い”を読む")
    gr.Markdown(
        "カメラに向かって笑ったり、視線を逸らしたりすると、見守り/寄り添い/問いかけ/会話のモードが切り替わります。"
    )

    with gr.Row():
        video = gr.Video(source="webcam", streaming=True, label="Webcam")
        image_out = gr.Image(label="Annotated", width=480, height=360)

    with gr.Row():
        mode_label = gr.Label(label="Talk Mode")
        ack_box = gr.Textbox(label="Ack / Response", interactive=False)

    gate_json = gr.JSON(label="GateContext / Metrics")
    listen_toggle = gr.Checkbox(label="見守り固定（TALKを抑制）", value=False)
    listen_status = gr.Textbox(label="現在のモード判定", value="自動切替 (AUTO)", interactive=False)

    video.stream(process_frame, inputs=video, outputs=[image_out, mode_label, ack_box, gate_json])
    listen_toggle.change(toggle_listening, inputs=listen_toggle, outputs=listen_status)

    gr.Markdown(
        "### ヒント\n"
        "- 笑顔や大きな表情で TALK モードへ移行します。\n"
        "- 無表情・視線を外すと WATCH モードで静かに見守ります。\n"
        "- 表情が硬いと SOOTHE、元気が落ちて見えると ASK の短い声かけが入ります。\n"
        "- 表示される GateContext を Nightly ログと照合すると、現場のしきい値調整に役立ちます。"
    )

demo.queue().launch()

