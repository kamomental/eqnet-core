# -*- coding: utf-8 -*-
"""
Gradio demo for the EQNet emotional hub.

Usage:
    python scripts/gradio_demo.py
    (open the URL displayed in the terminal)
"""

from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Deque, Dict, Optional, Tuple
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
import time
import os
import json
import random

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import yaml
import requests
import matplotlib.pyplot as plt
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import mediapipe as mp

    HAVE_MEDIAPIPE = True
except Exception:  # pragma: no cover
    HAVE_MEDIAPIPE = False

try:
    import pyttsx3

    HAVE_TTS = True
except Exception:  # pragma: no cover
    HAVE_TTS = False

from emot_terrain_lab.perception.text_affect import quick_text_affect_v2

from runtime.config import load_runtime_cfg
from hub import EmotionalHubRuntime, RuntimeConfig
from hub.perception import AffectSample
from terrain.emotion import AXES, AXIS_BOUNDS


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _build_radar_chart(emotion_vec: np.ndarray, qualia: Dict[str, Any]) -> go.Figure:
    """Create a 9-axis radar chart (7D emotion + 2D qualia)."""
    if emotion_vec is None or emotion_vec.size < len(AXES):
        return go.Figure()

    emotion_axes = ["sensory", "temporal", "spatial", "affective", "cognitive", "social", "meta"]
    emotion_values = [
        _normalize(float(emotion_vec[AXES.index(axis)]), *AXIS_BOUNDS[axis])
        for axis in emotion_axes
    ]

    magnitude = float(abs(qualia.get("magnitude", 0.0)))
    enthalpy = float(qualia.get("enthalpy", 0.0))
    qualia_mag = float(np.clip(np.tanh(magnitude), 0.0, 1.0))
    qualia_enthalpy = float(np.clip(np.tanh(enthalpy / 5.0), 0.0, 1.0))

    categories = emotion_axes + ["qualia_magnitude", "qualia_enthalpy"]
    values = emotion_values + [qualia_mag, qualia_enthalpy]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="EQNet (7D + Qualia)",
            line=dict(color="#ff7f0e"),
            opacity=0.75,
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
        showlegend=False,
    )
    return fig


def _build_basic_emotion_chart(valence: float, arousal: float) -> go.Figure:
    """Simplified radar chart for everyday emotions."""
    v = float(np.clip(valence, -1.0, 1.0))
    a = float(np.clip(arousal, -1.0, 1.0))

    joy = max(0.0, (v + a) / 2.0)
    calm = max(0.0, v * (1.0 - max(a, 0.0)))
    anger = max(0.0, (-v + a) / 2.0)
    sadness = max(0.0, (-v - a) / 2.0)
    surprise = max(0.0, a)

    basic_axes = ["Joy", "Calm", "Sadness", "Anger", "Surprise"]
    basic_values = [joy, calm, sadness, anger, surprise]
    basic_values.append(basic_values[0])
    basic_axes.append(basic_axes[0])

    fig = go.Figure(
        data=go.Scatterpolar(
            r=basic_values,
            theta=basic_axes,
            fill="toself",
            name="Basic emotions",
            line=dict(color="#1f77b4"),
            opacity=0.75,
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
        showlegend=False,
    )
    return fig


def _build_heatmap(snapshot: Optional[Dict[str, np.ndarray]]) -> go.Figure:
    """Render the field energy as a heatmap."""
    if not snapshot:
        return go.Figure()
    energy = snapshot.get("energy")
    if energy is None:
        return go.Figure()
    fig = go.Figure(data=go.Heatmap(z=energy, colorscale="Turbo"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        title="Energy Field Snapshot",
    )
    return fig


def _build_summary(metrics: Dict[str, Any], qualia: Dict[str, Any]) -> str:
    """Generate a short textual summary of metrics/qualia."""
    lines = []
    entropy = metrics.get("entropy")
    dissipation = metrics.get("dissipation")
    info_flux = metrics.get("info_flux")

    if entropy is not None:
        lines.append(f"- Entropy: {entropy:.3f} (spread of the field)")
    if dissipation is not None:
        lines.append(f"- Dissipation: {dissipation:.3f} (energy loss)")
    if info_flux is not None:
        lines.append(f"- Info Flux: {info_flux:.5f} (flow strength)")

    magnitude = qualia.get("magnitude")
    phase = qualia.get("phase")
    enthalpy = qualia.get("enthalpy")
    if magnitude is not None:
        lines.append(f"- Qualia magnitude: {magnitude:.3f} (intensity)")
    if phase is not None:
        lines.append(f"- Qualia phase: {phase:.3f} (direction)")
    if enthalpy is not None:
        lines.append(f"- Qualia enthalpy: {enthalpy:.3f} (memory coupling)")

    if not lines:
        lines.append("Metrics not available yet.")
    return "\n".join(lines)


EXECUTOR = ThreadPoolExecutor(max_workers=2)
ACK_TABLE: Dict[str, Any] = {}
ACK_CURSOR: defaultdict[Tuple[str, str, str], int] = defaultdict(int)
ACK_LOCK = threading.Lock()


def _load_ack_table() -> Dict[str, Any]:
    global ACK_TABLE
    if ACK_TABLE:
        return ACK_TABLE
    ack_path = ROOT.parent / "config" / "i18n" / "ack.yaml"
    if not ack_path.exists():
        ACK_TABLE = {}
        return ACK_TABLE
    try:
        with ack_path.open("r", encoding="utf-8") as handle:
            ACK_TABLE = yaml.safe_load(handle) or {}
    except Exception:
        ACK_TABLE = {}
    return ACK_TABLE


def _pick_ack(tone: str, culture: str, tag: str) -> str:
    table = _load_ack_table()
    tone_key = (tone or "casual").lower()
    culture_key = (culture or "default").lower()
    tag_key = tag if tag in {"happy_excited", "calm_positive", "angry_hot", "angry_quiet", "surprise", "curious", "neutral"} else "neutral"

    def resolve_keys() -> Tuple[str, str, str]:
        if (
            culture_key in table
            and isinstance(table[culture_key], dict)
            and tone_key in table[culture_key]
            and isinstance(table[culture_key][tone_key], dict)
            and tag_key in table[culture_key][tone_key]
        ):
            return culture_key, tone_key, tag_key
        if "default" in table:
            tone_map = table["default"].get("default", {})
            if isinstance(tone_map, dict) and tag_key in tone_map:
                return "default", "default", tag_key
        return "default", "default", "neutral"

    resolved = resolve_keys()
    phrases = table.get(resolved[0], {}).get(resolved[1], {}).get(resolved[2], [])
    if not phrases:
        return "はい、お話を受け取りました。"
    with ACK_LOCK:
        cursor = ACK_CURSOR[resolved]
        phrase = phrases[cursor % len(phrases)]
        ACK_CURSOR[resolved] = (cursor + 1) % len(phrases)
    return phrase


def _build_outputs(
    rt: EmotionalHubRuntime,
    result: Dict[str, Any],
    response_override: Optional[str],
    perceived_affect: Optional[Dict[str, float]],
) -> Tuple[
    str,
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    go.Figure,
    go.Figure,
    go.Figure,
    str,
]:
    affect = result["affect"]
    metrics = result["metrics"]
    controls = result["controls"]
    response_text = response_override
    if response_text is None:
        response_text = result["response"].text if result.get("response") else ""

    perceived = perceived_affect or getattr(rt, "perceived_affect", {}) or {}
    affect_payload = {
        "valence": round(perceived.get("valence", affect.valence), 4),
        "arousal": round(perceived.get("arousal", affect.arousal), 4),
        "confidence": round(perceived.get("confidence", affect.confidence), 4),
    }
    metrics_payload = {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in metrics.items()
    }
    controls_payload = {
        key: (round(value, 6) if isinstance(value, (int, float)) else str(value))
        for key, value in vars(controls).items()
    }

    radar_fig = _build_radar_chart(rt.last_E, rt.last_qualia)
    basic_fig = _build_basic_emotion_chart(affect.valence, affect.arousal)
    heatmap_fig = _build_heatmap(rt.last_snapshot)
    summary_text = _build_summary(metrics_payload, rt.last_qualia)

    return (
        response_text,
        affect_payload,
        metrics_payload,
        controls_payload,
        radar_fig,
        basic_fig,
        heatmap_fig,
        summary_text,
    )


runtime: Optional[EmotionalHubRuntime] = None


def _get_runtime() -> EmotionalHubRuntime:
    global runtime
    if runtime is None:
        external_cfg = None
        try:
            external_cfg = load_runtime_cfg()
        except Exception:
            external_cfg = None
        runtime = EmotionalHubRuntime(
            RuntimeConfig(
                use_eqnet_core=True,
                eqnet_state_dir="data/state_hub_gradio",
            )
        )
        if external_cfg is not None:
            runtime._runtime_cfg = external_cfg  # force override to reflect latest file
            latency_cfg = getattr(external_cfg, "latency", None)
            if latency_cfg is not None:
                setattr(runtime, "_loose_enabled", bool(getattr(latency_cfg, "enable_loose", True)))
        else:
            setattr(runtime, "_loose_enabled", True)
    return runtime


def _observe_frame(image: Optional[Image.Image]) -> None:
    if image is None:
        return
    frame = np.array(image.convert("RGB"), dtype=np.uint8)
    _get_runtime().observe_video(frame)


def _observe_audio(audio: Optional[Tuple[int, np.ndarray]]) -> None:
    if audio is None:
        return
    _, data = audio
    if data is None or data.size == 0:
        return
    if data.ndim > 1:
        data = data.mean(axis=1)
    chunk = data.astype(np.float32)
    _get_runtime().observe_audio(chunk)


def run_pipeline(
    user_text: str,
    image: Optional[Image.Image],
    audio: Optional[Tuple[int, np.ndarray]],
):
    rt = _get_runtime()
    tone_key = "casual"
    culture_key = "ja-JP"
    perceived_affect: Optional[Dict[str, float]] = None
    affect_tag = "neutral"
    precomputed: Optional[Tuple[Dict[str, float], str]] = None
    text_clean = (user_text or "").strip()
    runtime_cfg = getattr(rt, "_runtime_cfg", None)
    latency_cfg = getattr(runtime_cfg, "latency", None)
    loose_enabled = getattr(rt, "_loose_enabled", None)
    if loose_enabled is None:
        loose_enabled = bool(getattr(latency_cfg, "enable_loose", True))
    if text_clean:
        base_lang = culture_key.split("-")[0]
        metrics, affect_tag = quick_text_affect_v2(text_clean, lang=base_lang)
        perceived_affect = dict(metrics)
        perceived_affect["tag"] = affect_tag
        precomputed = (metrics, affect_tag)
        try:
            rt.observe_text(user_text, lang=culture_key, precomputed=precomputed)
        except Exception:
            pass

    _observe_frame(image)
    _observe_audio(audio)

    step_text = user_text if text_clean else None
    fast_result: Dict[str, Any]
    if text_clean and perceived_affect is not None:
        pending_sample = getattr(rt, "_pending_text_affect", None)
        if pending_sample is None:
            affect_metrics = perceived_affect
            pending_sample = AffectSample(
                valence=float(affect_metrics.get("valence", 0.0)),
                arousal=float(affect_metrics.get("arousal", 0.0)),
                confidence=float(affect_metrics.get("confidence", 0.0)),
                timestamp=time.time(),
            )
            rt._pending_text_affect = pending_sample
        fast_metrics = getattr(rt, "_last_metrics", {}) or {}
        fast_controls = rt.policy.affect_to_controls(pending_sample, fast_metrics)
        fast_result = {
            "affect": pending_sample,
            "metrics": fast_metrics,
            "controls": fast_controls,
            "response": None,
            "memory_reference": None,
        }
    else:
        fast_result = rt.step(user_text=step_text, fast_only=True)
    ack_text = _pick_ack(tone_key, culture_key, affect_tag)
    yield _build_outputs(rt, fast_result, ack_text, perceived_affect)

    if not loose_enabled or not text_clean:
        return

    def _run_loose() -> Dict[str, Any]:
        try:
            if text_clean and precomputed is not None:
                rt.observe_text(user_text, lang=culture_key, precomputed=precomputed)
            return rt.step(user_text=step_text, fast_only=False)
        except Exception as exc:
            return {"__error__": str(exc)}

    future = EXECUTOR.submit(_run_loose)
    loose_result = future.result()
    if "__error__" in loose_result:
        error_text = f"{ack_text}\n\n(backend error: {loose_result['__error__']})"
        yield _build_outputs(rt, fast_result, error_text, perceived_affect)
        return

    response_obj = loose_result.get("response")
    memory_ref = loose_result.get("memory_reference")
    response_override = None
    if response_obj is None and memory_ref and memory_ref.get("reply"):
        response_override = memory_ref["reply"]

    if response_obj is None and response_override is None:
        return

    final_perceived = getattr(rt, "perceived_affect", {}) or perceived_affect
    yield _build_outputs(rt, loose_result, response_override, final_perceived)


with gr.Blocks() as demo:
    gr.Markdown(
        "# EQNet Emotional Hub Demo\n"
        "Upload text, image, and/or audio to see EQNet's affect estimate, response, and field visualisation."
    )

    with gr.Row():
        text_in = gr.Textbox(label="User Text", placeholder="How are you feeling today?", lines=2)
        image_in = gr.Image(label="Image Frame (optional)", type="pil")

    audio_in = gr.Audio(label="Audio (optional)", type="numpy")
    submit_btn = gr.Button("Run")

    response_out = gr.Textbox(label="Response", lines=3)
    affect_out = gr.JSON(label="Affect (valence / arousal / confidence)")
    metrics_out = gr.JSON(label="EQNet Metrics")
    controls_out = gr.JSON(label="Behaviour Controls")
    advanced_radar_out = gr.Plot(label="EQNet 7D + Qualia Radar")
    basic_radar_out = gr.Plot(label="Basic Emotion Radar")
    heatmap_out = gr.Plot(label="Field Energy Heatmap")
    summary_out = gr.Markdown(label="Summary")

    submit_btn.click(
        fn=run_pipeline,
        inputs=[text_in, image_in, audio_in],
        outputs=[
            response_out,
            affect_out,
            metrics_out,
            controls_out,
            advanced_radar_out,
            basic_radar_out,
            heatmap_out,
            summary_out,
        ],
    )

    gr.Markdown(
        "For continuous demos, consider enabling Gradio's `live` mode or streaming camera/microphone input."
    )


if __name__ == "__main__":
    _get_runtime()
    demo.launch()
