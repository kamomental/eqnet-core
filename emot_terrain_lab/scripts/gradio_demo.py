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
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from hub import EmotionalHubRuntime, RuntimeConfig
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


runtime = EmotionalHubRuntime(
    RuntimeConfig(
        use_eqnet_core=True,
        eqnet_state_dir="data/state_hub_gradio",
    )
)


def _observe_frame(image: Optional[Image.Image]) -> None:
    if image is None:
        return
    frame = np.array(image.convert("RGB"), dtype=np.uint8)
    runtime.observe_video(frame)


def _observe_audio(audio: Optional[Tuple[int, np.ndarray]]) -> None:
    if audio is None:
        return
    _, data = audio
    if data is None or data.size == 0:
        return
    if data.ndim > 1:
        data = data.mean(axis=1)
    chunk = data.astype(np.float32)
    runtime.observe_audio(chunk)


def run_pipeline(
    user_text: str,
    image: Optional[Image.Image],
    audio: Optional[Tuple[int, np.ndarray]],
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
    _observe_frame(image)
    _observe_audio(audio)

    result = runtime.step(user_text=user_text or "")
    affect = result["affect"]
    metrics = result["metrics"]
    controls = result["controls"]
    response_text = result["response"].text if result.get("response") else ""

    affect_payload = {
        "valence": round(affect.valence, 4),
        "arousal": round(affect.arousal, 4),
        "confidence": round(affect.confidence, 4),
    }
    metrics_payload = {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in metrics.items()
    }
    controls_payload = {
        key: (round(value, 6) if isinstance(value, (int, float)) else str(value))
        for key, value in vars(controls).items()
    }

    radar_fig = _build_radar_chart(runtime.last_E, runtime.last_qualia)
    basic_fig = _build_basic_emotion_chart(affect.valence, affect.arousal)
    heatmap_fig = _build_heatmap(runtime.last_snapshot)
    summary_text = _build_summary(metrics_payload, runtime.last_qualia)

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
    demo.launch()
