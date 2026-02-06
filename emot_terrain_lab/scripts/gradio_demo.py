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
from typing import Any, Dict, Optional, Tuple, List
from collections import defaultdict
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
from emot_terrain_lab.hub.future_memory import FutureMemoryController
from emot_terrain_lab.ui.avatar import (
    AvatarAnimator,
    AvatarInputs,
    AvatarRenderer,
    load_avatar_config,
)

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
from eqnet.policy import ack_intent_for_tag, enforce_ack
from runtime.config import load_runtime_cfg
from hub import EmotionalHubRuntime, RuntimeConfig
from hub.perception import AffectSample
from terrain.emotion import AXES, AXIS_BOUNDS


HUB_URL = os.environ.get("EQNET_HUB_URL", "").strip()


class TalkMode(Enum):
    WATCH = auto()
    SOOTHE = auto()
    SOOTHE_DEEP = auto()
    ASK = auto()
    TALK = auto()


@dataclass
class GateContext:
    engaged: bool
    face: str
    dm: float
    jerk: float
    voice_energy: float
    since_last_user_ms: int
    S: float
    H: float
    rho: float
    ignition: float
    memory_anchor: str
    text_input: bool
    text_value: str = ""
    freeze_score: float = 0.0
    silence_ms: int = 0
    freeze_trend: float = 0.0


DM_ENTER = 0.35
DM_EXIT = 0.28
JK_ENTER = 0.20
JK_EXIT = 0.16

ASK_VOICE_TH = 0.20
ASK_MIN_SINCE = 800
TALK_MIN_SINCE = 400
TALK_COOLDOWN_MS = 1800
TALK_DWELL_MS = 3500
TALK_REPEAT_RESET_MS = 15000
TALK_SIGNAL_VOICE = 0.28
TALK_SIGNAL_DM = 0.22
TALK_SIGNAL_JERK = 0.18

FREEZE_ENTER = 0.58
FREEZE_EXIT = 0.42
FREEZE_TREND_ON = 0.10
SILENCE_ENTER_MS = 3500
SILENCE_EXIT_MS = 1800
SILENCE_VOICE_EPS = 0.12

IGNITION_LISTEN_HIGH = 0.55

_LAST_TALK_MS = 0
_LAST_MODE = TalkMode.WATCH
_LAST_TALK_TRIGGER: str = ""
_LAST_TALK_TRIGGER_MS: int = 0


ACK_TEMPLATES = {
    "listen": "うん、聞いているよ。必要なとき合囲してね。",
    "soothe": "表情が少し硬いかも。深呼吸して、肩の力を抜こう。私はここにいるよ。",
    "soothe_deep": "急がなくてもいいよ。ここは安全だから、目を閉じても大丈夫。声が出ないときは、ただ呼吸だけ意識しよう。",
    "ask": "さっきより元気が少なめに見える。何かあった？話せる範囲で教えてね。",
    "talk": "続きを整えよう。どこから話そうか。",
}

SOOTHE_VARIATIONS = [
    "呼吸を整えよう。肩を軽く回してみるのもいいかも。",
    "少し向き合おう。いまはゆっくりで大丈夫。",
    "緊張が続いているかもしれない。深呼吸を一緒にしよう。",
]

ASK_VARIATIONS = [
    "何か胸に引っかかることがあった？話せる範囲で大丈夫だよ。",
    "さっきより少し静かになった気がする。安心できている？",
    "もし変化があったら教えてね。ここで整えていこう。",
]

WATCH_VARIATIONS = [
    "うん、そばにいるよ。合囲があればいつでも続けよう。",
    "ここで見守っているね。必要なときに声をかけて。",
    "大丈夫、落ち着いたら教えて。いつでも耳を傾けるよ。",
]

_DEMO_FM_CFG = {
    "alpha": 0.6,
    "beta": 0.8,
    "gamma": 0.1,
    "future_ttl_sec": 90,
    "future_energy_threshold": 1.2,
}
_DEMO_FUTURE = FutureMemoryController(_DEMO_FM_CFG, clock=time.monotonic)


def _demo_embed_action(text: str) -> np.ndarray:
    text = text or ""
    features = [
        float(len(text)),
        float(text.count("。")),
        float(text.count("！")),
        float(text.count("？")),
        float(sum(ch.isdigit() for ch in text)),
    ]
    return np.asarray(features, dtype=float)


def _demo_novelty(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(u, v) / denom)


def select_action(
    candidates: List[str],
    m_past: Optional[np.ndarray],
    m_future: Optional[np.ndarray],
    gamma: float,
):
    ranked: List[Tuple[float, str, np.ndarray]] = []
    for text in candidates:
        vec = _demo_embed_action(text)
        score = 0.0
        if m_past is not None and m_past.size:
            score += _cosine(vec, m_past)
        if m_future is not None and m_future.size:
            score += _cosine(vec, m_future)
        score += float(gamma) * _demo_novelty(vec)
        ranked.append((score, text, vec))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return "", [], None
    best_score, best_text, best_vec = ranked[0]
    return best_text, ranked, best_vec


_LAST_DEMO_SNAPSHOT: Optional[Dict[str, Any]] = None

if HAVE_MEDIAPIPE:
    _MP_FACE = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
else:
    _MP_FACE = None

_TTS_ENGINE = None
_AVATAR_ANIMATOR: Optional[AvatarAnimator] = None
_AVATAR_RENDERER: Optional[AvatarRenderer] = None
_AVATAR_LAST_TICK: Optional[float] = None
_AVATAR_ERROR: Optional[str] = None


def _get_avatar_components() -> Optional[Tuple[AvatarAnimator, AvatarRenderer]]:
    global _AVATAR_ANIMATOR, _AVATAR_RENDERER, _AVATAR_LAST_TICK, _AVATAR_ERROR
    if _AVATAR_ERROR:
        return None
    if _AVATAR_ANIMATOR is None or _AVATAR_RENDERER is None:
        try:
            cfg = load_avatar_config()
            _AVATAR_ANIMATOR = AvatarAnimator(cfg)
            _AVATAR_RENDERER = AvatarRenderer(cfg)
            _AVATAR_LAST_TICK = time.monotonic()
        except Exception as exc:
            _AVATAR_ERROR = str(exc)
            return None
    return _AVATAR_ANIMATOR, _AVATAR_RENDERER


def _render_avatar(voice_energy: float, is_speaking: bool) -> Any:
    components = _get_avatar_components()
    if components is None:
        return gr.update()
    animator, renderer = components
    now = time.monotonic()
    global _AVATAR_LAST_TICK
    if _AVATAR_LAST_TICK is None:
        dt = 0.0
    else:
        dt = max(0.0, now - _AVATAR_LAST_TICK)
    _AVATAR_LAST_TICK = now

    state = animator.step(dt, AvatarInputs(voice_energy=voice_energy, is_speaking=is_speaking))
    return renderer.render(state, animator.time_sec)


class HeartbeatSynth:
    def __init__(self, sr: int = 100, window_sec: float = 4.0) -> None:
        self.sr = sr
        self.n = int(sr * window_sec)
        self.phase = 0.0
        self.env_phase = 0.0
        self.buf = np.zeros(self.n)

    def step(self, bpm: float, amp: float, hrv: float = 0.008) -> np.ndarray:
        f = max(0.15, bpm / 60.0)
        t = np.arange(self.n) / self.sr
        f = f * (1.0 + np.random.randn() * hrv)
        base = np.sin(2 * np.pi * (self.phase + f * t))
        overt = np.sin(2 * np.pi * 2.1 * (self.phase + f * t + 0.07))
        wave = 0.75 * base + 0.25 * np.tanh(2.5 * overt)
        breath = 0.7 + 0.3 * np.sin(2 * np.pi * (self.env_phase + 0.25 * t))
        y = amp * breath * wave
        self.buf = np.roll(self.buf, -len(t))
        self.buf[-len(t) :] = y
        self.phase = (self.phase + f * (len(t) / self.sr)) % 1.0
        self.env_phase = (self.env_phase + 0.25 * (len(t) / self.sr)) % 1.0
        return self.buf.copy()


_HB_SYNTH = HeartbeatSynth()

_UI_LAST_MODE: Optional[TalkMode] = None
_UI_LAST_MESSAGE: str = ""
_UI_LAST_UPDATE_MS: int = 0
_UI_MIN_INTERVAL_MS = 1500
_UI_LAST_TALK_MS: int = 0
_UI_TALK_HOLD_MS = 4000

_SESSION_STARTED = False
_SESSION_SILENCE_MS = 0
_SESSION_FREEZE_SMOOTH = 0.0
_SESSION_LAST_SAMPLE_MS: Optional[int] = None
_SESSION_LAST_FREEZE_RAW = 0.0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _update_session_state(
    voice_energy: float, instant_freeze: Optional[float] = None
) -> Tuple[int, float, float]:
    global _SESSION_SILENCE_MS, _SESSION_FREEZE_SMOOTH, _SESSION_LAST_SAMPLE_MS, _SESSION_LAST_FREEZE_RAW
    now = _now_ms()
    if _SESSION_LAST_SAMPLE_MS is None:
        delta = 0
    else:
        delta = max(0, now - _SESSION_LAST_SAMPLE_MS)
    _SESSION_LAST_SAMPLE_MS = now

    if voice_energy < SILENCE_VOICE_EPS:
        _SESSION_SILENCE_MS += delta
    else:
        _SESSION_SILENCE_MS = max(0, _SESSION_SILENCE_MS - int(delta * 0.6))

    if instant_freeze is None:
        instant_freeze = _SESSION_LAST_FREEZE_RAW
    alpha = 0.15
    prev = _SESSION_FREEZE_SMOOTH
    _SESSION_FREEZE_SMOOTH = (1 - alpha) * _SESSION_FREEZE_SMOOTH + alpha * float(
        instant_freeze
    )
    _SESSION_LAST_FREEZE_RAW = float(instant_freeze)
    freeze_trend = _SESSION_FREEZE_SMOOTH - prev
    return _SESSION_SILENCE_MS, _SESSION_FREEZE_SMOOTH, freeze_trend


def _decide_talk_mode_local(ctx: GateContext, guard_action: str = "ok") -> TalkMode:
    global _LAST_MODE, _LAST_TALK_MS, _LAST_TALK_TRIGGER, _LAST_TALK_TRIGGER_MS

    now = _now_ms()
    if _LAST_MODE == TalkMode.TALK and (now - _LAST_TALK_MS) < TALK_DWELL_MS:
        return TalkMode.TALK

    if guard_action in {"fallback_attention", "tighten_band"}:
        _LAST_MODE = (
            TalkMode.SOOTHE
            if (ctx.dm > DM_ENTER or ctx.jerk > JK_ENTER)
            else TalkMode.WATCH
        )
        return _LAST_MODE

    if not ctx.engaged or ctx.since_last_user_ms < TALK_MIN_SINCE:
        _LAST_MODE = TalkMode.WATCH
        return _LAST_MODE

    if _LAST_MODE == TalkMode.SOOTHE:
        if ctx.dm >= DM_EXIT or ctx.jerk >= JK_EXIT:
            _LAST_MODE = TalkMode.SOOTHE
            return _LAST_MODE
    else:
        if ctx.dm > DM_ENTER or ctx.jerk > JK_ENTER or ctx.face in {"sad", "fear"}:
            _LAST_MODE = TalkMode.SOOTHE
            return _LAST_MODE

    if _LAST_MODE == TalkMode.SOOTHE_DEEP:
        if ctx.freeze_score <= FREEZE_EXIT and ctx.silence_ms <= SILENCE_EXIT_MS:
            pass
        else:
            _LAST_MODE = TalkMode.SOOTHE_DEEP
            return _LAST_MODE
    else:
        if (
            ctx.freeze_score >= FREEZE_ENTER
            and ctx.silence_ms >= SILENCE_ENTER_MS
            and ctx.ignition >= IGNITION_LISTEN_HIGH
        ) or (
            ctx.freeze_trend >= FREEZE_TREND_ON and ctx.silence_ms >= SILENCE_ENTER_MS
        ):
            _LAST_MODE = TalkMode.SOOTHE_DEEP
            return _LAST_MODE

    if (
        ctx.face in {"neutral"}
        and ctx.voice_energy < ASK_VOICE_TH
        and ctx.since_last_user_ms > ASK_MIN_SINCE
    ):
        _LAST_MODE = TalkMode.ASK
        return _LAST_MODE

    trigger_key = None
    if ctx.text_input:
        normalized = ctx.text_value.strip()
        if normalized:
            trigger_key = f"text:{hash((normalized, ctx.memory_anchor))}"
    elif (
        ctx.voice_energy >= TALK_SIGNAL_VOICE
        or ctx.dm >= TALK_SIGNAL_DM
        or ctx.jerk >= TALK_SIGNAL_JERK
    ):
        trigger_key = f"sensor:{int(ctx.voice_energy * 100)}-{int(ctx.dm * 100)}-{int(ctx.jerk * 100)}"

    if trigger_key and ctx.freeze_score < FREEZE_EXIT:
        if (
            ctx.since_last_user_ms > TALK_MIN_SINCE
            and (now - _LAST_TALK_MS) > TALK_COOLDOWN_MS
        ):
            if (
                trigger_key != _LAST_TALK_TRIGGER
                or (now - _LAST_TALK_TRIGGER_MS) > TALK_REPEAT_RESET_MS
            ):
                _LAST_TALK_TRIGGER = trigger_key
                _LAST_TALK_TRIGGER_MS = now
                _LAST_MODE = TalkMode.TALK
                _LAST_TALK_MS = now
                return _LAST_MODE

    _LAST_MODE = TalkMode.WATCH
    return _LAST_MODE


def _face_metrics(frame: np.ndarray) -> Tuple[bool, float, float]:
    if _MP_FACE is None:
        return False, 0.0, 0.0
    result = _MP_FACE.process(frame[:, :, ::-1])
    if not result.multi_face_landmarks:
        return False, 0.0, 0.0
    lm = result.multi_face_landmarks[0].landmark
    mouth_open = abs(lm[13].y - lm[14].y)
    blink = abs(lm[159].y - lm[145].y)
    return True, float(mouth_open), float(blink)


def _tts_speak(text: str) -> None:
    global _TTS_ENGINE
    if not HAVE_TTS or not text:
        return
    try:
        if _TTS_ENGINE is None:
            _TTS_ENGINE = pyttsx3.init()
            _TTS_ENGINE.setProperty("rate", 180)
        threading.Thread(
            target=lambda: (_TTS_ENGINE.say(text), _TTS_ENGINE.runAndWait()),
            daemon=True,
        ).start()
    except Exception:  # pragma: no cover
        pass


def _render_heartbeat(y: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 1.8))
    x = np.linspace(-len(y) / _HB_SYNTH.sr, 0, len(y))
    ax.plot(x, y, color="#ff7f0e")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title("AI Internal Heartbeat")
    fig.tight_layout()
    return fig


def _compute_bpm_amp(ctx: GateContext, guard_action: str) -> Tuple[float, float]:
    bpm = 60.0 + 40.0 * ctx.ignition + 30.0 * ctx.dm + 15.0 * ctx.jerk
    if guard_action in {"fallback_attention", "tighten_band"}:
        bpm -= 10.0
    bpm = float(np.clip(bpm, 50.0, 130.0))
    amp = 0.6 * ctx.rho + 0.4 * ctx.H
    if guard_action in {"fallback_attention", "tighten_band"}:
        amp *= 0.7
    return bpm, float(np.clip(amp, 0.05, 1.0))


def _summarize_memory_tags(tags: str) -> str:
    items = [s.strip() for s in tags.split(",") if s.strip()]
    return ", ".join(items[:5]) if items else "-"


def _format_ack(mode: TalkMode, ctx: GateContext) -> str:
    if mode == TalkMode.SOOTHE:
        line = random.choice(SOOTHE_VARIATIONS)
        return f"{line} (delta_m={ctx.dm:.2f}, jerk={ctx.jerk:.2f})"
    if mode == TalkMode.SOOTHE_DEEP:
        line = ACK_TEMPLATES["soothe_deep"]
        return f"{line} (freeze={ctx.freeze_score:.2f}, silence={ctx.silence_ms}ms)"
    if mode == TalkMode.ASK:
        line = random.choice(ASK_VARIATIONS)
        return (
            f"{line} (voice={ctx.voice_energy:.2f}, since={ctx.since_last_user_ms}ms)"
        )
    line = random.choice(WATCH_VARIATIONS)
    return f"{line} (Ignition={ctx.ignition:.2f})"


def _llm_free_talk(ctx: GateContext) -> str:
    mood = (
        "静穏"
        if (ctx.S < 0.4 and ctx.H < 0.4)
        else "ざわつき" if (ctx.S > 0.6 or ctx.H > 0.6) else "ゆらぎ"
    )
    lines: List[str] = [
        f"今の空気は「{mood}」。S={ctx.S:.2f}, H={ctx.H:.2f}, rho={ctx.rho:.2f}, Ignition={ctx.ignition:.2f}",
    ]
    if ctx.face in {"sad", "fear"}:
        lines.append("すこし不安が混じっているかも。一緒に深呼吸して整えよう")
    elif ctx.face in {"joy"}:
        lines.append("いい表情。いまの流れ、続けられそう")
    if ctx.voice_energy < 0.2:
        lines.append("声はいま控えめ。こちらから少しだけ話すね")
    if ctx.freeze_score > FREEZE_ENTER or ctx.silence_ms > SILENCE_ENTER_MS:
        lines.append(
            "体が固まっている感じがあるね。呼吸を整えながら、分に一度だけそっと心の状態を見よう"
        )
    if ctx.dm > 0.35 or ctx.jerk > 0.2:
        lines.append("感情の波が少し大きい。大丈夫、私が隣で整えるよ")
    if ctx.memory_anchor and ctx.memory_anchor != "-":
        lines.append(
            f"最近鳴いている記憶は「{ctx.memory_anchor}」。そこに寄り添って考えてみよう"
        )
    lines.append("続ける準備ができたら、合図して")
    return " ".join(lines)


def _hub_step(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not HUB_URL:
        return None
    try:
        response = requests.post(
            f"{HUB_URL.rstrip('/')}/api/step", json=payload, timeout=10
        )
        if response.ok:
            return response.json()
    except Exception:  # pragma: no cover
        return None
    return None


@dataclass
class DemoState:
    prev_valence: float = 0.0
    prev_delta: float = 0.0
    voice_energy: float = 0.3

    def update_affect(self, valence: float) -> Tuple[float, float]:
        delta = abs(valence - self.prev_valence)
        jerk = abs(delta - self.prev_delta)
        self.prev_valence = valence
        self.prev_delta = delta
        return delta, jerk


_TALK_STATE = DemoState()


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _build_radar_chart(emotion_vec: np.ndarray, qualia: Dict[str, Any]) -> go.Figure:
    """Create a 9-axis radar chart (7D emotion + 2D qualia)."""
    if emotion_vec is None or emotion_vec.size < len(AXES):
        return go.Figure()

    emotion_axes = [
        "sensory",
        "temporal",
        "spatial",
        "affective",
        "cognitive",
        "social",
        "meta",
    ]
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
    tag_key = (
        tag
        if tag
        in {
            "happy_excited",
            "calm_positive",
            "angry_hot",
            "angry_quiet",
            "surprise",
            "curious",
            "neutral",
        }
        else "neutral"
    )

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
    prospective_payload: Dict[str, Any]
    prospective_payload = result.get("prospective") or {}
    if not prospective_payload:
        last_prospective = getattr(rt, "_last_prospective", None)
        if last_prospective:
            serialize = getattr(rt, "_serialize_prospective", None)
            if callable(serialize):
                prospective_payload = serialize(last_prospective) or {}

    radar_fig = _build_radar_chart(rt.last_E, rt.last_qualia)
    basic_fig = _build_basic_emotion_chart(affect.valence, affect.arousal)
    heatmap_fig = _build_heatmap(rt.last_snapshot)
    summary_text = _build_summary(metrics_payload, rt.last_qualia)
    return (
        response_text,
        affect_payload,
        metrics_payload,
        controls_payload,
        prospective_payload,
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
                setattr(
                    runtime,
                    "_loose_enabled",
                    bool(getattr(latency_cfg, "enable_loose", True)),
                )
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
    ack_intent = ack_intent_for_tag(affect_tag)
    ack_text = enforce_ack(ack_intent, ack_text, will_write_memory=False)
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


def talkmode_step_ui(
    face: str,
    voice_energy: float,
    dm: float,
    jerk: float,
    engaged: bool,
    since_ms: int,
    S: float,
    H: float,
    rho: float,
    ignition: float,
    memory_tags: str,
    user_text: str,
    force_refresh: bool = False,
    activate_session: bool = False,
) -> Tuple[Any, str, Any, str, Any]:
    global _SESSION_STARTED, _SESSION_SILENCE_MS, _SESSION_LAST_SAMPLE_MS, _SESSION_FREEZE_SMOOTH, _SESSION_LAST_FREEZE_RAW

    memory_anchor = _summarize_memory_tags(memory_tags)
    instant_freeze = (
        _SESSION_LAST_FREEZE_RAW
        if _SESSION_LAST_FREEZE_RAW > 0.0
        else float(
            np.clip(1.0 - min(1.0, 0.5 * float(dm) + 0.5 * float(jerk)), 0.0, 1.0)
        )
    )
    silence_ms, freeze_score, freeze_trend = _update_session_state(
        float(voice_energy), instant_freeze
    )

    payload = {
        "face": face,
        "voice_energy": voice_energy,
        "dm": dm,
        "jerk": jerk,
        "since_last_user_ms": since_ms,
        "engaged": engaged,
        "text_input": bool(user_text and user_text.strip()),
        "S": S,
        "H": H,
        "rho": rho,
        "Ignition": ignition,
        "memory_anchor": memory_anchor,
        "user_text": user_text or "",
        "freeze_score": freeze_score,
        "silence_ms": silence_ms,
        "freeze_trend": freeze_trend,
    }

    if activate_session:
        _SESSION_STARTED = True
        _SESSION_SILENCE_MS = 0
        _SESSION_LAST_SAMPLE_MS = None
        _SESSION_FREEZE_SMOOTH = 0.0
        _SESSION_LAST_FREEZE_RAW = 0.0
    if not _SESSION_STARTED:
        return gr.update(), gr.update(), gr.update(), "WATCH", gr.update()

    hub_response = _hub_step(payload)
    guard_action = "ok"
    ctx = GateContext(
        engaged=bool(engaged),
        face=face,
        dm=float(dm),
        jerk=float(jerk),
        voice_energy=float(voice_energy),
        since_last_user_ms=int(since_ms),
        S=float(S),
        H=float(H),
        rho=float(rho),
        ignition=float(ignition),
        memory_anchor=memory_anchor,
        text_input=bool(user_text and user_text.strip()),
        text_value=user_text or "",
        freeze_score=freeze_score,
        silence_ms=silence_ms,
        freeze_trend=freeze_trend,
    )
    ctx_json: Dict[str, Any] = dict(ctx.__dict__)

    now = _now_ms()
    if hub_response:
        gate = hub_response.get("gate_context") or {}
        ctx_json.update(gate)
        ctx_json.setdefault("freeze_score", freeze_score)
        ctx_json.setdefault("silence_ms", silence_ms)
        ctx_json.setdefault("freeze_trend", freeze_trend)
        guard_action = gate.get("guard_action", guard_action)
        mode_name = (hub_response.get("talk_mode") or "WATCH").upper()
        mode = (
            TalkMode[mode_name] if mode_name in TalkMode.__members__ else TalkMode.WATCH
        )
        resp_text = (
            hub_response.get("llm_text")
            or (hub_response.get("response") or {}).get("text")
            or hub_response.get("ack_text")
            or ""
        )
    else:
        mode = _decide_talk_mode_local(ctx, guard_action=guard_action)
        resp_text = ""
        ctx_json.setdefault("freeze_score", freeze_score)
        ctx_json.setdefault("silence_ms", silence_ms)
        ctx_json.setdefault("freeze_trend", freeze_trend)

    global _UI_LAST_TALK_MS, _UI_LAST_MESSAGE
    message = ""
    if mode == TalkMode.TALK:
        if resp_text:
            message = resp_text
        elif _UI_LAST_MESSAGE and (now - _UI_LAST_TALK_MS) < _UI_TALK_HOLD_MS:
            message = _UI_LAST_MESSAGE
        else:
            message = _llm_free_talk(ctx)
        _UI_LAST_TALK_MS = now
        _tts_speak(message)
    else:
        if _UI_LAST_TALK_MS and (now - _UI_LAST_TALK_MS) < _UI_TALK_HOLD_MS:
            mode = TalkMode.TALK
            message = _UI_LAST_MESSAGE or _llm_free_talk(ctx)
        else:
            message = _format_ack(mode, ctx)
            if resp_text:
                message = (resp_text + " " + message).strip()
            if mode in (TalkMode.SOOTHE, TalkMode.SOOTHE_DEEP, TalkMode.ASK):
                _tts_speak(message)

    bpm, amp = _compute_bpm_amp(ctx, guard_action)
    heartbeat = _HB_SYNTH.step(bpm=bpm, amp=amp, hrv=0.008)
    fig = _render_heartbeat(heartbeat)

    global _UI_LAST_MODE, _UI_LAST_UPDATE_MS
    allow_refresh = (
        force_refresh
        or _UI_LAST_MODE is None
        or mode != _UI_LAST_MODE
        or (now - _UI_LAST_UPDATE_MS) >= _UI_MIN_INTERVAL_MS
    )
    if not allow_refresh and _UI_LAST_MESSAGE:
        message_to_use = _UI_LAST_MESSAGE
    else:
        message_to_use = message
        _UI_LAST_MODE = mode
        _UI_LAST_MESSAGE = message_to_use
        _UI_LAST_UPDATE_MS = now

    display_text = f"[{mode.name}] {message_to_use}"
    if allow_refresh or force_refresh:
        talk_update = display_text if force_refresh else gr.update(value=display_text)
    else:
        talk_update = gr.update()

    context_payload = json.dumps(ctx_json, ensure_ascii=False, indent=2)
    avatar_image = _render_avatar(float(voice_energy), mode == TalkMode.TALK)
    return talk_update, context_payload, fig, mode.name, avatar_image


def _talkmode_refresh_start(*args):
    return talkmode_step_ui(*args, force_refresh=True, activate_session=True)


def _talkmode_refresh_passive(*args):
    return talkmode_step_ui(*args, force_refresh=True, activate_session=False)


def talkmode_auto_tick(
    auto_enabled: bool,
    face: str,
    voice_energy: float,
    dm: float,
    jerk: float,
    engaged: bool,
    since_ms: int,
    S: float,
    H: float,
    rho: float,
    ignition: float,
    memory_tags: str,
    user_text: str,
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    global _SESSION_STARTED
    if not auto_enabled or not _SESSION_STARTED:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value="WATCH"),
            gr.update(value=since_ms),
            gr.update(),
        )

    since_ms = min(int(since_ms) + 500, 4000)
    text, ctx_json, fig, mode, avatar_image = talkmode_step_ui(
        face,
        voice_energy,
        dm,
        jerk,
        engaged,
        since_ms,
        S,
        H,
        rho,
        ignition,
        memory_tags,
        user_text,
        force_refresh=False,
        activate_session=False,
    )
    return (
        text,
        ctx_json,
        fig,
        gr.update(value=mode),
        gr.update(value=since_ms),
        avatar_image,
    )


def talkmode_camera_stream(
    frame: Optional[np.ndarray], face_val: str, engaged_val: bool
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    if frame is None or not HAVE_MEDIAPIPE:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    if cv2 is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb = frame[:, :, ::-1]
    detected, mouth_open, blink = _face_metrics(rgb)
    valence = float(np.clip(mouth_open * 16.0 - 0.2, -1.0, 1.0))
    delta, jerk = _TALK_STATE.update_affect(valence)
    face = "joy" if mouth_open > 0.08 else "sad" if blink < 0.015 else "neutral"
    engaged = bool(detected)
    if engaged:
        mouth_norm = float(np.clip(mouth_open * 25.0, 0.0, 1.0))
        blink_norm = float(np.clip(blink * 150.0, 0.0, 1.0))
        freeze_raw = float(
            np.clip(1.0 - (0.6 * mouth_norm + 0.4 * blink_norm), 0.0, 1.0)
        )
    else:
        freeze_raw = 0.0
    global _SESSION_LAST_FREEZE_RAW
    _SESSION_LAST_FREEZE_RAW = freeze_raw
    return (
        rgb,
        gr.update(value=face),
        gr.update(value=float(np.clip(_TALK_STATE.voice_energy, 0.0, 1.0))),
        gr.update(value=float(np.clip(delta, 0.0, 1.0))),
        gr.update(value=float(np.clip(jerk, 0.0, 0.6))),
        gr.update(value=engaged),
    )


with gr.Blocks() as demo:
    with gr.Tab("Emotional Hub"):
        gr.Markdown(
            "# EQNet Emotional Hub Demo\n"
            "Upload text, image, and/or audio to see EQNet's affect estimate, response, and field visualisation."
        )

        with gr.Row():
            text_in = gr.Textbox(
                label="User Text", placeholder="How are you feeling today?", lines=2
            )
            image_in = gr.Image(label="Image Frame (optional)", type="pil")

        audio_in = gr.Audio(label="Audio (optional)", type="numpy")
        submit_btn = gr.Button("Run")

        response_out = gr.Textbox(label="Response", lines=3)
        affect_out = gr.JSON(label="Affect (valence / arousal / confidence)")
        metrics_out = gr.JSON(label="EQNet Metrics")
        controls_out = gr.JSON(label="Behaviour Controls")
        prospective_out = gr.JSON(label="Prospective State")
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
                prospective_out,
                advanced_radar_out,
                basic_radar_out,
                heatmap_out,
                summary_out,
            ],
        )

        gr.Markdown(
            "For continuous demos, consider enabling Gradio's `live` mode or streaming camera/microphone input."
        )

    with gr.Tab("TalkMode Resonance"):
        gr.Markdown(
            "### TalkMode × Emotion Terrain × Memory Resonance\n"
            "表情・情動地形・記憶タグから会話モードを切り替えます。"
        )

        with gr.Row():
            if HAVE_MEDIAPIPE:
                with gr.Column():
                    camera = gr.Video(
                        sources=["webcam"], streaming=True, label="Webcam", height=240
                    )
                    camera_preview = gr.Image(label="Webcam Preview")
            else:
                camera = None
                camera_preview = None
            with gr.Column():
                talk_mode_display = gr.Textbox(
                    label="TalkMode", value="WATCH", interactive=False
                )
            with gr.Column():
                avatar_out = gr.Image(label="Avatar", type="pil")

        gr.Markdown(
            "※ Auto update をオンにすると 0.5 秒ごとに再判定します。初回は下の Run ボタンを押してください。"
        )
        gr.Markdown(
            "※ Webカメラを使わない場合は下のスライダで手動入力できます。カメラ接続時は自動で上書きされます。"
        )

        with gr.Row():
            face_radio = gr.Radio(
                ["joy", "neutral", "sad", "surprise"], value="neutral", label="Face"
            )
            voice_slider = gr.Slider(0.0, 1.0, value=0.20, label="Voice energy")
            engaged_check = gr.Checkbox(True, label="Engaged")
            since_slider = gr.Slider(
                0, 4000, value=1200, step=50, label="Elapsed since last user [ms]"
            )

        with gr.Row():
            dm_slider = gr.Slider(0.0, 1.0, value=0.10, label="Delta m")
            jerk_slider = gr.Slider(0.0, 0.6, value=0.05, label="Jerk")
            S_slider = gr.Slider(0.0, 1.0, value=0.45, label="S (entropy)")
            H_slider = gr.Slider(0.0, 1.0, value=0.40, label="H (heat)")
            rho_slider = gr.Slider(0.0, 1.0, value=0.55, label="rho (sync)")
            ignition_slider = gr.Slider(0.0, 1.0, value=0.30, label="Ignition")

        memory_tags = gr.Textbox(
            value="work, family, project-X", label="Memory tags / recent events"
        )
        user_text = gr.Textbox(
            label="Text input (optional)", placeholder="ASR output can be placed here"
        )

        with gr.Row():
            talk_text = gr.Textbox(label="Response", lines=4)
            context_json = gr.Code(label="gate_context", language="json")

        heartbeat_plot = gr.Plot(label="AI heartbeat")
        go_button = gr.Button("Run (Decide & respond)")

        go_button.click(
            _talkmode_refresh_start,
            [
                face_radio,
                voice_slider,
                dm_slider,
                jerk_slider,
                engaged_check,
                since_slider,
                S_slider,
                H_slider,
                rho_slider,
                ignition_slider,
                memory_tags,
                user_text,
            ],
            [talk_text, context_json, heartbeat_plot, talk_mode_display, avatar_out],
        )

        auto_toggle = gr.Checkbox(True, label="Auto update (0.5 s)")
        timer = gr.Timer(0.5)
        timer.tick(
            talkmode_auto_tick,
            [
                auto_toggle,
                face_radio,
                voice_slider,
                dm_slider,
                jerk_slider,
                engaged_check,
                since_slider,
                S_slider,
                H_slider,
                rho_slider,
                ignition_slider,
                memory_tags,
                user_text,
            ],
            [
                talk_text,
                context_json,
                heartbeat_plot,
                talk_mode_display,
                since_slider,
                avatar_out,
            ],
        )

        controls = [
            face_radio,
            voice_slider,
            dm_slider,
            jerk_slider,
            engaged_check,
            since_slider,
            S_slider,
            H_slider,
            rho_slider,
            ignition_slider,
            memory_tags,
            user_text,
        ]
        for ctrl in controls:
            ctrl.change(
                _talkmode_refresh_passive,
                [
                    face_radio,
                    voice_slider,
                    dm_slider,
                    jerk_slider,
                    engaged_check,
                    since_slider,
                    S_slider,
                    H_slider,
                    rho_slider,
                    ignition_slider,
                    memory_tags,
                    user_text,
                ],
                [talk_text, context_json, heartbeat_plot, talk_mode_display, avatar_out],
            )

        if camera is not None and camera_preview is not None:
            camera.change(
                talkmode_camera_stream,
                inputs=[camera, face_radio, engaged_check],
                outputs=[
                    camera_preview,
                    face_radio,
                    voice_slider,
                    dm_slider,
                    jerk_slider,
                    engaged_check,
                ],
            )


if __name__ == "__main__":
    _get_runtime()
    demo.launch()
