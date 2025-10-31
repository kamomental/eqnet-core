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
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum, auto
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
import cv2
try:  # optional deps
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
    "listen": "\u3046\u3093\u3001\u805e\u3044\u3066\u3044\u308b\u3088\u3002\u5fc5\u8981\u306a\u3068\u304d\u5408\u56e0\u3057\u3066\u306d\u3002",
    "soothe": "\u8868\u60c5\u304c\u5c11\u3057\u786c\u3044\u304b\u3082\u3002\u6df1\u547c\u5438\u3057\u3066\u3001\u80a9\u306e\u529b\u3092\u629c\u3053\u3046\u3002\u79c1\u306f\u3053\u3053\u306b\u3044\u308b\u3088\u3002",
    "soothe_deep": "\u6025\u304c\u306a\u304f\u3066\u3082\u3044\u3044\u3088\u3002\u3053\u3053\u306f\u5b89\u5168\u3060\u304b\u3089\u3001\u76ee\u3092\u9589\u3058\u3066\u3082\u5927\u4e08\u592b\u3002\u58f0\u304c\u51fa\u306a\u3044\u3068\u304d\u306f\u3001\u305f\u3060\u547c\u5438\u3060\u3051\u610f\u8b58\u3057\u3088\u3046\u3002",
    "ask": "\u3055\u3063\u304d\u3088\u308a\u5143\u6c17\u304c\u5c11\u306a\u3081\u306b\u898b\u3048\u308b\u3002\u4f55\u304b\u3042\u3063\u305f\uff1f\u8a71\u305b\u308b\u7bc4\u56f2\u3067\u6559\u3048\u3066\u306d\u3002",
    "talk": "\u7d9a\u304d\u3092\u6574\u3048\u3088\u3046\u3002\u3069\u3053\u304b\u3089\u8a71\u305d\u3046\u304b\u3002",
}
SOOTHE_VARIATIONS = [
    "\u547c\u5438\u3092\u6574\u3048\u3088\u3046\u3002\u80a9\u3092\u8efd\u304f\u56de\u3057\u3066\u307f\u308b\u306e\u3082\u3044\u3044\u304b\u3082\u3002",
    "\u5c11\u3057\u5411\u304d\u5408\u304a\u3046\u3002\u3044\u307e\u306f\u3086\u3063\u304f\u308a\u3067\u5927\u4e08\u592b\u3002",
    "\u7dca\u5f35\u304c\u7d9a\u3044\u3066\u3044\u308b\u304b\u3082\u3057\u308c\u306a\u3044\u3002\u6df1\u547c\u5438\u3092\u4e00\u7dd2\u306b\u3057\u3088\u3046\u3002",
]
ASK_VARIATIONS = [
    "\u4f55\u304b\u80f8\u306b\u5f15\u3063\u304b\u304b\u308b\u3053\u3068\u304c\u3042\u3063\u305f\uff1f\u8a71\u305b\u308b\u7bc4\u56f2\u3067\u5927\u4e08\u592b\u3060\u3088\u3002",
    "\u3055\u3063\u304d\u3088\u308a\u5c11\u3057\u9759\u304b\u306b\u306a\u3063\u305f\u6c17\u304c\u3059\u308b\u3002\u5b89\u5fc3\u3067\u304d\u3066\u3044\u308b\uff1f",
    "\u3082\u3057\u5909\u5316\u304c\u3042\u3063\u305f\u3089\u6559\u3048\u3066\u306d\u3002\u3053\u3053\u3067\u6574\u3048\u3066\u3044\u3053\u3046\u3002",
]
WATCH_VARIATIONS = [
    "\u3046\u3093\u3001\u305d\u3070\u306b\u3044\u308b\u3088\u3002\u5408\u56e0\u304c\u3042\u308c\u3070\u3044\u3064\u3067\u3082\u7d9a\u3051\u3088\u3046\u3002",
    "\u3053\u3053\u3067\u898b\u5b88\u3063\u3066\u3044\u308b\u306d\u3002\u5fc5\u8981\u306a\u3068\u304d\u306b\u58f0\u3092\u304b\u3051\u3066\u3002",
    "\u5927\u4e08\u592b\u3001\u843d\u3061\u7740\u3044\u305f\u3089\u6559\u3048\u3066\u3002\u3044\u3064\u3067\u3082\u8033\u3092\u50be\u3051\u308b\u3088\u3002",
]
if HAVE_MEDIAPIPE:  # pragma: no cover
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
        self.buf[-len(t):] = y
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
def _update_session_state(voice_energy: float, instant_freeze: Optional[float] = None) -> Tuple[int, float, float]:
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
    _SESSION_FREEZE_SMOOTH = (1 - alpha) * _SESSION_FREEZE_SMOOTH + alpha * instant_freeze
    _SESSION_LAST_FREEZE_RAW = instant_freeze
    freeze_trend = _SESSION_FREEZE_SMOOTH - prev
    return _SESSION_SILENCE_MS, _SESSION_FREEZE_SMOOTH, freeze_trend
def _decide_talk_mode_local(ctx: GateContext, guard_action: str = "ok") -> TalkMode:
    global _LAST_MODE, _LAST_TALK_MS, _LAST_TALK_TRIGGER, _LAST_TALK_TRIGGER_MS
    now = _now_ms()
    if _LAST_MODE == TalkMode.TALK and (now - _LAST_TALK_MS) < TALK_DWELL_MS:
        return TalkMode.TALK
    if guard_action in {"fallback_attention", "tighten_band"}:
        _LAST_MODE = TalkMode.SOOTHE if (ctx.dm > DM_ENTER or ctx.jerk > JK_ENTER) else TalkMode.WATCH
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
    if ctx.face in {"neutral"} and ctx.voice_energy < ASK_VOICE_TH and ctx.since_last_user_ms > ASK_MIN_SINCE:
        _LAST_MODE = TalkMode.ASK
        return _LAST_MODE
    trigger_key = None
    if ctx.text_input:
        normalized = ctx.text_value.strip()
        if normalized:
            trigger_key = f"text:{hash((normalized, ctx.memory_anchor))}"
    elif ctx.voice_energy >= TALK_SIGNAL_VOICE or ctx.dm >= TALK_SIGNAL_DM or ctx.jerk >= TALK_SIGNAL_JERK:
        trigger_key = f"sensor:{int(ctx.voice_energy * 100)}-{int(ctx.dm * 100)}-{int(ctx.jerk * 100)}"
    if trigger_key and ctx.freeze_score < FREEZE_EXIT:
        if ctx.since_last_user_ms > TALK_MIN_SINCE and (now - _LAST_TALK_MS) > TALK_COOLDOWN_MS:
            if trigger_key != _LAST_TALK_TRIGGER or (now - _LAST_TALK_TRIGGER_MS) > TALK_REPEAT_RESET_MS:
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
        threading.Thread(target=lambda: (_TTS_ENGINE.say(text), _TTS_ENGINE.runAndWait()), daemon=True).start()
    except Exception:  # pragma: no cover
        pass
def _render_heartbeat(y: np.ndarray):
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
        return f"{line} (voice={ctx.voice_energy:.2f}, since={ctx.since_last_user_ms}ms)"
    line = random.choice(WATCH_VARIATIONS)
    return f"{line} (Ignition={ctx.ignition:.2f})"
def _llm_free_talk(ctx: GateContext) -> str:
    mood = "\u9759\u7a4f" if (ctx.S < 0.4 and ctx.H < 0.4) else "\u3056\u308f\u3064\u304d" if (ctx.S > 0.6 or ctx.H > 0.6) else "\u3086\u3089\u304e"
    lines: List[str] = [
        f"\u4eca\u306e\u7a7a\u6c17\u306f\u300c{mood}\u300d\u3002S={ctx.S:.2f}, H={ctx.H:.2f}, rho={ctx.rho:.2f}, Ignition={ctx.ignition:.2f}",
    ]
    if ctx.face in {"sad", "fear"}:
        lines.append("\u3059\u3053\u3057\u4e0d\u5b89\u304c\u6df7\u3058\u3063\u3066\u3044\u308b\u304b\u3082\u3002\u4e00\u7dd2\u306b\u6df1\u547c\u5438\u3057\u3066\u6574\u3048\u3088\u3046")
    elif ctx.face in {"joy"}:
        lines.append("\u3044\u3044\u8868\u60c5\u3002\u3044\u307e\u306e\u6d41\u308c\u3001\u7d9a\u3051\u3089\u308c\u305d\u3046")
    if ctx.voice_energy < 0.2:
        lines.append("\u58f0\u306f\u3044\u307e\u63a7\u3048\u3081\u3002\u3053\u3061\u3089\u304b\u3089\u5c11\u3057\u3060\u3051\u8a71\u3059\u306d")
    if ctx.freeze_score > FREEZE_ENTER or ctx.silence_ms > SILENCE_ENTER_MS:
        lines.append("\u4f53\u304c\u56fa\u307e\u3063\u3066\u3044\u308b\u611f\u3058\u304c\u3042\u308b\u306d\u3002\u547c\u5438\u3092\u6574\u3048\u306a\u304c\u3089\u3001\u5206\u306b\u4e00\u5ea6\u3060\u3051\u305d\u3063\u3068\u5fc3\u306e\u72b6\u614b\u3092\u898b\u3088\u3046")
    if ctx.dm > 0.35 or ctx.jerk > 0.2:
        lines.append("\u611f\u60c5\u306e\u6ce2\u304c\u5c11\u3057\u5927\u304d\u3044\u3002\u5927\u4e08\u592b\u3001\u79c1\u304c\u96a3\u3067\u6574\u3048\u308b\u3088")
    if ctx.memory_anchor and ctx.memory_anchor != "-":
        lines.append(f"\u6700\u8fd1\u9cf4\u3044\u3066\u3044\u308b\u8a18\u61b6\u306f\u300c{ctx.memory_anchor}\u300d\u3002\u305d\u3053\u306b\u5bc4\u308a\u6dfb\u3063\u3066\u8003\u3048\u3066\u307f\u3088\u3046")
    lines.append("\u7d9a\u3051\u308b\u6e96\u5099\u304c\u3067\u304d\u305f\u3089\u3001\u5408\u56e0\u3057\u3066")
    return " ".join(lines)
def _hub_step(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not HUB_URL:
        return None
    try:
        r = requests.post(f"{HUB_URL.rstrip('/')}/api/step", json=payload, timeout=10)
        if r.ok:
            return r.json()
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