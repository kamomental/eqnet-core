# -*- coding: utf-8 -*-
"""
Lightweight perception scaffold for the EQNet hub.

This module intentionally keeps the implementation minimal so it can run on
commodity webcams / microphones without GPU support. It exposes an API that
future, more sophisticated recognisers (MediaPipe, ONNX, etc.) can plug into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Iterable
import time

import numpy as np

from .video_backends import (
    create_video_backend,
    VideoBackendConfig,
    BaseVideoBackend,
)
from .audio_backends import (
    create_audio_backend,
    AudioBackendConfig,
    BaseAudioBackend,
)


@dataclass
class PerceptionConfig:
    """Configuration knobs for the perception bridge."""

    video_fps: float = 30.0
    """Input camera frame rate (used for timing heuristics)."""

    downscale: int = 256
    """Target edge length for face/landmark detection."""

    audio_sample_rate: int = 16_000
    """Audio sample rate expected for chunks (Hz)."""

    fusion_hz: float = 15.0
    """How often we surface fused affect samples (Hz)."""

    ema_beta: float = 0.7
    """Exponential moving-average coefficient for smoothing signals."""

    mode: str = "auto"
    """Operational mode: 'auto', 'community', or 'focus'."""

    video_backend: str = "opencv"
    """Video backend key (opencv / mediapipe / onnx / custom)."""

    video_backend_params: Dict[str, Any] = field(default_factory=dict)
    """Additional parameters for the selected video backend."""

    audio_backend: str = "lightweight"
    """Audio backend key (lightweight / webrtcvad / custom)."""

    audio_backend_params: Dict[str, Any] = field(default_factory=dict)
    """Additional parameters for the selected audio backend."""

    batch_enabled: bool = True


@dataclass
class AffectSample:
    """A compact representation of valence/arousal inferred from sensors."""

    valence: float  # [-1, 1]
    arousal: float  # [-1, 1]
    confidence: float  # [0, 1]
    timestamp: float


@dataclass
class PerceptionStats:
    """Collected timing / diagnostic information."""

    video_latency_ms: float = 0.0
    audio_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    video_latency_p95_ms: float = 0.0
    audio_latency_p95_ms: float = 0.0
    fusion_latency_p95_ms: float = 0.0
    last_video_metrics: dict[str, float] = field(default_factory=dict)
    last_audio_metrics: dict[str, float] = field(default_factory=dict)


class _SLOTracker:
    """Rolling latency tracker to monitor p95 targets."""

    def __init__(self, window: int = 100) -> None:
        self.window = max(window, 1)
        self.samples: list[float] = []

    def add(self, value_ms: float) -> None:
        self.samples.append(value_ms)
        if len(self.samples) > self.window:
            self.samples.pop(0)

    def p95(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        idx = max(int(round(0.95 * (len(ordered) - 1))), 0)
        return ordered[idx]


class PerceptionBridge:
    """
    Minimal perception bridge.

    - Video frames: we compute a very rough luminance + motion surrogate.
    - Audio chunks: root-mean-square energy (normalised) and zero-crossing rate.
    - Fusion: EMA smoothing yields a coarse valence/arousal estimate which can
      later be replaced by more sophisticated models.
    """

    def __init__(self, config: Optional[PerceptionConfig] = None) -> None:
        self.config = config or PerceptionConfig()
        self._last_video_ema: Optional[float] = None
        self._last_motion_ema: Optional[float] = None
        self._last_audio_ema: Optional[float] = None
        self._last_zcr_ema: Optional[float] = None
        self._last_timestamp: float = time.time()
        self._last_affect: Optional[AffectSample] = None
        self._last_video_metrics: dict[str, float] = {}
        self._last_audio_metrics: dict[str, float] = {}

        video_cfg = VideoBackendConfig(
            mediator=self.config.video_backend_params.get("mediator"),
            model_path=self.config.video_backend_params.get("model_path"),
            target_size=self.config.downscale,
        )
        self._video_backend: BaseVideoBackend = create_video_backend(self.config.video_backend, video_cfg)

        audio_cfg = AudioBackendConfig(
            vad_mode=self.config.audio_backend_params.get("vad_mode", 2),
            sample_rate=self.config.audio_sample_rate,
        )
        self._audio_backend: BaseAudioBackend = create_audio_backend(self.config.audio_backend, audio_cfg)

        self._video_slo = _SLOTracker()
        self._audio_slo = _SLOTracker()
        self._fusion_slo = _SLOTracker()

    # ------------------------------------------------------------------ #
    # Video / audio ingestion
    # ------------------------------------------------------------------ #

    def ingest_video_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Ingest a single RGB frame (numpy array). The bridge keeps tiny EMA
        statistics so that downstream affect fusion remains O(1).
        """
        if frame.size == 0:
            return
        start = time.perf_counter()
        metrics = self._video_backend.process(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._video_slo.add(elapsed_ms)
        self._last_video_metrics = metrics

        luminance = metrics.get("luminance", 0.0)
        if self._last_video_ema is None:
            self._last_video_ema = luminance
        else:
            self._last_video_ema = self._ema(self._last_video_ema, luminance)

        grad = metrics.get("motion", 0.0)
        if self._last_motion_ema is None:
            self._last_motion_ema = grad
        else:
            self._last_motion_ema = self._ema(self._last_motion_ema, grad)

        if timestamp is not None:
            self._last_timestamp = timestamp

    def ingest_audio_chunk(
        self,
        chunk: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Ingest a mono PCM chunk (float32 or int16). The method computes a
        normalised RMS energy and zero-crossing rate.
        """
        if chunk.size == 0:
            return
        start = time.perf_counter()
        metrics = self._audio_backend.process(chunk)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._audio_slo.add(elapsed_ms)
        self._last_audio_metrics = metrics

        rms = metrics.get("rms", 0.0)
        zcr = metrics.get("zcr", 0.0)

        if self._last_audio_ema is None:
            self._last_audio_ema = rms
            self._last_zcr_ema = zcr
        else:
            self._last_audio_ema = self._ema(self._last_audio_ema, rms)
            self._last_zcr_ema = self._ema(self._last_zcr_ema or zcr, zcr)

        if timestamp is not None:
            self._last_timestamp = timestamp

    # ------------------------------------------------------------------ #
    # Fusion
    # ------------------------------------------------------------------ #

    def sample_affect(self, timestamp: Optional[float] = None) -> AffectSample:
        """
        Produce the latest valence/arousal estimate. The mapping is deliberately
        simplistic; downstream policy code clamps values and adapts dynamics.
        """
        fusion_start = time.perf_counter()
        ts = timestamp or self._last_timestamp or time.time()
        brightness = self._last_video_ema or 0.0
        motion = self._last_motion_ema or 0.0
        energy = self._last_audio_ema or 0.0
        zcr = self._last_zcr_ema or 0.0

        # Heuristic normalisation.
        valence = self._scale(brightness, low=20.0, high=220.0)
        # Combine audio energy & motion for arousal (clip to [-1,1]).
        arousal_energy = self._scale(energy, low=0.005, high=0.15)
        arousal_motion = self._scale(motion, low=0.1, high=10.0)
        arousal = float(np.clip(0.6 * arousal_energy + 0.4 * arousal_motion, -1.0, 1.0))
        confidence = float(np.clip(0.5 + 0.5 * (abs(arousal) + abs(valence)) / 2.0, 0.0, 1.0))
        # ZCR pushes valence slightly toward neutral when speech is noisy.
        valence = float(np.clip(valence * (1.0 - min(zcr, 1.0) * 0.3), -1.0, 1.0))

        affect = AffectSample(valence=valence, arousal=arousal, confidence=confidence, timestamp=ts)
        self._last_affect = affect
        self._fusion_slo.add((time.perf_counter() - fusion_start) * 1000.0)
        return affect

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ema(self, prev: float, new: float) -> float:
        beta = float(np.clip(self.config.ema_beta, 0.0, 0.99))
        return beta * prev + (1.0 - beta) * new

    @staticmethod
    def _scale(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        norm = (value - low) / (high - low)
        return float(np.clip(norm * 2.0 - 1.0, -1.0, 1.0))

    # Public getters for logging / diagnostics

    @property
    def last_affect(self) -> Optional[AffectSample]:
        return self._last_affect

    def stats(self) -> PerceptionStats:
        return PerceptionStats(
            video_latency_ms=self._video_slo.samples[-1] if self._video_slo.samples else 0.0,
            audio_latency_ms=self._audio_slo.samples[-1] if self._audio_slo.samples else 0.0,
            fusion_latency_ms=self._fusion_slo.samples[-1] if self._fusion_slo.samples else 0.0,
            video_latency_p95_ms=self._video_slo.p95(),
            audio_latency_p95_ms=self._audio_slo.p95(),
            fusion_latency_p95_ms=self._fusion_slo.p95(),
            last_video_metrics=self._last_video_metrics,
            last_audio_metrics=self._last_audio_metrics,
        )

    # ------------------------------------------------------------------ #
    # Batch helpers
    # ------------------------------------------------------------------ #

    def process_video_batch(self, frames: Iterable[np.ndarray]) -> None:
        """
        Process a sequence of frames (e.g., overnight logs). Only available
        when ``batch_enabled`` is true in the configuration.
        """
        if not self.config.batch_enabled:
            raise RuntimeError("Batch mode is disabled in perception config.")
        for frame in frames:
            self.ingest_video_frame(frame)

    def process_audio_batch(self, chunks: Iterable[np.ndarray]) -> None:
        if not self.config.batch_enabled:
            raise RuntimeError("Batch mode is disabled in perception config.")
        for chunk in chunks:
            self.ingest_audio_chunk(chunk)
