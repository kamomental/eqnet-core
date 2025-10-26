# -*- coding: utf-8 -*-
"""
Video backends for the perception bridge.

Each backend exposes a ``process(frame)`` method returning a dict of features.
Backends are dynamically selected based on configuration, allowing easy swaps
between OpenCV/MediaPipe/ONNX pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

try:
    import cv2  # type: ignore
    HAS_OPENCV = True
except Exception:
    HAS_OPENCV = False

try:
    import mediapipe as mp  # type: ignore
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

try:
    import onnxruntime as ort  # type: ignore
    HAS_ONNXRUNTIME = True
except Exception:
    HAS_ONNXRUNTIME = False


@dataclass
class VideoBackendConfig:
    mediator: Optional[str] = None
    model_path: Optional[str] = None
    target_size: int = 256


class BaseVideoBackend:
    def __init__(self, config: VideoBackendConfig):
        self.config = config

    def process(self, frame: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError


class OpenCVBackend(BaseVideoBackend):
    """Fallback backend using simple luminance and motion heuristics."""

    def __init__(self, config: VideoBackendConfig):
        super().__init__(config)
        self._previous: Optional[np.ndarray] = None

    def process(self, frame: np.ndarray) -> Dict[str, float]:
        luminance = float(np.mean(frame))
        motion = 0.0
        if self._previous is not None and self._previous.shape == frame.shape:
            motion = float(np.abs(frame.astype(np.float32) - self._previous).mean())
        self._previous = frame.copy()
        return {"luminance": luminance, "motion": motion}


class MediaPipeBackend(BaseVideoBackend):
    """MediaPipe face/landmark backend (feature extraction simplified)."""

    def __init__(self, config: VideoBackendConfig):
        if not HAS_MEDIAPIPE:
            raise RuntimeError("mediapipe is not available.")
        super().__init__(config)
        self._solutions = mp.solutions
        self._face_mesh = self._solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame: np.ndarray) -> Dict[str, float]:
        frame_rgb = frame[:, :, ::-1]
        result = self._face_mesh.process(frame_rgb)
        luminance = float(np.mean(frame))
        mouth_open = 0.0
        blink = 0.0
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            mouth_open = abs(upper_lip.y - lower_lip.y)
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            blink = abs(left_eye_top.y - left_eye_bottom.y)
        return {
            "luminance": luminance,
            "motion": mouth_open,
            "blink": blink,
        }


class ONNXBackend(BaseVideoBackend):
    """Generic ONNX pose backend (placeholder)."""

    def __init__(self, config: VideoBackendConfig):
        if not HAS_ONNXRUNTIME:
            raise RuntimeError("onnxruntime is not available.")
        if not config.model_path:
            raise ValueError("model_path is required for ONNX backend.")
        super().__init__(config)
        self._session = ort.InferenceSession(config.model_path, providers=["CPUExecutionProvider"])

    def process(self, frame: np.ndarray) -> Dict[str, float]:
        # Placeholder: actual implementation depends on the ONNX model interface.
        # Here we simply return luminance and zero motion; extend as needed.
        return {"luminance": float(np.mean(frame)), "motion": 0.0}


def create_video_backend(name: str, config: VideoBackendConfig) -> BaseVideoBackend:
    name = name.lower()
    if name == "mediapipe":
        return MediaPipeBackend(config)
    if name == "onnx":
        return ONNXBackend(config)
    return OpenCVBackend(config)

