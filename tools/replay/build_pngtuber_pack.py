#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build Motion PNGTuber layer pack from a single source image and masks.

Usage examples:
  python tools/replay/build_pngtuber_pack.py ^
    --source character.png ^
    --out-dir assets/replay/character/default ^
    --manifest assets/replay/pngtuber_manifest.json

  python tools/replay/build_pngtuber_pack.py ^
    --source character.png ^
    --out-dir assets/replay/character/default ^
    --head-mask masks/head.png --body-mask masks/body.png --auto-split false

  python tools/replay/build_pngtuber_pack.py ^
    --source character.png ^
    --out-dir assets/replay/character/default ^
    --anchors-file assets/replay/masks/default/anchors.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageChops, ImageDraw

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None  # type: ignore[assignment]


LAYER_NAMES = (
    "base",
    "body",
    "head",
    "hair_back",
    "hair_front",
    "eye_open",
    "eye_half",
    "eye_open_wide",
)

MOUTH_NAMES = ("a", "i", "u", "e", "o", "n")
ANCHOR_KEYS = ("left_eye", "right_eye", "mouth")
AnchorMap = Dict[str, Tuple[float, float]]


@dataclass
class DetectorResult:
    anchors: AnchorMap
    detector_name: str
    confidence: float
    meta: Dict[str, object]


@dataclass
class Bounds:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return max(1, self.x1 - self.x0)

    @property
    def h(self) -> int:
        return max(1, self.y1 - self.y0)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def ensure_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def alpha_bounds(image: Image.Image) -> Bounds:
    alpha = image.getchannel("A")
    box = alpha.getbbox()
    if box is None:
        return Bounds(0, 0, image.width, image.height)
    return Bounds(box[0], box[1], box[2], box[3])


def new_mask(size: Tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 0)


def mask_rect(size: Tuple[int, int], box: Tuple[int, int, int, int], value: int = 255) -> Image.Image:
    m = new_mask(size)
    d = ImageDraw.Draw(m)
    d.rectangle(box, fill=value)
    return m


def mask_ellipse(size: Tuple[int, int], box: Tuple[int, int, int, int], value: int = 255) -> Image.Image:
    m = new_mask(size)
    d = ImageDraw.Draw(m)
    d.ellipse(box, fill=value)
    return m


def mask_rotated_ellipse(
    size: Tuple[int, int],
    center: Tuple[float, float],
    width: float,
    height: float,
    angle_deg: float = 0.0,
    value: int = 255,
) -> Image.Image:
    cx, cy = center
    w2 = max(1.0, width * 0.5)
    h2 = max(1.0, height * 0.5)
    box = (
        int(round(cx - w2)),
        int(round(cy - h2)),
        int(round(cx + w2)),
        int(round(cy + h2)),
    )
    m = mask_ellipse(size, clamp_box(box, size[0], size[1]), value=value)
    if abs(angle_deg) < 1e-3:
        return m
    return m.rotate(float(angle_deg), resample=Image.BICUBIC, center=(float(cx), float(cy)), fillcolor=0)


def mask_rotated_rect(
    size: Tuple[int, int],
    center: Tuple[float, float],
    width: float,
    height: float,
    angle_deg: float = 0.0,
    value: int = 255,
) -> Image.Image:
    cx, cy = center
    w2 = max(1.0, width * 0.5)
    h2 = max(1.0, height * 0.5)
    box = (
        int(round(cx - w2)),
        int(round(cy - h2)),
        int(round(cx + w2)),
        int(round(cy + h2)),
    )
    m = mask_rect(size, clamp_box(box, size[0], size[1]), value=value)
    if abs(angle_deg) < 1e-3:
        return m
    return m.rotate(float(angle_deg), resample=Image.BICUBIC, center=(float(cx), float(cy)), fillcolor=0)


def clamp_box(
    box: Tuple[int, int, int, int],
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, min(w, x0)),
        max(0, min(h, y0)),
        max(0, min(w, x1)),
        max(0, min(h, y1)),
    )


def multiply_alpha(base_alpha: Image.Image, mask: Image.Image) -> Image.Image:
    return ImageChops.multiply(base_alpha, mask)


def apply_mask_rgba(source: Image.Image, mask: Image.Image) -> Image.Image:
    out = source.copy()
    out.putalpha(multiply_alpha(source.getchannel("A"), mask))
    return out


def union_masks(masks: Dict[str, Image.Image], size: Tuple[int, int]) -> Image.Image:
    out = Image.new("L", size, 0)
    for m in masks.values():
        out = ImageChops.lighter(out, m)
    return out


def soften_mask(mask: Image.Image, radius: int = 2) -> Image.Image:
    # Pillow-only blur substitute using repeated box expand/shrink via min/max filters is unavailable without ImageFilter.
    # Keep mask as-is for deterministic output in minimal dependency environments.
    return mask


def erase_mouth_region(source: Image.Image, mouth_union_mask: Image.Image) -> Image.Image:
    """Create a mouth-erased variant by filling mouth area with sampled skin tone.

    This is a lightweight alternative to cv2 inpainting. For anime bust-up images,
    sampling just above the mouth is often acceptable as a base for mouth overlays.
    """
    rgba = source.copy().convert("RGBA")
    px = rgba.load()
    mask = mouth_union_mask.convert("L")
    mpx = mask.load()
    w, h = rgba.size
    box = mask.getbbox()
    if box is None:
        return rgba

    x0, y0, x1, y1 = box
    sample_y0 = max(0, y0 - max(4, (y1 - y0) * 2))
    sample_y1 = max(sample_y0 + 1, y0 - 1)
    sample_x0 = max(0, x0 - 2)
    sample_x1 = min(w, x1 + 2)
    samples = []
    for yy in range(sample_y0, sample_y1):
        for xx in range(sample_x0, sample_x1):
            if mpx[xx, yy] < 8:
                r, g, b, a = px[xx, yy]
                if a > 8:
                    samples.append((r, g, b, a))
    if samples:
        sr = int(sum(c[0] for c in samples) / len(samples))
        sg = int(sum(c[1] for c in samples) / len(samples))
        sb = int(sum(c[2] for c in samples) / len(samples))
        sa = int(sum(c[3] for c in samples) / len(samples))
    else:
        sr, sg, sb, sa = 220, 180, 170, 255

    # Fill masked pixels and add a tiny vertical gradient to reduce flat patches.
    height = max(1, y1 - y0)
    for yy in range(y0, y1):
        t = (yy - y0) / height
        rr = min(255, max(0, int(sr * (0.98 + 0.06 * t))))
        gg = min(255, max(0, int(sg * (0.98 + 0.04 * t))))
        bb = min(255, max(0, int(sb * (0.98 + 0.04 * t))))
        for xx in range(x0, x1):
            mv = mpx[xx, yy]
            if mv <= 0:
                continue
            if mv >= 200:
                px[xx, yy] = (rr, gg, bb, sa)
            else:
                # alpha blend border
                br, bg, bb0, ba = px[xx, yy]
                a = mv / 255.0
                px[xx, yy] = (
                    int(br * (1 - a) + rr * a),
                    int(bg * (1 - a) + gg * a),
                    int(bb0 * (1 - a) + bb * a),
                    ba,
                )
    return rgba


def load_mask(path: Path, size: Tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return mask


def _parse_anchor_point(raw: object) -> Tuple[float, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(raw, Mapping):
        try:
            return (float(raw.get("x")), float(raw.get("y")))
        except (TypeError, ValueError):
            return None
    return None


def load_anchors(path: Path, size: Tuple[int, int]) -> AnchorMap:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        return {}
    w, h = size
    out: AnchorMap = {}
    for key in ANCHOR_KEYS:
        pt = _parse_anchor_point(data.get(key))
        if pt is None:
            continue
        x = max(0.0, min(float(w - 1), float(pt[0])))
        y = max(0.0, min(float(h - 1), float(pt[1])))
        out[key] = (x, y)
    return canonicalize_eye_order(_validate_anchor_map(out, size))


def save_anchors(path: Path, anchors: AnchorMap) -> None:
    norm = canonicalize_eye_order(anchors)
    payload = {k: {"x": round(v[0], 2), "y": round(v[1], 2)} for k, v in norm.items() if k in ANCHOR_KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_anchor_map(anchors: AnchorMap, size: Tuple[int, int]) -> AnchorMap:
    w, h = size
    out: AnchorMap = {}
    for key in ANCHOR_KEYS:
        pt = anchors.get(key)
        if not pt:
            continue
        x = max(0.0, min(float(w - 1), float(pt[0])))
        y = max(0.0, min(float(h - 1), float(pt[1])))
        out[key] = (x, y)
    return out


def canonicalize_eye_order(anchors: AnchorMap) -> AnchorMap:
    out = dict(anchors)
    le = out.get("left_eye")
    re = out.get("right_eye")
    if le and re and float(le[0]) > float(re[0]):
        out["left_eye"] = (float(re[0]), float(re[1]))
        out["right_eye"] = (float(le[0]), float(le[1]))
    return out


def _pick_anchor_payload(raw: object) -> Mapping[str, object] | None:
    if isinstance(raw, Mapping):
        if any(k in raw for k in ANCHOR_KEYS):
            return raw
        for key in ("anchors", "anchor", "landmarks", "points"):
            node = raw.get(key)
            if isinstance(node, Mapping) and any(k in node for k in ANCHOR_KEYS):
                return node
    return None


def _decode_xy(raw_x: float, raw_y: float, size: Tuple[int, int]) -> Tuple[float, float]:
    w, h = size
    x = float(raw_x)
    y = float(raw_y)
    # Normalize [0,1] coordinates if needed.
    if 0.0 <= x <= 1.0:
        x = x * float(max(1, w - 1))
    if 0.0 <= y <= 1.0:
        y = y * float(max(1, h - 1))
    return (
        max(0.0, min(float(w - 1), x)),
        max(0.0, min(float(h - 1), y)),
    )


def _parse_anchor_point_any(raw: object, size: Tuple[int, int]) -> Tuple[float, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return _decode_xy(float(raw[0]), float(raw[1]), size)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, Mapping):
        x = raw.get("x", raw.get("X"))
        y = raw.get("y", raw.get("Y"))
        if x is not None and y is not None:
            try:
                return _decode_xy(float(x), float(y), size)
            except (TypeError, ValueError):
                return None
        # common variants
        xv = raw.get("u", raw.get("cx"))
        yv = raw.get("v", raw.get("cy"))
        if xv is not None and yv is not None:
            try:
                return _decode_xy(float(xv), float(yv), size)
            except (TypeError, ValueError):
                return None
    return None


def _read_point_from_mapping(mapping: Mapping[str, object], aliases: Tuple[str, ...], size: Tuple[int, int]) -> Tuple[float, float] | None:
    for key in aliases:
        if key in mapping:
            pt = _parse_anchor_point_any(mapping.get(key), size)
            if pt is not None:
                return pt
        # flattened style: left_eye_x / left_eye_y
        x_key = f"{key}_x"
        y_key = f"{key}_y"
        if x_key in mapping and y_key in mapping:
            try:
                return _decode_xy(float(mapping.get(x_key)), float(mapping.get(y_key)), size)
            except (TypeError, ValueError):
                pass
    return None


def _extract_anchor_triplet_from_mapping(mapping: Mapping[str, object], size: Tuple[int, int]) -> AnchorMap:
    out: AnchorMap = {}

    left_aliases = ("left_eye", "leftEye", "eye_left", "eyeLeft", "l_eye")
    right_aliases = ("right_eye", "rightEye", "eye_right", "eyeRight", "r_eye")
    mouth_aliases = ("mouth", "mouth_center", "mouthCenter", "lip_center", "lipCenter")

    left_pt = _read_point_from_mapping(mapping, left_aliases, size)
    right_pt = _read_point_from_mapping(mapping, right_aliases, size)
    mouth_pt = _read_point_from_mapping(mapping, mouth_aliases, size)

    # nested styles: eyes.left / eyes.right, landmarks.face.left_eye
    eyes = mapping.get("eyes")
    if isinstance(eyes, Mapping):
        left_pt = left_pt or _parse_anchor_point_any(eyes.get("left"), size)
        right_pt = right_pt or _parse_anchor_point_any(eyes.get("right"), size)

    landmarks = mapping.get("landmarks")
    if isinstance(landmarks, Mapping):
        left_pt = left_pt or _read_point_from_mapping(landmarks, left_aliases, size)
        right_pt = right_pt or _read_point_from_mapping(landmarks, right_aliases, size)
        mouth_pt = mouth_pt or _read_point_from_mapping(landmarks, mouth_aliases, size)
        face_landmarks = landmarks.get("face")
        if isinstance(face_landmarks, Mapping):
            left_pt = left_pt or _read_point_from_mapping(face_landmarks, left_aliases, size)
            right_pt = right_pt or _read_point_from_mapping(face_landmarks, right_aliases, size)
            mouth_pt = mouth_pt or _read_point_from_mapping(face_landmarks, mouth_aliases, size)

    if left_pt is not None:
        out["left_eye"] = left_pt
    if right_pt is not None:
        out["right_eye"] = right_pt
    if mouth_pt is not None:
        out["mouth"] = mouth_pt
    return _validate_anchor_map(out, size)


def _iter_mappings(root: object, max_nodes: int = 20000) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    stack = [root]
    seen = 0
    while stack and seen < max_nodes:
        node = stack.pop()
        seen += 1
        if isinstance(node, Mapping):
            out.append(node)
            for v in node.values():
                if isinstance(v, (Mapping, list, tuple)):
                    stack.append(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                if isinstance(v, (Mapping, list, tuple)):
                    stack.append(v)
    return out


def _anchor_geom_score(anchors: AnchorMap, size: Tuple[int, int]) -> float:
    w, h = size
    score = 0.0
    if "left_eye" in anchors and "right_eye" in anchors:
        lx, ly = anchors["left_eye"]
        rx, ry = anchors["right_eye"]
        if lx < rx:
            score += 1.0
        eye_dist = math.hypot(rx - lx, ry - ly)
        if w * 0.10 <= eye_dist <= w * 0.70:
            score += 1.0
        if abs(ry - ly) <= h * 0.18:
            score += 1.0
    if "mouth" in anchors and "left_eye" in anchors and "right_eye" in anchors:
        mx, my = anchors["mouth"]
        lx, ly = anchors["left_eye"]
        rx, ry = anchors["right_eye"]
        eye_mid_x = (lx + rx) * 0.5
        eye_mid_y = (ly + ry) * 0.5
        if my > eye_mid_y:
            score += 1.0
        if abs(mx - eye_mid_x) <= w * 0.30:
            score += 1.0
        if my - eye_mid_y <= h * 0.55:
            score += 1.0
    return score


def _median_anchor(candidates: list[AnchorMap], size: Tuple[int, int]) -> AnchorMap:
    out: AnchorMap = {}
    for key in ANCHOR_KEYS:
        xs = [c[key][0] for c in candidates if key in c]
        ys = [c[key][1] for c in candidates if key in c]
        if xs and ys:
            out[key] = (float(median(xs)), float(median(ys)))
    return _validate_anchor_map(out, size)


def load_replay_track_anchors(path: Path, size: Tuple[int, int]) -> AnchorMap:
    suffix = path.suffix.lower()
    roots: list[object] = []
    text_all = path.read_text(encoding="utf-8-sig")
    if suffix == ".jsonl":
        lines = text_all.splitlines()
        for line in lines:
            text = line.strip()
            if not text:
                continue
            try:
                roots.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    else:
        roots.append(json.loads(text_all))

    candidates: list[AnchorMap] = []
    for root in roots:
        payload = _pick_anchor_payload(root)
        if payload is not None:
            anchors = _extract_anchor_triplet_from_mapping(payload, size)
            if len(anchors) >= 2:
                candidates.append(anchors)
        for mapping in _iter_mappings(root):
            anchors = _extract_anchor_triplet_from_mapping(mapping, size)
            if len(anchors) >= 2:
                candidates.append(anchors)

    if not candidates:
        return {}

    full = [c for c in candidates if len(c) == 3]
    source = full if full else candidates
    if len(source) > 1:
        merged = _median_anchor(source, size)
        if len(merged) >= 2:
            return merged

    scored = sorted(source, key=lambda c: _anchor_geom_score(c, size), reverse=True)
    return _validate_anchor_map(scored[0], size)


def load_anime_points_map(path: Path) -> Dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("anime points map must be a JSON object")
    out: Dict[str, object] = {}
    if "output_name" in raw:
        out["output_name"] = str(raw.get("output_name"))
    points = raw.get("points")
    if not isinstance(points, Mapping):
        raise ValueError("anime points map must have 'points' object")
    converted: Dict[str, object] = {}
    for key in ANCHOR_KEYS:
        spec = points.get(key)
        if spec is None:
            continue
        if isinstance(spec, int):
            converted[key] = int(spec)
            continue
        if isinstance(spec, list) and spec:
            converted[key] = [int(v) for v in spec]
            continue
        raise ValueError(f"invalid points spec for '{key}'")
    out["points"] = converted
    return out


def _normalize_point_array(arr: object) -> object:
    if np is None:
        raise ValueError("numpy is required for anime_onnx detector")
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr.transpose(1, 0)
    if arr.ndim == 1 and arr.size >= 6 and arr.size % 2 == 0:
        return arr.reshape((-1, 2))
    raise ValueError(f"unsupported landmark output shape: {getattr(arr, 'shape', None)}")


def _extract_point_from_spec(points_xy: object, spec: object, net_w: int, net_h: int, out_w: int, out_h: int) -> Tuple[float, float] | None:
    if np is None:
        return None
    pts = points_xy
    if not isinstance(pts, np.ndarray):
        return None
    idxs: list[int] = []
    if isinstance(spec, int):
        idxs = [int(spec)]
    elif isinstance(spec, list):
        idxs = [int(v) for v in spec]
    else:
        return None
    valid = [i for i in idxs if 0 <= i < int(pts.shape[0])]
    if not valid:
        return None
    xy = pts[valid, :2].astype("float32")
    x = float(xy[:, 0].mean())
    y = float(xy[:, 1].mean())
    # Auto scale from normalized if needed.
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= float(net_w - 1)
        y *= float(net_h - 1)
    sx = float(out_w) / float(max(1, net_w))
    sy = float(out_h) / float(max(1, net_h))
    return (x * sx, y * sy)


class AnchorDetector(ABC):
    @abstractmethod
    def detect(self, source: Image.Image, size: Tuple[int, int]) -> DetectorResult:
        raise NotImplementedError


class AutoHeuristicDetector(AnchorDetector):
    def __init__(self, seed_anchors: AnchorMap | None = None) -> None:
        self._seed_anchors = seed_anchors or {}

    def detect(self, source: Image.Image, size: Tuple[int, int]) -> DetectorResult:
        det_debug: Dict[str, object] = {}
        _, _, anchors = make_auto_masks(
            source,
            anchors=self._seed_anchors if self._seed_anchors else None,
            debug_out=det_debug,
        )
        anchors = _validate_anchor_map(anchors, size)
        confidence = 0.70 if len(anchors) == 3 else 0.35
        return DetectorResult(
            anchors=anchors,
            detector_name="auto_heuristic",
            confidence=confidence,
            meta={"seeded": bool(self._seed_anchors), "debug": det_debug},
        )


class AnchorsFileDetector(AnchorDetector):
    def __init__(self, anchors_file: Path) -> None:
        self._anchors_file = anchors_file

    def detect(self, source: Image.Image, size: Tuple[int, int]) -> DetectorResult:
        anchors = load_anchors(self._anchors_file, size)
        confidence = 0.98 if len(anchors) == 3 else 0.60
        return DetectorResult(
            anchors=anchors,
            detector_name="anchors_file",
            confidence=confidence,
            meta={"path": str(self._anchors_file.as_posix())},
        )


class ReplayTrackDetector(AnchorDetector):
    def __init__(self, track_file: Path) -> None:
        self._track_file = track_file

    def detect(self, source: Image.Image, size: Tuple[int, int]) -> DetectorResult:
        anchors = load_replay_track_anchors(self._track_file, size)
        confidence = 0.85 if len(anchors) == 3 else 0.45
        return DetectorResult(
            anchors=anchors,
            detector_name="replay_track",
            confidence=confidence,
            meta={"path": str(self._track_file.as_posix())},
        )


class AnimeOnnxDetector(AnchorDetector):
    def __init__(self, model_file: Path, points_map_file: Path) -> None:
        if ort is None or np is None:
            raise ValueError("detector anime_onnx requires onnxruntime and numpy")
        if not model_file.exists():
            raise ValueError(f"anime_onnx model not found: {model_file.as_posix()}")
        if not points_map_file.exists():
            raise ValueError(f"anime_onnx points map not found: {points_map_file.as_posix()}")
        self._model_file = model_file
        self._points_map_file = points_map_file
        self._points_map = load_anime_points_map(points_map_file)
        self._session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])  # type: ignore[arg-type]
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def detect(self, source: Image.Image, size: Tuple[int, int]) -> DetectorResult:
        assert np is not None
        src = source.convert("RGB")
        in_meta = self._session.get_inputs()[0]
        shape = list(in_meta.shape)
        # Expected NCHW; fallback to 256 when dynamic.
        net_h = 256
        net_w = 256
        if len(shape) >= 4:
            h_raw = shape[-2]
            w_raw = shape[-1]
            if isinstance(h_raw, int) and h_raw > 0:
                net_h = int(h_raw)
            if isinstance(w_raw, int) and w_raw > 0:
                net_w = int(w_raw)
        resized = src.resize((net_w, net_h), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))[None, ...]
        outs = self._session.run(self._output_names, {self._input_name: chw})

        out_name = self._points_map.get("output_name")
        out_idx = 0
        if isinstance(out_name, str) and out_name in self._output_names:
            out_idx = self._output_names.index(out_name)
        points_xy = _normalize_point_array(outs[out_idx])

        anchors: AnchorMap = {}
        points_spec = self._points_map.get("points")
        if isinstance(points_spec, Mapping):
            for key in ANCHOR_KEYS:
                spec = points_spec.get(key)
                if spec is None:
                    continue
                pt = _extract_point_from_spec(points_xy, spec, net_w=net_w, net_h=net_h, out_w=size[0], out_h=size[1])
                if pt is not None:
                    anchors[key] = pt
        anchors = _validate_anchor_map(anchors, size)
        confidence = 0.93 if len(anchors) == 3 else 0.55
        return DetectorResult(
            anchors=anchors,
            detector_name="anime_onnx",
            confidence=confidence,
            meta={
                "model": str(self._model_file.as_posix()),
                "points_map": str(self._points_map_file.as_posix()),
                "output_names": self._output_names,
                "net_input_wh": {"w": int(net_w), "h": int(net_h)},
            },
        )


def detect_anime_cv_anchors(source: Image.Image, size: Tuple[int, int]) -> Tuple[AnchorMap, Dict[str, object]]:
    if cv2 is None or np is None:
        raise ValueError("detector anime_cv requires opencv-python and numpy")

    rgba = source.convert("RGBA")
    w, h = size
    arr = np.asarray(rgba, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    alpha_mask = (alpha > 8).astype(np.uint8) * 255

    bx, by, bw, bh = cv2.boundingRect(alpha_mask)
    hx0 = int(max(0, bx + bw * 0.18))
    hy0 = int(max(0, by + bh * 0.05))
    hx1 = int(min(w, bx + bw * 0.82))
    hy1 = int(min(h, by + bh * 0.72))
    hw = max(1, hx1 - hx0)
    hh = max(1, hy1 - hy0)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    dark = 255.0 - gray.astype(np.float32)
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)
    red_dom = np.maximum(0.0, red - np.maximum(green, blue))

    eye_like = dark * 0.38 + grad * 0.72 + sat * 0.20 - red_dom * 0.20
    eye_like = np.maximum(eye_like, 0.0)
    eye_win = np.zeros_like(eye_like, dtype=np.float32)
    ex0 = int(max(0, hx0 + hw * 0.06))
    ex1 = int(min(w, hx0 + hw * 0.98))
    ey0 = int(max(0, hy0 + hh * 0.22))
    ey1 = int(min(h, hy0 + hh * 0.56))
    eye_win[ey0:ey1, ex0:ex1] = 1.0
    eye_like *= eye_win
    eye_like[alpha_mask == 0] = 0.0
    eye_blur = cv2.GaussianBlur(eye_like, (0, 0), 3.0)

    # Find local maxima.
    dil = cv2.dilate(eye_blur, np.ones((9, 9), np.uint8))
    peaks_mask = (eye_blur >= dil - 1e-6) & (eye_blur > float(np.mean(eye_blur[ey0:ey1, ex0:ex1]) * 1.15 + 8.0))
    ys, xs = np.where(peaks_mask)
    peaks = [(float(eye_blur[y, x]), int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())]
    peaks.sort(key=lambda t: t[0], reverse=True)
    peaks = peaks[:40]

    # Fallback centers in face-local coordinates.
    left_eye = (hx0 + hw * 0.34, hy0 + hh * 0.42)
    right_eye = (hx0 + hw * 0.70, hy0 + hh * 0.45)

    best_pair = None
    best_score = -1e18
    for i in range(min(24, len(peaks))):
        si, xi, yi = peaks[i]
        for j in range(i + 1, min(24, len(peaks))):
            sj, xj, yj = peaks[j]
            lx, ly = (xi, yi) if xi <= xj else (xj, yj)
            rx, ry = (xj, yj) if xi <= xj else (xi, yi)
            gap = float(rx - lx)
            if gap < hw * 0.20 or gap > hw * 0.78:
                continue
            y_delta = abs(float(ry - ly))
            if y_delta > hh * 0.20:
                continue
            center_x = (lx + rx) * 0.5
            center_target = hx0 + hw * 0.62
            center_pen = abs(center_x - center_target) / max(1.0, hw * 0.30)
            gap_pen = abs(gap - hw * 0.50) / max(1.0, hw * 0.24)
            # Right-heavy portraits: keep right eye on right half.
            right_pen = max(0.0, (hx0 + hw * 0.58) - rx) / max(1.0, hw)
            score = si + sj - 30.0 * center_pen - 22.0 * gap_pen - 45.0 * right_pen
            if score > best_score:
                best_score = score
                best_pair = (lx, ly, rx, ry)

    if best_pair is not None:
        lx, ly, rx, ry = best_pair
        # Refine by weighted centroid around each peak.
        def refine(cx: int, cy: int) -> Tuple[float, float, float]:
            rx0 = int(max(ex0, cx - hw * 0.12))
            rx1 = int(min(ex1, cx + hw * 0.12))
            ry0 = int(max(ey0, cy - hh * 0.12))
            ry1 = int(min(ey1, cy + hh * 0.12))
            patch = eye_blur[ry0:ry1, rx0:rx1]
            if patch.size == 0:
                return (float(cx), float(cy), 0.0)
            yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
            ww = np.maximum(patch, 0.0)
            sw = float(np.sum(ww))
            if sw <= 1e-6:
                return (float(cx), float(cy), 0.0)
            xf = float(np.sum(xx * ww) / sw)
            yf = float(np.sum(yy * ww) / sw)
            return (xf, yf, sw)

        lxf, lyf, lw = refine(int(lx), int(ly))
        rxf, ryf, rw = refine(int(rx), int(ry))
        left_eye = (lxf, lyf)
        right_eye = (rxf, ryf)
    else:
        lw, rw = 0.0, 0.0

    # Mouth in face-local coordinates from eye roll.
    eye_dx = float(right_eye[0] - left_eye[0])
    eye_dy = float(right_eye[1] - left_eye[1])
    eye_mid_x = float((left_eye[0] + right_eye[0]) * 0.5)
    eye_mid_y = float((left_eye[1] + right_eye[1]) * 0.5)
    roll = math.atan2(eye_dy, max(1e-6, eye_dx))
    vx, vy = math.cos(roll), math.sin(roll)
    nx, ny = -vy, vx
    if ny < 0:
        nx, ny = -nx, -ny
    eye_gap = max(8.0, math.hypot(eye_dx, eye_dy))

    mouth_like = red_dom * 0.56 + sat * 0.28 + grad * 0.50 + dark * 0.15
    mouth_like = np.maximum(mouth_like, 0.0)
    my0 = int(max(0, hy0 + hh * 0.48))
    my1 = int(min(h, hy0 + hh * 0.90))
    mx0 = int(max(0, hx0 + hw * 0.12))
    mx1 = int(min(w, hx0 + hw * 0.90))
    mouth_like[:my0, :] = 0.0
    mouth_like[my1:, :] = 0.0
    mouth_like[:, :mx0] = 0.0
    mouth_like[:, mx1:] = 0.0
    mouth_like[alpha_mask == 0] = 0.0

    yy, xx = np.mgrid[0:h, 0:w]
    relx = xx.astype(np.float32) - eye_mid_x
    rely = yy.astype(np.float32) - eye_mid_y
    along = relx * vx + rely * vy
    depth = relx * nx + rely * ny
    geom_mask = (
        (np.abs(along) <= eye_gap * 0.60)
        & (depth >= hh * 0.10)
        & (depth <= hh * 0.54)
    )
    mouth_like *= geom_mask.astype(np.float32)
    mouth_blur = cv2.GaussianBlur(mouth_like, (0, 0), 2.4)
    swm = float(np.sum(mouth_blur))
    if swm > 1e-6:
        mx = float(np.sum(xx * mouth_blur) / swm)
        my = float(np.sum(yy * mouth_blur) / swm)
    else:
        mx = float(eye_mid_x + nx * hh * 0.27)
        my = float(eye_mid_y + ny * hh * 0.27)

    anchors: AnchorMap = {
        "left_eye": (float(left_eye[0]), float(left_eye[1])),
        "right_eye": (float(right_eye[0]), float(right_eye[1])),
        "mouth": (float(mx), float(my)),
    }
    anchors = _validate_anchor_map(anchors, size)
    debug = {
        "head_box": {"x0": int(hx0), "y0": int(hy0), "x1": int(hx1), "y1": int(hy1), "w": int(hw), "h": int(hh)},
        "eye_window": {"x0": int(ex0), "y0": int(ey0), "x1": int(ex1), "y1": int(ey1)},
        "peak_count": int(len(peaks)),
        "eye_pair_score": round(float(best_score), 4) if best_pair is not None else None,
        "left_mass": round(float(lw), 3),
        "right_mass": round(float(rw), 3),
    }
    return anchors, debug


class AnimeCvDetector(AnchorDetector):
    def detect(self, source: Image.Image, size: Tuple[int, int]) -> DetectorResult:
        anchors, debug = detect_anime_cv_anchors(source, size)
        confidence = 0.88 if len(anchors) == 3 else 0.40
        return DetectorResult(
            anchors=anchors,
            detector_name="anime_cv",
            confidence=confidence,
            meta={"debug": debug},
        )


class NullDetector(AnchorDetector):
    def detect(self, source: Image.Image, size: Tuple[int, int]) -> DetectorResult:
        return DetectorResult(anchors={}, detector_name="none", confidence=0.0, meta={})


def make_detector(
    detector_kind: str,
    anchors_file: str,
    replay_track_file: str,
    anime_model_file: str,
    anime_points_map_file: str,
    seed_anchors: AnchorMap | None = None,
) -> AnchorDetector:
    kind = detector_kind.strip().lower()
    if kind == "anchors":
        if not anchors_file:
            raise ValueError("--detector anchors requires --anchors-file")
        return AnchorsFileDetector(Path(anchors_file))
    if kind == "replay":
        if not replay_track_file:
            raise ValueError("--detector replay requires --detector-track-file")
        return ReplayTrackDetector(Path(replay_track_file))
    if kind == "anime_onnx":
        if not anime_model_file:
            raise ValueError("--detector anime_onnx requires --detector-model-file")
        if not anime_points_map_file:
            raise ValueError("--detector anime_onnx requires --detector-points-map-file")
        return AnimeOnnxDetector(Path(anime_model_file), Path(anime_points_map_file))
    if kind == "anime_cv":
        return AnimeCvDetector()
    if kind == "none":
        return NullDetector()
    return AutoHeuristicDetector(seed_anchors=seed_anchors)


def make_auto_masks(
    source: Image.Image,
    anchors: AnchorMap | None = None,
    debug_out: Dict[str, object] | None = None,
) -> Tuple[Dict[str, Image.Image], Dict[str, Image.Image], AnchorMap]:
    size = source.size
    b = alpha_bounds(source)
    x0, y0, x1, y1 = b.x0, b.y0, b.x1, b.y1
    w, h = b.w, b.h

    def rel_box(rx0: float, ry0: float, rx1: float, ry1: float) -> Tuple[int, int, int, int]:
        box = (
            int(x0 + w * rx0),
            int(y0 + h * ry0),
            int(x0 + w * rx1),
            int(y0 + h * ry1),
        )
        return clamp_box(box, source.width, source.height)

    layer_masks: Dict[str, Image.Image] = {}
    layer_masks["base"] = mask_rect(size, (0, 0, source.width, source.height))
    layer_masks["body"] = mask_rect(size, rel_box(0.05, 0.58, 0.95, 1.00))
    # Bust-up friendly head zone to keep eyes/mouth search windows on actual face.
    layer_masks["head"] = mask_rect(size, rel_box(0.18, 0.05, 0.82, 0.72))
    layer_masks["hair_back"] = mask_rect(size, rel_box(0.10, 0.00, 0.90, 0.52))
    layer_masks["hair_front"] = mask_rect(size, rel_box(0.18, 0.04, 0.82, 0.58))

    # Build mouth masks relative to HEAD region (not full body bbox),
    # so the auto mouth does not drift to neck/chest on bust-up images.
    head_box = rel_box(0.18, 0.05, 0.82, 0.72)
    hx0, hy0, hx1, hy1 = head_box
    hw = max(1, hx1 - hx0)
    hh = max(1, hy1 - hy0)
    auto_debug: Dict[str, object] = {
        "head_box": {"x0": int(hx0), "y0": int(hy0), "x1": int(hx1), "y1": int(hy1), "w": int(hw), "h": int(hh)}
    }

    def head_rel_box(rx0: float, ry0: float, rx1: float, ry1: float) -> Tuple[int, int, int, int]:
        box = (
            int(hx0 + hw * rx0),
            int(hy0 + hh * ry0),
            int(hx0 + hw * rx1),
            int(hy0 + hh * ry1),
        )
        return clamp_box(box, source.width, source.height)

    def estimate_eye_anchor_y() -> float:
        """Estimate eye line Y inside head region.

        Heuristic:
        - search upper-mid head area
        - prefer dark + saturated pixels (lashes/iris)
        - penalize strong red dominance to avoid mouth-like regions
        """
        rgb = source.convert("RGB")
        px = rgb.load()
        wx0 = int(hx0 + hw * 0.20)
        wx1 = int(hx0 + hw * 0.80)
        wy0 = int(hy0 + hh * 0.22)
        wy1 = int(hy0 + hh * 0.56)
        wx0, wy0, wx1, wy1 = clamp_box((wx0, wy0, wx1, wy1), source.width, source.height)
        samples = []
        for yy in range(wy0, wy1):
            for xx in range(wx0, wx1):
                r, g, b = px[xx, yy]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                mx = max(r, g, b)
                mn = min(r, g, b)
                sat = mx - mn
                dark = 255.0 - lum
                red_dom = max(0.0, r - max(g, b))
                score = dark * 0.55 + sat * 0.35 - red_dom * 0.25
                if score > 20:
                    samples.append((score, yy))
        auto_debug["eye_anchor_y_candidates"] = int(len(samples))
        if not samples:
            return hy0 + hh * 0.48
        samples.sort(key=lambda t: t[0], reverse=True)
        top_k = max(300, len(samples) // 20)  # top 5%
        chosen = samples[:top_k]
        sw = sum(s for s, _ in chosen)
        if sw <= 1e-6:
            return hy0 + hh * 0.48
        cy = sum(s * y for s, y in chosen) / sw
        ry = (cy - hy0) / max(1.0, hh)
        if not (0.26 <= ry <= 0.56):
            return hy0 + hh * 0.42
        return cy

    def estimate_eye_pair(eye_y: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        rgb = source.convert("RGB")
        px = rgb.load()
        y0w = int(max(hy0, eye_y - hh * 0.14))
        y1w = int(min(hy1, eye_y + hh * 0.14))
        alpha = source.getchannel("A")
        apx = alpha.load()

        def eye_saliency(xx: int, yy: int) -> float:
            r, g, b = px[xx, yy]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            mx = max(r, g, b)
            mn = min(r, g, b)
            sat = mx - mn
            dark = 255.0 - lum
            red_dom = max(0.0, r - max(g, b))
            lum_u = 0.299 * px[xx, yy - 1][0] + 0.587 * px[xx, yy - 1][1] + 0.114 * px[xx, yy - 1][2]
            lum_d = 0.299 * px[xx, yy + 1][0] + 0.587 * px[xx, yy + 1][1] + 0.114 * px[xx, yy + 1][2]
            lum_l = 0.299 * px[xx - 1, yy][0] + 0.587 * px[xx - 1, yy][1] + 0.114 * px[xx - 1, yy][2]
            lum_r = 0.299 * px[xx + 1, yy][0] + 0.587 * px[xx + 1, yy][1] + 0.114 * px[xx + 1, yy][2]
            edge = abs(lum_u - lum_d) * 0.68 + abs(lum_l - lum_r) * 0.32
            iris_like = max(0.0, 1.0 - abs(lum - 150.0) / 95.0) * max(0.0, 1.0 - sat / 90.0)
            y_penalty = abs(float(yy) - float(eye_y))
            return dark * 0.34 + sat * 0.18 + edge * 0.64 + iris_like * 32.0 - red_dom * 0.24 - y_penalty * 1.15

        # Face center: combine skin centroid and eye-band centroid.
        sx0 = int(hx0 + hw * 0.14)
        sx1 = int(hx0 + hw * 0.92)
        sy0 = int(hy0 + hh * 0.20)
        sy1 = int(hy0 + hh * 0.64)
        sx0, sy0, sx1, sy1 = clamp_box((sx0, sy0, sx1, sy1), source.width, source.height)
        skin_sum = 0.0
        skin_x = 0.0
        for yy in range(sy0, sy1, 2):
            for xx in range(sx0, sx1, 2):
                if apx[xx, yy] < 8:
                    continue
                r, g, b = px[xx, yy]
                mx = max(r, g, b)
                mn = min(r, g, b)
                sat = mx - mn
                if r < 70 or g < 50 or b < 35:
                    continue
                if not (r >= g >= b):
                    continue
                if sat > 95:
                    continue
                w_skin = (r - b + 8.0) * (1.0 - sat / 120.0)
                if w_skin <= 0.0:
                    continue
                skin_sum += w_skin
                skin_x += w_skin * float(xx)
        skin_cx = (skin_x / skin_sum) if skin_sum > 1e-6 else (hx0 + hw * 0.54)

        bx0 = int(hx0 + hw * 0.10)
        bx1 = int(hx0 + hw * 0.90)
        band_sum = 0.0
        band_x = 0.0
        for yy in range(y0w, y1w, 2):
            for xx in range(bx0, bx1, 2):
                if apx[xx, yy] < 8:
                    continue
                v = eye_saliency(xx, yy)
                if v > 6.0:
                    band_sum += v
                    band_x += v * float(xx)
        band_cx = (band_x / band_sum) if band_sum > 1e-6 else (hx0 + hw * 0.52)
        face_cx = max(hx0 + hw * 0.22, min(hx0 + hw * 0.78, band_cx * 0.88 + skin_cx * 0.12))
        auto_debug["face_center_x"] = round(float(face_cx), 3)
        auto_debug["face_center_components"] = {"skin_cx": round(float(skin_cx), 3), "band_cx": round(float(band_cx), 3)}

        # Collect column energies in a wide eye window.
        # Off-center portraits and perspective shots break center-locked windows.
        wx0 = int(hx0 + hw * 0.06)
        wx1 = int(hx1 - hw * 0.02)
        min_span = int(hw * 0.60)
        if wx1 - wx0 < min_span:
            mid = (hx0 + hx1) * 0.5
            wx0 = int(max(hx0 + hw * 0.04, mid - min_span * 0.5))
            wx1 = int(min(hx1 - hw * 0.02, mid + min_span * 0.5))
        auto_debug["eye_window"] = {"x0": int(wx0), "x1": int(wx1), "y0": int(y0w), "y1": int(y1w)}

        cols: list[float] = []
        for xx in range(wx0, wx1):
            s = 0.0
            for yy in range(y0w + 1, max(y0w + 2, y1w - 1), 2):
                if apx[xx, yy] < 8:
                    continue
                v = eye_saliency(xx, yy)
                if v > 10.0:
                    s += v
            cols.append(s)
        if not cols:
            auto_debug["eye_fallback_reason"] = "no_column_energy"
            fallback_y = eye_y
            return ((hx0 + hw * 0.42, fallback_y), (hx0 + hw * 0.72, fallback_y))

        # Moving-average smoothing.
        rad = 4
        smooth: list[float] = []
        for i in range(len(cols)):
            j0 = max(0, i - rad)
            j1 = min(len(cols), i + rad + 1)
            smooth.append(sum(cols[j0:j1]) / max(1, j1 - j0))

        mean_s = sum(smooth) / max(1, len(smooth))
        peaks: list[Tuple[float, int]] = []
        for i in range(1, len(smooth) - 1):
            v = smooth[i]
            if v < mean_s * 1.10:
                continue
            if v >= smooth[i - 1] and v >= smooth[i + 1]:
                peaks.append((v, i))
        peaks.sort(key=lambda t: t[0], reverse=True)
        auto_debug["eye_peak_count"] = int(len(peaks))

        # Peak-derived center is often more reliable than skin centroid on anime portraits.
        top_for_center = peaks[: min(12, len(peaks))]
        if top_for_center:
            sw_peak = sum(max(0.0, float(v)) for v, _ in top_for_center)
            if sw_peak > 1e-6:
                peak_center_x = sum(float(v) * float(wx0 + idx) for v, idx in top_for_center) / sw_peak
            else:
                peak_center_x = float(face_cx)
        else:
            peak_center_x = float(face_cx)
        pair_center_target = max(hx0 + hw * 0.38, min(hx0 + hw * 0.84, peak_center_x * 0.62 + face_cx * 0.38))
        auto_debug["peak_center_x"] = round(float(peak_center_x), 3)
        auto_debug["pair_center_target_x"] = round(float(pair_center_target), 3)

        best_pair: Tuple[int, int] | None = None
        best_pair_score = -1e9
        max_pairs = min(24, len(peaks))
        gap_min = hw * 0.24
        gap_max = hw * 0.76
        gap_target = hw * 0.49
        gap_sigma = max(1.0, hw * 0.11)
        center_guard = hw * 0.11
        for i in range(max_pairs):
            vi, pi = peaks[i]
            for j in range(i + 1, max_pairs):
                vj, pj = peaks[j]
                li = min(pi, pj)
                ri = max(pi, pj)
                gap = float(ri - li)
                if not (gap_min <= gap <= gap_max):
                    continue
                left_x = float(wx0 + li)
                right_x = float(wx0 + ri)
                if left_x > face_cx - center_guard:
                    continue
                if right_x < face_cx + center_guard:
                    continue
                cx = wx0 + (li + ri) * 0.5
                center_pen = abs(cx - pair_center_target) / max(1.0, hw * 0.30)
                center_score = max(0.0, 1.0 - center_pen) * 12.0
                gap_err = (gap - gap_target) / gap_sigma
                gap_score = math.exp(-(gap_err * gap_err)) * 36.0
                collapse_pen = max(0.0, hw * 0.34 - gap) * 0.42
                # Right-biased portraits: reject pairs that stay too far left.
                right_floor_pen = max(0.0, (hx0 + hw * 0.56) - right_x) * 0.60
                left_floor_pen = max(0.0, (hx0 + hw * 0.18) - left_x) * 0.30
                score = vi + vj + gap_score + center_score - collapse_pen - right_floor_pen - left_floor_pen
                if score > best_pair_score:
                    best_pair_score = score
                    best_pair = (li, ri)

        if best_pair is None:
            auto_debug["eye_fallback_reason"] = "no_valid_peak_pair"
            fallback_y = eye_y
            return ((hx0 + hw * 0.42, fallback_y), (hx0 + hw * 0.72, fallback_y))

        left_idx, right_idx = best_pair
        left_x0 = int(max(wx0, wx0 + left_idx - hw * 0.11))
        left_x1 = int(min(wx1, wx0 + left_idx + hw * 0.11))
        right_x0 = int(max(wx0, wx0 + right_idx - hw * 0.11))
        right_x1 = int(min(wx1, wx0 + right_idx + hw * 0.11))

        def weighted_quantile(samples: list[Tuple[float, float]], q: float) -> float:
            if not samples:
                return float(eye_y)
            qq = max(0.0, min(1.0, float(q)))
            sorted_samples = sorted(samples, key=lambda t: t[0])
            total_w = sum(max(0.0, w) for _, w in sorted_samples)
            if total_w <= 1e-6:
                return float(sorted_samples[len(sorted_samples) // 2][0])
            target = total_w * qq
            acc = 0.0
            for yv, wv in sorted_samples:
                acc += max(0.0, wv)
                if acc >= target:
                    return float(yv)
            return float(sorted_samples[-1][0])

        def refine_eye(cx0: int, cx1: int) -> Tuple[float, float, float, float]:
            x_start = max(wx0 + 1, cx0)
            x_end = min(wx1 - 1, cx1)
            if x_start >= x_end:
                return ((cx0 + cx1) * 0.5, eye_y, 0.0, eye_y)

            col_energy: list[Tuple[int, float, float]] = []
            sw = sx = sy = 0.0
            y_samples: list[Tuple[float, float]] = []
            for xx in range(x_start, x_end, 2):
                cs = 0.0
                csy = 0.0
                for yy in range(y0w + 1, max(y0w + 2, y1w - 1), 2):
                    if apx[xx, yy] < 8:
                        continue
                    v = eye_saliency(xx, yy)
                    if v <= 10.0:
                        continue
                    cs += v
                    csy += v * float(yy)
                    y_samples.append((float(yy), float(v)))
                col_energy.append((xx, cs, csy))
                sw += cs
                sx += cs * float(xx)
                sy += csy

            if sw <= 1e-6 or not col_energy:
                return ((cx0 + cx1) * 0.5, eye_y, 0.0, eye_y)

            x_weighted = sx / sw
            y_weighted = sy / sw
            y_lower = weighted_quantile(y_samples, 0.64)

            max_col = max(cs for _, cs, _ in col_energy)
            mean_col = sw / float(len(col_energy))
            support_thr = max(max_col * 0.42, mean_col * 1.12)
            support = [(xx, cs, csy) for xx, cs, csy in col_energy if cs >= support_thr]

            if len(support) >= 3:
                sx0 = float(support[0][0])
                sx1 = float(support[-1][0])
                span_w = sx1 - sx0
                support_sw = sum(cs for _, cs, _ in support)
                if support_sw > 1e-6 and span_w <= hw * 0.22:
                    span_mid = (sx0 + sx1) * 0.5
                    y_support = sum(csy for _, _, csy in support) / support_sw
                    x_center = x_weighted * 0.72 + span_mid * 0.28
                    y_center = y_weighted * 0.52 + y_support * 0.24 + y_lower * 0.24
                    return (x_center, y_center, sw, y_lower)

            y_center = y_weighted * 0.70 + y_lower * 0.30
            return (x_weighted, y_center, sw, y_lower)

        lx, ly, lw, ly_q = refine_eye(left_x0, left_x1)
        rx, ry, rw, ry_q = refine_eye(right_x0, right_x1)
        if rw < lw * 0.92:
            ry = ry * 0.50 + ry_q * 0.50
        elif lw < rw * 0.92:
            ly = ly * 0.50 + ly_q * 0.50

        # Guard against "outer-hair pull" where one eye drifts too far from face center.
        d_left = max(0.0, float(face_cx - lx))
        d_right = max(0.0, float(rx - face_cx))
        if d_left > 1e-6 and d_right > 1e-6:
            max_ratio = 1.72
            min_ratio = 1.0 / max_ratio
            ratio = d_left / d_right
            if ratio > max_ratio:
                target_left = face_cx - d_right * max_ratio
                blend = min(1.0, (ratio - max_ratio) / 0.60)
                lx = lx * (1.0 - 0.55 * blend) + target_left * (0.55 * blend)
            elif ratio < min_ratio:
                target_right = face_cx + d_left * max_ratio
                blend = min(1.0, (min_ratio - ratio) / 0.60)
                rx = rx * (1.0 - 0.55 * blend) + target_right * (0.55 * blend)

        # Roll-aware spread calibration: anime eyelid lines often under-estimate tilt.
        eye_mid_y_local = (ly + ry) * 0.5
        dy_meas = float(ry - ly)
        dy_quant = float(ry_q - ly_q)
        target_dy = dy_meas * 0.55 + dy_quant * 0.45
        target_dy *= 1.18
        dy_cap = hh * 0.22
        if target_dy > dy_cap:
            target_dy = dy_cap
        elif target_dy < -dy_cap:
            target_dy = -dy_cap
        ly = eye_mid_y_local - target_dy * 0.5
        ry = eye_mid_y_local + target_dy * 0.5

        # Re-center X by side mass imbalance to suppress left-side hair pull.
        side_sw = max(1e-6, float(lw + rw))
        side_imb = (float(lw) - float(rw)) / side_sw
        x_shift = max(-hw * 0.012, min(hw * 0.012, side_imb * hw * 0.060))
        lx += x_shift
        rx += x_shift

        left_pt = (lx, ly)
        right_pt = (rx, ry)

        mass_left = float(lw)
        mass_right = float(rw)
        auto_debug["eye_cluster"] = {
            "left_mass": round(float(mass_left), 3),
            "right_mass": round(float(mass_right), 3),
            "left": {"x": round(float(left_pt[0]), 3), "y": round(float(left_pt[1]), 3)},
            "right": {"x": round(float(right_pt[0]), 3), "y": round(float(right_pt[1]), 3)},
            "left_y_quantile": round(float(ly_q), 3),
            "right_y_quantile": round(float(ry_q), 3),
            "dy_measured": round(float(dy_meas), 3),
            "dy_quantile": round(float(dy_quant), 3),
            "dy_target": round(float(target_dy), 3),
            "x_shift_from_mass_imbalance": round(float(x_shift), 3),
        }

        max_dy = hh * 0.10
        if abs(left_pt[1] - right_pt[1]) > max_dy:
            avg_y = (left_pt[1] + right_pt[1]) * 0.5
            left_pt = (left_pt[0], avg_y)
            right_pt = (right_pt[0], avg_y)
        return (left_pt, right_pt)

    def estimate_mouth_anchor(
        eye_mid_x: float,
        eye_mid_y: float,
        vx: float,
        vy: float,
        nx: float,
        ny: float,
        eye_gap: float,
    ) -> Tuple[float, float]:
        """Estimate mouth center using roll-aligned geometry prior."""
        rgb = source.convert("RGB")
        px = rgb.load()
        alpha = source.getchannel("A")
        apx = alpha.load()
        wx0 = int(hx0 + hw * 0.12)
        wx1 = int(hx0 + hw * 0.88)
        wy0 = int(hy0 + hh * 0.50)
        wy1 = int(hy0 + hh * 0.86)
        wx0, wy0, wx1, wy1 = clamp_box((wx0, wy0, wx1, wy1), source.width, source.height)
        depth_prior = hh * 0.25
        samples: list[Tuple[float, float, float]] = []
        for yy in range(wy0 + 1, max(wy0 + 2, wy1 - 1), 2):
            for xx in range(wx0 + 1, max(wx0 + 2, wx1 - 1), 2):
                if apx[xx, yy] < 8:
                    continue
                relx = float(xx) - eye_mid_x
                rely = float(yy) - eye_mid_y
                along = relx * vx + rely * vy
                depth = relx * nx + rely * ny
                if depth < hh * 0.08 or depth > hh * 0.60:
                    continue
                if abs(along) > eye_gap * 0.65:
                    continue
                r, g, b = px[xx, yy]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                mx = max(r, g, b)
                mn = min(r, g, b)
                sat = mx - mn
                redness = r - (g + b) * 0.5
                dark = 255.0 - lum
                lum_u = 0.299 * px[xx, yy - 1][0] + 0.587 * px[xx, yy - 1][1] + 0.114 * px[xx, yy - 1][2]
                lum_d = 0.299 * px[xx, yy + 1][0] + 0.587 * px[xx, yy + 1][1] + 0.114 * px[xx, yy + 1][2]
                lum_l = 0.299 * px[xx - 1, yy][0] + 0.587 * px[xx - 1, yy][1] + 0.114 * px[xx - 1, yy][2]
                lum_r = 0.299 * px[xx + 1, yy][0] + 0.587 * px[xx + 1, yy][1] + 0.114 * px[xx + 1, yy][2]
                edge = abs(lum_u - lum_d) * 0.65 + abs(lum_l - lum_r) * 0.35
                depth_pen = abs(depth - depth_prior)
                along_pen = abs(along)
                score = (
                    redness * 0.48
                    + dark * 0.18
                    + sat * 0.26
                    + edge * 0.72
                    - depth_pen * 0.40
                    - along_pen * 0.30
                )
                if score > 24.0:
                    samples.append((score, float(xx), float(yy)))
        auto_debug["mouth_candidate_count"] = int(len(samples))
        if not samples:
            auto_debug["mouth_fallback_reason"] = "low_candidates"
            return (eye_mid_x + depth_prior * nx, eye_mid_y + depth_prior * ny)
        samples.sort(key=lambda t: t[0], reverse=True)
        top_k = max(120, len(samples) // 12)
        chosen = samples[:top_k]
        sw = sum(s for s, _, _ in chosen)
        if sw <= 1e-6:
            return (eye_mid_x + depth_prior * nx, eye_mid_y + depth_prior * ny)
        cx = sum(s * x for s, x, _ in chosen) / sw
        cy = sum(s * y for s, _, y in chosen) / sw
        auto_debug["mouth_raw"] = {"x": round(float(cx), 3), "y": round(float(cy), 3)}
        return (cx, cy)

    eye_anchor_left = anchors.get("left_eye") if anchors else None
    eye_anchor_right = anchors.get("right_eye") if anchors else None
    mouth_anchor = anchors.get("mouth") if anchors else None

    if eye_anchor_left and eye_anchor_right:
        if eye_anchor_left[0] <= eye_anchor_right[0]:
            left_eye = (eye_anchor_left[0], eye_anchor_left[1])
            right_eye = (eye_anchor_right[0], eye_anchor_right[1])
        else:
            left_eye = (eye_anchor_right[0], eye_anchor_right[1])
            right_eye = (eye_anchor_left[0], eye_anchor_left[1])
    else:
        eye_cy = estimate_eye_anchor_y()
        left_eye, right_eye = estimate_eye_pair(eye_cy)
    eye_lx, eye_ly = left_eye
    eye_rx, eye_ry = right_eye
    eye_mid_x = (eye_lx + eye_rx) * 0.5
    eye_mid_y = (eye_ly + eye_ry) * 0.5
    eye_gap = max(8.0, eye_rx - eye_lx)
    eye_dx = eye_rx - eye_lx
    eye_dy = eye_ry - eye_ly
    if abs(eye_dy) > eye_gap * 0.30:
        eye_ly = eye_mid_y
        eye_ry = eye_mid_y
        eye_dy = 0.0
    roll_rad = math.atan2(eye_dy, max(1e-6, eye_dx))
    max_roll_rad = math.radians(18.0)
    if roll_rad > max_roll_rad:
        roll_rad = max_roll_rad
    elif roll_rad < -max_roll_rad:
        roll_rad = -max_roll_rad
    # PIL.rotate uses CCW-positive, but image y-axis is downward.
    # Invert sign so mask tilt follows eye-line tilt visually.
    roll_deg = -math.degrees(roll_rad)
    vx = math.cos(roll_rad)
    vy = math.sin(roll_rad)
    nx = -vy
    ny = vx
    if ny < 0.0:
        nx = -nx
        ny = -ny
    if mouth_anchor:
        mcx, mcy = mouth_anchor
    else:
        mcx, mcy = estimate_mouth_anchor(
            eye_mid_x=eye_mid_x,
            eye_mid_y=eye_mid_y,
            vx=vx,
            vy=vy,
            nx=nx,
            ny=ny,
            eye_gap=eye_gap,
        )
    # Constrain mouth in roll-aligned face coordinates (improves tilted/perspective portraits).
    rel_mx = mcx - eye_mid_x
    rel_my = mcy - eye_mid_y
    along_eye = rel_mx * vx + rel_my * vy
    depth = rel_mx * nx + rel_my * ny
    if abs(along_eye) > eye_gap * 0.22:
        along_eye *= 0.45
    depth_prior = hh * 0.27
    if abs(depth - depth_prior) > hh * 0.14:
        depth = depth_prior * 0.68 + depth * 0.32
    # Anime child-like faces often have shorter eye-to-mouth distance.
    eye_ratio = eye_gap / max(1.0, hw)
    child_bias = max(0.0, min(1.0, (eye_ratio - 0.42) / 0.18))
    min_d = hh * (0.10 - 0.03 * child_bias)
    max_d = hh * (0.31 - 0.08 * child_bias)
    depth = max(min_d, min(max_d, depth))
    mcx = eye_mid_x + along_eye * vx + depth * nx
    mcy = eye_mid_y + along_eye * vy + depth * ny
    mcy = min(mcy, hy0 + hh * 0.86)
    mw = max(8, int(hw * 0.24))
    mh = max(5, int(hh * 0.10))

    per_eye_w = max(14, int(min(hw * 0.20, eye_gap * 0.38)))
    per_eye_h = max(6, int(hh * 0.09))
    left_center = (float(eye_lx), float(eye_ly))
    right_center = (float(eye_rx), float(eye_ry))

    eye_open_l = mask_rotated_ellipse(
        size=size,
        center=left_center,
        width=float(per_eye_w),
        height=float(per_eye_h),
        angle_deg=roll_deg,
    )
    eye_open_r = mask_rotated_ellipse(
        size=size,
        center=right_center,
        width=float(per_eye_w),
        height=float(per_eye_h),
        angle_deg=roll_deg,
    )
    layer_masks["eye_open"] = ImageChops.lighter(eye_open_l, eye_open_r)

    eye_half_l = mask_rotated_rect(
        size=size,
        center=left_center,
        width=float(per_eye_w * 0.96),
        height=float(per_eye_h * 0.42),
        angle_deg=roll_deg,
    )
    eye_half_r = mask_rotated_rect(
        size=size,
        center=right_center,
        width=float(per_eye_w * 0.96),
        height=float(per_eye_h * 0.42),
        angle_deg=roll_deg,
    )
    layer_masks["eye_half"] = ImageChops.lighter(eye_half_l, eye_half_r)

    eye_wide_l = mask_rotated_ellipse(
        size=size,
        center=left_center,
        width=float(per_eye_w * 1.10),
        height=float(per_eye_h * 1.24),
        angle_deg=roll_deg,
    )
    eye_wide_r = mask_rotated_ellipse(
        size=size,
        center=right_center,
        width=float(per_eye_w * 1.10),
        height=float(per_eye_h * 1.24),
        angle_deg=roll_deg,
    )
    layer_masks["eye_open_wide"] = ImageChops.lighter(eye_wide_l, eye_wide_r)

    def anchor_box(scale_w: float, scale_h: float, y_bias: float = 0.0) -> Tuple[int, int, int, int]:
        ww = max(4.0, float(mw) * scale_w)
        hh2 = max(3.0, float(mh) * scale_h)
        cx = float(mcx) + float(y_bias * hh) * nx
        cy = float(mcy) + float(y_bias * hh) * ny
        return (cx, cy, ww, hh2)

    mouth_specs = {
        "a": (1.06, 1.12, "ellipse", 0.00),
        "i": (0.62, 0.92, "ellipse", 0.00),
        "u": (0.78, 0.96, "ellipse", 0.02),
        "e": (0.94, 0.96, "ellipse", 0.00),
        "o": (1.12, 1.16, "ellipse", 0.00),
        "n": (0.60, 0.54, "rect", -0.02),
    }
    mouth_masks: Dict[str, Image.Image] = {}
    for key, (sw, sh, shape, yb) in mouth_specs.items():
        cx, cy, ww, hh2 = anchor_box(sw, sh, yb)
        if shape == "ellipse":
            mouth_masks[key] = mask_rotated_ellipse(
                size=size,
                center=(cx, cy),
                width=ww,
                height=hh2,
                angle_deg=roll_deg,
            )
        else:
            mouth_masks[key] = mask_rotated_rect(
                size=size,
                center=(cx, cy),
                width=ww,
                height=hh2,
                angle_deg=roll_deg,
            )

    used_anchors: AnchorMap = {
        "left_eye": (float(eye_lx), float(eye_ly)),
        "right_eye": (float(eye_rx), float(eye_ry)),
        "mouth": (float(mcx), float(mcy)),
    }
    auto_debug["anchors_used"] = {
        "left_eye": {"x": round(float(eye_lx), 3), "y": round(float(eye_ly), 3)},
        "right_eye": {"x": round(float(eye_rx), 3), "y": round(float(eye_ry), 3)},
        "mouth": {"x": round(float(mcx), 3), "y": round(float(mcy), 3)},
    }
    if debug_out is not None:
        debug_out.update(auto_debug)
    return layer_masks, mouth_masks, used_anchors


def build_manifest(manifest_path: Path, out_dir: Path) -> None:
    rel_dir = out_dir.as_posix().lstrip("./")
    manifest = {
        "version": 1,
        "character": out_dir.name,
        "transform": {"scale": 1.0},
        "layers": {name: f"/{rel_dir}/{name}.png" for name in LAYER_NAMES},
        "mouths": {name: f"/{rel_dir}/mouth_{name}.png" for name in MOUTH_NAMES},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def save_mask_guides(
    source: Image.Image,
    guide_dir: Path,
    mouth_masks: Dict[str, Image.Image],
    layer_masks: Dict[str, Image.Image],
    anchors: AnchorMap | None = None,
) -> None:
    guide_dir.mkdir(parents=True, exist_ok=True)

    # 1) per-mask template (white=paint target, black=background)
    for name, mask in mouth_masks.items():
        mask.save(guide_dir / f"mouth_{name}.png")

    # useful optional guides
    for key in ("eye_open", "eye_half", "head", "body", "hair_front"):
        if key in layer_masks:
            layer_masks[key].save(guide_dir / f"{key}.png")

    # 2) single preview sheet with overlays
    preview = source.convert("RGBA").copy()
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    color_map = {
        "a": (255, 120, 120, 140),
        "i": (120, 180, 255, 140),
        "u": (140, 255, 180, 140),
        "e": (255, 200, 120, 140),
        "o": (220, 150, 255, 140),
        "n": (255, 255, 255, 140),
    }
    for name, mask in mouth_masks.items():
        box = mask.getbbox()
        if box is None:
            continue
        od.rectangle(box, outline=color_map.get(name, (255, 255, 255, 120)), width=2)
        od.text((box[0], max(0, box[1] - 14)), name.upper(), fill=color_map.get(name, (255, 255, 255, 120)))
    if anchors:
        anchor_color = {
            "left_eye": (80, 220, 255, 220),
            "right_eye": (80, 220, 255, 220),
            "mouth": (255, 140, 140, 220),
        }
        for key in ANCHOR_KEYS:
            pt = anchors.get(key)
            if not pt:
                continue
            x = int(pt[0])
            y = int(pt[1])
            c = anchor_color.get(key, (255, 255, 255, 220))
            od.ellipse((x - 6, y - 6, x + 6, y + 6), outline=c, width=2)
            od.text((x + 8, y - 10), key, fill=c)
    preview = Image.alpha_composite(preview, overlay)
    preview.save(guide_dir / "mask_preview.png")


def alpha_coverage(img: Image.Image) -> float:
    alpha = img.getchannel("A")
    hist = alpha.histogram()
    total = max(1, img.width * img.height)
    non_zero = total - int(hist[0])
    return float(non_zero) / float(total)


def bbox_iou(a: Bounds, b: Bounds) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(1, a.w * a.h)
    area_b = max(1, b.w * b.h)
    union = max(1, area_a + area_b - inter)
    return inter / union


def bounds_to_dict(b: Bounds) -> Dict[str, int]:
    return {
        "x0": int(b.x0),
        "y0": int(b.y0),
        "x1": int(b.x1),
        "y1": int(b.y1),
        "w": int(b.w),
        "h": int(b.h),
    }


def _point_in_bounds(x: float, y: float, b: Bounds, pad: float = 0.0) -> bool:
    return (b.x0 - pad) <= x <= (b.x1 + pad) and (b.y0 - pad) <= y <= (b.y1 + pad)


def _point_to_bounds_distance(x: float, y: float, b: Bounds) -> float:
    dx = 0.0
    if x < b.x0:
        dx = b.x0 - x
    elif x > b.x1:
        dx = x - b.x1
    dy = 0.0
    if y < b.y0:
        dy = b.y0 - y
    elif y > b.y1:
        dy = y - b.y1
    return float(math.hypot(dx, dy))


def _alpha_centroid(mask_or_rgba: Image.Image) -> Tuple[float, float] | None:
    img = mask_or_rgba
    if img.mode != "L":
        img = img.getchannel("A")
    px = img.load()
    box = img.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    sw = 0.0
    sx = 0.0
    sy = 0.0
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            w = float(px[xx, yy])
            if w <= 0.0:
                continue
            sw += w
            sx += w * float(xx)
            sy += w * float(yy)
    if sw <= 1e-6:
        return None
    return (sx / sw, sy / sw)


def evaluate_landmark_alignment_strict(
    layer_outputs: Dict[str, Image.Image],
    mouth_outputs: Dict[str, Image.Image],
    reference_anchors: AnchorMap | None,
) -> Dict[str, object]:
    if not reference_anchors:
        return {"available": False, "reason": "missing_reference_anchors"}
    if not all(k in reference_anchors for k in ANCHOR_KEYS):
        return {"available": False, "reason": "incomplete_reference_anchors"}
    if "head" not in layer_outputs or "eye_open" not in layer_outputs:
        return {"available": False, "reason": "missing_required_layers"}
    if "n" not in mouth_outputs:
        return {"available": False, "reason": "missing_required_mouth_n"}

    head_b = alpha_bounds(layer_outputs["head"])
    eye_alpha = layer_outputs["eye_open"].getchannel("A")
    mouth_alpha = mouth_outputs["n"].getchannel("A")
    eye_px = eye_alpha.load()
    box = eye_alpha.getbbox()
    if box is None:
        return {"available": False, "reason": "empty_eye_mask"}
    x0, y0, x1, y1 = box

    lx_ref, ly_ref = reference_anchors["left_eye"]
    rx_ref, ry_ref = reference_anchors["right_eye"]
    mx_ref, my_ref = reference_anchors["mouth"]
    x_mid_ref = (lx_ref + rx_ref) * 0.5

    lsw = lsx = lsy = 0.0
    rsw = rsx = rsy = 0.0
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            w = float(eye_px[xx, yy])
            if w <= 0.0:
                continue
            if float(xx) <= x_mid_ref:
                lsw += w
                lsx += w * float(xx)
                lsy += w * float(yy)
            else:
                rsw += w
                rsx += w * float(xx)
                rsy += w * float(yy)
    if lsw <= 1e-6 or rsw <= 1e-6:
        return {"available": False, "reason": "failed_eye_side_split"}

    lx_pred = lsx / lsw
    ly_pred = lsy / lsw
    rx_pred = rsx / rsw
    ry_pred = rsy / rsw
    mouth_cent = _alpha_centroid(mouth_alpha)
    if mouth_cent is None:
        return {"available": False, "reason": "empty_mouth_mask"}
    mx_pred, my_pred = mouth_cent

    left_err = math.hypot(lx_pred - lx_ref, ly_pred - ly_ref)
    right_err = math.hypot(rx_pred - rx_ref, ry_pred - ry_ref)
    mouth_err = math.hypot(mx_pred - mx_ref, my_pred - my_ref)

    eye_gap_ref = math.hypot(rx_ref - lx_ref, ry_ref - ly_ref)
    eye_gap_pred = math.hypot(rx_pred - lx_pred, ry_pred - ly_pred)
    eye_gap_err = eye_gap_pred - eye_gap_ref

    ref_mid_x = (lx_ref + rx_ref) * 0.5
    ref_mid_y = (ly_ref + ry_ref) * 0.5
    pred_mid_x = (lx_pred + rx_pred) * 0.5
    pred_mid_y = (ly_pred + ry_pred) * 0.5
    ref_depth = my_ref - ref_mid_y
    pred_depth = my_pred - pred_mid_y
    depth_err = pred_depth - ref_depth

    hw = max(1.0, float(head_b.w))
    hh = max(1.0, float(head_b.h))
    return {
        "available": True,
        "center_error_px": {
            "left_eye": round(left_err, 4),
            "right_eye": round(right_err, 4),
            "mouth_n": round(mouth_err, 4),
        },
        "center_error_ratio_head": {
            "left_eye": round(left_err / hw, 4),
            "right_eye": round(right_err / hw, 4),
            "mouth_n": round(mouth_err / hh, 4),
        },
        "geometry_error_px": {
            "eye_gap": round(eye_gap_err, 4),
            "mouth_depth": round(depth_err, 4),
            "eye_mid_x_shift": round(pred_mid_x - ref_mid_x, 4),
            "eye_mid_y_shift": round(pred_mid_y - ref_mid_y, 4),
        },
        "predicted_centers": {
            "left_eye": {"x": round(lx_pred, 2), "y": round(ly_pred, 2)},
            "right_eye": {"x": round(rx_pred, 2), "y": round(ry_pred, 2)},
            "mouth_n": {"x": round(mx_pred, 2), "y": round(my_pred, 2)},
        },
        "reference_anchors": {
            "left_eye": {"x": round(lx_ref, 2), "y": round(ly_ref, 2)},
            "right_eye": {"x": round(rx_ref, 2), "y": round(ry_ref, 2)},
            "mouth": {"x": round(mx_ref, 2), "y": round(my_ref, 2)},
        },
    }


def evaluate_reference_anchors(
    source: Image.Image,
    reference_anchors: AnchorMap | None,
    layer_outputs: Dict[str, Image.Image],
) -> Dict[str, object]:
    if not reference_anchors:
        return {"available": False, "reason": "missing_reference_anchors"}
    if not all(k in reference_anchors for k in ANCHOR_KEYS):
        return {"available": False, "reason": "incomplete_reference_anchors"}
    if "head" not in layer_outputs:
        return {"available": False, "reason": "missing_head_layer"}

    rgb = source.convert("RGB")
    arr = None
    if np is not None:
        arr = np.asarray(rgb, dtype=np.float32)
    if arr is None:
        return {"available": False, "reason": "numpy_unavailable"}

    head_b = alpha_bounds(layer_outputs["head"])
    x0, y0, x1, y1 = head_b.x0, head_b.y0, head_b.x1, head_b.y1
    if x1 <= x0 or y1 <= y0:
        return {"available": False, "reason": "invalid_head_bounds"}
    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        return {"available": False, "reason": "empty_head_patch"}

    gray = patch[:, :, 0] * 0.299 + patch[:, :, 1] * 0.587 + patch[:, :, 2] * 0.114
    dark = 255.0 - gray
    if cv2 is not None:
        gx = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(np.maximum(0.0, gx * gx + gy * gy))
    else:
        edge = np.zeros_like(gray)
    red = patch[:, :, 0]
    green = patch[:, :, 1]
    blue = patch[:, :, 2]
    red_dom = np.maximum(0.0, red - np.maximum(green, blue))
    eye_map = dark * 0.45 + edge * 0.65
    mouth_map = red_dom * 0.55 + edge * 0.35 + dark * 0.10

    def local_mean(m: object, x: float, y: float, r: int = 8) -> float:
        mm = m
        if not isinstance(mm, np.ndarray):
            return 0.0
        cx = int(round(x)) - x0
        cy = int(round(y)) - y0
        xa = max(0, cx - r)
        xb = min(mm.shape[1], cx + r + 1)
        ya = max(0, cy - r)
        yb = min(mm.shape[0], cy + r + 1)
        if xb <= xa or yb <= ya:
            return 0.0
        return float(mm[ya:yb, xa:xb].mean())

    le = reference_anchors["left_eye"]
    re = reference_anchors["right_eye"]
    mo = reference_anchors["mouth"]
    score_le = local_mean(eye_map, le[0], le[1])
    score_re = local_mean(eye_map, re[0], re[1])
    score_mo = local_mean(mouth_map, mo[0], mo[1])
    eye_p50 = float(np.percentile(eye_map, 50))
    eye_p75 = float(np.percentile(eye_map, 75))
    mouth_p50 = float(np.percentile(mouth_map, 50))
    mouth_p75 = float(np.percentile(mouth_map, 75))
    checks = {
        "left_eye_on_eye_like_region": bool(score_le >= eye_p50),
        "right_eye_on_eye_like_region": bool(score_re >= eye_p50),
        "mouth_on_mouth_like_region": bool(score_mo >= mouth_p50),
        "eye_points_have_enough_contrast": bool(score_le >= eye_p75 * 0.72 and score_re >= eye_p75 * 0.72),
        "mouth_point_has_enough_contrast": bool(score_mo >= mouth_p75 * 0.70),
    }
    pass_count = sum(1 for v in checks.values() if v)
    return {
        "available": True,
        "checks": checks,
        "pass_rate": round(float(pass_count) / float(max(1, len(checks))), 4),
        "scores": {
            "left_eye": round(score_le, 4),
            "right_eye": round(score_re, 4),
            "mouth": round(score_mo, 4),
        },
        "thresholds": {
            "eye_p50": round(eye_p50, 4),
            "eye_p75": round(eye_p75, 4),
            "mouth_p50": round(mouth_p50, 4),
            "mouth_p75": round(mouth_p75, 4),
        },
    }


def judge_landmark_alignment_strict(
    strict_eval: Dict[str, object],
    left_eye_max_px: float,
    right_eye_max_px: float,
    mouth_n_max_px: float,
) -> Dict[str, object]:
    if not strict_eval.get("available"):
        return {"available": False, "reason": strict_eval.get("reason", "strict_eval_unavailable")}
    center_err = strict_eval.get("center_error_px")
    if not isinstance(center_err, Mapping):
        return {"available": False, "reason": "missing_center_error_px"}
    left = float(center_err.get("left_eye", 1e9))
    right = float(center_err.get("right_eye", 1e9))
    mouth_n = float(center_err.get("mouth_n", 1e9))
    checks = {
        "left_eye_ok": left <= float(left_eye_max_px),
        "right_eye_ok": right <= float(right_eye_max_px),
        "mouth_n_ok": mouth_n <= float(mouth_n_max_px),
    }
    passed = all(checks.values())
    return {
        "available": True,
        "thresholds_px": {
            "left_eye_max": float(left_eye_max_px),
            "right_eye_max": float(right_eye_max_px),
            "mouth_n_max": float(mouth_n_max_px),
        },
        "checks": checks,
        "passed": bool(passed),
    }


def collect_part_bboxes(
    layer_outputs: Dict[str, Image.Image],
    mouth_outputs: Dict[str, Image.Image],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    layers = {name: bounds_to_dict(alpha_bounds(img)) for name, img in layer_outputs.items()}
    mouths = {name: bounds_to_dict(alpha_bounds(img)) for name, img in mouth_outputs.items()}
    return {"layers": layers, "mouths": mouths}


def _estimate_roll_from_anchors(anchors: AnchorMap | None) -> float:
    if not anchors:
        return 0.0
    le = anchors.get("left_eye")
    re = anchors.get("right_eye")
    if not le or not re:
        return 0.0
    dx = float(re[0] - le[0])
    dy = float(re[1] - le[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    roll = math.atan2(dy, dx)
    max_roll = math.radians(18.0)
    if roll > max_roll:
        roll = max_roll
    elif roll < -max_roll:
        roll = -max_roll
    return float(math.degrees(roll))


def _oriented_box_from_alpha(
    img: Image.Image,
    angle_deg: float,
    sample_step: int = 2,
) -> list[Tuple[float, float]] | None:
    a = img.getchannel("A")
    box = a.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5

    rad = math.radians(float(angle_deg))
    c = math.cos(rad)
    s = math.sin(rad)

    u_min = 1e18
    u_max = -1e18
    v_min = 1e18
    v_max = -1e18

    apx = a.load()
    found = False
    step = max(1, int(sample_step))
    for yy in range(y0, y1, step):
        for xx in range(x0, x1, step):
            if apx[xx, yy] < 8:
                continue
            found = True
            dx = float(xx) - cx
            dy = float(yy) - cy
            u = dx * c + dy * s
            v = -dx * s + dy * c
            if u < u_min:
                u_min = u
            if u > u_max:
                u_max = u
            if v < v_min:
                v_min = v
            if v > v_max:
                v_max = v

    if not found:
        return None

    corners_uv = [
        (u_min, v_min),
        (u_max, v_min),
        (u_max, v_max),
        (u_min, v_max),
    ]
    corners_xy: list[Tuple[float, float]] = []
    for u, v in corners_uv:
        dx = u * c - v * s
        dy = u * s + v * c
        corners_xy.append((cx + dx, cy + dy))
    return corners_xy


def save_bbox_preview(
    source: Image.Image,
    guide_dir: Path,
    layer_outputs: Dict[str, Image.Image],
    mouth_outputs: Dict[str, Image.Image],
    anchors: AnchorMap | None = None,
) -> None:
    guide_dir.mkdir(parents=True, exist_ok=True)
    preview = source.convert("RGBA").copy()
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    debug_bg = source.convert("RGBA").copy()
    dim = Image.new("RGBA", preview.size, (0, 0, 0, 150))
    debug_bg = Image.alpha_composite(debug_bg, dim)
    debug_overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(debug_overlay)

    layer_color = {
        "head": (120, 220, 255, 220),
        "body": (140, 255, 180, 220),
        "hair_front": (255, 220, 120, 220),
        "hair_back": (255, 170, 120, 220),
        "eye_open": (120, 180, 255, 220),
        "eye_half": (120, 180, 255, 220),
        "eye_open_wide": (120, 180, 255, 220),
    }
    mouth_color = {
        "a": (255, 120, 120, 220),
        "i": (120, 180, 255, 220),
        "u": (140, 255, 180, 220),
        "e": (255, 200, 120, 220),
        "o": (220, 150, 255, 220),
        "n": (255, 255, 255, 220),
    }
    roll_deg = _estimate_roll_from_anchors(anchors)

    for name, img in layer_outputs.items():
        b = alpha_bounds(img)
        c = layer_color.get(name, (220, 220, 220, 220))
        tint = Image.new("RGBA", preview.size, (c[0], c[1], c[2], 0))
        tint_alpha = img.getchannel("A").point(lambda p: int(min(72, p * 0.30)))
        tint.putalpha(tint_alpha)
        if hasattr(overlay, "alpha_composite"):
            overlay.alpha_composite(tint)
        else:
            overlay = Image.alpha_composite(overlay, tint)
            d = ImageDraw.Draw(overlay)
        d.rectangle((b.x0, b.y0, b.x1, b.y1), outline=c, width=2)
        dd.rectangle((b.x0, b.y0, b.x1, b.y1), outline=(c[0], c[1], c[2], 255), width=4)
        if name in {"eye_open", "eye_half", "eye_open_wide"}:
            poly = _oriented_box_from_alpha(img, angle_deg=roll_deg, sample_step=2)
            if poly:
                d.polygon(poly, outline=(40, 255, 255, 220), width=2)
                dd.polygon(poly, outline=(40, 255, 255, 255), width=4)
        d.text((b.x0, max(0, b.y0 - 14)), f"L:{name}", fill=c)
        dd.text((b.x0 + 2, max(0, b.y0 - 16)), f"L:{name}", fill=(255, 255, 255, 255))

    for name, img in mouth_outputs.items():
        b = alpha_bounds(img)
        c = mouth_color.get(name, (255, 255, 255, 220))
        tint = Image.new("RGBA", preview.size, (c[0], c[1], c[2], 0))
        tint_alpha = img.getchannel("A").point(lambda p: int(min(60, p * 0.24)))
        tint.putalpha(tint_alpha)
        if hasattr(overlay, "alpha_composite"):
            overlay.alpha_composite(tint)
        else:
            overlay = Image.alpha_composite(overlay, tint)
            d = ImageDraw.Draw(overlay)
        d.rectangle((b.x0, b.y0, b.x1, b.y1), outline=c, width=2)
        dd.rectangle((b.x0, b.y0, b.x1, b.y1), outline=(c[0], c[1], c[2], 255), width=4)
        if name in {"a", "i", "u", "e", "o", "n"}:
            poly = _oriented_box_from_alpha(img, angle_deg=roll_deg, sample_step=2)
            if poly:
                d.polygon(poly, outline=(255, 255, 255, 220), width=2)
                dd.polygon(poly, outline=(255, 255, 255, 255), width=4)
        d.text((b.x0, max(0, b.y0 - 14)), f"M:{name}", fill=c)
        dd.text((b.x0 + 2, max(0, b.y0 - 16)), f"M:{name}", fill=(255, 255, 255, 255))

    if anchors:
        anchor_color = {
            "left_eye": (0, 180, 255, 255),
            "right_eye": (0, 255, 180, 255),
            "mouth": (255, 80, 120, 255),
        }
        for key in ANCHOR_KEYS:
            pt = anchors.get(key)
            if not pt:
                continue
            x, y = int(pt[0]), int(pt[1])
            c = anchor_color.get(key, (255, 255, 255, 255))
            d.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(0, 0, 0, 255), width=3)
            d.ellipse((x - 7, y - 7, x + 7, y + 7), outline=c, width=2)
            d.line((x - 10, y, x + 10, y), fill=c, width=2)
            d.line((x, y - 10, x, y + 10), fill=c, width=2)
            d.text((x + 10, y - 14), f"{key}({x},{y})", fill=c)
            dd.ellipse((x - 10, y - 10, x + 10, y + 10), outline=(0, 0, 0, 255), width=4)
            dd.ellipse((x - 9, y - 9, x + 9, y + 9), outline=c, width=3)
            dd.line((x - 14, y, x + 14, y), fill=c, width=3)
            dd.line((x, y - 14, x, y + 14), fill=c, width=3)
            dd.text((x + 12, y - 18), f"{key}({x},{y})", fill=(255, 255, 255, 255))

    preview = Image.alpha_composite(preview, overlay)
    preview.save(guide_dir / "bbox_preview.png")
    debug_img = Image.alpha_composite(debug_bg, debug_overlay)
    debug_img.save(guide_dir / "bbox_preview_debug.png")


def save_alignment_overlay(
    source: Image.Image,
    guide_dir: Path,
    predicted_anchors: AnchorMap | None,
    reference_anchors: AnchorMap | None,
    mouth_outputs: Dict[str, Image.Image] | None = None,
) -> None:
    if not predicted_anchors or not reference_anchors:
        return
    if not all(k in predicted_anchors for k in ANCHOR_KEYS):
        return
    if not all(k in reference_anchors for k in ANCHOR_KEYS):
        return

    guide_dir.mkdir(parents=True, exist_ok=True)
    base = source.convert("RGBA").copy()
    dim = Image.new("RGBA", base.size, (0, 0, 0, 120))
    base = Image.alpha_composite(base, dim)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    color_map = {
        "left_eye": (0, 190, 255, 255),
        "right_eye": (0, 255, 170, 255),
        "mouth": (255, 90, 130, 255),
    }

    for key in ANCHOR_KEYS:
        rx, ry = reference_anchors[key]
        px, py = predicted_anchors[key]
        c = color_map.get(key, (255, 255, 255, 255))

        # Reference point: square
        d.rectangle((rx - 7, ry - 7, rx + 7, ry + 7), outline=(0, 0, 0, 255), width=3)
        d.rectangle((rx - 6, ry - 6, rx + 6, ry + 6), outline=c, width=2)

        # Predicted point: circle + cross
        d.ellipse((px - 9, py - 9, px + 9, py + 9), outline=(0, 0, 0, 255), width=3)
        d.ellipse((px - 8, py - 8, px + 8, py + 8), outline=c, width=2)
        d.line((px - 12, py, px + 12, py), fill=c, width=2)
        d.line((px, py - 12, px, py + 12), fill=c, width=2)

        # Residual vector
        d.line((rx, ry, px, py), fill=(255, 255, 255, 255), width=3)
        d.line((rx, ry, px, py), fill=c, width=1)

        err = math.hypot(px - rx, py - ry)
        label = f"{key} err={err:.1f}px"
        tx = int(min(base.width - 220, max(8, rx + 14)))
        ty = int(min(base.height - 24, max(8, ry - 16)))
        d.text((tx, ty), label, fill=(255, 255, 255, 255))

    # Additional visualization: actual mouth_n mask center vs reference mouth anchor.
    if mouth_outputs and "n" in mouth_outputs:
        mouth_ref = reference_anchors["mouth"]
        mouth_center = _alpha_centroid(mouth_outputs["n"])
        if mouth_center is not None:
            rx, ry = mouth_ref
            mx, my = mouth_center
            c = (255, 220, 80, 255)
            # mouth_n center marker: diamond-like cross
            d.line((mx - 10, my, mx + 10, my), fill=c, width=3)
            d.line((mx, my - 10, mx, my + 10), fill=c, width=3)
            d.ellipse((mx - 7, my - 7, mx + 7, my + 7), outline=c, width=2)
            # residual to reference mouth
            d.line((rx, ry, mx, my), fill=(255, 255, 255, 255), width=3)
            d.line((rx, ry, mx, my), fill=c, width=1)
            err = math.hypot(mx - rx, my - ry)
            tx = int(min(base.width - 260, max(8, rx + 14)))
            ty = int(min(base.height - 24, max(8, ry + 10)))
            d.text((tx, ty), f"mouth_n err={err:.1f}px", fill=c)

    out = Image.alpha_composite(base, overlay)
    out.save(guide_dir / "alignment_overlay.png")


def evaluate_bboxes_against_reference(
    layer_outputs: Dict[str, Image.Image],
    mouth_outputs: Dict[str, Image.Image],
    reference_anchors: AnchorMap | None,
) -> Dict[str, object]:
    if not reference_anchors:
        return {"available": False, "reason": "missing_reference_anchors"}
    if not all(k in reference_anchors for k in ANCHOR_KEYS):
        return {"available": False, "reason": "incomplete_reference_anchors"}
    if "head" not in layer_outputs or "eye_open" not in layer_outputs:
        return {"available": False, "reason": "missing_required_layers"}
    if "n" not in mouth_outputs:
        return {"available": False, "reason": "missing_required_mouth_n"}

    head_b = alpha_bounds(layer_outputs["head"])
    eye_b = alpha_bounds(layer_outputs["eye_open"])
    mouth_b = alpha_bounds(mouth_outputs["n"])

    lx, ly = reference_anchors["left_eye"]
    rx, ry = reference_anchors["right_eye"]
    mx, my = reference_anchors["mouth"]

    margin = max(4.0, head_b.w * 0.02)
    eye_ref_mid_y = (ly + ry) * 0.5
    eye_box_mid_y = (eye_b.y0 + eye_b.y1) * 0.5
    mouth_box_mid_y = (mouth_b.y0 + mouth_b.y1) * 0.5
    mouth_box_mid_x = (mouth_b.x0 + mouth_b.x1) * 0.5

    checks = {
        "eye_bbox_contains_left_eye_ref": _point_in_bounds(lx, ly, eye_b, pad=margin),
        "eye_bbox_contains_right_eye_ref": _point_in_bounds(rx, ry, eye_b, pad=margin),
        "eye_bbox_spans_eye_refs_x": (eye_b.x0 - margin) <= min(lx, rx) and (eye_b.x1 + margin) >= max(lx, rx),
        "eye_bbox_mid_y_near_eye_ref_line": abs(eye_box_mid_y - eye_ref_mid_y) <= head_b.h * 0.12,
        "mouth_bbox_contains_mouth_ref": _point_in_bounds(mx, my, mouth_b, pad=margin),
        "mouth_bbox_below_eye_bbox": mouth_box_mid_y > eye_box_mid_y,
        "mouth_bbox_inside_head": _point_in_bounds(mouth_box_mid_x, mouth_box_mid_y, head_b, pad=0.0),
    }

    total = max(1, len(checks))
    failed = sum(1 for ok in checks.values() if not ok)
    pass_rate = float(total - failed) / float(total)
    fail_rate = float(failed) / float(total)

    return {
        "available": True,
        "checks": checks,
        "pass_rate": round(pass_rate, 4),
        "fail_rate": round(fail_rate, 4),
        "distance_px": {
            "left_eye_ref_to_eye_bbox": round(_point_to_bounds_distance(lx, ly, eye_b), 4),
            "right_eye_ref_to_eye_bbox": round(_point_to_bounds_distance(rx, ry, eye_b), 4),
            "mouth_ref_to_mouth_n_bbox": round(_point_to_bounds_distance(mx, my, mouth_b), 4),
        },
        "reference_anchors": {
            "left_eye": {"x": round(lx, 2), "y": round(ly, 2)},
            "right_eye": {"x": round(rx, 2), "y": round(ry, 2)},
            "mouth": {"x": round(mx, 2), "y": round(my, 2)},
        },
    }


def evaluate_anchor_alignment_modes(
    predicted_anchors: AnchorMap | None,
    reference_anchors: AnchorMap | None,
) -> Dict[str, object]:
    if not predicted_anchors or not reference_anchors:
        return {"available": False, "reason": "missing_predicted_or_reference"}
    if not all(k in predicted_anchors for k in ANCHOR_KEYS):
        return {"available": False, "reason": "incomplete_predicted_anchors"}
    if not all(k in reference_anchors for k in ANCHOR_KEYS):
        return {"available": False, "reason": "incomplete_reference_anchors"}

    def _rmse(
        mapping: Dict[str, str],
        shift_x: float = 0.0,
        shift_y: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        sq_sum = 0.0
        per_point: Dict[str, float] = {}
        for key in ANCHOR_KEYS:
            src = predicted_anchors[mapping[key]]
            ref = reference_anchors[key]
            dx = (src[0] + shift_x) - ref[0]
            dy = (src[1] + shift_y) - ref[1]
            d = math.hypot(dx, dy)
            per_point[key] = d
            sq_sum += dx * dx + dy * dy
        rmse = math.sqrt(sq_sum / float(len(ANCHOR_KEYS)))
        return rmse, per_point

    mappings: Dict[str, Dict[str, str]] = {
        "normal": {"left_eye": "left_eye", "right_eye": "right_eye", "mouth": "mouth"},
        "swap_eyes": {"left_eye": "right_eye", "right_eye": "left_eye", "mouth": "mouth"},
    }

    out_cases: Dict[str, object] = {}
    best_case = ""
    best_rmse = 1e18
    for mode_name, mode_map in mappings.items():
        rmse_no_shift, point_no_shift = _rmse(mode_map)
        shift_x = sum(reference_anchors[k][0] - predicted_anchors[mode_map[k]][0] for k in ANCHOR_KEYS) / float(len(ANCHOR_KEYS))
        shift_y = sum(reference_anchors[k][1] - predicted_anchors[mode_map[k]][1] for k in ANCHOR_KEYS) / float(len(ANCHOR_KEYS))
        rmse_shift, point_shift = _rmse(mode_map, shift_x=shift_x, shift_y=shift_y)
        out_cases[mode_name] = {
            "rmse_no_shift": round(rmse_no_shift, 4),
            "rmse_with_best_shift": round(rmse_shift, 4),
            "best_shift": {"dx": round(shift_x, 4), "dy": round(shift_y, 4)},
            "distance_no_shift_px": {k: round(v, 4) for k, v in point_no_shift.items()},
            "distance_with_best_shift_px": {k: round(v, 4) for k, v in point_shift.items()},
        }
        if rmse_no_shift < best_rmse:
            best_rmse = rmse_no_shift
            best_case = mode_name

    normal = out_cases["normal"]["rmse_no_shift"]  # type: ignore[index]
    swapped = out_cases["swap_eyes"]["rmse_no_shift"]  # type: ignore[index]
    swapped_better = float(swapped) + 1e-6 < float(normal)

    return {
        "available": True,
        "modes": out_cases,
        "best_mode_no_shift": best_case,
        "swap_eyes_better_than_normal": bool(swapped_better),
        "predicted": {
            "left_eye": {"x": round(predicted_anchors["left_eye"][0], 2), "y": round(predicted_anchors["left_eye"][1], 2)},
            "right_eye": {"x": round(predicted_anchors["right_eye"][0], 2), "y": round(predicted_anchors["right_eye"][1], 2)},
            "mouth": {"x": round(predicted_anchors["mouth"][0], 2), "y": round(predicted_anchors["mouth"][1], 2)},
        },
        "reference": {
            "left_eye": {"x": round(reference_anchors["left_eye"][0], 2), "y": round(reference_anchors["left_eye"][1], 2)},
            "right_eye": {"x": round(reference_anchors["right_eye"][0], 2), "y": round(reference_anchors["right_eye"][1], 2)},
            "mouth": {"x": round(reference_anchors["mouth"][0], 2), "y": round(reference_anchors["mouth"][1], 2)},
        },
    }


def build_anchor_diagnostics(
    anchors: AnchorMap,
    layer_outputs: Dict[str, Image.Image],
    image_size: Tuple[int, int],
) -> Dict[str, object]:
    out: Dict[str, object] = {"available": False}
    if not all(k in anchors for k in ANCHOR_KEYS):
        out["reason"] = "missing_anchor_keys"
        return out

    head_img = layer_outputs.get("head")
    if head_img is None:
        out["reason"] = "missing_head_layer"
        return out

    head_b = alpha_bounds(head_img)
    hw = max(1.0, float(head_b.w))
    hh = max(1.0, float(head_b.h))
    w, h = image_size

    lx, ly = anchors["left_eye"]
    rx, ry = anchors["right_eye"]
    mx, my = anchors["mouth"]

    dx = rx - lx
    dy = ry - ly
    eye_dist = math.hypot(dx, dy)
    roll_deg = math.degrees(math.atan2(dy, max(1e-6, dx)))
    eye_mid_x = (lx + rx) * 0.5
    eye_mid_y = (ly + ry) * 0.5
    mouth_offset_x = mx - eye_mid_x
    mouth_offset_y = my - eye_mid_y

    # Face local coordinates: v=eye-line direction, n=downward normal.
    r = math.atan2(dy, max(1e-6, dx))
    vx, vy = math.cos(r), math.sin(r)
    nx, ny = -vy, vx
    if ny < 0.0:
        nx, ny = -nx, -ny
    mouth_along_eye = mouth_offset_x * vx + mouth_offset_y * vy
    mouth_depth = mouth_offset_x * nx + mouth_offset_y * ny

    eye_dist_ratio_w = eye_dist / hw
    mouth_depth_ratio_h = mouth_depth / hh
    mouth_center_x_ratio = (mx - head_b.x0) / hw
    mouth_center_y_ratio = (my - head_b.y0) / hh

    checks = {
        "left_right_order_ok": bool(lx < rx),
        "eye_dist_ok": bool(0.10 <= eye_dist_ratio_w <= 0.72),
        "roll_ok": bool(abs(roll_deg) <= 24.0),
        "mouth_below_eyes_ok": bool(mouth_depth_ratio_h >= 0.10),
        "mouth_depth_ok": bool(0.10 <= mouth_depth_ratio_h <= 0.55),
        "mouth_center_ok": bool(0.18 <= mouth_center_x_ratio <= 0.82 and 0.50 <= mouth_center_y_ratio <= 0.90),
    }
    pass_count = sum(1 for v in checks.values() if v)
    total = max(1, len(checks))
    plausibility = float(pass_count) / float(total)

    out.update(
        {
            "available": True,
            "roll_deg": round(roll_deg, 4),
            "eye_dist_px": round(eye_dist, 4),
            "eye_dist_ratio_head_w": round(eye_dist_ratio_w, 4),
            "mouth_depth_px": round(mouth_depth, 4),
            "mouth_depth_ratio_head_h": round(mouth_depth_ratio_h, 4),
            "mouth_along_eye_px": round(mouth_along_eye, 4),
            "mouth_center_ratio_in_head": {
                "x": round(mouth_center_x_ratio, 4),
                "y": round(mouth_center_y_ratio, 4),
            },
            "checks": checks,
            "plausibility": round(plausibility, 4),
            "image_size": {"w": int(w), "h": int(h)},
        }
    )
    return out


def build_quality_report(
    source_path: Path,
    layer_outputs: Dict[str, Image.Image],
    mouth_outputs: Dict[str, Image.Image],
) -> Dict[str, object]:
    layer_cov = {k: round(alpha_coverage(v), 4) for k, v in layer_outputs.items()}
    mouth_cov = {k: round(alpha_coverage(v), 4) for k, v in mouth_outputs.items()}
    layer_bounds = {k: alpha_bounds(v) for k, v in layer_outputs.items()}
    mouth_bounds = {k: alpha_bounds(v) for k, v in mouth_outputs.items()}
    iou_n = {}
    ref = mouth_bounds.get("n")
    if ref is not None:
        for k, b in mouth_bounds.items():
            iou_n[k] = round(bbox_iou(ref, b), 4)

    warnings = []
    mouth_center_ratio = {}
    for name, cov in layer_cov.items():
        if cov < 0.01:
            warnings.append(f"layer '{name}' coverage too small: {cov}")
    for name, cov in mouth_cov.items():
        if cov < 0.001:
            warnings.append(f"mouth '{name}' coverage too small: {cov}")
    if iou_n:
        same_like = [k for k, v in iou_n.items() if k != "n" and v > 0.97]
        if same_like:
            warnings.append(f"mouth masks too similar to 'n': {', '.join(same_like)}")

    # Semantic mouth placement checks.
    head_b = layer_bounds.get("head")
    eye_b = layer_bounds.get("eye_open")
    n_b = mouth_bounds.get("n")
    semantic_penalty = 0
    if head_b and n_b:
        mx = (n_b.x0 + n_b.x1) / 2.0
        my = (n_b.y0 + n_b.y1) / 2.0
        rx = (mx - head_b.x0) / max(1.0, head_b.w)
        ry = (my - head_b.y0) / max(1.0, head_b.h)
        mouth_center_ratio = {"x": round(rx, 4), "y": round(ry, 4)}
        in_head_x = (head_b.x0 + head_b.w * 0.20) <= mx <= (head_b.x1 - head_b.w * 0.20)
        in_head_y = (head_b.y0 + head_b.h * 0.52) <= my <= (head_b.y0 + head_b.h * 0.84)
        if not (in_head_x and in_head_y):
            warnings.append("mouth center seems outside expected head/lower-face zone")
            semantic_penalty += 40
    if head_b and eye_b and n_b:
        eye_cy = (eye_b.y0 + eye_b.y1) / 2.0
        mouth_cy = (n_b.y0 + n_b.y1) / 2.0
        min_y = eye_cy + head_b.h * 0.10
        max_y = eye_cy + head_b.h * 0.45
        if not (min_y <= mouth_cy <= max_y):
            warnings.append("eye-mouth vertical relation looks invalid")
            semantic_penalty += 35

    # rough score
    base_score = 100
    base_score -= len(warnings) * 8
    base_score -= semantic_penalty
    if iou_n:
        diversity = 1.0 - (sum(iou_n.values()) / max(1, len(iou_n)))
        base_score += int(diversity * 20)
    score = max(0, min(100, base_score))

    return {
        "tool": "build_pngtuber_pack",
        "source": str(source_path.as_posix()),
        "score": score,
        "layer_coverage": layer_cov,
        "mouth_coverage": mouth_cov,
        "mouth_iou_to_n": iou_n,
        "mouth_center_ratio_in_head": mouth_center_ratio,
        "warnings": warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Source portrait image (RGBA recommended)")
    ap.add_argument("--out-dir", required=True, help="Output directory for generated PNGTuber pack")
    ap.add_argument("--manifest", default="", help="Optional manifest output path")
    ap.add_argument("--auto-split", default="true", help="Use automatic region split when masks are missing")
    ap.add_argument("--erase-mouth", default="true", help="Erase mouth from head layer for overlay lipsync")
    ap.add_argument(
        "--pipeline-mode",
        default="offline_build",
        choices=("offline_build", "runtime"),
        help="offline_build=detector run and save anchors, runtime=prefer precomputed anchors",
    )
    ap.add_argument(
        "--detector",
        default="auto",
        choices=("auto", "anchors", "replay", "anime_cv", "anime_onnx", "none"),
        help="Anchor detector backend",
    )
    ap.add_argument("--detector-track-file", default="", help="Optional track JSON for detector=replay")
    ap.add_argument("--detector-model-file", default="", help="Model file path for detector=anime_onnx")
    ap.add_argument(
        "--detector-points-map-file",
        default="",
        help="JSON points map for detector=anime_onnx (left_eye/right_eye/mouth point indices)",
    )
    ap.add_argument("--clean-out-dir", default="true", help="Remove old png/json artifacts in out-dir before writing")
    ap.add_argument("--report-file", default="build_report.json", help="Quality report filename under out-dir")
    ap.add_argument("--strict-left-eye-max-px", type=float, default=6.0, help="Pass threshold for strict left_eye error")
    ap.add_argument("--strict-right-eye-max-px", type=float, default=6.0, help="Pass threshold for strict right_eye error")
    ap.add_argument("--strict-mouth-max-px", type=float, default=8.0, help="Pass threshold for strict mouth_n error")
    ap.add_argument(
        "--eval-anchors-file",
        default="",
        help="Optional reference anchors JSON used for bbox/anchor quality evaluation",
    )
    ap.add_argument("--emit-guides", default="true", help="Write mask guide files for manual refinement")
    ap.add_argument("--guide-dir", default="", help="Guide output directory (default: assets/replay/masks/<character>)")
    ap.add_argument("--anchors-file", default="", help="Optional JSON with left_eye/right_eye/mouth points")
    ap.add_argument(
        "--save-auto-anchors",
        default="",
        help="Optional path to write detected anchors JSON (for manual tweak + rerun)",
    )

    for name in LAYER_NAMES:
        ap.add_argument(f"--{name.replace('_', '-')}-mask", default="", help=f"Mask image for layer '{name}'")
    for name in MOUTH_NAMES:
        ap.add_argument(f"--mouth-{name}-mask", default="", help=f"Mask image for mouth '{name}'")

    args = ap.parse_args()

    source_path = Path(args.source)
    out_dir = Path(args.out_dir)
    auto_split = parse_bool(args.auto_split)
    erase_mouth = parse_bool(args.erase_mouth)
    clean_out_dir = parse_bool(args.clean_out_dir)
    emit_guides = parse_bool(args.emit_guides)
    pipeline_mode = str(args.pipeline_mode).strip().lower()

    src = ensure_rgba(source_path)
    size = src.size
    out_dir.mkdir(parents=True, exist_ok=True)
    if clean_out_dir:
        for old in out_dir.glob("*"):
            if old.is_file() and old.suffix.lower() in {".png", ".json"}:
                old.unlink(missing_ok=True)
            elif old.is_dir() and old.name == "__pycache__":
                shutil.rmtree(old, ignore_errors=True)

    auto_layer_masks: Dict[str, Image.Image] = {}
    auto_mouth_masks: Dict[str, Image.Image] = {}
    auto_split_debug: Dict[str, object] = {}
    anchors_input: AnchorMap = {}
    anchors_used: AnchorMap = {}

    if args.anchors_file:
        anchor_path = Path(args.anchors_file)
        if anchor_path.exists():
            anchors_input = load_anchors(anchor_path, size)

    detector_meta: Dict[str, object] = {"pipeline_mode": pipeline_mode, "detector": str(args.detector)}
    detector_result: DetectorResult | None = None
    try:
        # Runtime should prefer precomputed anchors to keep CPU path stable and fast.
        detector_kind = str(args.detector)
        if pipeline_mode == "runtime" and detector_kind == "auto" and anchors_input:
            detector_kind = "anchors"
        detector = make_detector(
            detector_kind=detector_kind,
            anchors_file=str(args.anchors_file),
            replay_track_file=str(args.detector_track_file),
            anime_model_file=str(args.detector_model_file),
            anime_points_map_file=str(args.detector_points_map_file),
            seed_anchors=anchors_input if anchors_input else None,
        )
        detector_result = detector.detect(src, size)
    except ValueError as exc:
        raise SystemExit(f"[error] {exc}") from exc

    if detector_result is not None:
        detector_meta.update(
            {
                "detector_name": detector_result.detector_name,
                "detector_confidence": round(float(detector_result.confidence), 4),
            }
        )
        if detector_result.meta:
            detector_meta["detector_meta"] = detector_result.meta
        if detector_result.anchors:
            anchors_input = detector_result.anchors

    if auto_split:
        auto_layer_masks, auto_mouth_masks, anchors_used = make_auto_masks(
            src,
            anchors=anchors_input,
            debug_out=auto_split_debug,
        )

    manual_mask_count = 0
    layer_outputs: Dict[str, Image.Image] = {}
    layer_masks_used: Dict[str, Image.Image] = {}
    for name in LAYER_NAMES:
        mask_arg = getattr(args, f"{name}_mask")
        if mask_arg:
            manual_mask_count += 1
            mask = load_mask(Path(mask_arg), size)
        else:
            mask = auto_layer_masks.get(name)
        if mask is None:
            mask = mask_rect(size, (0, 0, src.width, src.height))
        layer_masks_used[name] = mask
        layer_outputs[name] = apply_mask_rgba(src, mask)

    mouth_outputs: Dict[str, Image.Image] = {}
    mouth_masks_for_union: Dict[str, Image.Image] = {}
    for name in MOUTH_NAMES:
        mask_arg = getattr(args, f"mouth_{name}_mask")
        if mask_arg:
            manual_mask_count += 1
            mask = load_mask(Path(mask_arg), size)
        else:
            mask = auto_mouth_masks.get(name)
        if mask is None:
            mask = auto_mouth_masks.get("n") or mask_rect(size, (0, 0, src.width, src.height))
        mouth_masks_for_union[name] = mask
        mouth_outputs[name] = apply_mask_rgba(src, mask)

    if erase_mouth and "head" in layer_outputs:
        mouth_union = union_masks(mouth_masks_for_union, size)
        mouth_union = soften_mask(mouth_union, radius=2)
        head_clean = erase_mouth_region(layer_outputs["head"], mouth_union)
        layer_outputs["head"] = head_clean

    for name, img in layer_outputs.items():
        img.save(out_dir / f"{name}.png")
    for name, img in mouth_outputs.items():
        img.save(out_dir / f"mouth_{name}.png")

    if emit_guides:
        if args.guide_dir:
            guide_dir = Path(args.guide_dir)
        else:
            guide_dir = Path("assets/replay/masks") / out_dir.name
        save_mask_guides(
            source=src,
            guide_dir=guide_dir,
            mouth_masks=mouth_masks_for_union,
            layer_masks=layer_masks_used,
            anchors=anchors_used,
        )
        save_bbox_preview(
            source=src,
            guide_dir=guide_dir,
            layer_outputs=layer_outputs,
            mouth_outputs=mouth_outputs,
            anchors=anchors_used,
        )
    if args.save_auto_anchors and anchors_used:
        save_anchors(Path(args.save_auto_anchors), anchors_used)
    elif emit_guides and anchors_used:
        # Keep manual/approved anchors stable; write fresh auto estimate to a separate file.
        save_anchors(guide_dir / "anchors.last_auto.json", anchors_used)

    if args.manifest:
        build_manifest(Path(args.manifest), out_dir)

    report = build_quality_report(source_path=source_path, layer_outputs=layer_outputs, mouth_outputs=mouth_outputs)
    report["detector"] = detector_meta
    report["part_bboxes"] = collect_part_bboxes(layer_outputs=layer_outputs, mouth_outputs=mouth_outputs)

    eval_anchors: AnchorMap = {}
    eval_anchor_path: Path | None = None
    if args.eval_anchors_file:
        cand = Path(args.eval_anchors_file)
        if cand.exists():
            eval_anchor_path = cand
    else:
        default_gt = Path("assets/replay/masks") / out_dir.name / "anchors.gt.json"
        default_auto = Path("assets/replay/masks") / out_dir.name / "anchors.auto.json"
        if default_gt.exists():
            eval_anchor_path = default_gt
        elif default_auto.exists():
            eval_anchor_path = default_auto
    if eval_anchor_path is not None:
        eval_anchors = load_anchors(eval_anchor_path, size)
        report["bbox_eval_reference"] = str(eval_anchor_path.as_posix())
        if emit_guides:
            save_alignment_overlay(
                source=src,
                guide_dir=guide_dir,
                predicted_anchors=anchors_used if anchors_used else None,
                reference_anchors=eval_anchors if eval_anchors else None,
                mouth_outputs=mouth_outputs,
            )
    report["reference_anchor_eval"] = evaluate_reference_anchors(
        source=src,
        reference_anchors=eval_anchors if eval_anchors else None,
        layer_outputs=layer_outputs,
    )

    report["bbox_eval"] = evaluate_bboxes_against_reference(
        layer_outputs=layer_outputs,
        mouth_outputs=mouth_outputs,
        reference_anchors=eval_anchors if eval_anchors else None,
    )
    report["landmark_eval_strict"] = evaluate_landmark_alignment_strict(
        layer_outputs=layer_outputs,
        mouth_outputs=mouth_outputs,
        reference_anchors=eval_anchors if eval_anchors else None,
    )
    report["landmark_eval_strict_gate"] = judge_landmark_alignment_strict(
        strict_eval=report["landmark_eval_strict"],
        left_eye_max_px=float(args.strict_left_eye_max_px),
        right_eye_max_px=float(args.strict_right_eye_max_px),
        mouth_n_max_px=float(args.strict_mouth_max_px),
    )
    report["anchor_alignment_eval"] = evaluate_anchor_alignment_modes(
        predicted_anchors=anchors_used if anchors_used else None,
        reference_anchors=eval_anchors if eval_anchors else None,
    )

    if auto_split_debug:
        report["auto_split_debug"] = auto_split_debug
    if anchors_input:
        report["anchors_input"] = {k: {"x": round(v[0], 2), "y": round(v[1], 2)} for k, v in anchors_input.items()}
    if anchors_used:
        report["anchors_used"] = {k: {"x": round(v[0], 2), "y": round(v[1], 2)} for k, v in anchors_used.items()}
        report["anchor_diagnostics"] = build_anchor_diagnostics(
            anchors=anchors_used,
            layer_outputs=layer_outputs,
            image_size=size,
        )
    if auto_split and manual_mask_count == 0:
        report["warnings"].append("auto split only: score capped to avoid overconfidence")
        report["score"] = min(int(report.get("score", 0)), 85)
    ref_eval = report.get("reference_anchor_eval")
    if isinstance(ref_eval, Mapping) and bool(ref_eval.get("available")):
        if float(ref_eval.get("pass_rate", 0.0)) < 0.6:
            report["warnings"].append("reference anchors likely misaligned; verify anchors.gt.json/anchors.auto.json")
    report_path = out_dir / str(args.report_file)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] generated PNGTuber pack: {out_dir}")
    if emit_guides:
        print(f"[OK] wrote mask guides: {guide_dir}")
    print(f"[OK] wrote quality report: {report_path} (score={report.get('score')})")
    for w in report.get("warnings", []):
        print(f"[warn] {w}")
    if args.manifest:
        print(f"[OK] wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
