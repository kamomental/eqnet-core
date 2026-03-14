#!/usr/bin/env python3
"""Unified vision bridge for telemetry webhooks and frame-driven runtime turns."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from fastapi import FastAPI, HTTPException, Request
from scripts.vision_frame_contract import build_sensor_frame, summarize_sensor_frame

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROOT_PKG = ROOT / "emot_terrain_lab"
if ROOT_PKG.exists():
    pkg_path = str(ROOT_PKG)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

try:
    from telemetry import event as telemetry_event
except Exception:
    telemetry_event = None  # type: ignore


app = FastAPI(title="Vision EQNet Bridge", version="0.2.0")


def _load_runtime_classes() -> tuple[type[Any], type[Any]]:
    module = importlib.import_module("emot_terrain_lab.hub.runtime")
    return module.EmotionalHubRuntime, module.RuntimeConfig


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timestamp(payload: Dict[str, Any]) -> float:
    ts = payload.get("timestamp")
    if ts is None:
        ts = payload.get("time")
    if ts is None:
        ts = datetime.utcnow().timestamp()
    return _coerce_float(ts, default=datetime.utcnow().timestamp())


def _summarise_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(detections)
    classes: Dict[str, int] = {}
    avg_conf = 0.0
    for det in detections:
        cls = str(det.get("class") or det.get("label") or "unknown")
        classes[cls] = classes.get(cls, 0) + 1
        avg_conf += _coerce_float(det.get("confidence"), default=0.0)
    if total:
        avg_conf /= total
    return {"total": total, "by_class": classes, "mean_confidence": avg_conf}


def _extract_pose_metrics(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    pose_entries = [det.get("pose") for det in detections if det.get("pose")]
    if not pose_entries:
        return {}
    speeds = [_coerce_float(pose.get("speed"), default=0.0) for pose in pose_entries]  # type: ignore[arg-type]
    directions = [pose.get("direction") for pose in pose_entries if isinstance(pose.get("direction"), str)]  # type: ignore[arg-type]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    return {
        "pose_count": len(pose_entries),
        "mean_speed": avg_speed,
        "directions": directions,
    }


def _enqueue_telemetry(payload: Dict[str, Any]) -> None:
    if telemetry_event is None:
        return
    ts = _timestamp(payload)
    detections: List[Dict[str, Any]] = payload.get("detections") or []
    emotion: Dict[str, Any] = payload.get("emotion") or {}
    meta: Dict[str, Any] = payload.get("meta") or {}

    summary = _summarise_detections(detections)
    pose_summary = _extract_pose_metrics(detections)

    payload_out: Dict[str, Any] = {
        "ts": ts,
        "valence": _coerce_float(emotion.get("valence"), default=0.0),
        "arousal": _coerce_float(emotion.get("arousal"), default=0.0),
        "dominance": _coerce_float(emotion.get("dominance"), default=0.0),
        "detections_total": summary["total"],
        "detections_by_class": summary["by_class"],
        "detections_mean_confidence": summary["mean_confidence"],
    }
    if meta:
        payload_out["meta"] = meta
    if pose_summary:
        payload_out["pose"] = pose_summary

    telemetry_event("vision.metrics", payload_out)


@dataclass
class VisionRuntimeBridgeConfig:
    state_dir: str = "data/state_hub"
    log_path: Path = Path("logs/streaming_vlm_turns.jsonl")
    use_eqnet_core: bool = True


class VisionRuntimeBridge:
    def __init__(self, config: Optional[VisionRuntimeBridgeConfig] = None) -> None:
        self.config = config or VisionRuntimeBridgeConfig()
        self._runtime: Any = None

    def _runtime_instance(self) -> Any:
        if self._runtime is None:
            EmotionalHubRuntime, RuntimeConfig = _load_runtime_classes()
            self._runtime = EmotionalHubRuntime(
                RuntimeConfig(
                    use_eqnet_core=self.config.use_eqnet_core,
                    eqnet_state_dir=self.config.state_dir,
                )
            )
        return self._runtime

    def process_image(
        self,
        *,
        image_path: str,
        prompt: str,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        sensor_frame: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        runtime = self._runtime_instance()
        if sensor_frame:
            runtime.on_sensor_tick(dict(sensor_frame))
        result = runtime.process_turn(
            user_text=prompt,
            context=context,
            intent=intent,
            fast_only=fast_only,
            image_path=image_path,
        )
        payload = _serialize_result(result)
        event = {
            "timestamp": time.time(),
            "image_path": image_path,
            "prompt": prompt,
            "context": context,
            "intent": intent,
            "sensor_summary": summarize_sensor_frame(sensor_frame),
            "result": payload,
        }
        _append_jsonl(self.config.log_path, event)
        return payload

    def get_2d_state(self) -> Dict[str, Any]:
        runtime = self._runtime_instance()
        payload = runtime.serialize_2d_state()
        return dict(payload)

    def ingest_2d_event(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        runtime = self._runtime_instance()
        result = runtime.ingest_2d_event(payload)
        return dict(result)

    def shutdown(self) -> None:
        if self._runtime is None:
            return
        try:
            self._runtime.shutdown()
        finally:
            self._runtime = None


_runtime_bridge: Optional[VisionRuntimeBridge] = None


def _runtime_bridge_enabled() -> bool:
    raw = os.getenv("EQNET_VISION_BRIDGE_RUNTIME_ENABLED", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_runtime_bridge() -> VisionRuntimeBridge:
    global _runtime_bridge
    if _runtime_bridge is None:
        _runtime_bridge = VisionRuntimeBridge(
            VisionRuntimeBridgeConfig(
                state_dir=os.getenv("EQNET_VISION_BRIDGE_STATE_DIR", "data/state_hub"),
                log_path=Path(os.getenv("EQNET_VISION_BRIDGE_LOG_PATH", "logs/streaming_vlm_turns.jsonl")),
                use_eqnet_core=os.getenv("EQNET_VISION_BRIDGE_USE_EQNET_CORE", "1").strip().lower() not in {"0", "false", "no", "off"},
            )
        )
    return _runtime_bridge


def _serialize_result(result: Any) -> Dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return dict(result)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_candidates(frames_dir: Path, patterns: Iterable[str]) -> List[Path]:
    items: List[Path] = []
    for pattern in patterns:
        items.extend(path for path in frames_dir.glob(pattern) if path.is_file())
    return sorted(set(items), key=lambda path: path.stat().st_mtime, reverse=True)


def select_stable_frame(frames_dir: Path, patterns: Iterable[str], *, settle_seconds: float) -> Optional[Path]:
    now = time.time()
    for path in iter_candidates(frames_dir, patterns):
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age >= settle_seconds:
            return path
    return None


def run_frame_watch(
    *,
    frames_dir: Path,
    patterns: Iterable[str],
    prompt: str,
    context: Optional[str],
    intent: Optional[str],
    fast_only: bool,
    interval: float,
    settle_seconds: float,
    once: bool,
    bridge: Optional[VisionRuntimeBridge] = None,
) -> int:
    runtime_bridge = bridge or get_runtime_bridge()
    seen_key: Optional[tuple[str, float]] = None
    while True:
        frame_path = select_stable_frame(frames_dir, patterns, settle_seconds=max(0.0, settle_seconds))
        if frame_path is None:
            if once:
                return 1
            time.sleep(max(0.1, interval))
            continue

        stat = frame_path.stat()
        frame_key = (str(frame_path.resolve()), float(stat.st_mtime))
        if frame_key == seen_key:
            if once:
                return 0
            time.sleep(max(0.1, interval))
            continue

        payload = runtime_bridge.process_image(
            image_path=str(frame_path),
            prompt=prompt,
            context=context,
            intent=intent,
            fast_only=fast_only,
        )
        response = payload.get("response") or {}
        print(
            json.dumps(
                {
                    "image_path": str(frame_path),
                    "response_route": payload.get("response_route"),
                    "text": response.get("text"),
                    "perception_summary": response.get("perception_summary"),
                    "retrieval_summary": response.get("retrieval_summary"),
                },
                ensure_ascii=True,
            )
        )

        seen_key = frame_key
        if once:
            return 0
        time.sleep(max(0.1, interval))


@app.post("/vision-webhook")
async def ingest_vision(req: Request) -> Dict[str, str]:
    payload: Dict[str, Any] = await req.json()
    _enqueue_telemetry(payload)
    return {"status": "ok"}


@app.post("/vision-frame")
async def ingest_frame(req: Request) -> Dict[str, Any]:
    if not _runtime_bridge_enabled():
        raise HTTPException(status_code=503, detail="runtime_bridge_disabled")
    payload: Dict[str, Any] = await req.json()
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise HTTPException(status_code=400, detail="image_path_required")
    frame = Path(image_path)
    if not frame.exists() or not frame.is_file():
        raise HTTPException(status_code=404, detail="image_not_found")
    sensor_frame = build_sensor_frame(payload)
    result = get_runtime_bridge().process_image(
        image_path=str(frame),
        prompt=str(payload.get("prompt") or "Observe the frame and respond briefly."),
        context=payload.get("context"),
        intent=payload.get("intent"),
        fast_only=bool(payload.get("fast_only", False)),
        sensor_frame=sensor_frame,
    )
    return {
        "status": "ok",
        "sensor_schema": "vision_sensor_frame/v1",
        "sensor_summary": summarize_sensor_frame(sensor_frame),
        "result": result,
    }


@app.get("/project-atri/2d-state")
async def project_atri_2d_state() -> Dict[str, Any]:
    if not _runtime_bridge_enabled():
        raise HTTPException(status_code=503, detail="runtime_bridge_disabled")
    return {
        "status": "ok",
        "state": get_runtime_bridge().get_2d_state(),
    }


@app.post("/project-atri/2d-event")
async def project_atri_2d_event(req: Request) -> Dict[str, Any]:
    if not _runtime_bridge_enabled():
        raise HTTPException(status_code=503, detail="runtime_bridge_disabled")
    payload: Dict[str, Any] = await req.json()
    try:
        result = get_runtime_bridge().ingest_2d_event(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "ok",
        "result": result,
        "state": get_runtime_bridge().get_2d_state(),
    }


@app.get("/health")
async def healthcheck() -> Dict[str, Any]:
    return {
        "status": "ok",
        "runtime_bridge_enabled": _runtime_bridge_enabled(),
    }


__all__ = [
    "VisionRuntimeBridge",
    "VisionRuntimeBridgeConfig",
    "app",
    "get_runtime_bridge",
    "iter_candidates",
    "run_frame_watch",
    "select_stable_frame",
]






