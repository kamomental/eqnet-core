#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-Agent webhook bridge for EQNet telemetry.

This small FastAPI application receives callbacks from Stream Vision-Agents
and forwards the useful signals (valence, arousal, detections, poses, etc.)
to EQNet's telemetry pipeline so that the emotional field / nightly reports
can ingest them.

Usage:

    uvicorn scripts.vision_bridge:app --host 0.0.0.0 --port 8000

Configure Vision-Agents to POST JSON events to
`http://<bridge-host>:8000/vision-webhook`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request

try:
    from telemetry import event as telemetry_event
except Exception:  # pragma: no cover - telemetry unavailable in certain envs
    telemetry_event = None  # type: ignore


app = FastAPI(title="Vision→EQNet Bridge", version="0.1.0")


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

    valence = _coerce_float(emotion.get("valence"), default=0.0)
    arousal = _coerce_float(emotion.get("arousal"), default=0.0)
    dominance = _coerce_float(emotion.get("dominance"), default=0.0)

    payload_out: Dict[str, Any] = {
        "ts": ts,
        "valence": valence,
        "arousal": arousal,
        "dominance": dominance,
        "detections_total": summary["total"],
        "detections_by_class": summary["by_class"],
        "detections_mean_confidence": summary["mean_confidence"],
    }
    if meta:
        payload_out["meta"] = meta
    if pose_summary:
        payload_out["pose"] = pose_summary

    telemetry_event("vision.metrics", payload_out)


@app.post("/vision-webhook")
async def ingest_vision(req: Request) -> Dict[str, str]:
    payload: Dict[str, Any] = await req.json()
    _enqueue_telemetry(payload)
    return {"status": "ok"}


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


__all__ = ["app"]
