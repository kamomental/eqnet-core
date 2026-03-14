from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from scripts.vision_bridge import get_runtime_bridge

router = APIRouter()


def _runtime_bridge_enabled() -> bool:
    raw = os.getenv("EQNET_VISION_BRIDGE_RUNTIME_ENABLED", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@router.get("/project-atri/2d-state")
async def project_atri_2d_state() -> Dict[str, Any]:
    if not _runtime_bridge_enabled():
        raise HTTPException(status_code=503, detail="runtime_bridge_disabled")
    return {
        "status": "ok",
        "state": get_runtime_bridge().get_2d_state(),
    }


@router.post("/project-atri/2d-event")
async def project_atri_2d_event(req: Request) -> Dict[str, Any]:
    if not _runtime_bridge_enabled():
        raise HTTPException(status_code=503, detail="runtime_bridge_disabled")
    payload = await req.json()
    try:
        result = get_runtime_bridge().ingest_2d_event(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "ok",
        "result": result,
        "state": get_runtime_bridge().get_2d_state(),
    }


__all__ = ["router"]
