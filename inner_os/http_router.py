from __future__ import annotations

from typing import Any, Dict, Mapping

from fastapi import APIRouter, HTTPException, Request

from .http_contract import build_inner_os_manifest
from .service import InnerOSService

router = APIRouter()
_service = InnerOSService()


def _coerce_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("payload_must_be_mapping")
    return payload


async def _read_payload(req: Request) -> Mapping[str, Any]:
    try:
        payload = await req.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_json") from exc
    try:
        return _coerce_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/inner-os/manifest")
async def inner_os_manifest() -> Dict[str, Any]:
    return {"status": "ok", "manifest": build_inner_os_manifest()}


@router.post("/inner-os/pre-turn-update")
async def inner_os_pre_turn_update(req: Request) -> Dict[str, Any]:
    payload = await _read_payload(req)
    return {"status": "ok", "result": _service.pre_turn_update(payload)}


@router.post("/inner-os/memory-recall")
async def inner_os_memory_recall(req: Request) -> Dict[str, Any]:
    payload = await _read_payload(req)
    return {"status": "ok", "result": _service.memory_recall(payload)}


@router.post("/inner-os/response-gate")
async def inner_os_response_gate(req: Request) -> Dict[str, Any]:
    payload = await _read_payload(req)
    return {"status": "ok", "result": _service.response_gate(payload)}


@router.post("/inner-os/post-turn-update")
async def inner_os_post_turn_update(req: Request) -> Dict[str, Any]:
    payload = await _read_payload(req)
    return {"status": "ok", "result": _service.post_turn_update(payload)}


__all__ = ["router"]
