#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastMCP-compatible HTTP server exposing EQNet inner-state resources and A2A tools.

Usage:
    uvicorn scripts.run_mcp_agent:app --host 0.0.0.0 --port 8055 --reload
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ops.a2a_router import A2ARouter

try:
    from telemetry import event as telemetry_event
except Exception:  # pragma: no cover - telemetry is optional during unit tests
    telemetry_event = None  # type: ignore


READ_ONLY = os.getenv("EQNET_MCP_READ_ONLY", "").lower() in {"1", "true", "yes", "on"}

app = FastAPI(
    title="EQNet FastMCP Agent",
    version="2025.10",
    description="Expose EQNet internal state via Model Context Protocol resources/tools.",
)
router = A2ARouter(log_dir=os.getenv("EQNET_A2A_LOG_DIR", "logs/a2a"))


# --------------------------------------------------------------------------- util
def _load_nightly(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        override = os.getenv("EQNET_NIGHTLY_PATH")
        path = Path(override) if override else Path("reports/nightly.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="nightly report not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"failed to parse nightly report: {exc}") from exc


def _extract(res: Dict[str, Any], key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    value = res.get(key) if isinstance(res, dict) else None
    if isinstance(value, dict):
        return value
    return default or {}


def _ensure_writable(tool_name: str) -> None:
    if READ_ONLY:
        raise HTTPException(status_code=403, detail=f"{tool_name} disabled in read-only mode")


# ----------------------------------------------------------------------- resources
@app.get("/mcp/capabilities")
def get_capabilities() -> Dict[str, Any]:
    caps = router.capabilities()
    caps.setdefault("prompts", [{"name": "narrative:alert_explainer", "io": "json"}])
    return caps


@app.get("/mcp/resources/resonance/summary")
def get_resonance_summary() -> Dict[str, Any]:
    report = _load_nightly()
    resonance = _extract(report, "resonance")
    summary = resonance.get("summary") or resonance
    if not summary:
        raise HTTPException(status_code=404, detail="resonance summary unavailable")
    return summary


@app.get("/mcp/resources/vision/snapshot")
def get_vision_snapshot() -> Dict[str, Any]:
    report = _load_nightly()
    snapshot = _extract(report, "vision_snapshot")
    if not snapshot:
        raise HTTPException(status_code=404, detail="vision snapshot unavailable")
    return snapshot


@app.get("/mcp/resources/culture/feedback")
def get_culture_feedback() -> Dict[str, Any]:
    report = _load_nightly()
    policy = _extract(report, "policy_feedback")
    if not policy:
        raise HTTPException(status_code=404, detail="policy feedback unavailable")
    return policy


# --------------------------------------------------------------------------- tools
@app.post("/mcp/tools/telemetry/vision.push")
def push_vision_metrics(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    _ensure_writable("telemetry:vision.push")
    if telemetry_event is None:
        raise HTTPException(status_code=503, detail="telemetry subsystem not available")
    telemetry_event("vision.metrics", payload)
    return {"status": "accepted"}


@app.post("/mcp/tools/a2a/contract.open")
def open_contract(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    try:
        return router.open_contract(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/mcp/tools/a2a/turn.post")
def post_turn(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    _ensure_writable("a2a:turn.post")
    try:
        return router.append_turn(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/mcp/tools/a2a/score.report")
def post_score(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    _ensure_writable("a2a:score.report")
    try:
        return router.record_score(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/mcp/tools/a2a/session.close")
def close_session(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    _ensure_writable("a2a:session.close")
    session_id = str(payload.get("session_id"))
    reason = str(payload.get("reason", "completed"))
    try:
        return router.close(session_id, reason=reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ------------------------------------------------------------------------ misc api
@app.get("/mcp/a2a/session/{session_id}")
def get_session_snapshot(session_id: str) -> Dict[str, Any]:
    try:
        return router.session_snapshot(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/healthz")
def healthcheck() -> JSONResponse:
    try:
        _ = _load_nightly()
    except HTTPException:
        nightly_ok = False
    else:
        nightly_ok = True
    return JSONResponse({"ok": True, "nightly": nightly_ok})
