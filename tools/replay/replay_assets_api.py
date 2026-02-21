#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replay asset API for browser-side save operations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ANCHOR_KEYS = ("left_eye", "right_eye", "mouth")


def _norm_name(value: str, fallback: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip()).strip("_").lower()
    return s or fallback


def _parse_point(raw: Any, key: str) -> Dict[str, float]:
    if isinstance(raw, dict):
        x = raw.get("x")
        y = raw.get("y")
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        x = raw[0]
        y = raw[1]
    else:
        raise ValueError(f"{key} must be object {{x,y}} or [x,y]")
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} has non-numeric x/y") from exc
    return {"x": round(xf, 2), "y": round(yf, 2)}


def _validate_anchors(data: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(data, dict):
        raise ValueError("anchors must be a JSON object")
    out: Dict[str, Dict[str, float]] = {}
    for key in ANCHOR_KEYS:
        if key not in data:
            raise ValueError(f"missing key: {key}")
        out[key] = _parse_point(data[key], key)
    return out


class AnchorRegisterRequest(BaseModel):
    character: str = "default"
    pattern: str = "base"
    anchors: Dict[str, Any]
    set_default: bool = True
    masks_root: str = "assets/replay/masks"


app = FastAPI(title="Replay Assets API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/replay/anchors/register")
def register_anchor_profile(payload: AnchorRegisterRequest) -> Dict[str, Any]:
    character = _norm_name(payload.character, "default")
    pattern = _norm_name(payload.pattern, "base")
    masks_root = Path(payload.masks_root)
    try:
        anchors = _validate_anchors(payload.anchors)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    char_dir = masks_root / character
    char_dir.mkdir(parents=True, exist_ok=True)

    profile_path = char_dir / f"anchors.{pattern}.json"
    profile_path.write_text(json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")

    default_path = char_dir / "anchors.gt.json"
    if payload.set_default:
        default_path.write_text(json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "character": character,
        "pattern": pattern,
        "profile_path": profile_path.as_posix(),
        "default_path": default_path.as_posix() if payload.set_default else None,
        "anchors": anchors,
    }

