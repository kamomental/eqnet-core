#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Register anchor JSON into canonical replay mask path.

Examples:
  python tools/replay/register_anchor_profile.py ^
    --character default ^
    --pattern base ^
    --anchors-file C:\\Users\\you\\Downloads\\anchors.default.base.json ^
    --set-default

  python tools/replay/register_anchor_profile.py ^
    --character default ^
    --pattern angle_left ^
    --anchors-json "{\"left_eye\":{\"x\":230.4,\"y\":482.9},\"right_eye\":{\"x\":487.2,\"y\":522.9},\"mouth\":{\"x\":353.1,\"y\":749.7}}"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

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


def _read_input(anchors_file: str, anchors_json: str) -> Dict[str, Dict[str, float]]:
    if anchors_file:
        payload = json.loads(Path(anchors_file).read_text(encoding="utf-8"))
        return _validate_anchors(payload)
    if anchors_json:
        payload = json.loads(anchors_json)
        return _validate_anchors(payload)
    raise ValueError("either --anchors-file or --anchors-json is required")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True, help="Character id (folder name)")
    ap.add_argument("--pattern", required=True, help="Pattern id (file suffix)")
    ap.add_argument("--anchors-file", default="", help="Source anchors JSON file")
    ap.add_argument("--anchors-json", default="", help="Source anchors JSON string")
    ap.add_argument(
        "--masks-root",
        default="assets/replay/masks",
        help="Root directory for masks (default: assets/replay/masks)",
    )
    ap.add_argument(
        "--set-default",
        action="store_true",
        help="Also update anchors.gt.json to this profile",
    )
    args = ap.parse_args()

    character = _norm_name(args.character, "default")
    pattern = _norm_name(args.pattern, "base")
    anchors = _read_input(args.anchors_file, args.anchors_json)

    char_dir = Path(args.masks_root) / character
    char_dir.mkdir(parents=True, exist_ok=True)
    profile_path = char_dir / f"anchors.{pattern}.json"
    profile_path.write_text(json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")

    default_path = char_dir / "anchors.gt.json"
    if args.set_default:
        default_path.write_text(json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] profile: {profile_path.as_posix()}")
    if args.set_default:
        print(f"[OK] default: {default_path.as_posix()}")


if __name__ == "__main__":
    main()

