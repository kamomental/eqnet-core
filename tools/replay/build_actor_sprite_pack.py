#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build pixel-style actor sprite sheet + manifest entry.

Examples:
  python tools/replay/build_actor_sprite_pack.py ^
    --source assets/replay/src/default.png ^
    --actor-id default

  python tools/replay/build_actor_sprite_pack.py ^
    --source-dir assets/replay/character/default ^
    --actor-id default ^
    --manifest assets/replay/actors/actors_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image, ImageChops, ImageDraw

ROLE_PRESETS = {
    "default": {
        "helmet": None,
        "hat": (108, 122, 148, 255),
        "skin": (236, 199, 182, 255),
        "torso": (215, 228, 245, 255),
        "torso_line": None,
        "arms": (78, 95, 128, 255),
        "legs": (65, 86, 118, 255),
        "boots": (36, 40, 50, 255),
        "tool": None,
        "cheek": False,
    },
    "worker": {
        "helmet": (244, 194, 71, 255),
        "helmet_shadow": (214, 163, 48, 255),
        "skin": (245, 209, 194, 255),
        "torso": (249, 143, 52, 255),
        "torso_line": (255, 236, 171, 255),
        "arms": (78, 95, 128, 255),
        "legs": (78, 95, 128, 255),
        "boots": (40, 46, 58, 255),
        "tool": ("hammer", (170, 172, 183, 255), (101, 82, 54, 255)),
        "cheek": True,
    },
    "guard": {
        "helmet": (52, 74, 112, 255),
        "helmet_shadow": (38, 56, 89, 255),
        "skin": (230, 194, 180, 255),
        "torso": (46, 77, 132, 255),
        "torso_line": None,
        "arms": (34, 61, 110, 255),
        "legs": (39, 60, 93, 255),
        "boots": (29, 35, 45, 255),
        "tool": ("spear", (181, 188, 199, 255), (112, 88, 62, 255)),
        "cheek": False,
    },
    "vendor": {
        "helmet": None,
        "hat": (195, 150, 104, 255),
        "skin": (236, 199, 182, 255),
        "torso": (82, 142, 79, 255),
        "torso_line": (240, 236, 220, 255),
        "arms": (116, 84, 63, 255),
        "legs": (120, 74, 52, 255),
        "boots": (52, 42, 36, 255),
        "tool": ("bag", (160, 126, 77, 255), None),
        "cheek": True,
    },
    "farmer": {
        "helmet": None,
        "hat": (203, 171, 104, 255),
        "skin": (236, 200, 182, 255),
        "torso": (101, 153, 83, 255),
        "torso_line": None,
        "arms": (116, 88, 62, 255),
        "legs": (78, 114, 75, 255),
        "boots": (63, 52, 44, 255),
        "tool": ("hoe", (152, 154, 158, 255), (116, 88, 58, 255)),
        "cheek": False,
    },
    "fisher": {
        "helmet": (84, 126, 170, 255),
        "helmet_shadow": (61, 96, 137, 255),
        "skin": (235, 198, 182, 255),
        "torso": (70, 122, 160, 255),
        "torso_line": None,
        "arms": (58, 96, 130, 255),
        "legs": (61, 93, 124, 255),
        "boots": (33, 42, 56, 255),
        "tool": ("rod", (193, 198, 210, 255), (89, 74, 55, 255)),
        "cheek": False,
    },
    "medic": {
        "helmet": None,
        "hat": (236, 236, 244, 255),
        "skin": (239, 203, 186, 255),
        "torso": (226, 233, 246, 255),
        "torso_line": (228, 94, 94, 255),
        "arms": (90, 102, 125, 255),
        "legs": (103, 120, 146, 255),
        "boots": (45, 56, 73, 255),
        "tool": ("crossbag", (228, 94, 94, 255), None),
        "cheek": True,
    },
    "engineer": {
        "helmet": (237, 179, 78, 255),
        "helmet_shadow": (200, 144, 57, 255),
        "skin": (238, 201, 183, 255),
        "torso": (84, 137, 168, 255),
        "torso_line": (244, 222, 158, 255),
        "arms": (68, 96, 128, 255),
        "legs": (72, 95, 125, 255),
        "boots": (36, 42, 53, 255),
        "tool": ("wrench", (181, 185, 196, 255), (102, 82, 58, 255)),
        "cheek": False,
    },
    "banker": {
        "helmet": None,
        "hat": (68, 76, 96, 255),
        "skin": (236, 198, 180, 255),
        "torso": (78, 90, 124, 255),
        "torso_line": (217, 223, 236, 255),
        "arms": (70, 80, 109, 255),
        "legs": (70, 80, 109, 255),
        "boots": (32, 36, 45, 255),
        "tool": ("brief", (145, 108, 74, 255), None),
        "cheek": False,
    },
    "blacksmith": {
        "helmet": (86, 92, 108, 255),
        "helmet_shadow": (63, 68, 83, 255),
        "skin": (233, 193, 176, 255),
        "torso": (127, 84, 67, 255),
        "torso_line": (206, 165, 113, 255),
        "arms": (101, 74, 61, 255),
        "legs": (84, 86, 104, 255),
        "boots": (31, 35, 44, 255),
        "tool": ("hammer", (176, 179, 186, 255), (82, 62, 44, 255)),
        "cheek": False,
    },
    "teacher": {
        "helmet": None,
        "hat": (142, 118, 95, 255),
        "skin": (236, 198, 182, 255),
        "torso": (164, 133, 95, 255),
        "torso_line": (236, 214, 176, 255),
        "arms": (118, 96, 74, 255),
        "legs": (92, 85, 104, 255),
        "boots": (42, 40, 50, 255),
        "tool": ("book", (220, 96, 96, 255), None),
        "cheek": True,
    },
}


def _ensure_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _alpha_bbox(img: Image.Image) -> Tuple[int, int, int, int]:
    alpha = img.getchannel("A")
    box = alpha.getbbox()
    if box is None:
        return (0, 0, img.width, img.height)
    return box


def _composite_from_layers(source_dir: Path) -> Image.Image:
    layer_order = [
        "base.png",
        "body.png",
        "hair_back.png",
        "head.png",
        "eye_open.png",
        "mouth_n.png",
        "hair_front.png",
    ]
    existing = [source_dir / name for name in layer_order if (source_dir / name).exists()]
    if not existing:
        raise FileNotFoundError(f"no known layer files found in: {source_dir}")
    base = _ensure_rgba(existing[0])
    for layer_path in existing[1:]:
        base.alpha_composite(_ensure_rgba(layer_path))
    return base


def _template_actor(canvas_px: int, role: str, style_mode: str = "normal") -> Image.Image:
    key = str(role or "default").strip().lower()
    role_style = ROLE_PRESETS.get(key, ROLE_PRESETS["default"])
    style_mode = str(style_mode or "normal").strip().lower()
    is_cute = style_mode == "cute"
    img = Image.new("RGBA", (canvas_px, canvas_px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = canvas_px // 2
    by = int(canvas_px * 0.92)
    # Base chibi silhouette. "cute" shifts toward bigger head + softer body.
    leg_top = by - (22 if is_cute else 24)
    torso_top = by - (45 if is_cute else 43)
    head_top = by - (70 if is_cute else 66)
    head_bottom = by - (41 if is_cute else 42)
    d.rectangle((cx - 11, leg_top, cx - 4, by - 11), fill=role_style["legs"])
    d.rectangle((cx + 4, leg_top, cx + 11, by - 11), fill=role_style["legs"])
    d.rounded_rectangle((cx - 12, by - 11, cx - 3, by - 4), radius=2, fill=role_style["boots"])
    d.rounded_rectangle((cx + 3, by - 11, cx + 12, by - 4), radius=2, fill=role_style["boots"])
    d.rounded_rectangle((cx - 17, torso_top, cx + 17, by - 24), radius=3 if is_cute else 2, fill=role_style["torso"])
    if role_style.get("torso_line") is not None:
        d.rectangle((cx - 4, torso_top, cx + 4, by - 24), fill=role_style["torso_line"])
    d.rounded_rectangle((cx - 22, by - 40, cx - 16, by - 27), radius=2, fill=role_style["arms"])
    d.rounded_rectangle((cx + 16, by - 40, cx + 22, by - 27), radius=2, fill=role_style["arms"])

    # Face + head.
    d.ellipse((cx - 15, head_top, cx + 15, head_bottom), fill=role_style["skin"])
    eye_y = by - (56 if is_cute else 56)
    if is_cute:
        d.rectangle((cx - 8, eye_y, cx - 5, eye_y + 3), fill=(45, 54, 76, 255))
        d.rectangle((cx + 5, eye_y, cx + 8, eye_y + 3), fill=(45, 54, 76, 255))
        d.rectangle((cx - 1, by - 50, cx + 1, by - 49), fill=(160, 82, 92, 255))
    else:
        d.rectangle((cx - 7, eye_y, cx - 4, eye_y + 3), fill=(48, 59, 82, 255))
        d.rectangle((cx + 4, eye_y, cx + 7, eye_y + 3), fill=(48, 59, 82, 255))
    if role_style.get("cheek") or is_cute:
        d.rectangle((cx - 12, by - 52, cx - 9, by - 49), fill=(242, 170, 166, 235))
        d.rectangle((cx + 9, by - 52, cx + 12, by - 49), fill=(242, 170, 166, 235))

    if role_style.get("helmet") is not None:
        d.ellipse((cx - 19, by - 78, cx + 19, by - 54), fill=role_style["helmet"])
        d.rectangle((cx - 19, by - 63, cx + 19, by - 58), fill=role_style.get("helmet_shadow", role_style["helmet"]))
        d.rectangle((cx - 9, by - 55, cx + 9, by - 52), fill=(57, 62, 76, 255))
    elif role_style.get("hat") is not None:
        d.rectangle((cx - 15, by - 72, cx + 15, by - 64), fill=role_style["hat"])

    # Dedicated cute worker silhouette so the difference is explicit.
    if is_cute and key == "worker":
        d.rounded_rectangle((cx - 18, by - 47, cx + 18, by - 22), radius=5, fill=(247, 141, 56, 255))
        d.rounded_rectangle((cx - 5, by - 47, cx + 5, by - 22), radius=3, fill=(255, 233, 166, 255))
        d.ellipse((cx - 18, by - 76, cx + 18, by - 50), fill=(246, 199, 76, 255))
        d.rectangle((cx - 17, by - 64, cx + 17, by - 58), fill=(216, 168, 52, 255))
        d.ellipse((cx - 12, by - 60, cx + 12, by - 43), fill=(244, 206, 192, 255))
        d.rectangle((cx - 8, by - 54, cx - 5, by - 51), fill=(42, 52, 74, 255))
        d.rectangle((cx + 5, by - 54, cx + 8, by - 51), fill=(42, 52, 74, 255))
        d.rectangle((cx - 1, by - 49, cx + 1, by - 48), fill=(162, 84, 93, 255))
        d.rectangle((cx - 13, by - 50, cx - 10, by - 47), fill=(244, 170, 166, 235))
        d.rectangle((cx + 10, by - 50, cx + 13, by - 47), fill=(244, 170, 166, 235))
        d.rounded_rectangle((cx - 13, by - 22, cx - 4, by - 4), radius=2, fill=(75, 95, 128, 255))
        d.rounded_rectangle((cx + 4, by - 22, cx + 13, by - 4), radius=2, fill=(75, 95, 128, 255))
        d.rounded_rectangle((cx - 14, by - 8, cx - 3, by - 3), radius=2, fill=(33, 39, 52, 255))
        d.rounded_rectangle((cx + 3, by - 8, cx + 14, by - 3), radius=2, fill=(33, 39, 52, 255))

    # Tool silhouette per role.
    tool = role_style.get("tool")
    if tool:
        ttype = tool[0]
        main = tool[1]
        sub = tool[2] if len(tool) > 2 else None
        if ttype == "hammer":
            d.rectangle((cx + 20, by - 36, cx + 23, by - 18), fill=sub or (101, 82, 54, 255))
            d.rounded_rectangle((cx + 16, by - 36, cx + 28, by - 32), radius=1, fill=main)
        elif ttype == "spear":
            d.rectangle((cx + 22, by - 42, cx + 24, by - 16), fill=sub or (112, 88, 62, 255))
            d.polygon([(cx + 21, by - 42), (cx + 25, by - 42), (cx + 23, by - 48)], fill=main)
        elif ttype == "rod":
            d.rectangle((cx + 22, by - 46, cx + 24, by - 16), fill=sub or (89, 74, 55, 255))
            d.arc((cx + 20, by - 50, cx + 36, by - 36), 220, 355, fill=main, width=1)
        elif ttype == "hoe":
            d.rectangle((cx + 22, by - 44, cx + 24, by - 16), fill=sub or (116, 88, 58, 255))
            d.rectangle((cx + 17, by - 44, cx + 30, by - 40), fill=main)
        elif ttype == "wrench":
            d.rectangle((cx + 21, by - 40, cx + 24, by - 17), fill=sub or (102, 82, 58, 255))
            d.ellipse((cx + 18, by - 41, cx + 28, by - 35), outline=main, width=1)
        elif ttype == "brief":
            d.rounded_rectangle((cx + 18, by - 36, cx + 29, by - 28), radius=1, fill=main)
        elif ttype == "book":
            d.rounded_rectangle((cx + 18, by - 36, cx + 29, by - 27), radius=1, fill=main)
            d.line((cx + 23, by - 36, cx + 23, by - 27), fill=(245, 230, 208, 255), width=1)
        elif ttype == "crossbag":
            d.rounded_rectangle((cx + 18, by - 36, cx + 29, by - 27), radius=1, fill=(245, 233, 220, 255))
            d.rectangle((cx + 22, by - 33, cx + 25, by - 30), fill=main)
    if is_cute:
        # tiny highlight to make eyes look lively in low-res.
        d.point((cx - 6, by - 56), fill=(240, 240, 248, 255))
        d.point((cx + 6, by - 56), fill=(240, 240, 248, 255))
    return img


def _fit_to_canvas(img: Image.Image, canvas_px: int = 96) -> Image.Image:
    x0, y0, x1, y1 = _alpha_bbox(img)
    crop = img.crop((x0, y0, x1, y1))
    out = Image.new("RGBA", (canvas_px, canvas_px), (0, 0, 0, 0))
    if crop.width < 1 or crop.height < 1:
        return out
    target_h = int(canvas_px * 0.82)
    target_w = max(1, int(crop.width * (target_h / max(1, crop.height))))
    resized = crop.resize((target_w, target_h), Image.LANCZOS)
    ox = (canvas_px - target_w) // 2
    oy = canvas_px - target_h - int(canvas_px * 0.04)
    out.alpha_composite(resized, (ox, oy))
    return out


def _pixelate(img: Image.Image, frame_px: int, dot_scale: int) -> Image.Image:
    small_px = max(8, frame_px // max(1, dot_scale))
    tiny = img.resize((small_px, small_px), Image.NEAREST)
    pal = tiny.convert("P", palette=Image.ADAPTIVE, colors=48)
    return pal.convert("RGBA").resize((frame_px, frame_px), Image.NEAREST)


def _shift(img: Image.Image, dx: int, dy: int) -> Image.Image:
    return ImageChops.offset(img, dx, dy)


def _frame_idle(base: Image.Image) -> Image.Image:
    return base.copy()


def _frame_walk(base: Image.Image, phase: int) -> Image.Image:
    dy = -1 if phase % 2 == 0 else 1
    dx = -1 if phase % 2 == 0 else 1
    return _shift(base, dx, dy)


def _frame_talk(base: Image.Image) -> Image.Image:
    out = base.copy()
    d = ImageDraw.Draw(out)
    cx = out.width // 2
    cy = int(out.height * 0.62)
    d.ellipse((cx - 2, cy - 1, cx + 2, cy + 2), fill=(55, 28, 34, 255))
    return out


def _frame_blink(base: Image.Image) -> Image.Image:
    out = base.copy()
    d = ImageDraw.Draw(out)
    cy = int(out.height * 0.46)
    cx = out.width // 2
    d.line((cx - 6, cy, cx - 2, cy), fill=(25, 34, 52, 255), width=1)
    d.line((cx + 2, cy, cx + 6, cy), fill=(25, 34, 52, 255), width=1)
    return out


def _apply_role_style(base: Image.Image, role: str) -> Image.Image:
    key = str(role or "default").strip().lower()
    if key in {"default", ""}:
        return base
    out = base.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    cx = w // 2

    if key == "worker":
        # Strong override silhouette for clear role readability.
        # Head/helmet area
        d.ellipse((cx - int(w * 0.28), int(h * 0.12), cx + int(w * 0.28), int(h * 0.32)), fill=(236, 188, 56, 255))
        d.rectangle((cx - int(w * 0.28), int(h * 0.24), cx + int(w * 0.28), int(h * 0.30)), fill=(208, 159, 36, 255))
        d.rectangle((cx - int(w * 0.12), int(h * 0.30), cx + int(w * 0.12), int(h * 0.34)), fill=(58, 63, 78, 250))
        # Face opening
        d.ellipse((cx - int(w * 0.13), int(h * 0.32), cx + int(w * 0.13), int(h * 0.48)), fill=(238, 198, 180, 255))
        # Body/overall (opaque repaint to avoid "same character" feel)
        d.rectangle((cx - int(w * 0.24), int(h * 0.50), cx + int(w * 0.24), int(h * 0.82)), fill=(242, 130, 36, 255))
        d.rectangle((cx - int(w * 0.07), int(h * 0.50), cx + int(w * 0.07), int(h * 0.82)), fill=(246, 226, 148, 255))
        # Arms
        d.rectangle((cx - int(w * 0.30), int(h * 0.56), cx - int(w * 0.22), int(h * 0.74)), fill=(52, 72, 98, 255))
        d.rectangle((cx + int(w * 0.22), int(h * 0.56), cx + int(w * 0.30), int(h * 0.74)), fill=(52, 72, 98, 255))
        # Pants + boots
        d.rectangle((cx - int(w * 0.20), int(h * 0.82), cx + int(w * 0.20), int(h * 0.95)), fill=(66, 84, 116, 255))
        d.rectangle((cx - int(w * 0.22), int(h * 0.93), cx - int(w * 0.04), int(h * 0.98)), fill=(30, 36, 48, 255))
        d.rectangle((cx + int(w * 0.04), int(h * 0.93), cx + int(w * 0.22), int(h * 0.98)), fill=(30, 36, 48, 255))
        # Tool silhouette
        d.rectangle((cx + int(w * 0.24), int(h * 0.62), cx + int(w * 0.30), int(h * 0.88)), fill=(82, 68, 40, 255))
        d.rectangle((cx + int(w * 0.20), int(h * 0.60), cx + int(w * 0.34), int(h * 0.65)), fill=(146, 146, 156, 255))
    elif key == "guard":
        d.rectangle((cx - int(w * 0.22), int(h * 0.52), cx + int(w * 0.22), int(h * 0.83)), fill=(44, 74, 128, 245))
        d.rectangle((cx - int(w * 0.20), int(h * 0.22), cx + int(w * 0.20), int(h * 0.28)), fill=(60, 82, 118, 255))
    elif key == "vendor":
        d.rectangle((cx - int(w * 0.23), int(h * 0.55), cx + int(w * 0.23), int(h * 0.83)), fill=(78, 136, 74, 245))
        d.rectangle((cx - int(w * 0.23), int(h * 0.22), cx + int(w * 0.23), int(h * 0.27)), fill=(238, 232, 214, 255))
    return out


def _build_frames(base_px: Image.Image) -> Dict[str, Image.Image]:
    return {
        "idle": _frame_idle(base_px),
        "walk1": _frame_walk(base_px, 1),
        "walk2": _frame_walk(base_px, 2),
        "talk": _frame_talk(base_px),
        "blink": _frame_blink(base_px),
    }


def _sheet_from_frames(frames: Dict[str, Image.Image]) -> Tuple[Image.Image, Dict[str, int]]:
    order = ["idle", "walk1", "walk2", "talk", "blink"]
    frame_w = next(iter(frames.values())).width
    frame_h = next(iter(frames.values())).height
    sheet = Image.new("RGBA", (frame_w * len(order), frame_h), (0, 0, 0, 0))
    idx_map: Dict[str, int] = {}
    for i, key in enumerate(order):
        sheet.alpha_composite(frames[key], (i * frame_w, 0))
        idx_map[key] = i
    return sheet, idx_map


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _rel_web_path(path: Path) -> str:
    parts: Iterable[str] = path.as_posix().split("/")
    return "/" + "/".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build world actor sprite pack from source image/layers.")
    ap.add_argument("--source", type=Path, help="Single source image path.")
    ap.add_argument("--source-dir", type=Path, help="Layer directory (base/body/head...).")
    ap.add_argument(
        "--template",
        type=str,
        default="",
        help="Template actor type (default/worker/... ). If set, source is not required.",
    )
    ap.add_argument(
        "--list-roles",
        action="store_true",
        help="Print available template roles and exit.",
    )
    ap.add_argument(
        "--batch-professions",
        action="store_true",
        help="Generate a batch of profession actors into manifest (template mode).",
    )
    ap.add_argument("--actor-id", type=str, default="", help="Actor id key (e.g., default).")
    ap.add_argument("--frame-size", type=int, default=24, help="Output frame width/height.")
    ap.add_argument("--dot-scale", type=int, default=2, help="Pixelation scale factor.")
    ap.add_argument(
        "--style",
        type=str,
        default="normal",
        choices=["normal", "cute"],
        help="Template style mode.",
    )
    ap.add_argument(
        "--role",
        type=str,
        default="default",
        help="Role style overlay: default / worker / guard / vendor",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("assets/replay/actors"),
        help="Output root dir for actor sprite assets.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/replay/actors/actors_manifest.json"),
        help="Actors manifest path.",
    )
    args = ap.parse_args()

    if args.list_roles:
      print(",".join(sorted(ROLE_PRESETS.keys())))
      return

    if args.batch_professions:
        roles = [
            "worker",
            "guard",
            "vendor",
            "farmer",
            "fisher",
            "medic",
            "engineer",
            "banker",
            "blacksmith",
            "teacher",
        ]
        for role in roles:
            cmd_args = argparse.Namespace(**vars(args))
            cmd_args.template = role
            cmd_args.actor_id = f"{role}_01"
            cmd_args.role = "default"
            _run_single(cmd_args)
        return

    _run_single(args)


def _run_single(args: argparse.Namespace) -> None:
    if not str(args.actor_id or "").strip():
        raise SystemExit("--actor-id is required unless --list-roles or --batch-professions is used")
    if args.template:
        src = _template_actor(96, args.template, style_mode=args.style)
        px = _pixelate(src, frame_px=int(args.frame_size), dot_scale=max(1, int(args.dot_scale)))
    else:
        if not args.source and not args.source_dir:
            raise SystemExit("set either --source/--source-dir, or --template")
        if args.source:
            src = _ensure_rgba(args.source)
        else:
            src = _composite_from_layers(args.source_dir)
        fitted = _fit_to_canvas(src, canvas_px=96)
        px = _pixelate(fitted, frame_px=int(args.frame_size), dot_scale=int(args.dot_scale))
        px = _apply_role_style(px, args.role)
    frames = _build_frames(px)
    sheet, idx_map = _sheet_from_frames(frames)

    actor_dir = args.out_dir / args.actor_id
    actor_dir.mkdir(parents=True, exist_ok=True)
    out_sheet = actor_dir / "sprite_sheet.png"
    sheet.save(out_sheet)

    manifest = _load_json(args.manifest)
    actors = manifest.setdefault("actors", {})
    if not isinstance(actors, dict):
        actors = {}
        manifest["actors"] = actors
    manifest["version"] = int(manifest.get("version", 1) or 1)
    actors[args.actor_id] = {
        "sheet": _rel_web_path(out_sheet),
        "frame_w": px.width,
        "frame_h": px.height,
        "scale": 2.2,
        "animations": {
            "idle": [idx_map["idle"]],
            "walk": [idx_map["walk1"], idx_map["walk2"]],
            "talk": [idx_map["talk"], idx_map["idle"]],
            "blink": [idx_map["blink"], idx_map["idle"]],
        },
    }
    _write_json(args.manifest, manifest)
    print(f"[OK] wrote {out_sheet}")
    mode = f"template={args.template},style={args.style}" if getattr(args, "template", "") else f"role={args.role}"
    print(f"[OK] updated {args.manifest} actor={args.actor_id} {mode}")


if __name__ == "__main__":
    main()
