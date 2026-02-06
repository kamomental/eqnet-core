from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from .cfg import AvatarConfig
from .contracts import AvatarState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class AvatarRenderer:
    def __init__(self, cfg: AvatarConfig) -> None:
        self._cfg = cfg
        self._base = self._load_base(cfg.base_image)

    def _load_base(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Avatar base image not found: {path}")
        return Image.open(path).convert("RGBA")

    def render(self, state: AvatarState, t_sec: float) -> Image.Image:
        cfg = self._cfg
        base = self._base.copy()
        draw = ImageDraw.Draw(base)

        if state.blink > 0.001:
            lid_h = int(cfg.eyes.h * _clamp01(state.blink))
            draw.rectangle(
                [
                    cfg.eyes.x,
                    cfg.eyes.y,
                    cfg.eyes.x + cfg.eyes.w,
                    cfg.eyes.y + lid_h,
                ],
                fill=cfg.colors.bg,
            )

        mouth_center_y = cfg.mouth.y + int(cfg.mouth.h * 0.5)
        mouth_open_px = int(cfg.mouth.h * 0.8 * _clamp01(state.mouth_open))
        amp = cfg.mouth_cfg.wavy_amp_px * (1.0 + 2.0 * _clamp01(state.wavy))
        steps = max(2, cfg.mouth_cfg.steps)

        points = []
        for i in range(steps + 1):
            x = cfg.mouth.x + int((cfg.mouth.w * i) / steps)
            phase = (i / steps) * math.pi * 2.0
            y = mouth_center_y + int(math.sin(phase + t_sec * 6.0) * amp) + mouth_open_px
            points.append((x, y))
        draw.line(points, fill=cfg.colors.green, width=1)

        if state.fang_skin:
            fang_x = cfg.mouth.x + int(cfg.mouth.w * cfg.mouth_cfg.fang_x_ratio)
            fang_y = mouth_center_y + mouth_open_px - 1
            draw.rectangle(
                [fang_x, fang_y, fang_x + 1, fang_y + 1], fill=cfg.colors.skin
            )

        scale = max(1, int(cfg.canvas_scale))
        scaled = base.resize(
            (base.width * scale, base.height * scale), resample=Image.NEAREST
        )

        bob_px = cfg.motion.bob_px * _clamp01(state.bob)
        hop_px = cfg.motion.hop_px * _clamp01(state.hop)
        y_off = int(round((bob_px + hop_px) * scale))
        canvas = Image.new("RGB", scaled.size, cfg.colors.bg)
        canvas.paste(scaled, (0, y_off), scaled)
        return canvas
