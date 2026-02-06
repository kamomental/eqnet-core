from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass(frozen=True)
class AvatarColors:
    bg: Tuple[int, int, int]
    green: Tuple[int, int, int]
    skin: Tuple[int, int, int]


@dataclass(frozen=True)
class AvatarRegion:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class AvatarBlinkCfg:
    base_interval_s: Tuple[float, float]
    close_s: float
    open_s: float


@dataclass(frozen=True)
class AvatarMotionCfg:
    bob_px: float
    hop_px: float
    bob_period_s: float
    hop_s: float
    blink: AvatarBlinkCfg


@dataclass(frozen=True)
class AvatarMouthCfg:
    wavy_amp_px: float
    steps: int
    fang_enable: bool
    fang_x_ratio: float


@dataclass(frozen=True)
class AvatarConfig:
    base_image: Path
    canvas_scale: int
    colors: AvatarColors
    eyes: AvatarRegion
    mouth: AvatarRegion
    motion: AvatarMotionCfg
    mouth_cfg: AvatarMouthCfg


def _to_int_tuple(values: Any) -> Tuple[int, int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("Expected 3-item color list.")
    return (int(values[0]), int(values[1]), int(values[2]))


def _to_float_pair(values: Any) -> Tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("Expected 2-item float list.")
    return (float(values[0]), float(values[1]))


def _load_cfg_data(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Avatar config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "avatar" not in data:
        raise ValueError("Missing 'avatar' in config.")
    return data["avatar"]


def load_avatar_config(path: Path | None = None) -> AvatarConfig:
    if path is None:
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / "config" / "avatar.yaml"

    raw = _load_cfg_data(path)
    colors_raw = raw["canvas"]
    regions_raw = raw["regions"]
    motion_raw = raw["motion"]
    blink_raw = motion_raw["blink"]
    mouth_raw = raw["mouth"]

    colors = AvatarColors(
        bg=_to_int_tuple(colors_raw["bg"]),
        green=_to_int_tuple(colors_raw["green"]),
        skin=_to_int_tuple(colors_raw["skin"]),
    )
    eyes = AvatarRegion(**{k: int(v) for k, v in regions_raw["eyes"].items()})
    mouth = AvatarRegion(**{k: int(v) for k, v in regions_raw["mouth"].items()})
    blink = AvatarBlinkCfg(
        base_interval_s=_to_float_pair(blink_raw["base_interval_s"]),
        close_s=float(blink_raw["close_s"]),
        open_s=float(blink_raw["open_s"]),
    )
    motion = AvatarMotionCfg(
        bob_px=float(motion_raw["bob_px"]),
        hop_px=float(motion_raw["hop_px"]),
        bob_period_s=float(motion_raw["bob_period_s"]),
        hop_s=float(motion_raw["hop_s"]),
        blink=blink,
    )
    mouth_cfg = AvatarMouthCfg(
        wavy_amp_px=float(mouth_raw["wavy_amp_px"]),
        steps=int(mouth_raw["steps"]),
        fang_enable=bool(mouth_raw["fang"]["enable"]),
        fang_x_ratio=float(mouth_raw["fang"]["x_ratio"]),
    )

    base_image = Path(str(raw["base_image"]))
    if not base_image.is_absolute():
        base_image = path.parent.parent / base_image
    return AvatarConfig(
        base_image=base_image,
        canvas_scale=int(colors_raw["scale"]),
        colors=colors,
        eyes=eyes,
        mouth=mouth,
        motion=motion,
        mouth_cfg=mouth_cfg,
    )
