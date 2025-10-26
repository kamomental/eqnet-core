# -*- coding: utf-8 -*-
"""Bud-aware demo loop for EQNet."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.green_kernel import LowRankGreen
from core.prune_gate import PruneGate
from plugins.text.bud_detector import BudDetector
from terrain.field import detect_buds_2d

try:
    import websockets
except Exception:  # noqa: BLE001
    websockets = None


@dataclass
class Config:
    bud_thresh: float = 0.42
    decay_sec: float = 180.0
    grid_size: int = 64
    lr: float = 0.03
    rho_max: float = 1.8
    gain: float = 0.6
    centre: Tuple[int, int] = (32, 32)


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    text_conf = (data or {}).get("text", {})
    thresh = float(text_conf.get("bud_thresh", 0.42))
    decay = float(text_conf.get("decay_sec", 180))
    grid = int((data or {}).get("sampling", {}).get("grid_size", 64) or 64)
    return Config(bud_thresh=thresh, decay_sec=decay, grid_size=grid)


def gaussian_bases(size: int) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    centres = [(-0.5, -0.5, 0.4), (0.0, 0.0, 0.35), (0.5, 0.5, 0.45)]
    bases = []
    for cx, cy, sigma in centres:
        gx = np.exp(-((xs - cx) ** 2) / (2 * sigma ** 2))
        gy = np.exp(-((xs - cy) ** 2) / (2 * sigma ** 2))
        kernel = np.outer(gx, gy)
        norm = np.max(np.abs(kernel)) or 1.0
        bases.append(kernel / norm)
    return np.stack(bases, axis=0)


def estimate_coords(utterance: str, grid: int) -> Tuple[int, int]:
    """Map an utterance to a stable coordinate without storing text."""
    digest = hashlib.sha256(utterance.encode("utf-8", errors="ignore")).digest()
    x_raw = int.from_bytes(digest[:4], "big")
    y_raw = int.from_bytes(digest[4:8], "big")
    x = x_raw % max(grid, 1)
    y = y_raw % max(grid, 1)
    return int(x), int(y)


async def send_event(uri: str, payload: dict) -> None:
    if not websockets:
        return
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(payload, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to send bud event: {exc}", file=sys.stderr)


def prune_old(events: List[Tuple[float, dict]], decay: float) -> List[Tuple[float, dict]]:
    now = time.time()
    return [(ts, data) for ts, data in events if now - ts <= decay]


async def process_utterances(
    lines: Iterable[str],
    config: Config,
    events_uri: Optional[str],
) -> None:
    detector = BudDetector()
    bases = gaussian_bases(config.grid_size)
    kernel = LowRankGreen(bases=bases, lr=config.lr)
    gate = PruneGate(rho_max=config.rho_max)
    history: List[Tuple[float, dict]] = []

    for raw in lines:
        utter = raw.strip()
        if not utter:
            continue
        metrics = detector.observe(utter)
        bud_score = metrics["bud_score"]
        if bud_score <= config.bud_thresh:
            continue

        x_idx, y_idx = estimate_coords(utter, config.grid_size)
        kernel.update_local((x_idx, y_idx), gain=config.gain)
        rho = kernel.spectral_radius()
        ok = gate.check(rho)
        if not ok:
            kernel.w *= 0.9

        field = kernel.field()
        buds = detect_buds_2d(field, min_prominence=0.05)
        payload = {
            "type": "bud",
            "ts": int(time.time() * 1000),
            "score": bud_score,
            "rho": rho,
            "ok": ok,
            "novelty": metrics["novelty"],
            "meta": metrics["meta"],
            "bud_coords": buds[:5],
        }
        history = prune_old(history, config.decay_sec)
        history.append((time.time(), payload))

        print(
            json.dumps(
                {
                    "score": bud_score,
                    "rho": rho,
                    "ok": ok,
                    "novelty": metrics["novelty"],
                    "meta": metrics["meta"],
                    "coords": [x_idx, y_idx],
                    "active_buds": buds[:5],
                },
                ensure_ascii=False,
            )
        )
        if events_uri:
            await send_event(events_uri, payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet bud-aware demo")
    parser.add_argument(
        "--events-uri",
        type=str,
        default="",
        help="WebSocket URI for sending bud events (e.g. ws://127.0.0.1:8765/events)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional path to a UTF-8 text file; otherwise stdin is used.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/sensibility.yaml"),
        help="Path to sensibility configuration with text.bud_thresh settings.",
    )
    return parser.parse_args(argv)


def iter_utterances(path: Optional[Path]) -> Iterable[str]:
    if path and path.exists():
        yield from path.read_text(encoding="utf-8").splitlines()
    else:
        print("Enter utterances (Ctrl+D to finish):", file=sys.stderr)
        for line in sys.stdin:
            yield line


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    events_uri = args.events_uri or None
    lines = iter_utterances(args.input)
    try:
        asyncio.run(process_utterances(lines, config, events_uri))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
