#!/usr/bin/env python
"""CLI wrapper for the unified vision bridge frame watcher."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision_bridge import run_frame_watch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch a frame directory and send the newest stable image into EmotionalHubRuntime.",
    )
    parser.add_argument("--frames-dir", required=True, help="Directory that receives image frames.")
    parser.add_argument(
        "--glob",
        dest="globs",
        action="append",
        default=None,
        help="Glob pattern for candidate frames. Repeatable. Defaults to common image patterns.",
    )
    parser.add_argument(
        "--prompt",
        default="Observe the newest frame and respond briefly.",
        help="Prompt passed into runtime.process_turn for each selected frame.",
    )
    parser.add_argument("--context", default=None, help="Optional shared context.")
    parser.add_argument("--intent", default=None, help="Optional intent label.")
    parser.add_argument("--interval", type=float, default=1.5, help="Polling interval in seconds.")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Minimum age before a frame is considered stable enough to process.",
    )
    parser.add_argument("--fast-only", action="store_true", help="Use runtime fast path only.")
    parser.add_argument("--once", action="store_true", help="Process one stable frame and exit.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists() or not frames_dir.is_dir():
        parser.error(f"frames directory not found: {frames_dir}")

    patterns = tuple(args.globs or ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"])
    return run_frame_watch(
        frames_dir=frames_dir,
        patterns=patterns,
        prompt=args.prompt,
        context=args.context,
        intent=args.intent,
        fast_only=args.fast_only,
        interval=args.interval,
        settle_seconds=args.settle_seconds,
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
