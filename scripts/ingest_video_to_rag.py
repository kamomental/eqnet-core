#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ingest a video via frame sampling → Markdown timeline → RAG index.

This script focuses on a workable baseline:
- Extract frames at fixed FPS using ffmpeg (preferred) or a simple fallback.
- Optional near-duplicate filtering via simple frame difference.
- Convert selected frames to Markdown using VisionToMarkdown backend.
- Aggregate into a timeline Markdown (sections per timestamp) and index.

Usage:
  python scripts/ingest_video_to_rag.py --video ./talk.mp4 \
    --backend deepseek-http --endpoint http://localhost:8000/infer \
    --fps 1 --dedupe_ssim 0.92 --roi_mode text
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

from emot_terrain_lab.ingest.vision_to_md import VisionToMarkdown, VisionToMarkdownConfig
from emot_terrain_lab.ingest.deepseek_backend import select_backend, get_http_endpoint_from_env, get_cli_bin_from_env
from emot_terrain_lab.rag.indexer import IndexedDocument, RagIndex, NumericMeasurement


def encode_md(text: str, dim: int = 384) -> torch.Tensor:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rep = (dim + len(h) - 1) // len(h)
    raw = (h * rep)[: dim]
    vec = torch.tensor([((b / 255.0) * 2.0 - 1.0) for b in raw], dtype=torch.float32)
    return torch.tanh(vec)


def extract_frames_ffmpeg(video: Path, out_dir: Path, fps: float) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        str(pattern),
    ]
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        raise RuntimeError(f"ffmpeg failed: {e}")
    return sorted(out_dir.glob("frame_*.jpg"))


def simple_dedupe(paths: List[Path], threshold: float = 0.08) -> List[Path]:
    """Remove near-duplicates using mean absolute difference on downscaled grayscale.

    threshold is difference fraction (0..1). Smaller threshold → more aggressive dedupe.
    """
    keep: List[Path] = []
    last_vec = None
    for p in paths:
        try:
            im = Image.open(p).convert("L").resize((64, 64))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        except Exception:
            continue
        if last_vec is None:
            keep.append(p)
            last_vec = arr
        else:
            diff = float(np.mean(np.abs(arr - last_vec)))
            if diff >= threshold:
                keep.append(p)
                last_vec = arr
    return keep


def timeline_markdown(frame_paths: List[Path]) -> str:
    lines: List[str] = ["# Timeline"]
    for i, p in enumerate(frame_paths):
        # approximate timestamp from index (1 fps assumption)
        t_sec = i
        mm = t_sec // 60
        ss = t_sec % 60
        lines.append(f"\n## t={mm:02d}:{ss:02d} (frame {i+1})\n")
        # backend will fill the content per frame; we leave section headers here
    return "\n".join(lines)


def split_sections(md: str) -> List[Tuple[str | None, str]]:
    lines = md.splitlines()
    sections: List[Tuple[str | None, str]] = []
    current_title = None
    current_body: List[str] = []
    for ln in lines:
        if ln.startswith("## "):
            if current_title is not None or current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
                current_body = []
            current_title = ln[3:].strip()
        else:
            current_body.append(ln)
    if current_title is not None or current_body:
        sections.append((current_title, "\n".join(current_body).strip()))
    return sections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--dedupe_ssim", type=float, default=0.92, help="SSIM-like threshold: higher keeps more frames")
    ap.add_argument("--roi_mode", type=str, default="text", choices=["text", "full"])  # placeholder
    ap.add_argument("--backend", type=str, default="dummy", choices=["dummy", "deepseek-http", "deepseek-cli"]) 
    ap.add_argument("--endpoint", type=str, default="", help="DeepSeek-OCR HTTP endpoint (for deepseek-http)")
    ap.add_argument("--bin", type=str, default="", help="DeepSeek-OCR CLI path (for deepseek-cli)")
    args = ap.parse_args()

    vid = Path(args.video)
    if args.backend == "deepseek-http" and not args.endpoint:
        args.endpoint = get_http_endpoint_from_env()
    if args.backend == "deepseek-cli" and not args.bin:
        args.bin = get_cli_bin_from_env()
    backend = select_backend(args.backend, endpoint=args.endpoint, bin_path=args.bin)
    v2md = VisionToMarkdown(backend, VisionToMarkdownConfig())

    with tempfile.TemporaryDirectory() as td:
        frames_dir = Path(td)
        frames = extract_frames_ffmpeg(vid, frames_dir, args.fps)
        # Map dedupe_ssim to diff-threshold (simple): diff_th = 1 - ssim
        diff_th = max(0.0, min(1.0, 1.0 - float(args.dedupe_ssim)))
        frames_kept = simple_dedupe(frames, threshold=diff_th)

        # Convert each kept frame to Markdown and build a single timeline markdown
        sections: List[str] = ["# Timeline: " + vid.name]
        total = 0
        for i, fpath in enumerate(frames_kept):
            t_sec = i
            mm = t_sec // 60
            ss = t_sec % 60
            doc = v2md.convert_file(fpath)
            sections.append(f"\n## t={mm:02d}:{ss:02d} ({fpath.name})\n")
            sections.append(doc.markdown)
            total += 1
        md = "\n".join(sections)

    # Index into RAG
    index = RagIndex()
    parts = split_sections(md)
    count_secs = 0
    for i, (title, body) in enumerate(parts):
        text = f"# {title}\n\n{body}" if title else body
        emb = encode_md(text)
        meta = {"source": str(vid), "section": i, "title": title or "", "type": "video_timeline"}
        numeric = [NumericMeasurement(label="md_tokens_approx", value=float(len(body)/4.0), unit="tok")]
        index.add(IndexedDocument(doc_id=f"{vid.stem}#t{i}", text=text, embedding=emb, numeric=numeric, metadata=meta))
        count_secs += 1
    index.build()
    print(f"Video sections indexed: {count_secs}")


if __name__ == "__main__":
    main()

