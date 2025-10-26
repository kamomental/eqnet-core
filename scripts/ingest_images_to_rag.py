#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ingest images via Vision→Markdown into the RAG index.

Usage (HTTP):
  python scripts/ingest_images_to_rag.py --img_dir ./images \
    --backend deepseek-http --endpoint http://localhost:8000/infer

Usage (CLI):
  python scripts/ingest_images_to_rag.py --img_dir ./images \
    --backend deepseek-cli --bin deepseek-ocr
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from emot_terrain_lab.ingest.vision_to_md import VisionToMarkdown, VisionToMarkdownConfig
from emot_terrain_lab.ingest.deepseek_backend import select_backend, get_http_endpoint_from_env, get_cli_bin_from_env
from emot_terrain_lab.rag.indexer import IndexedDocument, RagIndex, NumericMeasurement


def iter_images(root: Path, exts: List[str]) -> Iterable[Path]:
    for ext in exts:
        for p in root.rglob(f"*{ext}"):
            if p.is_file():
                yield p


def encode_md(text: str, dim: int = 384) -> torch.Tensor:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rep = (dim + len(h) - 1) // len(h)
    raw = (h * rep)[: dim]
    vec = torch.tensor([((b / 255.0) * 2.0 - 1.0) for b in raw], dtype=torch.float32)
    return torch.tanh(vec)


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
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.bmp")
    ap.add_argument("--backend", type=str, default="dummy", choices=["dummy", "deepseek-http", "deepseek-cli"]) 
    ap.add_argument("--endpoint", type=str, default="", help="DeepSeek-OCR HTTP endpoint (for deepseek-http)")
    ap.add_argument("--bin", type=str, default="", help="DeepSeek-OCR CLI path (for deepseek-cli)")
    args = ap.parse_args()

    root = Path(args.img_dir)
    if args.backend == "deepseek-http" and not args.endpoint:
        args.endpoint = get_http_endpoint_from_env()
    if args.backend == "deepseek-cli" and not args.bin:
        args.bin = get_cli_bin_from_env()

    backend = select_backend(args.backend, endpoint=args.endpoint, bin_path=args.bin)
    v2md = VisionToMarkdown(backend, VisionToMarkdownConfig())
    index = RagIndex()

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    count_docs = 0
    count_secs = 0
    for img_path in iter_images(root, exts):
        doc = v2md.convert_file(img_path)
        parts = split_sections(doc.markdown)
        for i, (title, body) in enumerate(parts):
            text = f"# {title}\n\n{body}" if title else body
            emb = encode_md(text)
            meta = {"source": str(img_path), "section": i, "title": title or ""}
            numeric = [NumericMeasurement(label="md_tokens_approx", value=float(len(body)/4.0), unit="tok")]
            index.add(IndexedDocument(doc_id=f"{img_path.stem}#sec{i}", text=text, embedding=emb, numeric=numeric, metadata=meta))
            count_secs += 1
        count_docs += 1

    index.build()
    print(f"Indexed image docs: {count_docs}, sections: {count_secs}")


if __name__ == "__main__":
    main()

