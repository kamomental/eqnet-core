#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Import PDFs via Vision→Markdown into the RAG index.

This script wires the `VisionToMarkdown` interface to the in-memory `RagIndex`.
Plug in a DeepSeek‑OCR backend that returns structured Markdown to turn PDFs
into knowledge sources ready for retrieval.

Usage:
  python scripts/ingest_pdf_to_rag.py --pdf_dir ./docs --ext .pdf

Notes:
- Embedding uses a deterministic pseudo-encoder in this environment.
  Replace `encode_md` with your embedding model.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, Tuple

import torch

from emot_terrain_lab.ingest.vision_to_md import VisionToMarkdown, VisionToMarkdownConfig, dummy_backend
from emot_terrain_lab.ingest.deepseek_backend import (
    deepseek_http_backend,
    deepseek_cli_backend,
    select_backend,
    get_http_endpoint_from_env,
    get_cli_bin_from_env,
)
from emot_terrain_lab.rag.indexer import IndexedDocument, RagIndex, NumericMeasurement


def iter_files(root: Path, ext: str) -> Iterable[Path]:
    for p in root.rglob(f"*{ext}"):
        if p.is_file():
            yield p


def encode_md(text: str, dim: int = 384) -> torch.Tensor:
    """Deterministic pseudo-embedding: useful as a placeholder.

    Do not use for production retrieval quality. Replace with a real encoder.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # expand/repeat to desired dim
    rep = (dim + len(h) - 1) // len(h)
    raw = (h * rep)[:dim]
    vec = torch.tensor([((b / 255.0) * 2.0 - 1.0) for b in raw], dtype=torch.float32)
    # add a tiny position-dependent variation to avoid many near-duplicates
    vec = torch.tanh(vec)
    return vec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, required=True)
    ap.add_argument("--ext", type=str, default=".pdf")
    ap.add_argument("--target_tokens", type=int, default=800)
    ap.add_argument("--backend", type=str, default="dummy", choices=["dummy", "deepseek-http", "deepseek-cli"]) 
    ap.add_argument("--endpoint", type=str, default="", help="DeepSeek-OCR HTTP endpoint (for deepseek-http)")
    ap.add_argument("--bin", type=str, default="", help="DeepSeek-OCR CLI path (for deepseek-cli)")
    args = ap.parse_args()

    root = Path(args.pdf_dir)
    # Select backend
    if args.backend == "deepseek-http" and not args.endpoint:
        # pick from environment when --endpoint is omitted
        args.endpoint = get_http_endpoint_from_env()
    if args.backend == "deepseek-cli" and not args.bin:
        args.bin = get_cli_bin_from_env()
    backend = select_backend(args.backend, endpoint=args.endpoint, bin_path=args.bin)
    v2md = VisionToMarkdown(backend, VisionToMarkdownConfig(max_tokens_hint=args.target_tokens))

    index = RagIndex()
    total_docs = 0
    total_sections = 0

    for path in iter_files(root, args.ext):
        doc = v2md.convert_file(path)
        md = doc.markdown
        # simple section split by level-2 headings
        parts = split_sections(md)
        for i, (title, body) in enumerate(parts):
            text = f"# {title}\n\n{body}" if title else body
            emb = encode_md(text)
            meta = {"source": str(path), "section": i, "title": title or ""}
            numeric = [
                NumericMeasurement(label="md_tokens_approx", value=float(len(body) / 4.0), unit="tok"),
            ]
            index.add(IndexedDocument(doc_id=f"{path.stem}#sec{i}", text=text, embedding=emb, numeric=numeric, metadata=meta))
            total_sections += 1
        total_docs += 1

    index.build()
    print(f"Indexed documents: {total_docs}, sections: {total_sections}")


def split_sections(md: str) -> Iterable[Tuple[str | None, str]]:
    lines = md.splitlines()
    sections = []
    current_title = None
    current_body = []
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


if __name__ == "__main__":
    main()
