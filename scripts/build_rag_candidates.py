#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build QUBO candidates JSONL for RAG re-ranking (candidates.v1).

Input formats (lightweight, dependency-free):
- Index JSONL: each line is a chunk with fields:
    {"id": str, "text": str, "tokens": int (opt), "emb": [float,...]}
- Queries JSONL: each line is a query with fields:
    {"id": str, "text": str (opt), "emb": [float,...]}

Output JSONL (candidates.v1): one line per (query, chunk) pair selected by top-K similarity
    {"schema":"candidates.v1","version":"1.0"}
    {"id": "<chunk_id>", "type": "rag", "gain": <float>, "cost": <float>, "emb": [...],
     "meta": {"query_id": "...", "source": "rag", "title": ""}}

Notes
- This is a minimal, emb-only generator to avoid extra dependencies.
- If BM25 scores are available in the index (field "bm25" keyed per query), you can extend
  the gain computation externally or post-hoc adjust the gain field.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Any
import json
import math

import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, required=True, help="chunks JSONL (id,text,tokens,emb)")
    ap.add_argument("--queries", type=str, required=True, help="queries JSONL (id,text,emb)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--topk", type=int, default=20)
    # gain = w_emb * cos + w_bm25 * bm25 + w_struct * S(text)
    ap.add_argument("--w_emb", type=float, default=0.7, help="weight for embedding similarity in gain")
    ap.add_argument("--w_bm25", type=float, default=0.3, help="weight for BM25 score in gain")
    ap.add_argument("--w_struct", type=float, default=0.1, help="weight for structural bonus in gain")
    ap.add_argument("--b_table", type=float, default=0.3, help="struct bonus for tables")
    ap.add_argument("--b_math", type=float, default=0.2, help="struct bonus for math")
    ap.add_argument("--b_heading", type=float, default=0.1, help="struct bonus for headings")
    args = ap.parse_args()

    idx = Path(args.index)
    qry = Path(args.queries)
    out = Path(args.out)

    chunks = read_jsonl(idx)
    queries = read_jsonl(qry)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": "candidates.v1", "version": "1.0"}, ensure_ascii=False) + "\n")

        # Build a simple matrix of chunk embeddings
        chunk_embs = []
        for ch in chunks:
            emb = np.asarray(ch.get("emb", []), dtype=np.float64)
            chunk_embs.append(emb)
        # Pre-normalize
        norms = [np.linalg.norm(e) for e in chunk_embs]
        chunk_embs = [e / (n if n > 1e-9 else 1.0) for e, n in zip(chunk_embs, norms)]

        for q in queries:
            qid = q.get("id") or f"q:{len(q)}"
            qemb = np.asarray(q.get("emb", []), dtype=np.float64)
            if qemb.size == 0:
                continue
            qemb = qemb / (np.linalg.norm(qemb) if np.linalg.norm(qemb) > 1e-9 else 1.0)
            sims = [float(np.dot(qemb, e)) if e.size else 0.0 for e in chunk_embs]
            # Select topk indices
            order = np.argsort(sims)[::-1][: int(args.topk)]
            for idx_i in order:
                ch = chunks[int(idx_i)]
                sim = sims[int(idx_i)]
                # BM25 score sources: bm25_by_query[qid] > meta.bm25 > ch.bm25 > 0
                bm25_by_q = ch.get("bm25_by_query", {}) if isinstance(ch.get("bm25_by_query", {}), dict) else {}
                bm25_raw = None
                if qid in bm25_by_q:
                    bm25_raw = float(bm25_by_q[qid])
                elif isinstance(ch.get("meta", {}), dict) and "bm25" in ch["meta"]:
                    bm25_raw = float(ch["meta"]["bm25"])
                elif "bm25" in ch:
                    try:
                        bm25_raw = float(ch["bm25"])
                    except Exception:
                        bm25_raw = None
                bm25_norm = float(bm25_raw / (bm25_raw + 1.0)) if bm25_raw is not None and bm25_raw >= 0 else 0.0

                # Structural bonus from meta.struct or heuristics on text
                meta_struct = {}
                if isinstance(ch.get("meta", {}), dict):
                    meta_struct = ch["meta"].get("struct", {}) or {}
                text = ch.get("text") or ""
                has_table = int(bool(meta_struct.get("table", 0)) or ("|" in text and "---" in text))
                has_math = int(bool(meta_struct.get("math", 0)) or ("$" in text))
                has_heading = int(bool(meta_struct.get("heading", 0)) or ("\n#" in ("\n" + text)))
                struct_score = args.b_table * has_table + args.b_math * has_math + args.b_heading * has_heading

                gain = float(args.w_emb) * sim + float(args.w_bm25) * bm25_norm + float(args.w_struct) * struct_score
                tokens = int(ch.get("tokens", 0)) if isinstance(ch.get("tokens", 0), int) else 0
                cost = (tokens / 1000.0) if tokens > 0 else max(1.0, len((ch.get("text") or "")) / 400.0)
                record = {
                    "id": ch.get("id"),
                    "type": "rag",
                    "gain": gain,
                    "cost": float(cost),
                    "emb": ch.get("emb", []),
                    "meta": {
                        "source": ch.get("source", "rag"),
                        "title": ch.get("title", ""),
                        "query_id": qid,
                        "bm25": bm25_raw if bm25_raw is not None else 0.0,
                        "struct": {"table": has_table, "math": has_math, "heading": has_heading},
                    },
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
