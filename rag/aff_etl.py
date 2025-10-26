"""Δaff ETL: upsert episodes with strong affect into the RAG index.

Minimal implementation for nightly sleep ETL:
- scans episode JSON files under logs/episodes
- for records with |delta_affect| > tau, upsert as documents
- attaches numeric measurement for delta_aff

Embedding is passed by a callable encoder(text) -> torch.Tensor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch

from emot_terrain_lab.rag.indexer import IndexedDocument, NumericMeasurement, RagIndex


@dataclass
class AffEtlConfig:
    episodes_root: Path = Path("logs/episodes")
    tau: float = 0.25  # gate threshold for |Δaff|
    channel: str = "utterance"  # text source; fallback to tokens


def _episode_paths(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for p in root.rglob("*.json"):
        if p.is_file():
            yield p


def upsert_aff_episodes(
    index: RagIndex,
    encoder: Callable[[str], torch.Tensor],
    *,
    config: Optional[AffEtlConfig] = None,
) -> int:
    cfg = config or AffEtlConfig()
    inserted = 0
    for path in _episode_paths(cfg.episodes_root):
        try:
            with path.open("r", encoding="utf-8") as fh:
                ep = json.load(fh)
        except Exception:
            continue
        delta_aff = float(ep.get("delta_affect", 0.0))
        if abs(delta_aff) <= cfg.tau:
            continue
        # text payload
        text = None
        if cfg.channel == "utterance":
            text = str(ep.get("utterance", ""))
        if not text:
            tokens = ep.get("tokens", [])
            text = "tokens:" + ",".join(map(str, tokens))
        embedding = encoder(text)
        if embedding is None:
            continue
        doc_id = f"ep:{path.stem}"
        numeric = (
            NumericMeasurement(
                label="delta_aff",
                value=delta_aff,
                unit="",
                direction="up" if delta_aff >= 0 else "down",
            ),
        )
        meta = {
            "stage": ep.get("stage", ""),
            "timestamp": ep.get("timestamp", ""),
        }
        index.add(
            IndexedDocument(
                doc_id=doc_id,
                text=text,
                embedding=embedding,
                numeric=numeric,
                metadata=meta,
            )
        )
        inserted += 1
    # finalize index for retrieval
    index.build()
    return inserted


__all__ = ["AffEtlConfig", "upsert_aff_episodes"]

