# -*- coding: utf-8 -*-
"""Human-readable explanations for RAG selection."""

from __future__ import annotations

from typing import Iterable, Sequence

try:
    from emot_terrain_lab.rag.retriever import RetrievalHit
except Exception:  # pragma: no cover
    RetrievalHit = object  # type: ignore


def explain_selection(hits: Sequence["RetrievalHit"], *, limit: int = 2) -> str:
    """Produce a short textual summary of why hits were chosen."""
    if not hits:
        return ""
    lines = []
    for hit in hits[:limit]:
        meta = getattr(hit, "metadata", {}) or {}
        trust = meta.get("trust_score")
        junk = meta.get("junk_prob")
        cue = getattr(hit, "cue", None) or meta.get("title") or meta.get("site")
        snippet = cue or str(hit.doc_id)
        pieces = [snippet]
        if trust is not None:
            pieces.append(f"trust {float(trust):.2f}")
        if junk is not None:
            pieces.append(f"junk {float(junk):.2f}")
        reason = getattr(hit, "reason", None)
        if reason:
            pieces.append(reason)
        lines.append(" / ".join(pieces))
    return "; ".join(lines)


__all__ = ["explain_selection"]

