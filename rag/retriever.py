"""Retriever utilities implementing numeric gating + semantic MMR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set
import os
import json

import torch

try:
    from emot_terrain_lab.rag.indexer import (
        IndexedDocument,
        NumericConstraint,
        NumericMeasurement,
        RagIndex,
    )
except ImportError:  # fallback to local package when running in-place
    from .indexer import (
        IndexedDocument,
        NumericConstraint,
        NumericMeasurement,
        RagIndex,
    )


@dataclass(frozen=True)
class RetrievalHit:
    """Structured retrieval result consumed by downstream modules."""

    doc_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any]
    numeric: Sequence[Dict[str, Any]]
    suggestion: Optional[str] = None
    cue: Optional[str] = None
    reason: Optional[str] = None


class RagRetriever:
    """Two-layer retrieval: numeric gating followed by semantic MMR."""

    def __init__(
        self,
        index: RagIndex,
        *,
        mmr_lambda: float = 0.35,
        oversample: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        if not (0.0 < mmr_lambda <= 1.0):
            raise ValueError("mmr_lambda must be in (0, 1].")
        if oversample < 1:
            raise ValueError("oversample must be >= 1.")
        self.index = index
        self.mmr_lambda = mmr_lambda
        self.oversample = oversample
        self.device = device or index.device
        self._prior_ids: Set[str] = set()
        self._prior_epsilon: float = 0.0
        # Optional warm-cache prior from env
        try:
            path = os.getenv("EQNET_RAG_WARM_CACHE", "").strip()
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ids = set(data.get("ids", []))
                eps = float(data.get("epsilon", 0.05))
                self.set_prior(ids, eps)
        except Exception:
            pass

    def set_prior(self, ids: Iterable[str], epsilon: float = 0.05) -> None:
        """Bias selected ids by adding epsilon to their scores before MMR.

        Keep epsilon small (e.g., 0.03–0.08) to avoid overwhelming relevance.
        """
        self._prior_ids = set(ids)
        self._prior_epsilon = max(0.0, float(epsilon))

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        *,
        top_k: int = 6,
        numeric_constraints: Optional[Sequence[NumericConstraint]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        if top_k <= 0:
            return []
        pool_k = min(len(self.index), max(top_k * self.oversample, top_k))
        candidates, scores = self.index.search(
            query_embedding,
            top_k=pool_k,
            numeric_constraints=numeric_constraints,
        )
        if not candidates:
            return []

        filtered, filtered_scores = self._filter_metadata(candidates, scores, metadata_filter)
        if not filtered:
            return []

        # Apply prior bias before MMR (small epsilon)
        if self._prior_ids and self._prior_epsilon > 0:
            bias = torch.zeros_like(filtered_scores)
            for i, doc in enumerate(filtered):
                if doc.doc_id in self._prior_ids:
                    bias[i] = self._prior_epsilon
            filtered_scores = filtered_scores + bias

        selected_indices = self._mmr(filtered_scores, torch.stack([doc.embedding for doc in filtered]), top_k)
        hits: List[RetrievalHit] = []
        for rank, idx in enumerate(selected_indices, start=1):
            doc = filtered[idx]
            score = float(filtered_scores[idx].item())
            hits.append(self._build_hit(doc, score, rank, numeric_constraints))
        return hits

    def _filter_metadata(
        self,
        docs: Sequence[IndexedDocument],
        scores: torch.Tensor,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> Tuple[List[IndexedDocument], torch.Tensor]:
        if not metadata_filter:
            return list(docs), scores
        mask = []
        for doc in docs:
            include = True
            for key, expected in metadata_filter.items():
                if doc.metadata.get(key) != expected:
                    include = False
                    break
            mask.append(include)
        if not any(mask):
            return [], torch.empty(0, device=scores.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=scores.device)
        filtered_docs = [doc for doc, keep in zip(docs, mask) if keep]
        filtered_scores = scores.masked_select(mask_tensor)
        return filtered_docs, filtered_scores

    def _mmr(self, scores: torch.Tensor, embeddings: torch.Tensor, top_k: int) -> List[int]:
        """Greedy Maximal Marginal Relevance."""
        top_k = min(top_k, scores.size(0))
        selected: List[int] = []
        candidate_indices = list(range(scores.size(0)))
        sim_matrix = torch.matmul(embeddings, embeddings.t())

        while candidate_indices and len(selected) < top_k:
            best_idx = None
            best_value = float("-inf")
            for idx in candidate_indices:
                relevance = scores[idx].item()
                diversity = 0.0
                if selected:
                    diversity = max(sim_matrix[idx, j].item() for j in selected)
                value = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * diversity
                if value > best_value:
                    best_value = value
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            candidate_indices.remove(best_idx)
        return selected

    def _build_hit(
        self,
        doc: IndexedDocument,
        score: float,
        rank: int,
        constraints: Optional[Sequence[NumericConstraint]],
    ) -> RetrievalHit:
        numeric_payload = [_numeric_to_payload(measure) for measure in doc.numeric]
        cue = doc.metadata.get("title") or doc.metadata.get("headline")
        suggestion = doc.metadata.get("suggestion")
        reason = None
        if constraints:
            matched = []
            for constraint in constraints:
                for measure in doc.numeric:
                    if constraint.matches(measure):
                        matched.append(f"{constraint.label}:{measure.value}{measure.unit}")
                        break
            if matched:
                reason = "matched " + ", ".join(matched)
        return RetrievalHit(
            doc_id=doc.doc_id,
            text=doc.text,
            score=score,
            rank=rank,
            metadata=dict(doc.metadata),
            numeric=numeric_payload,
            suggestion=suggestion,
            cue=cue,
            reason=reason,
        )


def _numeric_to_payload(measure: NumericMeasurement) -> Dict[str, Any]:
    lo, hi = measure.range()
    payload: Dict[str, Any] = {
        "label": measure.label,
        "value": measure.value,
        "unit": measure.unit,
        "lower": lo,
        "upper": hi,
        "direction": measure.direction,
    }
    return payload
