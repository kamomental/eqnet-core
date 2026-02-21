"""Retriever utilities implementing numeric gating + semantic MMR."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set, Mapping
import os
import json

import torch
from rag.assoc_safety import clamp_score, sanitize_weights

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
        assoc_context: Optional[Dict[str, Any]] = None,
        assoc_weights: Optional[Dict[str, Any]] = None,
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

        if assoc_context:
            filtered_scores = self._compose_association_scores(
                docs=filtered,
                semantic_scores=filtered_scores,
                assoc_context=assoc_context,
                assoc_weights=assoc_weights,
            )

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

    def retrieve_with_assoc(
        self,
        *,
        runtime_cfg: Any,
        temporal_state: Optional[Any],
        query_embedding: torch.Tensor,
        top_k: int = 6,
        numeric_constraints: Optional[Sequence[NumericConstraint]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        """Thin wrapper for runtime-configured associative retrieval."""
        assoc_kwargs = build_assoc_kwargs(runtime_cfg=runtime_cfg, temporal_state=temporal_state)
        return self.retrieve(
            query_embedding,
            top_k=top_k,
            numeric_constraints=numeric_constraints,
            metadata_filter=metadata_filter,
            **assoc_kwargs,
        )

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

    def _compose_association_scores(
        self,
        *,
        docs: Sequence[IndexedDocument],
        semantic_scores: torch.Tensor,
        assoc_context: Dict[str, Any],
        assoc_weights: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Compose semantic + temporal + affective + value + open-loop signals."""
        weights = _normalize_assoc_weights(assoc_weights)
        use_normalized = bool(assoc_context.get("normalize_weights", True))
        weights = sanitize_weights(weights, normalize=use_normalized, fallback_key="semantic")
        tau = _safe_float(assoc_context.get("temporal_tau_sec"), 86400.0)
        if tau <= 0.0:
            tau = 86400.0
        clamp_min = _safe_float(assoc_context.get("clamp_min"), -5.0)
        clamp_max = _safe_float(assoc_context.get("clamp_max"), 5.0)
        if clamp_min > clamp_max:
            clamp_min, clamp_max = clamp_max, clamp_min
        query_ts = _coerce_timestamp_sec(assoc_context)
        query_valence = _safe_float(assoc_context.get("valence"), 0.0)
        query_arousal = _safe_float(assoc_context.get("arousal"), 0.0)
        query_open_loops = max(0.0, _safe_float(assoc_context.get("open_loops"), 0.0))
        query_tags = _coerce_tag_map(assoc_context.get("value_tags"))

        out = semantic_scores.clone()
        for idx, doc in enumerate(docs):
            meta = doc.metadata or {}
            temporal = _temporal_score(query_ts, _coerce_timestamp_sec(meta), tau)
            affective = _affective_score(
                query_valence=query_valence,
                query_arousal=query_arousal,
                doc_valence=_safe_float(meta.get("valence"), 0.0),
                doc_arousal=_safe_float(meta.get("arousal"), 0.0),
            )
            value = _tag_cosine(query_tags, _coerce_tag_map(meta.get("value_tags")))
            open_loop = _open_loop_score(
                query_open_loops=query_open_loops,
                doc_open_loops=max(0.0, _safe_float(meta.get("open_loops"), 0.0)),
            )
            score = (
                weights["semantic"] * float(semantic_scores[idx].item())
                + weights["temporal"] * temporal
                + weights["affective"] * affective
                + weights["value"] * value
                + weights["open_loop"] * open_loop
            )
            out[idx] = clamp_score(score, clamp_min, clamp_max)
        return out

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


def _normalize_assoc_weights(raw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    base = {
        "semantic": 1.0,
        "temporal": 0.10,
        "affective": 0.12,
        "value": 0.15,
        "open_loop": 0.08,
    }
    if not raw:
        return base
    out = dict(base)
    for key in out:
        if key in raw:
            out[key] = _safe_float(raw.get(key), out[key])
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_tag_map(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        label = str(key).strip()
        if not label:
            continue
        out[label] = _safe_float(value, 0.0)
    return out


def _coerce_timestamp_sec(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        for key in ("timestamp_sec", "timestamp", "created_at", "ts", "time"):
            if key in payload:
                maybe = _coerce_timestamp_sec(payload.get(key))
                if maybe is not None:
                    return maybe
        if "timestamp_ms" in payload:
            value = _safe_float(payload.get("timestamp_ms"), float("nan"))
            if math.isfinite(value):
                return value / 1000.0
        return None
    if isinstance(payload, (int, float)):
        value = float(payload)
        if not math.isfinite(value):
            return None
        if value > 10_000_000_000:
            return value / 1000.0
        return value
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                return None
        if numeric > 10_000_000_000:
            return numeric / 1000.0
        return numeric
    return None


def _temporal_score(query_ts: Optional[float], doc_ts: Optional[float], tau: float) -> float:
    if query_ts is None or doc_ts is None:
        return 0.0
    dt = abs(query_ts - doc_ts)
    return float(math.exp(-dt / max(tau, 1.0)))


def _affective_score(
    *,
    query_valence: float,
    query_arousal: float,
    doc_valence: float,
    doc_arousal: float,
) -> float:
    # Expected normalized domain [-1, 1]. Clamp prevents outliers from dominating.
    qv = max(-1.0, min(1.0, query_valence))
    qa = max(-1.0, min(1.0, query_arousal))
    dv = max(-1.0, min(1.0, doc_valence))
    da = max(-1.0, min(1.0, doc_arousal))
    distance = (abs(qv - dv) + abs(qa - da)) / 2.0
    return 1.0 - min(1.0, distance / 2.0)


def _tag_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    dot = 0.0
    an = 0.0
    bn = 0.0
    for key in keys:
        va = _safe_float(a.get(key), 0.0)
        vb = _safe_float(b.get(key), 0.0)
        dot += va * vb
        an += va * va
        bn += vb * vb
    if an <= 0.0 or bn <= 0.0:
        return 0.0
    return dot / math.sqrt(an * bn)


def _open_loop_score(*, query_open_loops: float, doc_open_loops: float) -> float:
    if query_open_loops <= 0.0 and doc_open_loops <= 0.0:
        return 0.0
    target = max(0.0, min(1.0, query_open_loops))
    observed = max(0.0, min(1.0, doc_open_loops))
    return 1.0 - abs(target - observed)


def build_assoc_kwargs(
    *,
    runtime_cfg: Any,
    temporal_state: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build `retrieve()` kwargs from runtime config + temporal state.

    Returns empty dict when assoc score is disabled.
    """
    rag_cfg = getattr(runtime_cfg, "rag", None)
    assoc = getattr(rag_cfg, "assoc_score", None) if rag_cfg is not None else None
    if assoc is None:
        return {}
    if not bool(getattr(assoc, "enabled", False)):
        return {}
    weights_cfg = getattr(assoc, "weights", None)
    weights = {
        "semantic": _safe_float(getattr(weights_cfg, "semantic", 1.0), 1.0),
        "temporal": _safe_float(getattr(weights_cfg, "temporal", 0.10), 0.10),
        "affective": _safe_float(getattr(weights_cfg, "affective", 0.12), 0.12),
        "value": _safe_float(getattr(weights_cfg, "value", 0.15), 0.15),
        "open_loop": _safe_float(getattr(weights_cfg, "open_loop", 0.08), 0.08),
    }
    use_normalized = bool(getattr(assoc, "normalize_weights", True))
    weights = sanitize_weights(weights, normalize=use_normalized, fallback_key="semantic")
    context: Dict[str, Any] = {
        "temporal_tau_sec": _safe_float(getattr(assoc, "temporal_tau_sec", 86400.0), 86400.0),
        "normalize_weights": use_normalized,
    }
    clamp_cfg = getattr(assoc, "clamp", None)
    context["clamp_min"] = _safe_float(getattr(clamp_cfg, "min", -5.0), -5.0)
    context["clamp_max"] = _safe_float(getattr(clamp_cfg, "max", 5.0), 5.0)
    if temporal_state is not None:
        if hasattr(temporal_state, "to_assoc_context"):
            payload = temporal_state.to_assoc_context(
                temporal_tau_sec=context["temporal_tau_sec"]
            )
            if isinstance(payload, Mapping):
                context.update(dict(payload))
        elif isinstance(temporal_state, Mapping):
            context.update(dict(temporal_state))
    return {
        "assoc_context": context,
        "assoc_weights": weights,
    }
