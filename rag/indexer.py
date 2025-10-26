"""Two-layer RAG indexer with numeric gating support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class NumericMeasurement:
    """Structured numeric fact attached to a document."""

    label: str
    value: float
    unit: str = ""
    direction: str = "any"  # e.g. up/down/steady
    lower: Optional[float] = None
    upper: Optional[float] = None

    def range(self) -> Tuple[float, float]:
        lo = self.lower if self.lower is not None else self.value
        hi = self.upper if self.upper is not None else self.value
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi


@dataclass(frozen=True)
class NumericConstraint:
    """Numeric gate used during retrieval."""

    label: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: Optional[str] = None
    direction: str = "any"

    def matches(self, measurement: NumericMeasurement) -> bool:
        if self.unit and measurement.unit and self.unit.lower() != measurement.unit.lower():
            return False
        if self.direction != "any" and measurement.direction != "any":
            if measurement.direction.lower() != self.direction.lower():
                return False
        lo, hi = measurement.range()
        if self.min_value is not None and hi < self.min_value:
            return False
        if self.max_value is not None and lo > self.max_value:
            return False
        return True


@dataclass
class IndexedDocument:
    """Entry stored in the retrieval index."""

    doc_id: str
    text: str
    embedding: torch.Tensor
    numeric: Sequence[NumericMeasurement] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_device(self, device: torch.device) -> "IndexedDocument":
        if self.embedding.device == device:
            return self
        return IndexedDocument(
            doc_id=self.doc_id,
            text=self.text,
            embedding=self.embedding.to(device),
            numeric=self.numeric,
            metadata=self.metadata,
        )


class RagIndex:
    """In-memory index supporting numeric gating and semantic retrieval."""

    def __init__(self, *, device: Optional[torch.device] = None, normalize: bool = True) -> None:
        self.device = device or torch.device("cpu")
        self.normalize = normalize
        self._documents: List[IndexedDocument] = []
        self._matrix: Optional[torch.Tensor] = None
        self._built = False

    def add(self, doc: IndexedDocument) -> None:
        """Add a document to the index (lazy build)."""
        if doc.embedding.dim() != 1:
            raise ValueError("Document embedding must be 1-D (feature vector).")
        embedding = doc.embedding.to(self.device)
        if self.normalize:
            embedding = _safe_normalize(embedding)
        stored = IndexedDocument(
            doc_id=doc.doc_id,
            text=doc.text,
            embedding=embedding,
            numeric=list(doc.numeric),
            metadata=dict(doc.metadata),
        )
        self._documents.append(stored)
        self._built = False

    def extend(self, docs: Iterable[IndexedDocument]) -> None:
        for doc in docs:
            self.add(doc)

    def __len__(self) -> int:
        return len(self._documents)

    def build(self) -> None:
        """Finalize the index (stack embeddings)."""
        if not self._documents:
            self._matrix = None
            self._built = True
            return
        matrix = torch.stack([doc.embedding for doc in self._documents], dim=0)
        self._matrix = matrix
        self._built = True

    @property
    def documents(self) -> Sequence[IndexedDocument]:
        return self._documents

    def search(
        self,
        query: torch.Tensor,
        *,
        top_k: int = 10,
        numeric_constraints: Optional[Sequence[NumericConstraint]] = None,
    ) -> Tuple[Sequence[IndexedDocument], torch.Tensor]:
        """Return candidate documents and similarity scores (before MMR)."""
        if not self._built:
            self.build()
        if self._matrix is None or not self._documents:
            return [], torch.empty(0, device=self.device)

        query = query.to(self.device)
        if self.normalize:
            query = _safe_normalize(query)
        scores = torch.mv(self._matrix, query)

        if numeric_constraints:
            mask = self._numeric_mask(numeric_constraints)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))

        top_k = min(top_k, len(self._documents))
        values, indices = torch.topk(scores, k=top_k)
        candidates = [self._documents[int(idx)] for idx in indices]
        return candidates, values

    def _numeric_mask(self, constraints: Sequence[NumericConstraint]) -> Optional[torch.Tensor]:
        if not constraints:
            return None
        mask = torch.ones(len(self._documents), dtype=torch.bool, device=self.device)
        for idx, doc in enumerate(self._documents):
            for constraint in constraints:
                if not any(constraint.matches(measure) for measure in doc.numeric):
                    mask[idx] = False
                    break
        return mask


def _safe_normalize(vec: torch.Tensor) -> torch.Tensor:
    denom = torch.norm(vec).clamp_min(1e-9)
    return vec / denom
