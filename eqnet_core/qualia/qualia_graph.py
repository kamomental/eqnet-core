"""QualiaGraph manages relational qualia prototypes and their distance profiles."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class QualiaGraph:
    max_prototypes: int = 128
    smoothing: float = 0.25
    _prototypes: List[np.ndarray] = field(default_factory=list)
    _distances: Optional[np.ndarray] = None

    def add_sample(self, vector: Iterable[float]) -> int:
        """Insert a new experience vector and update the relational graph."""
        sample = np.asarray(vector, dtype=float)
        self._prototypes.append(sample)
        if len(self._prototypes) > self.max_prototypes:
            self._prototypes.pop(0)
        self._recompute_distances()
        return len(self._prototypes) - 1

    def phi(self, index: int) -> Optional[np.ndarray]:
        """Return the relation profile D[i,:] for prototype i."""
        if self._distances is None:
            return None
        if index < 0 or index >= self._distances.shape[0]:
            return None
        return np.copy(self._distances[index, :])

    def nightly_update(
        self,
        batch: Optional[Iterable[Iterable[float]]] = None,
        num_clusters: int = 8,
    ) -> dict:
        """Fold in a batch of samples and emit lightweight cluster metadata."""
        if batch is not None:
            for sample in batch:
                self.add_sample(sample)

        if not self._prototypes:
            return {
                "num_prototypes": 0,
                "mean_distance": 0.0,
                "clusters": [],
            }

        if self._distances is None:
            self._recompute_distances()

        metadata = {
            "num_prototypes": len(self._prototypes),
            "mean_distance": float(self._distances.mean()),
            "clusters": self._cluster_metadata(num_clusters),
        }
        return metadata

    def _recompute_distances(self) -> None:
        count = len(self._prototypes)
        if count == 0:
            self._distances = None
            return
        stacked = np.stack(self._prototypes)
        diff = stacked[:, None, :] - stacked[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        if self._distances is None or self._distances.shape != distances.shape:
            self._distances = distances
        else:
            beta = self.smoothing
            self._distances = (1.0 - beta) * self._distances + beta * distances

    def _cluster_metadata(self, num_clusters: int) -> List[dict]:
        if not self._prototypes:
            return []
        data = np.stack(self._prototypes)
        k = max(1, min(num_clusters, data.shape[0]))
        centroids = data[np.linspace(0, data.shape[0] - 1, k, dtype=int)]
        labels = np.zeros(data.shape[0], dtype=int)

        for _ in range(6):
            distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            for idx in range(k):
                members = data[labels == idx]
                if members.size:
                    centroids[idx] = members.mean(axis=0)

        summary: List[dict] = []
        for idx in range(k):
            member_idx = np.where(labels == idx)[0]
            if member_idx.size == 0:
                continue
            summary.append(
                {
                    "cluster_id": int(idx),
                    "size": int(member_idx.size),
                    "representative_idx": int(member_idx[0]),
                }
            )
        return summary
