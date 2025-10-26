"""Quantised association link between body field and lexical tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class AssocConfig:
    vocab_size: int = 256
    embedding_dim: int = 16
    alpha: float = 0.1
    seed: int = 0


@dataclass
class AssocVQLink:
    config: AssocConfig = field(default_factory=AssocConfig)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        self.prototypes = rng.normal(size=(self.config.vocab_size, self.config.embedding_dim)).astype(np.float32)
        self.link_strength: Dict[int, float] = {}

    def encode(self, field_state: np.ndarray) -> Tuple[List[int], float]:
        features = self._extract_features(field_state)
        if features.size == 0:
            return [], 0.0
        dots = features @ self.prototypes.T
        labels = np.argmax(dots, axis=1)
        strength = float(np.mean(np.max(dots, axis=1)))
        self._update_prototypes(features, labels)
        tokens = labels[: min(5, labels.size)].tolist()
        if tokens:
            key = int(tokens[0])
            self.link_strength[key] = 0.9 * self.link_strength.get(key, 0.0) + 0.1 * strength
        return tokens, strength

    def _update_prototypes(self, features: np.ndarray, labels: np.ndarray) -> None:
        for idx in range(self.config.vocab_size):
            mask = labels == idx
            if not np.any(mask):
                continue
            delta = features[mask].mean(axis=0) - self.prototypes[idx]
            self.prototypes[idx] += self.config.alpha * delta

    def _extract_features(self, field_state: np.ndarray) -> np.ndarray:
        channel = field_state[0]
        h, w = channel.shape
        patch_h, patch_w = max(1, h // 4), max(1, w // 4)
        features = []
        for i in range(0, h, patch_h):
            for j in range(0, w, patch_w):
                patch = channel[i : min(i + patch_h, h), j : min(j + patch_w, w)]
                vec = [
                    float(patch.mean()),
                    float(patch.std()),
                    float(np.max(patch)),
                    float(np.min(patch)),
                ]
                features.append(vec)
        feats = np.asarray(features, dtype=np.float32)
        if feats.size == 0:
            return feats
        if feats.shape[1] < self.config.embedding_dim:
            pad = self.config.embedding_dim - feats.shape[1]
            feats = np.pad(feats, ((0, 0), (0, pad)))
        elif feats.shape[1] > self.config.embedding_dim:
            feats = feats[:, : self.config.embedding_dim]
        return feats
