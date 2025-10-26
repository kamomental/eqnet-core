"""Composer stub mapping body states to proto-lexicon tokens and responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class ComposerConfig:
    vocab_size: int = 32
    embedding_dim: int = 16
    top_k: int = 3
    seed: int = 0


class SimpleComposer:
    """Transforms body embeddings into proto tokens and scripted responses."""

    def __init__(self, config: ComposerConfig | None = None) -> None:
        self.config = config or ComposerConfig()
        rng = np.random.default_rng(self.config.seed)
        self.prototypes = rng.normal(size=(self.config.vocab_size, self.config.embedding_dim)).astype(np.float32)
        self.lexicon = [f"body_proto_{i}" for i in range(self.config.vocab_size)]

    def bodylex_map(self, body_state: np.ndarray) -> List[str]:
        embedding = self._embed(body_state)
        distances = np.linalg.norm(self.prototypes - embedding[None, :], axis=1)
        top_indices = np.argsort(distances)[: self.config.top_k]
        return [self.lexicon[int(idx)] for idx in top_indices]

    def dialog(
        self,
        body_tokens: Sequence[str],
        affect: Dict[str, float],
        *,
        world_tokens: Sequence[str] | None = None,
    ) -> Tuple[str, Dict[str, Dict[str, float]]]:
        world_tokens = world_tokens or []
        combined = list(body_tokens) + list(world_tokens)
        valence = affect.get("H_valence", 0.0)
        arousal = affect.get("H_arousal", 0.0)
        novelty = affect.get("H_novelty", 0.0)
        mood = "soft" if valence >= 0 else "steady"
        text = f"[{mood}] {' '.join(combined)} | val={valence:.2f} nov={novelty:.2f}"
        delta = {
            "d_aff": {
                "valence": float(0.03 * np.tanh(valence)),
                "arousal": float(0.02 * np.tanh(arousal)),
                "novelty": float(0.01 * np.tanh(novelty)),
            }
        }
        return text, delta

    # ------------------------------------------------------------------ helpers
    def _embed(self, body_state: np.ndarray) -> np.ndarray:
        channel = body_state[0]
        h, w = channel.shape
        patch_h, patch_w = h // 4, w // 4
        patches: List[float] = []
        for i in range(4):
            for j in range(4):
                patch = channel[i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
                patches.append(float(patch.mean()))
                patches.append(float(patch.std()))
        embedding = np.array(patches, dtype=np.float32)
        if embedding.size < self.config.embedding_dim:
            embedding = np.pad(embedding, (0, self.config.embedding_dim - embedding.size))
        else:
            embedding = embedding[: self.config.embedding_dim]
        return embedding
