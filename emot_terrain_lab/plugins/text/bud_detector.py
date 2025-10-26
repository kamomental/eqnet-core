"""Lightweight text-based bud (芽吹き) detector for EQNet.

The detector looks for novelty in conversation tokens and simple self-disclosure /
metaphor markers, returning only numeric indicators so raw text never persists.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, deque
from typing import Deque, Dict, List

META_PATTERNS = [
    r"(正直|実は|ふと|なぜか|はじめて|本音|気づいた)",
    r"(比喩|メタファ|喩えると|まるで|みたいな)",
]
META_REGEX = [re.compile(pattern) for pattern in META_PATTERNS]


class BudDetector:
    """Detects budding signals without retaining the original utterance."""

    def __init__(self, hist_len: int = 128, window: int = 32) -> None:
        self.hist_tokens: Deque[str] = deque(maxlen=hist_len)
        self.window = window
        self.last_ts: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Approximate tokenisation by overlapping bigrams."""
        stripped = re.sub(r"\s+", "", text)
        if not stripped:
            return []
        if len(stripped) == 1:
            return [stripped]
        return [stripped[i : i + 2] for i in range(len(stripped) - 1)]

    def _distribution(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        total = float(sum(counts.values())) or 1.0
        return {token: count / total for token, count in counts.items()}

    def _kl_novelty(self, tokens_now: List[str]) -> float:
        """Symmetric KL divergence between history and current tokens."""
        history = list(self.hist_tokens) if self.hist_tokens else tokens_now
        dist_hist = self._distribution(history)
        dist_now = self._distribution(tokens_now)
        keys = set(dist_hist) | set(dist_now)

        def kl_div(a: Dict[str, float], b: Dict[str, float]) -> float:
            score = 0.0
            for key in keys:
                p = a.get(key, 1e-9)
                q = b.get(key, 1e-9)
                score += p * math.log(p / q)
            return max(0.0, score)

        return 0.5 * (kl_div(dist_hist, dist_now) + kl_div(dist_now, dist_hist))

    def _meta_score(self, text: str) -> float:
        return float(any(regex.search(text) for regex in META_REGEX))

    def observe(self, utterance: str) -> Dict[str, float]:
        """Return bud metrics for an utterance (raw text is discarded)."""
        tokens = self._tokenize(utterance)
        novelty = self._kl_novelty(tokens) if tokens else 0.0
        meta = self._meta_score(utterance)
        bud_score = 0.7 * min(1.0, novelty) + 0.3 * meta

        for token in tokens:
            self.hist_tokens.append(token)
        self.last_ts = time.time()

        return {
            "bud_score": float(bud_score),
            "novelty": float(novelty),
            "meta": float(meta),
        }

