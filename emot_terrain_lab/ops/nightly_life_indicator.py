"""LifeIndicator calculation helpers for Nightly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from eqnet.runtime.life_indicator import LifeIndicator


def load_qualia_log(path: Path) -> List[dict]:
    """Load telemetry/qualia-*.jsonl into a list of dicts."""

    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _score_identity(num_diary_entries: int) -> float:
    # 30“ú˜A‘±‚ðãŒÀ‚Æ‚µ‚½’Pƒ‚È–O˜aŠÖ”
    return max(0.0, min(1.0, 0.2 + 0.8 * min(num_diary_entries, 30) / 30.0))


def _score_qualia_dispersion(qualia_vectors: Iterable[np.ndarray]) -> float:
    vectors = [np.asarray(vec, dtype=float) for vec in qualia_vectors]
    if not vectors:
        return 0.0
    X = np.stack(vectors)
    var = float(np.mean(np.var(X, axis=0)))
    # ’†—fƒŒƒ“ƒW around 0.3 } 0.2 ‚ð‰¼’è‚µ‚½ƒKƒEƒX“IƒXƒRƒA‰»
    center, width = 0.3, 0.2
    score = float(np.exp(-((var - center) ** 2) / (2 * width ** 2)))
    return max(0.0, min(1.0, score))


def _score_meta(num_self_reflections: int, total_events: int) -> float:
    if total_events <= 0:
        return 0.0
    return max(0.0, min(1.0, num_self_reflections / total_events))


def compute_life_indicator_for_day(
    qualia_records: List[dict],
    num_diary_entries: int,
    num_self_reflection_entries: int,
) -> LifeIndicator:
    """Compute v0 LifeIndicator from simple heuristics."""

    qualia_vectors = [rec.get("qualia_vec", []) for rec in qualia_records]
    identity = _score_identity(num_diary_entries)
    qualia = _score_qualia_dispersion(qualia_vectors)
    meta = _score_meta(num_self_reflection_entries, len(qualia_records))

    return LifeIndicator(
        identity_score=identity,
        qualia_score=qualia,
        meta_awareness_score=meta,
    ).clamp()
