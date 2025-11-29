"""Helpers that wire QFS imagery replay into Nightly."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import numpy as np

from eqnet.qualia_model import FutureReplayConfig, ReplayMode, simulate_future
from eqnet.runtime.policy import PolicyPrior, apply_imagery_update
from eqnet.runtime.state import QualiaState


Array = np.ndarray
EmbeddingFn = Callable[[str], Array]


def build_intention_vec_from_story(
    story_text: str,
    embed_text_fn: EmbeddingFn,
    projector: Array,
    *,
    scale: float = 0.1,
) -> Array:
    """Convert narrative intent into a small qualia-space offset."""

    projector = np.asarray(projector, dtype=float)
    if projector.ndim != 2:
        raise ValueError("projector must be a 2-D matrix")
    if projector.shape[1] == 0:
        raise ValueError("projector must have non-zero width")

    if not story_text.strip():
        return np.zeros(projector.shape[1], dtype=float)

    emb = np.asarray(embed_text_fn(story_text), dtype=float)
    if emb.ndim != 1:
        emb = emb.reshape(-1)
    if emb.shape[0] != projector.shape[0]:
        raise ValueError("embedding dimension and projector rows must match")

    raw = emb @ projector
    norm = float(np.linalg.norm(raw)) + 1e-8
    return (scale / norm) * raw


def _average(values: Sequence[float]) -> float:
    return float(np.mean(list(values))) if values else 0.0


def run_imagery_replay(
    history: List[QualiaState],
    intention_texts: Iterable[str],
    embed_text_fn: EmbeddingFn,
    projector: Array,
    policy_prior: PolicyPrior,
    life_indicator_score: float,
    cfg: FutureReplayConfig | None = None,
    *,
    potential_fn: Callable[[Array], float],
) -> PolicyPrior:
    """Run imagery-mode rollouts for each intention and nudge the policy prior."""

    if not history:
        raise ValueError("history must not be empty")

    cfg = cfg or FutureReplayConfig()
    projector = np.asarray(projector, dtype=float)
    if projector.ndim != 2:
        raise ValueError("projector must be 2-D")
    qualia_dim = history[-1].qualia_vec.shape[0]
    if projector.shape[1] != qualia_dim:
        raise ValueError(
            "projector second dimension must equal qualia dimension"
        )

    for text in intention_texts:
        intention_vec = build_intention_vec_from_story(
            text, embed_text_fn, projector
        )
        traj = simulate_future(
            history=history,
            mode=ReplayMode.IMAGERY,
            cfg=cfg,
            intention_vec=intention_vec,
        )
        potentials = [potential_fn(vec) for vec in traj]
        avg_potential = _average(potentials)
        policy_prior = apply_imagery_update(
            policy_prior,
            avg_potential=avg_potential,
            avg_life_indicator=life_indicator_score,
        )
    return policy_prior
