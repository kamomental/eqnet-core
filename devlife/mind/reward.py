from __future__ import annotations


def aesthetic_reward(feat: dict[str, float], alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.2) -> float:
    """Compute a simple aesthetic reward from heuristic features.

    novelty     = feat["nov"]
    inconsistency ≈ max(0, repetition - 0.5)
    harmony     ≈ 1 - |0.5 - granularity|
    """
    novelty = float(feat.get("nov", 0.0))
    repetition = float(feat.get("rep", 0.0))
    incons = max(0.0, repetition - 0.5)
    harmony = 1.0 - abs(0.5 - float(feat.get("gran", 0.0)))
    reward = alpha * novelty - beta * incons + gamma * harmony
    return float(max(-1.0, min(1.0, reward)))


__all__ = ["aesthetic_reward"]

