from __future__ import annotations


def aesthetic_features(text_or_image: object) -> dict[str, float]:
    """Lightweight heuristic features for initial aesthetic reward.

    This is a placeholder to bootstrap experiments. Replace with a proper
    detector when available.
    """
    t = str(text_or_image or "")
    symmetry = 0.1 if t == t[::-1] and len(t) > 4 else 0.0
    punctuation = t.count(",") + t.count("・") + t.count("。")
    repetition = min(1.0, (punctuation / max(1, len(t))) * 5.0)
    novelty = min(1.0, 1.0 - (len(set(t)) / max(1, len(t))))
    granularity = min(1.0, len(t) / 1000.0)
    return {"sym": float(symmetry), "rep": float(repetition), "nov": float(novelty), "gran": float(granularity)}


__all__ = ["aesthetic_features"]

