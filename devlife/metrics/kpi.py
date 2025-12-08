"""KPI computations over episode records."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def reconstruction_error(tracks: List[Dict]) -> float:
    values = [float(tr.get("recon_err", 0.0)) for tr in tracks if "recon_err" in tr]
    return float(np.mean(values)) if values else 0.0


def self_other_misclassification(tracks: List[Dict], gold_key: str = "true_self_flag") -> float:
    entries: List[int] = []
    for tr in tracks:
        if gold_key in tr:
            p_self = float(tr.get("self_event", 0.5))
            predicted = p_self >= 0.5
            entries.append(int(predicted != bool(tr[gold_key])))
    return float(np.mean(entries)) if entries else 0.0


def counterfactual_match(tracks: List[Dict]) -> float:
    matches = [
        float(tr.get("plan", {}).get("match_score"))
        for tr in tracks
        if "plan" in tr and tr["plan"].get("match_score") is not None
    ]
    return float(np.mean(matches)) if matches else 0.0


def meta_brier(tracks: List[Dict], target_key: str = "answer_correct") -> float:
    scores: List[float] = []
    for tr in tracks:
        target = tr.get(target_key, None)
        conf = tr.get("meta", {}).get("meta_conf", None)
        if target is not None and conf is not None:
            scores.append((float(conf) - float(target)) ** 2)
    return float(np.mean(scores)) if scores else 0.0


def taste_violation_rate(tracks: List[Dict]) -> float:
    flags = [
        bool(tr.get("taste", {}).get("violations"))
        for tr in tracks
        if isinstance(tr.get("taste"), dict)
    ]
    return float(np.mean(flags)) if flags else 0.0


def mood_total_variation(tracks: List[Dict]) -> float:
    moods = [tr.get("mood") for tr in tracks]
    diffs: List[float] = []
    for i in range(1, len(moods)):
        cur = _mood_vector(moods[i])
        prev = _mood_vector(moods[i - 1])
        if cur is None or prev is None:
            continue
        if cur.shape != prev.shape:
            continue
        diffs.append(float(np.mean(np.abs(cur - prev))))
    return float(np.mean(diffs)) if diffs else 0.0


def _mood_vector(entry: Any) -> Optional[np.ndarray]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        ordered = [float(entry.get("mood_v", 0.0)), float(entry.get("mood_a", 0.0))]
        extra_keys = [k for k in entry.keys() if k not in {"mood_v", "mood_a"}]
        for key in sorted(extra_keys):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                ordered.append(float(value))
        return np.asarray(ordered, dtype=np.float32)
    if isinstance(entry, (list, tuple)) and entry:
        return np.asarray([float(x) for x in entry], dtype=np.float32)
    return None



def body_field_R(tracks: List[Dict]) -> float:
    values: List[float] = []
    for tr in tracks:
        stats = tr.get("sensory_stats") or {}
        err = float(stats.get("homeo_error", 0.0))
        values.append(float(np.exp(-abs(err))))
    return float(np.mean(values)) if values else 0.0


def body_field_rho(tracks: List[Dict]) -> float:
    values: List[float] = []
    for tr in tracks:
        field = tr.get("field") or {}
        rho = field.get("rho")
        if rho is None:
            rho = field.get("rho_norm")
        if rho is None:
            grad = float(field.get("grad_norm", 0.0))
            rho = float(np.clip(1.0 - grad, 0.0, 1.0))
        try:
            values.append(float(np.clip(float(rho), 0.0, 1.0)))
        except Exception:
            continue
    return float(np.mean(values)) if values else 0.0


def affect_love_signal(tracks: List[Dict]) -> float:
    values: List[float] = []
    for tr in tracks:
        love = tr.get("love_mode")
        if love is None:
            hormones = tr.get("hormones") or {}
            valence = float(hormones.get("H_valence", 0.0))
            novelty = float(((tr.get("delta_affect") or {}).get("d_aff") or {}).get("novelty", 0.0))
            entrain = 1.0 - float((tr.get("field") or {}).get("grad_norm", 0.0))
            love = 0.5 * np.clip(valence, 0.0, 1.0)
            love += 0.3 * np.clip(novelty, 0.0, 1.0)
            love += 0.2 * np.clip(entrain, 0.0, 1.0)
        try:
            values.append(float(np.clip(float(love), 0.0, 1.0)))
        except Exception:
            continue
    return float(np.mean(values)) if values else 0.0


def value_intent_trust(tracks: List[Dict]) -> float:
    trust_values: List[float] = []
    for tr in tracks:
        tom = tr.get("tom") or {}
        trust = tom.get("intent_trust_smoothed")
        if trust is None:
            trust = tom.get("intent_trust")
        if trust is None:
            continue
        trust_values.append(float(np.clip(float(trust), 0.0, 1.0)))
    return float(np.mean(trust_values)) if trust_values else 0.5


def safety_violation_rate(tracks: List[Dict]) -> float:
    flags: List[float] = []
    for tr in tracks:
        violation = False
        taste = tr.get("taste")
        if isinstance(taste, dict):
            violation = violation or bool(taste.get("violations"))
        value_vote = tr.get("value_committee")
        if isinstance(value_vote, dict):
            violation = violation or bool(value_vote.get("violation"))
        safety = tr.get("safety")
        if isinstance(safety, dict):
            violation = violation or bool(safety.get("violation"))
        policy = tr.get("policy_feedback")
        if isinstance(policy, dict):
            violation = violation or bool(policy.get("violation"))
        flags.append(1.0 if violation else 0.0)
    return float(np.mean(flags)) if flags else 0.0

def compute_all(tracks: List[Dict]) -> Dict[str, float]:
    return {
        "recon_err": reconstruction_error(tracks),
        "self_other_mis": self_other_misclassification(tracks),
        "cf_match": counterfactual_match(tracks),
        "meta_brier": meta_brier(tracks),
        "taste_violation": taste_violation_rate(tracks),
        "mood_total_var": mood_total_variation(tracks),
        "body.R": body_field_R(tracks),
        "body.rho": body_field_rho(tracks),
        "affect.love": affect_love_signal(tracks),
        "value.intent_trust": value_intent_trust(tracks),
        "ethics.safety_violation": safety_violation_rate(tracks),
        "suffering": suffering_rate(tracks),
        "tension": tension_level(tracks),
    }


# ------------------------------ Δaff RAG NDCG utilities
def ndcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """Compute NDCG@k given a list of gains ordered by ranked results.

    Args:
        relevances: list of graded relevances (gains) in ranked order.
        k: cutoff rank.
    Returns:
        NDCG@k value in [0, 1].
    """
    k = min(k, len(relevances))
    if k == 0:
        return 0.0
    rel = np.asarray(relevances[:k], dtype=np.float32)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float32))
    dcg = float(np.sum(rel * discounts))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum(ideal * discounts))
    return float(dcg / idcg) if idcg > 0 else 0.0


def ndcg_for_queries(
    rankings: Sequence[Sequence[Tuple[str, float]]],
    relevances: Sequence[Dict[str, float]],
    k: int = 10,
) -> float:
    """Average NDCG@k over queries.

    Args:
        rankings: per-query ranked list of (doc_id, score) pairs.
        relevances: per-query mapping doc_id -> gain.
    """
    if not rankings:
        return 0.0
    ndcgs: List[float] = []
    for rank_list, rel_map in zip(rankings, relevances):
        gains = [float(rel_map.get(doc_id, 0.0)) for doc_id, _ in rank_list]
        ndcgs.append(ndcg_at_k(gains, k=k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ------------------------------ Ignition ↔ Success regression
def ignition_success_regression(tracks: List[Dict]) -> Dict[str, float]:
    """Return correlation-based effect size between ignition index and success proxy.

    - I: episode["ignite"]["I"]
    - success: ΔR or R proxy from sensory_stats/homeo_error
    Returns r, r2, f2, n.
    """
    I_vals: List[float] = []
    succ_vals: List[float] = []
    prev_R: float | None = None
    for tr in tracks:
        ignite = tr.get("ignite", {}) if isinstance(tr, dict) else {}
        I = ignite.get("I", None)
        ss = tr.get("sensory_stats", {}) if isinstance(tr, dict) else {}
        homeo_err = float(ss.get("homeo_error", 0.0))
        R_t = -abs(homeo_err)
        if I is None:
            continue
        if prev_R is None:
            prev_R = R_t
            continue
        I_vals.append(float(I))
        succ_vals.append(float(R_t - prev_R))
        prev_R = R_t
    if len(I_vals) < 3:
        return {"r": 0.0, "r2": 0.0, "f2": 0.0, "n": float(len(I_vals))}
    x = np.asarray(I_vals, dtype=np.float32)
    y = np.asarray(succ_vals, dtype=np.float32)
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    r = float(np.clip(np.dot(x, y) / max(1, len(x) - 1), -1.0, 1.0))
    r2 = r * r
    f2 = float(r2 / max(1e-6, 1.0 - r2))
    return {"r": r, "r2": r2, "f2": f2, "n": float(len(x))}


# ------------------------------ Suffering / Tension KPIs
def suffering_rate(tracks: List[Dict]) -> float:
    """Average negative affect portion from mood vector (mood_v if present)."""
    vals: List[float] = []
    for tr in tracks:
        mood = tr.get("mood")
        if isinstance(mood, dict):
            v = float(mood.get("mood_v", 0.0))
        elif isinstance(mood, list) and mood:
            v = float(mood[0])
        else:
            v = 0.0
        vals.append(max(0.0, -v))
    return float(np.mean(vals)) if vals else 0.0


def tension_level(tracks: List[Dict]) -> float:
    """Average absolute homeostatic error as a tension proxy."""
    vals: List[float] = []
    for tr in tracks:
        ss = tr.get("sensory_stats", {}) if isinstance(tr, dict) else {}
        vals.append(abs(float(ss.get("homeo_error", 0.0))))
    return float(np.mean(vals)) if vals else 0.0
