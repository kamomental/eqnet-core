from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping


DEFAULT_INTERACTION_POLICY: Dict[str, Any] = {
    "version": "interaction_tools_v1",
    "reflex": {
        "latency_target_ms": 150,
        "ack_tokens": {
            "neutral": "了解です。",
            "care": "それは大変でしたね。",
            "excited": "お、いいですね。",
            "urgent": "すぐ整理します。",
        },
    },
    "resonance": {
        "baseline": {
            "valence": 0.5,
            "arousal": 0.5,
            "safety": 0.5,
        },
        "smoothing_alpha": 0.35,
    },
    "response_shaper": {
        "max_sentences_default": 4,
        "max_sentences_low_energy": 2,
    },
}


def estimate_resonance_state(
    *,
    text: str,
    prev_state: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = _deep_merge(DEFAULT_INTERACTION_POLICY, policy or {})
    base = ((cfg.get("resonance") or {}).get("baseline") or {})
    alpha = _safe_float(((cfg.get("resonance") or {}).get("smoothing_alpha")), 0.35)
    prev = prev_state if isinstance(prev_state, Mapping) else {}
    prev_valence = _safe_float(prev.get("valence"), _safe_float(base.get("valence"), 0.5))
    prev_arousal = _safe_float(prev.get("arousal"), _safe_float(base.get("arousal"), 0.5))
    prev_safety = _safe_float(prev.get("safety"), _safe_float(base.get("safety"), 0.5))

    raw = _resonance_signal(text)
    valence = _clamp01((1.0 - alpha) * prev_valence + alpha * raw["valence"])
    arousal = _clamp01((1.0 - alpha) * prev_arousal + alpha * raw["arousal"])
    safety = _clamp01((1.0 - alpha) * prev_safety + alpha * raw["safety"])

    confidence = _clamp01(raw["confidence"])
    uncertainty = 1.0 - confidence
    return {
        "valence": float(valence),
        "arousal": float(arousal),
        "safety": float(safety),
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "reason_codes": list(raw["reason_codes"]),
    }


def build_reflex_signal(
    *,
    resonance: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = _deep_merge(DEFAULT_INTERACTION_POLICY, policy or {})
    reflex_cfg = cfg.get("reflex") if isinstance(cfg.get("reflex"), Mapping) else {}
    tokens = reflex_cfg.get("ack_tokens") if isinstance(reflex_cfg.get("ack_tokens"), Mapping) else {}
    arousal = _safe_float(resonance.get("arousal"), 0.5)
    safety = _safe_float(resonance.get("safety"), 0.5)
    valence = _safe_float(resonance.get("valence"), 0.5)
    mode = "neutral"
    if safety < 0.35:
        mode = "urgent"
    elif valence < 0.4:
        mode = "care"
    elif arousal > 0.72 and valence >= 0.55:
        mode = "excited"
    reflex_text = str(tokens.get(mode) or tokens.get("neutral") or "了解です。")
    return {
        "mode": mode,
        "text": reflex_text,
        "latency_target_ms": int(_safe_float(reflex_cfg.get("latency_target_ms"), 150)),
    }


def shape_response_profile(
    *,
    resonance: Mapping[str, Any],
    metabolism: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = _deep_merge(DEFAULT_INTERACTION_POLICY, policy or {})
    shaper_cfg = cfg.get("response_shaper") if isinstance(cfg.get("response_shaper"), Mapping) else {}
    metabolism_map = metabolism if isinstance(metabolism, Mapping) else {}
    energy = _safe_float((metabolism_map.get("resource_budget") or {}).get("attention", {}).get("level"), 1.0)
    load = _safe_float((metabolism_map.get("attention_budget_used")), 0.0)
    arousal = _safe_float(resonance.get("arousal"), 0.5)
    safety = _safe_float(resonance.get("safety"), 0.5)
    valence = _safe_float(resonance.get("valence"), 0.5)

    mode = "balanced"
    pace = "steady"
    max_sentences = int(_safe_float(shaper_cfg.get("max_sentences_default"), 4))
    strategy = "brief_then_detail"
    if energy < 0.25 or load > 0.25:
        mode = "economy"
        pace = "short"
        max_sentences = int(_safe_float(shaper_cfg.get("max_sentences_low_energy"), 2))
        strategy = "short_confirm"
    elif safety < 0.4 or arousal > 0.75:
        mode = "calming"
        pace = "slow"
        max_sentences = min(max_sentences, 3)
        strategy = "validate_then_options"
    elif valence > 0.7 and safety > 0.6:
        mode = "engaged"
        pace = "lively"
        strategy = "support_then_action"
    return {
        "mode": mode,
        "pace": pace,
        "strategy": strategy,
        "max_sentences": max(1, max_sentences),
    }


def interaction_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(_normalize(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _resonance_signal(text: str) -> Dict[str, Any]:
    lower = (text or "").lower()
    valence = 0.5
    arousal = 0.5
    safety = 0.5
    reason_codes: list[str] = []
    confidence = 0.4

    if any(token in lower for token in ("ありがとう", "嬉しい", "助かった", "great", "thanks", "love")):
        valence += 0.25
        safety += 0.08
        reason_codes.append("POSITIVE_AFFECT")
        confidence += 0.18
    if any(token in lower for token in ("不安", "怖い", "つらい", "sad", "anxious", "angry", "panic")):
        valence -= 0.25
        safety -= 0.18
        arousal += 0.15
        reason_codes.append("DISTRESS_CUE")
        confidence += 0.2
    punct = text.count("!") + text.count("！") + text.count("?") + text.count("？")
    if punct >= 2:
        arousal += 0.2
        reason_codes.append("HIGH_PUNCT_AROUSAL")
        confidence += 0.1
    if len(text.strip()) <= 6:
        confidence -= 0.08
        reason_codes.append("SHORT_UTTERANCE")
    return {
        "valence": _clamp01(valence),
        "arousal": _clamp01(arousal),
        "safety": _clamp01(safety),
        "confidence": _clamp01(confidence),
        "reason_codes": reason_codes,
    }


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda i: str(i[0]))}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, float):
        # Fingerprint stability: quantize floating point noise.
        return round(value, 4)
    return value


__all__ = [
    "DEFAULT_INTERACTION_POLICY",
    "estimate_resonance_state",
    "build_reflex_signal",
    "shape_response_profile",
    "interaction_digest",
]
