# -*- coding: utf-8 -*-
"""
Observer layer: Turns observable cues + EQNet field metrics into explainable hypotheses.

Important: this is NOT hidden "thinking" output. Every suggestion is paired with the
evidence and confidence (SES) so that humans can verify or override it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from runtime.config import ObserverDisclaimerCfg, load_runtime_cfg
from emot_terrain_lab.terrain.llm import chat_json

PLUTCHIK_CENTROIDS = {
    "Joy": (0.8, 0.6),
    "Trust": (0.6, 0.2),
    "Fear": (-0.4, 0.8),
    "Surprise": (0.0, 1.0),
    "Sadness": (-0.8, -0.4),
    "Disgust": (-0.6, -0.2),
    "Anger": (-0.6, 0.6),
    "Anticipation": (0.4, 0.4),
}

DEFAULT_GATE_CFG = {
    "threshold": float(os.getenv("OFFER_GATE_THRESHOLD", 0.55)),
    "cool_off_s": float(os.getenv("OFFER_GATE_COOLOFF_S", 300)),
    "listen_bias": float(os.getenv("OFFER_GATE_LISTEN_BIAS", 0.20)),
    "play_bias": float(os.getenv("OFFER_GATE_PLAYTEST_BIAS", 0.15)),
    "community_bias": float(os.getenv("OFFER_GATE_COMMUNITY_BIAS", 0.05)),
    "safety_override": os.getenv("OFFER_GATE_SAFETY_OVERRIDE", "true").lower() != "false",
}

_DEFAULT_DISCLAIMER_CFG = ObserverDisclaimerCfg()
_DISCLAIMER_SOFTENERS = {
    "ja": ("kari", "suisoku", "moshi", "oshiete", "kudasai", "uketomete"),
    "en": ("hypothesis", "guess", "if", "please", "correct me"),
}

_MAX_DISCLAIMER_LEN = 140
_OBSERVER_CFG_CACHE: Optional[ObserverDisclaimerCfg] = None


def _get_observer_cfg() -> ObserverDisclaimerCfg:
    global _OBSERVER_CFG_CACHE
    if _OBSERVER_CFG_CACHE is not None:
        return _OBSERVER_CFG_CACHE
    try:
        _OBSERVER_CFG_CACHE = load_runtime_cfg().observer
    except Exception:
        _OBSERVER_CFG_CACHE = ObserverDisclaimerCfg()
    return _OBSERVER_CFG_CACHE


def _effective_mode(default_mode: str) -> str:
    mode = os.getenv("OBSERVER_DISCLAIMER_MODE") or default_mode
    mode = (mode or "llm_first").strip().lower()
    valid = {"llm_first", "llm_only", "template_first", "template_only", "fixed_only"}
    if mode not in valid:
        return "llm_first"
    return mode


def _templates_path(cfg: ObserverDisclaimerCfg) -> str:
    return os.getenv("OBSERVER_DISCLAIMER_TEMPLATES") or cfg.templates_path


@lru_cache(maxsize=4)
def _load_templates(path: str) -> Dict[str, Dict[str, List[str]]]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    normalized: Dict[str, Dict[str, List[str]]] = {}
    for culture, tones in data.items():
        if not isinstance(tones, dict):
            continue
        culture_key = str(culture).lower()
        normalized[culture_key] = {}
        for tone, values in tones.items():
            if isinstance(values, list):
                normalized[culture_key][str(tone).lower()] = [str(v) for v in values if v]
    return normalized


def _rotation_index(key: str, length: int, rotation: str) -> int:
    if length <= 0:
        return 0
    if rotation == "random":
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        return rng.randrange(length)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % length


def _template_disclaimer(
    state: Dict[str, Any],
    tone: str,
    culture: str,
    cfg: ObserverDisclaimerCfg,
) -> Optional[str]:
    path = _templates_path(cfg)
    templates = _load_templates(path)
    if not templates:
        return None
    culture_candidates = [culture, culture.split("-")[0], "en-us", "en"]
    tone_candidates = [tone, "default", "casual"]
    offer_gate = state.get("offer_gate", {})
    seed_key = "|".join(
        [
            culture,
            tone,
            str(offer_gate.get("suggestion_allowed", False)),
            offer_gate.get("suppress_reason", "none"),
        ]
    )
    for culture_key in culture_candidates:
        entries = templates.get(culture_key)
        if not entries:
            continue
        for tone_key in tone_candidates:
            options = entries.get(tone_key)
            if options:
                idx = _rotation_index(seed_key, len(options), cfg.rotation)
                return options[idx]
    return None


def _culture_bucket(culture: str) -> str:
    culture = (culture or "en-us").lower()
def _validate_disclaimer(text: str, culture: str) -> bool:
    if not text:
        return False
    if len(text) > _MAX_DISCLAIMER_LEN:
        return False
    bucket = _culture_bucket(culture)
    softeners = _DISCLAIMER_SOFTENERS.get(bucket, ())
    return any(token in text for token in softeners)



def _fixed_disclaimer_text(culture: str | None = None, tone: str | None = None) -> str:
    tone = (tone or "").lower()
    culture = (culture or "").lower()
    if culture.startswith("ja"):
        if tone == "support":
            return "感じ取った手がかりをそっとまとめてお伝えします。違和感があれば遠慮なく教えてくださいね。"
        if tone == "polite":
            return "これは仮説です。違っていたら遠慮なく教えてください。"
        return "これは感じ取った仮説だよ。違っていれば遠慮なく指摘してね。"
    if tone == "support":
        return "This is an observation-based hypothesis. Please let me know if it feels off; I'll stay with you."
    if tone == "polite":
        return "This is an interpretable hypothesis. Kindly correct me if it does not feel right."
    return "This is a working hypothesis. If it feels off, please tell me and I will keep listening."


    if culture.startswith("ja"):
        if tone == "support":
            return "感じ取った手がかりをそっとまとめてお伝えします。違和感があれば遠慮なく教えてくださいね。"
        if tone == "polite":
            return "これは仮説です。もし違っていたら遠慮なくお知らせくださいね。"
        return "いま感じ取った仮説だよ。もし違っていたら遠慮なくつっこんでね。"
    if tone == "support":
        return "This is an observation-based hypothesis. Please let me know if it feels off; I'll stay with you."
    if tone == "polite":
        return "This is an interpretable hypothesis. Kindly correct me if it does not feel right."
    return "This is a working hypothesis. If it feels off, please tell me and I will keep listening."


def _generate_disclaimer_text(state: Dict[str, Any], tone: str, culture: str) -> Tuple[str, str]:
    cfg = _get_observer_cfg()
    mode = _effective_mode(cfg.mode)
    order_map = {
        "llm_first": ("llm", "template", "fixed"),
        "template_first": ("template", "llm", "fixed"),
        "llm_only": ("llm", "fixed"),
        "template_only": ("template", "fixed"),
        "fixed_only": ("fixed",),
    }
    sequence = order_map.get(mode, ("llm", "template", "fixed"))
    for strategy in sequence:
        if strategy == "llm":
            text = _attempt_llm_disclaimer(state, tone, culture, cfg)
            if text:
                return text, "llm"
        elif strategy == "template":
            text = _template_disclaimer(state, tone, culture, cfg)
            if text and _validate_disclaimer(text, culture):
                return text, "template"
        else:
            return _fixed_disclaimer_text(culture, tone), "fixed"
    return _fixed_disclaimer_text(culture, tone), "fixed"


def generate_disclaimer(state: Dict[str, Any], tone: str | None = None, culture: str | None = None) -> str:
    tone_norm = (tone or "casual").lower()
    culture_norm = (culture or "ja-JP").lower()
    cached = state.get("disclaimer")
    if isinstance(cached, str) and cached.strip():
        return cached
    text, source = _generate_disclaimer_text(state, tone_norm, culture_norm)
    state["disclaimer"] = text
    state["disclaimer_source"] = source
    return text


def plutchik_scores(valence: float, arousal: float) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name, (vx, ax) in PLUTCHIK_CENTROIDS.items():
        dist = ((valence - vx) ** 2 + (arousal - ax) ** 2) ** 0.5
        score = max(0.0, 1.0 - min(1.5, dist) / 1.5)
        scores[name] = score
    return scores


def _intent_hypotheses(user_text: str, valence: float, arousal: float) -> List[Tuple[str, float]]:
    text_norm = user_text.strip().lower()
    intents: List[Tuple[str, float]] = []
    question = bool("?" in user_text or re.search(r"(how|why|what|縺ｩ縺・＠縺ｦ|縺ｩ縺・ｄ縺｣縺ｦ)", text_norm))
    guidance_keywords = ("help", "蜉ｩ縺代※", "縺ｩ縺・☆繧後・", "縺ｩ縺・＠縺溘ｉ", "謨吶∴縺ｦ", "謇倶ｼ昴▲縺ｦ")
    if any(keyword in text_norm for keyword in guidance_keywords):
        intents.append(("seek_guidance", 0.72))
    if valence < -0.2:
        intents.append(("seek_reassurance", 0.64))
    if question and valence >= -0.2:
        intents.append(("seek_information", 0.58))
    if not intents:
        intents.append(("thinking_aloud", 0.48))
    if re.search(r"(縺ｩ縺・☆繧後・|縺ｩ縺・＠縺溘ｉ|蜉ｩ縺代※|謇倶ｼ昴▲縺ｦ|help me|please advise)", text_norm):
        intents.insert(0, ("seek_guidance", 0.82))
    return intents


def _cues_from_context(user_text: str, valence: float, arousal: float) -> List[str]:
    cues: List[str] = []
    if "..." in user_text or "窶ｦ" in user_text:
        cues.append("hesitation_tokens")
    if "?" in user_text:
        cues.append("questioning_speech")
    if len(user_text.strip()) == 0:
        cues.append("silence")
    if valence < -0.2:
        cues.append("negative_valence")
    if arousal > 0.4:
        cues.append("heightened_energy")
    elif arousal < -0.2:
        cues.append("low_energy")
    return cues[:4]


def _choose_action(intents: List[Tuple[str, float]], conflict: float) -> Dict[str, Any]:
    if not intents:
        return {"type": "listen_first_2s", "rationale": "insufficient_signals"}
    top_label, top_conf = intents[0]
    if conflict >= 0.65:
        return {"type": "reflect_back", "rationale": "conflict_high"}
    if top_label == "seek_guidance" and top_conf >= 0.6:
        return {"type": "offer_one_idea", "rationale": "guidance_requested"}
    if top_label == "seek_reassurance":
        return {"type": "listen_first_2s", "rationale": "reassurance_priority"}
    if top_label == "thinking_aloud":
        return {"type": "stay_with_silence", "rationale": "thinking_aloud"}
    return {"type": "reflect_back", "rationale": "default_care"}


def _estimate_urgency(user_text: str) -> float:
    text_norm = user_text.lower()
    urgent_keywords = ("help me", "urgent", "danger", "emergency", "asap")
    if any(keyword in text_norm for keyword in urgent_keywords):
        return 0.85
    if "deadline" in text_norm:
        return 0.65
    return 0.25 if "?" in text_norm else 0.0


def _estimate_novelty(user_text: str) -> float:
    if re.search(r"\d", user_text):
        return 0.4
    if len(user_text.split()) > 18:
        return 0.3
    return 0.1


def _within_cooloff(recent_reject: float, cool_off: float) -> bool:
    return recent_reject < cool_off


def _compute_ses(
    intents: List[Tuple[str, float]],
    conflict: float,
    recent_reject: float,
    gate_cfg: Dict[str, float],
    urgency: float,
    trust: float,
    novelty: float,
) -> Tuple[float, str]:
    base = 0.5
    suppress_reason = "intent_bias"
    if intents:
        label, conf = intents[0]
        if label in {"seek_guidance", "seek_information"}:
            base += 0.18 * conf
        if label in {"seek_reassurance"}:
            base -= gate_cfg.get("listen_bias", 0.2) * conf
        if label in {"thinking_aloud"}:
            base -= gate_cfg.get("play_bias", 0.15) * conf
    if _within_cooloff(recent_reject, gate_cfg.get("cool_off_s", 300.0)):
        base -= 0.30 * (1.0 - recent_reject / max(1.0, gate_cfg.get("cool_off_s", 300.0)))
        suppress_reason = "recent_reject"
    base += 0.18 * conflict
    base += 0.15 * urgency
    base += 0.10 * novelty
    base += 0.10 * trust
    base += gate_cfg.get("community_bias", 0.0)
    ses = max(0.0, min(1.0, base))
    return ses, suppress_reason


def infer_observer_state(
    user_text: str,
    affect: Any,
    metrics: Dict[str, float],
    controls: Any,
    recent_reject_seconds: float = 9999.0,
    gate_cfg: Dict[str, float] | None = None,
    trust_score: float = 0.0,
) -> Dict[str, Any]:
    gate_cfg = dict(DEFAULT_GATE_CFG | (gate_cfg or {}))
    observer_cfg = _get_observer_cfg()
    if observer_cfg is not None:
        gate_cfg.setdefault("community_bias", getattr(observer_cfg, "community_bias", gate_cfg.get("community_bias", 0.0)))
        gate_cfg.setdefault("i_message_bias", getattr(observer_cfg, "i_message_bias", 0.0))
    ts = int(time.time() * 1000)
    valence = getattr(affect, "valence", 0.0)
    arousal = getattr(affect, "arousal", 0.0)
    plutchik = plutchik_scores(valence, arousal)
    top_plutchik = [
        name for name, score in sorted(plutchik.items(), key=lambda kv: kv[1], reverse=True) if score > 0.15
    ][:3]

    intents = _intent_hypotheses(user_text, valence, arousal)
    cues = _cues_from_context(user_text, valence, arousal)
    conflict = float(metrics.get("conflict_level", metrics.get("info_flux", 0.0)))
    conflict = max(0.0, min(1.0, conflict))
    action = _choose_action(intents, conflict)
    urgency = _estimate_urgency(user_text)
    novelty = _estimate_novelty(user_text)
    ses, suppress_reason = _compute_ses(
        intents,
        conflict,
        recent_reject_seconds,
        gate_cfg,
        urgency,
        trust_score,
        novelty,
    )
    threshold = gate_cfg.get("threshold", 0.55)
    suggestion_allowed = ses >= threshold
    if gate_cfg.get("safety_override", True) and urgency > 0.82:
        suggestion_allowed = True
        suppress_reason = "safety_override"
    elif not suggestion_allowed:
        if _within_cooloff(recent_reject_seconds, gate_cfg.get("cool_off_s", 300.0)):
            suppress_reason = "recent_reject"
        elif intents and intents[0][0] in {"seek_reassurance", "thinking_aloud"}:
            suppress_reason = "listen_bias"
        else:
            suppress_reason = suppress_reason or "low_confidence"

    eq_fields = {
        "H": round(metrics.get("H", 0.0), 3),
        "R": round(metrics.get("R", 0.0), 3),
        "kappa": round(metrics.get("kappa", 0.0), 3),
        "entropy": round(metrics.get("entropy", 0.0), 3),
        "ignition": round(metrics.get("ignition", metrics.get("info_flux", 0.0)), 3),
    }
    if hasattr(affect, "confidence"):
        eq_fields["perception_confidence"] = round(getattr(affect, "confidence"), 3)
    if hasattr(controls, "temperature"):
        eq_fields["temperature"] = round(controls.temperature, 3)
    if hasattr(controls, "warmth"):
        eq_fields["warmth"] = round(controls.warmth, 3)

    state = {
        "ts_ms": ts,
        "perceived_affect": {
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "plutchik_top": top_plutchik,
        },
        "intent_hypotheses": [
            {"label": label, "confidence": round(conf, 3)} for label, conf in intents
        ],
        "conflict_level": round(conflict, 3),
        "suggested_action": action if suggestion_allowed else {"type": "listen_first_2s", "rationale": "offer_gate_suppressed"},
        "offer_gate": {
            "ses": round(ses, 3),
            "threshold": threshold,
            "suggestion_allowed": suggestion_allowed,
            "suppress_reason": suppress_reason,
            "recent_reject_s": round(recent_reject_seconds, 1),
            "community_bias": round(gate_cfg.get("community_bias", 0.0), 3),
            "i_message_bias": round(gate_cfg.get("i_message_bias", 0.0), 3),
        },
        "evidence": {
            "cues": cues,
            "eqnet_fields": eq_fields,
        },
        "disclaimer": None,
    }
    return state



def observer_markdown(state: Dict[str, Any], tone: str | None = None, culture: str | None = None) -> str:
    lines = ["### Observer Commentary"]
    offer = state.get("offer_gate", {})
    ses = offer.get("ses", 0.0)
    allowed = offer.get("suggestion_allowed", False)
    threshold = offer.get("threshold", 0.55)
    reason = offer.get("suppress_reason", "intent_bias")
    lines.append(f"- Suggestion eligibility (SES): {ses:.2f} {'OK' if allowed else 'HOLD'} (threshold {threshold:.2f})")
    if not allowed:
        lines.append(f"  reason: {reason} (cool-off {offer.get('recent_reject_s', 0)}s)")
    perceived = state.get("perceived_affect", {})
    lines.append("- Perceived affect (user): "
                 f"valence {perceived.get('valence', 0.0):+.2f}, arousal {perceived.get('arousal', 0.0):+.2f}, "
                 f"Plutchik {perceived.get('plutchik_top', [])}")
    intents = state.get("intent_hypotheses", [])
    if intents:
        lines.append("- Intent hypotheses:")
        for hyp in intents[:3]:
            lines.append(f"  窶｢ {hyp['label']} ({hyp['confidence']*100:.0f}%)")
    lines.append(f"- Conflict level: {state.get('conflict_level', 0.0):.2f}")
    if allowed:
        action = state.get("suggested_action", {})
        lines.append(f"- Suggested action: {action.get('type', 'listen_first_2s')} ({action.get('rationale', 'rationale')})")
    else:
        lines.append("- Suggested action: withheld (listening stance)")
    evidence = state.get("evidence", {})
    cues = evidence.get("cues", [])
    lines.append(f"- Evidence cues: {cues}")
    eq_fields = evidence.get("eqnet_fields", {})
    lines.append(f"- EQNet fields: {eq_fields}")
    culture_summary = state.get("culture_summary")
    if culture_summary:
        lines.append("")
        lines.append("**Culture cues**")
        summary_lines = culture_summary.get("lines") or []
        if summary_lines:
            for item in summary_lines[:3]:
                lines.append(f"- {item}")
        else:
            tag = culture_summary.get("tag")
            stats = culture_summary.get("stats") or {}
            if tag:
                lines.append(f"- {tag}: valence {stats.get('mean_valence', 0.0):+.2f}")
    lines.append("")
    lines.append(generate_disclaimer(state, tone, culture))
    return "\n".join(lines)

__all__ = [
    "plutchik_scores",
    "infer_observer_state",
    "observer_markdown",
    "generate_disclaimer",
    "DEFAULT_GATE_CFG",
]
