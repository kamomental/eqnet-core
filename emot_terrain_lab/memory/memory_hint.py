"""Memory hint rendering that avoids verbatim recall."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple

from emot_terrain_lab.i18n.locale import lookup_text, truncate_text


def _compile_patterns(patterns: Any) -> list[re.Pattern]:
    compiled: list[re.Pattern] = []
    if not patterns:
        return compiled
    for raw in patterns:
        try:
            compiled.append(re.compile(str(raw)))
        except re.error:
            continue
    return compiled


def _clamp(value: float, floor: float, ceiling: float) -> float:
    if value != value or value is None:
        return float(floor)
    return float(max(floor, min(ceiling, value)))


def _normalize_social_mode(mode: Any) -> str:
    return str(mode or "solo").strip().lower()


def _normalize_tags(value: Any) -> Sequence[str]:
    if not value:
        return ()
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return (str(value),)


def _select_hint_category(
    *,
    reason: Optional[str],
    confidence: float,
    pressure: float,
    cfg: Any,
    gate_ctx: Optional[Dict[str, Any]],
) -> str:
    if reason in {"turn_taking", "interrupt_cost", "social_bandwidth"}:
        return "boundary"
    if reason == "low_confidence":
        return "presence"
    gate_ctx = gate_ctx or {}
    novelty_score = gate_ctx.get("novelty_score")
    link_tags = _normalize_tags(gate_ctx.get("link_tags"))
    value_seed_tags = _normalize_tags(gate_ctx.get("value_seed_tags"))
    try:
        novelty_score = float(novelty_score) if novelty_score is not None else None
    except (TypeError, ValueError):
        novelty_score = None
    if novelty_score is not None and novelty_score >= 0.6:
        return "value_external"
    if link_tags:
        return "link"
    if value_seed_tags:
        return "value_internal"
    if pressure > float(getattr(cfg, "pressure_threshold", 1.0) or 1.0):
        return "boundary"
    if confidence >= 0.65:
        return "value_internal"
    return "presence"


def _select_hint_key(
    *,
    prefix: str,
    category: Optional[str],
    variant: str,
    locale: str,
) -> Tuple[Optional[str], Optional[str]]:
    keys = []
    if category:
        keys.append(f"{prefix}{category}.{variant}")
    keys.append(f"{prefix}{variant}")
    for key in keys:
        text = lookup_text(locale, key)
        if text:
            return key, text
    return None, None


def update_pressure(
    *,
    prev_pressure: float,
    dt_s: float,
    interrupt_cost: float,
    social_bandwidth: float,
    low_confidence: float,
    cfg: Any,
) -> Tuple[float, float]:
    decay = float(getattr(cfg, "pressure_decay_per_s", 1.0) or 1.0)
    decay = _clamp(decay, 0.0, 1.0)
    floor = float(getattr(cfg, "pressure_floor", 0.0) or 0.0)
    ceiling = float(getattr(cfg, "pressure_ceiling", 1.0) or 1.0)
    dt_s = max(0.0, float(dt_s))
    p = float(prev_pressure)
    if dt_s > 0.0:
        p *= decay**dt_s
    p += float(getattr(cfg, "pressure_w_interrupt", 0.0) or 0.0) * interrupt_cost
    p += float(getattr(cfg, "pressure_w_social", 0.0) or 0.0) * social_bandwidth
    p += float(getattr(cfg, "pressure_w_lowconf", 0.0) or 0.0) * low_confidence
    p = _clamp(p, floor, ceiling)
    return p, p - prev_pressure


def decide_block_by_pressure(
    *,
    pressure: float,
    prev_blocked: bool,
    cfg: Any,
) -> bool:
    threshold = float(getattr(cfg, "pressure_threshold", 1.0) or 1.0)
    margin = float(getattr(cfg, "pressure_margin", 0.0) or 0.0)
    if prev_blocked:
        return not (pressure < (threshold - margin))
    return pressure > threshold


def evaluate_hint_suppression(
    *,
    cfg: Any,
    gate_ctx: Optional[Dict[str, Any]],
    shadow_uncertainty: Optional[float],
    last_speaker: Optional[str],
    prev_pressure: float,
    dt_s: float,
    prev_blocked: bool,
) -> Tuple[bool, Optional[str], float, float, float, float]:
    gate_ctx = gate_ctx or {}
    since_last_user_ms = float(gate_ctx.get("since_last_user_ms", 0.0) or 0.0)
    text_input = bool(gate_ctx.get("text_input", False))
    min_silence = float(getattr(cfg, "min_silence_ms_for_hint", 0.0) or 0.0)
    silence_ratio = 0.0
    if min_silence > 0.0:
        silence_ratio = 1.0 - _clamp(since_last_user_ms / min_silence, 0.0, 1.0)
    interrupt_cost = _clamp(silence_ratio, 0.0, 1.0)

    social_mode = _normalize_social_mode(getattr(cfg, "social_mode", "solo"))
    social_bandwidth = 0.0
    if social_mode == "community":
        social_bandwidth = 0.2
    elif social_mode == "meeting":
        social_bandwidth = 0.4

    confidence = 0.5
    if shadow_uncertainty is not None:
        confidence = 1.0 - _clamp(float(shadow_uncertainty), 0.0, 1.0)
    low_confidence = _clamp(1.0 - confidence, 0.0, 1.0)

    turn_mode = _normalize_social_mode(getattr(cfg, "turn_taking_mode", "none"))
    if turn_mode == "last_speaker_not_ai" and str(last_speaker).lower() == "ai":
        return True, "turn_taking", interrupt_cost, confidence, prev_pressure, 0.0

    if text_input or since_last_user_ms < min_silence:
        return True, "interrupt_cost", interrupt_cost, confidence, prev_pressure, 0.0

    new_pressure, delta = update_pressure(
        prev_pressure=prev_pressure,
        dt_s=dt_s,
        interrupt_cost=interrupt_cost,
        social_bandwidth=social_bandwidth,
        low_confidence=low_confidence,
        cfg=cfg,
    )

    blocked = decide_block_by_pressure(
        pressure=new_pressure, prev_blocked=prev_blocked, cfg=cfg
    )
    if blocked:
        reason = "pressure"
        contributions = {
            "interrupt_cost": interrupt_cost,
            "social_bandwidth": social_bandwidth,
            "low_confidence": low_confidence,
        }
        top = max(contributions, key=contributions.get)
        if contributions.get(top, 0.0) > 0.0:
            reason = top
        return True, reason, interrupt_cost, confidence, new_pressure, delta

    return False, None, interrupt_cost, confidence, new_pressure, delta


def render_memory_hint(
    label: Optional[str],
    fidelity: float,
    *,
    locale: str,
    cfg: Any,
    gate_ctx: Optional[Dict[str, Any]] = None,
    shadow_uncertainty: Optional[float] = None,
    last_speaker: Optional[str] = None,
    prev_pressure: float = 0.0,
    dt_s: float = 0.0,
    prev_blocked: bool = False,
) -> Optional[Dict[str, Any]]:
    if cfg is None or not getattr(cfg, "enable", False):
        return None
    if getattr(cfg, "hint_style", "temperature") == "none":
        return None
    allow_verbatim = bool(getattr(cfg, "allow_verbatim", False))
    if allow_verbatim:
        return {
            "enabled": True,
            "shown": False,
            "blocked": False,
            "reason": "verbatim_allowed",
        }

    blocked, reason, interrupt_cost, confidence, pressure, pressure_delta = evaluate_hint_suppression(
        cfg=cfg,
        gate_ctx=gate_ctx,
        shadow_uncertainty=shadow_uncertainty,
        last_speaker=last_speaker,
        prev_pressure=prev_pressure,
        dt_s=dt_s,
        prev_blocked=prev_blocked,
    )
    if blocked:
        style_when_suppressed = _normalize_social_mode(
            getattr(cfg, "style_when_suppressed", "none")
        )
        if style_when_suppressed == "minimal":
            key = f"{getattr(cfg, 'hint_key_prefix', 'memory_hint.')}minimal"
            text = lookup_text(locale, key)
            if text:
                return {
                    "enabled": True,
                    "shown": True,
                    "blocked": True,
                    "reason": reason,
                    "key": key,
                    "style": "minimal",
                    "text": truncate_text(
                        text, int(getattr(cfg, "max_reply_chars", 0)) or None
                    ),
                    "interrupt_cost": interrupt_cost,
                    "confidence": confidence,
                    "social_mode": _normalize_social_mode(
                        getattr(cfg, "social_mode", "solo")
                    ),
                    "pressure": pressure,
                    "pressure_delta": pressure_delta,
                }
        return {
            "enabled": True,
            "shown": False,
            "blocked": True,
            "reason": reason,
            "interrupt_cost": interrupt_cost,
            "confidence": confidence,
            "social_mode": _normalize_social_mode(getattr(cfg, "social_mode", "solo")),
            "pressure": pressure,
            "pressure_delta": pressure_delta,
        }

    if fidelity >= 0.65:
        suffix = "warm"
    elif fidelity >= 0.45:
        suffix = "anchor"
    else:
        suffix = "faint"

    prefix = str(getattr(cfg, "hint_key_prefix", "memory_hint."))
    category = _select_hint_category(
        reason=reason,
        confidence=confidence,
        pressure=pressure,
        cfg=cfg,
        gate_ctx=gate_ctx,
    )
    key, text = _select_hint_key(
        prefix=prefix,
        category=category,
        variant=suffix,
        locale=locale,
    )
    if not text:
        return {
            "enabled": True,
            "shown": False,
            "blocked": True,
            "reason": "missing_key",
            "key": key,
            "style": str(getattr(cfg, "hint_style", "temperature")),
            "interrupt_cost": interrupt_cost,
            "confidence": confidence,
            "social_mode": _normalize_social_mode(getattr(cfg, "social_mode", "solo")),
            "pressure": pressure,
            "pressure_delta": pressure_delta,
        }

    ban_patterns = _compile_patterns(getattr(cfg, "ban_patterns", []))
    for pattern in ban_patterns:
        if pattern.search(text):
            return {
                "enabled": True,
                "shown": False,
                "blocked": True,
                "reason": "ban_pattern",
                "key": key,
                "style": str(getattr(cfg, "hint_style", "temperature")),
                "interrupt_cost": interrupt_cost,
                "confidence": confidence,
                "social_mode": _normalize_social_mode(getattr(cfg, "social_mode", "solo")),
                "pressure": pressure,
                "pressure_delta": pressure_delta,
            }

    max_reply_chars = int(getattr(cfg, "max_reply_chars", 0)) or None
    hint_text = truncate_text(text, max_reply_chars)
    if not hint_text:
        return {
            "enabled": True,
            "shown": False,
            "blocked": True,
            "reason": "empty",
            "key": key,
            "style": str(getattr(cfg, "hint_style", "temperature")),
            "interrupt_cost": interrupt_cost,
            "confidence": confidence,
            "social_mode": _normalize_social_mode(getattr(cfg, "social_mode", "solo")),
            "pressure": pressure,
            "pressure_delta": pressure_delta,
        }

    payload = {
        "enabled": True,
        "shown": True,
        "blocked": False,
        "reason": "ok",
        "text": hint_text,
        "key": key,
        "style": str(getattr(cfg, "hint_style", "temperature")),
        "interrupt_cost": interrupt_cost,
        "confidence": confidence,
        "social_mode": _normalize_social_mode(getattr(cfg, "social_mode", "solo")),
        "pressure": pressure,
        "pressure_delta": pressure_delta,
    }
    if label:
        max_label_chars = int(getattr(cfg, "max_label_chars", 0)) or None
        if max_label_chars:
            payload["label_preview"] = truncate_text(label, max_label_chars)
    return payload
