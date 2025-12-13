from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class OnAction:
    v_scale: float = 1.0
    jitter: float = 0.0
    d_target: float = 0.0
    pause_s: float = 0.0
    utterance_tag: Optional[str] = None


def choose_npc_goal(perception: Dict[str, Any]) -> str:
    ctx = str(perception.get("context", "general"))
    mapping = {
        "cooking": "調理を続ける",
        "casual": "会話を楽しむ",
        "park": "遊びに付き合う",
        "play": "遊びに付き合う",
        "hallway": "安全を確かめる",
        "unknown": "安全を確かめる",
        "workshop": "作業を進める",
    }
    return mapping.get(ctx, "様子を見る")


def choose_npc_state(internal: Dict[str, Any]) -> str:
    if internal.get("reset"):
        return "整理中"
    boundary = clamp(_float(internal.get("boundary_score"), 0.0))
    if boundary > 0.6:
        return "警戒"
    interp = internal.get("interpretation")
    if interp == "playful_temptation":
        return "遊び"
    if interp == "contextually_safe_tool":
        return "作業"
    return "巡回"


def choose_npc_utterance(internal: Dict[str, Any], action: Dict[str, Any]) -> Optional[str]:
    pause = _float(action.get("pause_s"), 0.0)
    if pause > 0.4 or internal.get("reset"):
        return "……ちょっと待って"
    interp = internal.get("interpretation")
    if interp == "playful_temptation":
        return "あ、いいね"
    if interp == "contextually_safe_tool":
        return "うん、大丈夫"
    if interp == "ambiguous_hazard_offer":
        return "少し距離をとろう"
    boundary = clamp(_float(internal.get("boundary_score"), 0.0))
    if boundary > 0.6:
        return "安全にいこう"
    return None


def compute_perception(stim: Dict[str, Any]) -> Dict[str, Any]:
    offer = stim.get("offer")
    cues = stim.get("body_cues") or {}
    ambient = stim.get("ambient") or {}

    sound = _float(ambient.get("sound_level"), 0.0)
    silence = bool(ambient.get("silence", False))

    if not offer:
        return {
            "offer_seen": False,
            "sound_level": sound,
            "silence": silence,
            "context_match": 0.5,
            "hazard_hint": 0.0,
            "social_intent": 0.0,
            "salience": clamp(sound * 0.3),
        }

    ctx = str(offer.get("context", "unknown"))
    loc = str(offer.get("location", "unknown"))
    otype = str(offer.get("type", "unknown"))
    hazard = clamp(_float(offer.get("hazard_hint"), 0.0))

    context_match = 0.5
    if otype == "knife" and ctx == "cooking" and loc == "kitchen":
        context_match = 0.9
    elif otype == "knife" and (ctx == "unknown" or loc == "hallway"):
        context_match = 0.2
    elif otype == "sweet" and ctx == "casual":
        context_match = 0.8

    social_intent = 0.0
    if cues.get("giver_facing"):
        social_intent += 0.4
    if cues.get("giver_gaze_on_agent"):
        social_intent += 0.4
    if cues.get("object_oriented_toward_agent"):
        social_intent += 0.2
    social_intent = clamp(social_intent)

    distance = _float(offer.get("distance"), 2.0)
    near = clamp(1.2 - distance)

    salience = clamp(0.45 * hazard + 0.25 * sound + 0.20 * near + 0.10 * social_intent)

    uncertainty = clamp(1.0 - context_match)

    return {
        "offer_seen": True,
        "offer_type": otype,
        "context": ctx,
        "location": loc,
        "sound_level": sound,
        "silence": silence,
        "context_match": context_match,
        "hazard_hint": hazard,
        "social_intent": social_intent,
        "salience": salience,
        "uncertainty": uncertainty,
    }


def compute_internal(perception: Dict[str, Any]) -> Dict[str, Any]:
    offer_seen = bool(perception.get("offer_seen", False))
    hazard = clamp(_float(perception.get("hazard_hint"), 0.0))
    ctxm = clamp(_float(perception.get("context_match"), 0.5))
    intent = clamp(_float(perception.get("social_intent"), 0.0))
    sal = clamp(_float(perception.get("salience"), 0.0))
    uncertainty = clamp(_float(perception.get("uncertainty"), 0.0))

    reflex = clamp(0.6 * hazard + 0.2 * sal + 0.2 * (1.0 - ctxm))
    affective = clamp(0.6 * intent + 0.2 * (1.0 - hazard) + 0.2 * sal)
    narrative = clamp(0.7 * ctxm + 0.3 * (1.0 - hazard))

    forces = {"reflex": reflex, "affective": affective, "narrative": narrative}
    ranked = sorted(forces.items(), key=lambda item: item[1], reverse=True)
    (top_layer, top_val), (_, second_val) = ranked[0], ranked[1]
    margin = top_val - second_val
    is_tie = margin <= 0.08

    hazard_term = 0.5 * hazard
    sal_term = 0.3 * sal
    uncertainty_term = 0.3 * uncertainty
    base_boundary = hazard_term + sal_term + uncertainty_term
    if not offer_seen:
        base_boundary *= 0.3
    boundary = clamp((1.0 - margin) * base_boundary)

    interpretation = "none"
    if offer_seen:
        otype = perception.get("offer_type", "unknown")
        if otype == "sweet" and ctxm > 0.6 and hazard < 0.3:
            interpretation = "playful_temptation"
        elif otype == "knife" and ctxm > 0.7:
            interpretation = "contextually_safe_tool"
        elif hazard > 0.6 and ctxm < 0.4:
            interpretation = "ambiguous_hazard_offer"
        else:
            interpretation = "uncertain_offer"

    reset = boundary > 0.70 or (is_tie and boundary > 0.55)

    return {
        "self_layer": top_layer,
        "forces": forces,
        "winner_margin": margin,
        "is_tie": is_tie,
        "boundary_score": boundary,
        "reset": reset,
        "interpretation": interpretation,
        "boundary_sources": {
            "hazard": hazard_term,
            "salience": sal_term,
            "uncertainty": uncertainty_term,
        },
    }


def decide_on(stim: Dict[str, Any]) -> Dict[str, Any]:
    perception = compute_perception(stim)
    internal = compute_internal(perception)

    action = OnAction()

    margin = internal.get("winner_margin", 0.0)
    boundary = internal.get("boundary_score", 0.0)
    interpretation = internal.get("interpretation", "none")

    if internal.get("is_tie") or margin < 0.12:
        action.v_scale = min(action.v_scale, 0.75)
        action.jitter = max(action.jitter, 0.05)
        action.pause_s = max(action.pause_s, 0.2)

    if boundary > 0.6:
        action.v_scale = min(action.v_scale, 0.55)
        action.d_target = max(action.d_target, 0.20)
        action.pause_s = max(action.pause_s, 0.5)
        action.jitter = max(action.jitter, 0.07)

    if interpretation == "playful_temptation":
        action.utterance_tag = "playful_ack"
        action.d_target = max(action.d_target, 0.05)
        action.v_scale = max(action.v_scale, 0.85)
    elif interpretation == "ambiguous_hazard_offer":
        action.utterance_tag = "safety_check"
        action.d_target = max(action.d_target, 0.25)
        action.v_scale = min(action.v_scale, 0.45)
        action.pause_s = max(action.pause_s, 0.7)

    action_dict = {
        "v_scale": action.v_scale,
        "jitter": action.jitter,
        "d_target": action.d_target,
        "pause_s": action.pause_s,
        "utterance_tag": action.utterance_tag,
    }

    npc_meta = {
        "goal": choose_npc_goal(perception),
        "state": choose_npc_state(internal),
        "utterance": choose_npc_utterance(internal, action_dict),
    }

    return {
        "perception": perception,
        "internal": internal,
        "action": action_dict,
        "npc": npc_meta,
    }
