from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OffAction:
    v_scale: float = 1.0
    jitter: float = 0.0
    d_target: float = 0.0
    pause_s: float = 0.0
    utterance_tag: Optional[str] = None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def choose_off_goal(perception: Dict[str, Any]) -> str:
    ctx = str(perception.get("context", "general"))
    mapping = {
        "cooking": "調理を続ける",
        "casual": "会話を続ける",
        "hallway": "周囲を確認",
        "unknown": "周囲を確認",
        "park": "見守る",
    }
    return mapping.get(ctx, "巡回する")


def choose_off_state(perception: Dict[str, Any], action: Dict[str, Any]) -> str:
    hazard = _float(perception.get("hazard_hint"), 0.0)
    if hazard > 0.6 or _float(action.get("d_target"), 0.0) > 0.25:
        return "警戒"
    if perception.get("offer_seen"):
        return "応答"
    return "巡回"


def choose_off_utterance(perception: Dict[str, Any], action: Dict[str, Any]) -> Optional[str]:
    hazard = _float(perception.get("hazard_hint"), 0.0)
    if hazard > 0.6 or _float(action.get("d_target"), 0.0) > 0.25:
        return "気を付けよう"
    offer = perception.get("offer_type")
    if offer == "sweet":
        return "あ、ありがとう"
    return None


def decide_off(stim: Dict[str, Any]) -> Dict[str, Any]:
    offer = stim.get("offer") or {}
    ambient = stim.get("ambient") or {}

    action = OffAction()

    if offer:
        dist = _float(offer.get("distance"), 999.0)
        hazard = _float(offer.get("hazard_hint"), 0.0)

        if dist < 0.8:
            action.d_target = 0.25
        if hazard > 0.6:
            action.d_target = max(action.d_target, 0.35)
            action.v_scale = min(action.v_scale, 0.6)
            action.utterance_tag = action.utterance_tag or "reactive_avoid"

    sound = _float(ambient.get("sound_level"), 0.0)
    if sound > 0.5:
        action.v_scale = min(action.v_scale, 0.7)

    perception = {
        "offer_seen": bool(offer),
        "offer_type": offer.get("type") if offer else None,
        "context": offer.get("context") if offer else None,
        "location": offer.get("location") if offer else None,
        "hazard_hint": offer.get("hazard_hint") if offer else None,
        "sound_level": sound,
    }

    internal = {"mode": "reactive"}

    act = {
        "v_scale": action.v_scale,
        "jitter": action.jitter,
        "d_target": action.d_target,
        "pause_s": action.pause_s,
        "utterance_tag": action.utterance_tag,
    }

    npc_meta = {
        "goal": choose_off_goal(perception),
        "state": choose_off_state(perception, act),
        "utterance": choose_off_utterance(perception, act),
    }

    return {
        "perception": perception,
        "internal": internal,
        "action": act,
        "npc": npc_meta,
    }
