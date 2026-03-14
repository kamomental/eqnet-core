from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping


OBJECT_RELATION_WEIGHTS = {
    "graspable": 0.18,
    "fragile": 0.26,
    "avoidable": 0.24,
    "inviting": 0.16,
    "tool_like": 0.18,
    "ritually_sensitive": 0.14,
    "attachment": 0.12,
    "avoidance": 0.14,
}

FRAGILE_TERMS = {"glass", "lantern", "cup", "bowl", "vase", "mirror", "ceramic", "shell"}
AVOIDABLE_TERMS = {"knife", "blade", "fire", "flame", "wire", "needle", "hazard", "thorn"}
INVITING_TERMS = {"apple", "book", "blanket", "pillow", "toy", "lantern", "mug", "cup"}
TOOL_TERMS = {"pen", "brush", "tool", "key", "rope", "stick", "bag", "map"}
RITUAL_TERMS = {"altar", "relic", "offering", "lantern", "monument", "token", "ring"}
ATTACHMENT_TERMS = {"gift", "photo", "keepsake", "letter", "ring", "lantern", "token"}


@dataclass
class ObjectRelationSnapshot:
    object_affordance_bias: float = 0.0
    fragility_guard: float = 0.0
    object_attachment: float = 0.0
    object_avoidance: float = 0.0
    tool_extension_bias: float = 0.0
    ritually_sensitive_bias: float = 0.0
    handle_gently: bool = False
    dominant_object: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ObjectRelationCore:
    def snapshot(
        self,
        *,
        relational_world: Mapping[str, Any] | None = None,
        current_state: Mapping[str, Any] | None = None,
        recall_payload: Mapping[str, Any] | None = None,
    ) -> ObjectRelationSnapshot:
        world = dict(relational_world or {})
        state = dict(current_state or {})
        recall = dict(recall_payload or {})
        nearby = list(world.get("nearby_objects") or [])
        referenced = []
        if recall:
            for key in ("summary", "text", "memory_anchor"):
                value = str(recall.get(key) or "").strip()
                if value:
                    referenced.append(value)
        source = " ".join([*nearby, *referenced]).lower()
        terms = {token for token in source.replace("|", " ").replace(",", " ").split() if token}

        fragility_guard = _clamp01(sum(OBJECT_RELATION_WEIGHTS["fragile"] for token in terms if token in FRAGILE_TERMS))
        object_avoidance = _clamp01(sum(OBJECT_RELATION_WEIGHTS["avoidable"] for token in terms if token in AVOIDABLE_TERMS))
        object_attachment = _clamp01(sum(OBJECT_RELATION_WEIGHTS["attachment"] for token in terms if token in ATTACHMENT_TERMS))
        tool_extension_bias = _clamp01(sum(OBJECT_RELATION_WEIGHTS["tool_like"] for token in terms if token in TOOL_TERMS))
        inviting_bias = _clamp01(sum(OBJECT_RELATION_WEIGHTS["inviting"] for token in terms if token in INVITING_TERMS))
        ritual_bias = _clamp01(sum(OBJECT_RELATION_WEIGHTS["ritually_sensitive"] for token in terms if token in RITUAL_TERMS))
        graspable_bias = _clamp01(min(len(nearby), 3) * OBJECT_RELATION_WEIGHTS["graspable"])
        object_affordance_bias = _clamp01(graspable_bias + inviting_bias * 0.6 + tool_extension_bias * 0.5)

        if state:
            object_attachment = _clamp01(max(object_attachment, _float_from(state, "object_attachment", 0.0)))
            object_avoidance = _clamp01(max(object_avoidance, _float_from(state, "object_avoidance", 0.0)))
            fragility_guard = _clamp01(max(fragility_guard, _float_from(state, "fragility_guard", 0.0)))
            tool_extension_bias = _clamp01(max(tool_extension_bias, _float_from(state, "tool_extension_bias", 0.0)))
            ritual_bias = _clamp01(max(ritual_bias, _float_from(state, "ritually_sensitive_bias", 0.0)))
            object_affordance_bias = _clamp01(max(object_affordance_bias, _float_from(state, "object_affordance_bias", 0.0)))

        dominant_object = str(nearby[0]).strip() if nearby else ""
        handle_gently = fragility_guard >= 0.24 or ritual_bias >= 0.22
        return ObjectRelationSnapshot(
            object_affordance_bias=round(object_affordance_bias, 4),
            fragility_guard=round(fragility_guard, 4),
            object_attachment=round(object_attachment, 4),
            object_avoidance=round(object_avoidance, 4),
            tool_extension_bias=round(tool_extension_bias, 4),
            ritually_sensitive_bias=round(ritual_bias, 4),
            handle_gently=handle_gently,
            dominant_object=dominant_object,
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float_from(mapping: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default
