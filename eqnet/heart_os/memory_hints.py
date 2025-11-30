"""Memory hint scaffolding for Heart OS conversations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

from .session_config import ConversationMode, VoiceStyle
from eqnet.memory.moment_knn import Neighbor
from eqnet.memory.terrain_state import TerrainState
from eqnet.memory.monuments import Monument

_NEG_EMOTION = {"anger", "fear", "sadness"}
_POS_EMOTION = {"joy", "trust", "anticipation"}


@dataclass(slots=True)
class MemoryHint:
    """Lightweight hint payload passed downstream to the LLM."""

    type: str
    source: str
    intent: str
    confidence: float
    emotion_tag: Optional[str]
    text: str

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_kg_facts_from_rag_docs(
    rag_docs: Sequence[str],
    context: str,
    emotion_tag: Optional[str],
) -> List[Dict[str, Any]]:
    """Compress raw ``rag_docs`` identifiers into KG fact snippets."""

    facts: List[Dict[str, Any]] = []
    for doc_id in rag_docs:
        if doc_id.startswith("mem-family"):
            facts.append(
                {
                    "type": "family_episode",
                    "confidence": 0.9,
                    "emotion_tag": emotion_tag or "joy",
                    "text": (
                        "Evenings at the window with mom, humming the new song, "
                        "are the ritual Nazuna recalls whenever she wants to soften her heart."
                    ),
                }
            )
        elif doc_id.startswith("mem-safety"):
            facts.append(
                {
                    "type": "safety_episode",
                    "confidence": 0.85,
                    "emotion_tag": emotion_tag or "trust",
                    "text": (
                        "During rehearsal when her breathing grew shallow, Ria brewed herb tea "
                        "and said 'let's rest a bit', a cue that it's okay not to push too hard."
                    ),
                }
            )
        elif doc_id.startswith("mem-unit"):
            facts.append(
                {
                    "type": "unit_bond",
                    "confidence": 0.8,
                    "emotion_tag": emotion_tag or "love",
                    "text": (
                        "Singing with Kotone is the sound that gives Nazuna confidence in her own voice."
                    ),
                }
            )
        else:
            facts.append(
                {
                    "type": "kg_fact",
                    "confidence": 0.7,
                    "emotion_tag": emotion_tag,
                    "text": context.strip() or "recent memory",
                }
            )
    return facts[:2]


def _topic_for(moment: Any) -> Optional[str]:
    topic = getattr(moment, "topic", None)
    if topic:
        return str(topic)
    culture = getattr(moment, "culture", None)
    if isinstance(culture, dict):
        return culture.get("topic")
    if hasattr(culture, "get"):
        try:
            return culture.get("topic")  # type: ignore[attr-defined]
        except Exception:
            return None
    rag = getattr(moment, "rag", None)
    if isinstance(rag, dict):
        return rag.get("topic")
    return None


def should_use_memory(
    moment: Any,
    neighbors: Sequence[Neighbor],
    mode: ConversationMode,
    last_used_episode_id: Optional[str],
) -> bool:
    if not neighbors:
        return False
    max_sim = max((n.similarity for n in neighbors), default=0.0)
    topic = _topic_for(moment)
    has_topic_match = any(n.topic == topic for n in neighbors if topic)
    if mode == "stream":
        if max_sim < 0.6:
            return False
        if topic and not has_topic_match:
            return False
        if last_used_episode_id and neighbors[0].episode_id == last_used_episode_id:
            return False
        return True
    if max_sim < 0.3:
        return False
    if last_used_episode_id and neighbors[0].episode_id == last_used_episode_id and max_sim < 0.9:
        return False
    return True


def build_memory_hints(
    *,
    mode: ConversationMode,
    emotion_tag: Optional[str],
    kg_facts: Sequence[Dict[str, Any]],
    neighbors: Optional[Sequence[Neighbor]] = None,
    moment_topic: Optional[str] = None,
    episodes_by_id: Optional[Dict[str, Any]] = None,
    monuments_by_id: Optional[Dict[str, Any]] = None,
    voice_style: VoiceStyle = "normal",
) -> List[MemoryHint]:
    """Convert available facts (currently KG only) into ``MemoryHint`` objects."""

    hints: List[MemoryHint] = []
    for fact in kg_facts:
        f_type = fact.get("type", "kg_fact")
        base_intent = "expand"
        if f_type in {"blocked_share", "off_topic"}:
            base_intent = "ack_only"
        hint = MemoryHint(
            type=f_type,
            source="KG",
            intent=base_intent,
            confidence=float(fact.get("confidence", 0.7)),
            emotion_tag=fact.get("emotion_tag", emotion_tag),
            text=str(fact.get("text", "")),
        )
        hints.append(_decorate_hint_with_emotion(hint))

    if neighbors:
        for idx, neighbor in enumerate(neighbors):
            if idx >= 3:
                break
            source = "L1"
            if neighbor.monument_id:
                source = "L3"
            elif neighbor.episode_id:
                source = "L2"
            intent = "expand"
            if voice_style == "whisper" and idx > 0:
                intent = "ack_only"
            hint_type = "memory"
            if moment_topic and neighbor.topic and neighbor.topic != moment_topic and mode == "meet":
                hint_type = "off_topic"
                intent = "ack_only"
            confidence = float(max(0.0, min(1.0, neighbor.similarity)))
            hint = MemoryHint(
                type=hint_type,
                source=source,
                intent=intent,
                confidence=confidence,
                emotion_tag=neighbor.emotion_tag,
                text=neighbor.summary or neighbor.topic or "recent memory",
            )
            hints.append(_decorate_hint_with_emotion(hint))
        if mode == "meet" and moment_topic and not any(n.topic == moment_topic for n in neighbors):
            hints.append(
                MemoryHint(
                    type="off_topic",
                    source="meta",
                    intent="ack_only",
                    confidence=0.4,
                    emotion_tag=None,
                    text=f"pivot gently back to {moment_topic}",
                )
            )

    return hints


def _decorate_hint_with_emotion(hint: MemoryHint) -> MemoryHint:
    if hint.emotion_tag in _NEG_EMOTION:
        hint.intent = "ack_only"
    elif hint.emotion_tag in _POS_EMOTION and hint.intent != "ack_only":
        hint.intent = "expand"
    return hint


__all__ = [
    "MemoryHint",
    "build_memory_hints",
    "build_kg_facts_from_rag_docs",
    "should_use_memory",
    "build_place_memory_hints",
]


def build_place_memory_hints(
    place_id: Optional[str], terrain_state: Optional[TerrainState], k: int = 2
) -> List[MemoryHint]:
    """Return brief hints for monuments connected to ``place_id``."""

    if not place_id or terrain_state is None:
        return []
    monuments: List[Monument] = terrain_state.find_monuments_by_place(place_id, k)
    hints: List[MemoryHint] = []
    for mon in monuments:
        hints.append(
            MemoryHint(
                type="place_memory",
                source="L3",
                intent="expand",
                confidence=float(mon.importance),
                emotion_tag=mon.core_emotion,
                text=mon.summary,
            )
        )
    return hints
