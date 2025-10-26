# -*- coding: utf-8 -*-
"""
Community Orchestrator
----------------------

Lightweight orchestration layer that keeps track of multi-speaker conversations,
shared canon, and group synchrony. The implementation focuses on transparent
data structures and heuristic updates that can be extended later with richer
models (e.g., graph embeddings, trained AKOrN controllers).
"""

from __future__ import annotations

import math
import re
import os
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import yaml

from .emotion import AXES, extract_emotion


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """Single utterance observed by the orchestrator."""

    speaker_id: str
    text: str
    prosody: Optional[Mapping[str, float]] = None
    timestamp: Optional[datetime] = None

    def resolved_timestamp(self) -> datetime:
        return self.timestamp or datetime.utcnow()


@dataclass
class CommunityPolicy:
    spoiler_mode: str = "warn"  # warn | soft_block | free
    rating: str = "G"  # G | PG | PG-13
    banter_style: str = "medium"  # low | medium | high
    shipping_sensitivity: float = 0.3

    @staticmethod
    def from_profile(profile: "CommunityProfile") -> "CommunityPolicy":
        norms = profile.norms
        return CommunityPolicy(
            spoiler_mode=norms.get("spoiler_mode", "warn"),
            rating=norms.get("rating", "G"),
            banter_style=norms.get("banter_style", "medium"),
            shipping_sensitivity=profile.shipping_sensitivity,
        )


@dataclass
class NowCard:
    title: str
    summary: str
    intensity: float
    trend_topics: List[str]


@dataclass
class LoreCard:
    headline: str
    references: List[str]
    freshness_days: float


@dataclass
class CommunityCards:
    nowcard: NowCard
    lorecard: LoreCard
    spoiler_gate: str  # "ok" | "warn" | "block"


@dataclass
class CommunityReply:
    text: str
    cards: CommunityCards
    metadata: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Profile / state structures
# ---------------------------------------------------------------------------


@dataclass
class CommunityProfile:
    name: str
    norms: Dict[str, str]
    fandom_mix: Dict[str, float]
    meme_ttl_days: int
    shipping_sensitivity: float

    @staticmethod
    def default() -> "CommunityProfile":
        return CommunityProfile(
            name="default_otaku_club",
            norms={
                "spoiler_mode": "warn",
                "rating": "G",
                "banter_style": "medium",
            },
            fandom_mix={"mecha": 0.25, "isekai": 0.25, "vtuber": 0.25, "slice_of_life": 0.25},
            meme_ttl_days=14,
            shipping_sensitivity=0.3,
        )


@dataclass
class SpeakerState:
    speaker_id: str
    culture_profile: Dict[str, float]
    stance: Dict[str, float]
    safety_pref: float
    phase: float = 0.0  # [0, 2π)
    energy: float = 0.0
    last_seen: Optional[datetime] = None

    def update_energy(self, energy: float, timestamp: datetime) -> None:
        self.energy = 0.7 * self.energy + 0.3 * energy
        self.phase = (self.phase + self.energy * 0.4) % (2 * math.pi)
        self.last_seen = timestamp


@dataclass
class ThreadNode:
    topic: str
    mentions: int = 0
    last_seen: Optional[datetime] = None
    speakers: Dict[str, int] = field(default_factory=dict)

    def touch(self, speaker_id: str, timestamp: datetime) -> None:
        self.mentions += 1
        self.last_seen = timestamp
        self.speakers[speaker_id] = self.speakers.get(speaker_id, 0) + 1


@dataclass
class ThreadEdge:
    speaker_id: str
    topic: str
    relation: str
    timestamp: datetime


@dataclass
class ThreadGraph:
    nodes: Dict[str, ThreadNode] = field(default_factory=dict)
    edges: List[ThreadEdge] = field(default_factory=list)

    def touch_topic(self, topic: str, speaker_id: str, relation: str, timestamp: datetime) -> None:
        node = self.nodes.setdefault(topic, ThreadNode(topic=topic))
        node.touch(speaker_id, timestamp)
        self.edges.append(ThreadEdge(speaker_id=speaker_id, topic=topic, relation=relation, timestamp=timestamp))

    def top_topics(self, limit: int = 3, since: Optional[datetime] = None) -> List[str]:
        if not self.nodes:
            return []
        entries = []
        for node in self.nodes.values():
            if since and node.last_seen and node.last_seen < since:
                continue
            entries.append((node.mentions, node.last_seen or datetime.min, node.topic))
        entries.sort(key=lambda item: (-item[0], -item[1].timestamp()))
        return [topic for _, _, topic in entries[:limit]]


@dataclass
class MemeItem:
    count: int
    first_seen: datetime
    last_seen: datetime


@dataclass
class MemeTracker:
    items: Dict[str, MemeItem] = field(default_factory=dict)

    def update(self, tokens: Iterable[str], timestamp: datetime, ttl_days: int) -> None:
        for token in tokens:
            if token in self.items:
                item = self.items[token]
                item.count += 1
                item.last_seen = timestamp
            else:
                self.items[token] = MemeItem(count=1, first_seen=timestamp, last_seen=timestamp)

        # TTL pruning
        ttl = timedelta(days=ttl_days)
        for key in list(self.items.keys()):
            if timestamp - self.items[key].last_seen > ttl:
                del self.items[key]

    def trending(self, limit: int = 3) -> List[str]:
        if not self.items:
            return []
        ranked = sorted(
            self.items.items(),
            key=lambda item: (-item[1].count, -item[1].last_seen.timestamp()),
        )
        return [token for token, _ in ranked[:limit]]


@dataclass
class SharedCanon:
    glossary: Dict[str, str] = field(default_factory=dict)
    facts: Dict[str, str] = field(default_factory=dict)

    def lookup(self, topic: Optional[str]) -> List[str]:
        if topic and topic in self.glossary:
            return [self.glossary[topic]]
        if topic and topic in self.facts:
            return [self.facts[topic]]
        # Fallback: top 2 glossary entries
        return list(self.glossary.values())[:2]


# ---------------------------------------------------------------------------
# Community orchestrator
# ---------------------------------------------------------------------------


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9一-龯ぁ-んァ-ヶー]+")
SPOILER_KEYWORDS = {"ネタバレ", "spoiler", "重大展開", "最終回"}


DEFAULT_TEMPLATE = {
    "greeting": {
        "high": "{speaker}さん、今日もテンション高めでいきましょう！",
        "low": "{speaker}さん、落ち着いて一緒に考えてみましょうね。",
        "neutral": "{speaker}さん、ちょうどいいリズムで話せていますよ。",
    },
    "topic": "今の話題は「{topic}」。",
    "no_topic": "",
    "advice": "設定メモ：{reference}",
    "no_advice": "",
    "spoiler": "重要な展開に触れるときは合図をお願いしますね。",
}


class CommunityOrchestrator:
    """Stateful coordinator for multi-speaker diary / chat sessions."""

    def __init__(
        self,
        profile: Optional[CommunityProfile] = None,
        shared_canon: Optional[SharedCanon] = None,
        window_size: int = 64,
        template_path: Optional[str] = None,
    ) -> None:
        self.profile = profile or CommunityProfile.default()
        self.policy = CommunityPolicy.from_profile(self.profile)
        self.shared_canon = shared_canon or SharedCanon()
        self.thread_graph = ThreadGraph()
        self.meme_tracker = MemeTracker()
        self.speakers: Dict[str, SpeakerState] = {}
        self.recent_turns: Deque[Turn] = deque(maxlen=window_size)
        self.group_vector = np.zeros(len(AXES), dtype=float)
        self.group_order_param = 0.0
        self.templates = self._load_templates(template_path)

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#

    def step(self, turn: Turn, policies: Optional[CommunityPolicy] = None) -> CommunityReply:
        """Ingest a turn and produce a reply suggestion plus cards."""
        timestamp = turn.resolved_timestamp()
        active_policy = self._merge_policy(policies)
        speaker = self._ensure_speaker(turn.speaker_id)

        tokens = list(self._tokenise(turn.text))
        topic_guess = tokens[0].lower() if tokens else None

        energy = self._estimate_energy(turn, tokens)
        speaker.update_energy(energy, timestamp)

        self._update_thread(topic_guess, turn.speaker_id, timestamp)
        self.meme_tracker.update(tokens, timestamp, ttl_days=self.profile.meme_ttl_days)
        self.recent_turns.append(turn)

        # Update group dynamics
        self._update_group_vector(turn.text, energy)
        self._update_group_order()

        now_card = self._build_now_card(topic_guess, active_policy)
        lore_card = self._build_lore_card(topic_guess, timestamp)
        spoiler_gate = self._evaluate_spoiler(turn.text, active_policy)
        reply_text = self._compose_reply(turn, speaker, now_card, lore_card, active_policy)

        metadata = {
            "energy": round(energy, 3),
            "group_order": round(self.group_order_param, 3),
            "topic": topic_guess or "",
        }
        cards = CommunityCards(nowcard=now_card, lorecard=lore_card, spoiler_gate=spoiler_gate)
        return CommunityReply(text=reply_text, cards=cards, metadata=metadata)

    def observe(self, turn: Turn, policies: Optional[CommunityPolicy] = None) -> None:
        """Update state without caring about the textual reply."""
        _ = self.step(turn, policies=policies)

    def summary(self) -> Dict[str, object]:
        """Snapshot used for dashboards or diary exports."""
        top_topics = self.thread_graph.top_topics(limit=3)
        trending_memes = self.meme_tracker.trending(limit=3)
        return {
            "profile": self.profile.name,
            "speakers": {
                speaker_id: {
                    "energy": state.energy,
                    "phase": state.phase,
                    "last_seen": state.last_seen.isoformat() if state.last_seen else None,
                }
                for speaker_id, state in self.speakers.items()
            },
            "group_order": self.group_order_param,
            "top_topics": top_topics,
            "trending_memes": trending_memes,
        }

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#

    def _merge_policy(self, override: Optional[CommunityPolicy]) -> CommunityPolicy:
        if not override:
            return self.policy
        merged = CommunityPolicy.from_profile(self.profile)
        merged.spoiler_mode = override.spoiler_mode or merged.spoiler_mode
        merged.rating = override.rating or merged.rating
        merged.banter_style = override.banter_style or merged.banter_style
        merged.shipping_sensitivity = override.shipping_sensitivity
        return merged

    def _ensure_speaker(self, speaker_id: str) -> SpeakerState:
        if speaker_id not in self.speakers:
            self.speakers[speaker_id] = SpeakerState(
                speaker_id=speaker_id,
                culture_profile=dict(self.profile.fandom_mix),
                stance={},
                safety_pref=0.8,
            )
        return self.speakers[speaker_id]

    def _tokenise(self, text: str) -> Iterable[str]:
        for match in TOKEN_PATTERN.finditer(text.lower()):
            token = match.group(0)
            if len(token) < 2:
                continue
            yield token

    def _estimate_energy(self, turn: Turn, tokens: Sequence[str]) -> float:
        if turn.prosody:
            tempo = float(turn.prosody.get("tempo", 120.0))
            energy = float(turn.prosody.get("energy", 0.5))
            return min(1.0, (tempo / 240.0) * 0.6 + energy * 0.4)
        # Fallback: use emotion extractor as a heuristic
        vec = extract_emotion(turn.text)
        magnitude = float(np.linalg.norm(vec))
        if tokens:
            magnitude += min(0.5, len(tokens) / 50.0)
        return float(np.clip(magnitude / 3.5, 0.0, 1.0))

    def _update_thread(self, topic: Optional[str], speaker_id: str, timestamp: datetime) -> None:
        if not topic:
            return
        relation = "reply_to" if self.recent_turns else "start"
        self.thread_graph.touch_topic(topic, speaker_id, relation, timestamp)

    def _update_group_vector(self, text: str, energy: float) -> None:
        emotion_vec = extract_emotion(text)
        self.group_vector = 0.8 * self.group_vector + 0.2 * (energy * emotion_vec)

    def _update_group_order(self) -> None:
        if not self.speakers:
            self.group_order_param = 0.0
            return
        complex_sum = sum(
            complex(math.cos(state.phase), math.sin(state.phase)) for state in self.speakers.values()
        )
        self.group_order_param = abs(complex_sum) / max(1, len(self.speakers))

    def _build_now_card(self, primary_topic: Optional[str], policy: CommunityPolicy) -> NowCard:
        window_start = datetime.utcnow() - timedelta(minutes=20)
        trend_topics = self.thread_graph.top_topics(limit=3, since=window_start)
        if primary_topic and primary_topic not in trend_topics:
            trend_topics = [primary_topic] + trend_topics[:-1]
        intensity = float(np.clip(self.group_order_param * 1.5, 0.0, 1.0))
        banter_multiplier = {"low": 0.8, "medium": 1.0, "high": 1.2}.get(policy.banter_style, 1.0)
        summary = " / ".join(trend_topics) if trend_topics else "フリートーク"
        return NowCard(
            title="Current Vibe",
            summary=summary,
            intensity=float(np.clip(intensity * banter_multiplier, 0.0, 1.0)),
            trend_topics=trend_topics,
        )

    def _build_lore_card(self, primary_topic: Optional[str], timestamp: datetime) -> LoreCard:
        references = self.shared_canon.lookup(primary_topic)
        if not references and self.thread_graph.nodes:
            fallback_topic = max(self.thread_graph.nodes.values(), key=lambda node: node.mentions).topic
            references = self.shared_canon.lookup(fallback_topic)
        freshness_days = 0.0
        if primary_topic and primary_topic in self.thread_graph.nodes:
            node = self.thread_graph.nodes[primary_topic]
            if node.last_seen:
                freshness_days = (timestamp - node.last_seen).total_seconds() / 86400.0
        return LoreCard(
            headline=primary_topic or "集いの設定メモ",
            references=references,
            freshness_days=freshness_days,
        )

    def _template_for_profile(self) -> Dict[str, str]:
        key = self.profile.norms.get("reply_template", "default")
        if key in self.templates:
            return self.templates[key]
        return self.templates.get("default", DEFAULT_TEMPLATE)

    def _load_templates(self, template_path: Optional[str]) -> Dict[str, Dict[str, str]]:
        defaults = {"default": DEFAULT_TEMPLATE}
        path_str = template_path or os.getenv("COMMUNITY_TEMPLATE_FILE", "resources/community_reply_templates.yaml")
        path = Path(path_str)
        if not path.exists():
            return defaults
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return defaults
        if isinstance(raw, dict) and "presets" in raw:
            raw = raw["presets"]
        if not isinstance(raw, dict):
            return defaults
        templates: Dict[str, Dict[str, str]] = {}
        for name, payload in raw.items():
            if isinstance(payload, dict):
                templates[name] = {**DEFAULT_TEMPLATE, **payload}
        if "default" not in templates:
            templates["default"] = DEFAULT_TEMPLATE
        return templates

    def _evaluate_spoiler(self, text: str, policy: CommunityPolicy) -> str:
        if policy.spoiler_mode == "free":
            return "ok"
        contains_spoiler = any(keyword.lower() in text.lower() for keyword in SPOILER_KEYWORDS)
        if not contains_spoiler:
            return "ok"
        return "warn" if policy.spoiler_mode == "warn" else "block"

    def _compose_reply(
        self,
        turn: Turn,
        speaker: SpeakerState,
        now_card: NowCard,
        lore_card: LoreCard,
        policy: CommunityPolicy,
    ) -> str:
        template_set = self._template_for_profile()
        if speaker.energy > 0.6 and policy.banter_style == "high":
            greeting_template = template_set["greeting"].get("high", DEFAULT_TEMPLATE["greeting"]["high"])
        elif speaker.energy < 0.3:
            greeting_template = template_set["greeting"].get("low", DEFAULT_TEMPLATE["greeting"]["low"])
        else:
            greeting_template = template_set["greeting"].get("neutral", DEFAULT_TEMPLATE["greeting"]["neutral"])
        greeting = greeting_template.format(speaker=turn.speaker_id)

        if now_card.trend_topics:
            topic_template = template_set.get("topic", DEFAULT_TEMPLATE["topic"])
            topic_line = topic_template.format(topic=now_card.trend_topics[0])
        else:
            topic_line = template_set.get("no_topic", DEFAULT_TEMPLATE["no_topic"])

        if lore_card.references:
            advice_template = template_set.get("advice", DEFAULT_TEMPLATE["advice"])
            advice_line = advice_template.format(reference=lore_card.references[0][:80])
        else:
            advice_line = template_set.get("no_advice", DEFAULT_TEMPLATE["no_advice"])

        spoiler_line = ""
        if now_card.trend_topics and self._evaluate_spoiler(turn.text, policy) != "ok":
            spoiler_line = template_set.get("spoiler", DEFAULT_TEMPLATE["spoiler"])

        parts = [greeting, topic_line, advice_line, spoiler_line]
        return " ".join(fragment for fragment in parts if fragment)


__all__ = [
    "CommunityOrchestrator",
    "CommunityReply",
    "CommunityCards",
    "CommunityPolicy",
    "CommunityProfile",
    "SharedCanon",
    "SpeakerState",
    "ThreadGraph",
    "Turn",
]
