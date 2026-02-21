#!/usr/bin/env python
"""EQNet Heart OS本番体験用のセッションランナー。"""
from __future__ import annotations

from pathlib import Path as _PathShim
import sys

ROOT = _PathShim(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from eqnet.hub.multi_tenant import EQNetHubManager
from eqnet.logs.moment_log import MomentLogEntry, MomentLogWriter
from eqnet.persona.loader import PersonaConfig, load_persona_from_dir
from eqnet.qualia_model import FutureReplayConfig, ReplayMode, simulate_future
from eqnet.runtime.policy import PolicyPrior, apply_imagery_update
from eqnet.heart_os.session_config import HeartOSSessionConfig
from eqnet.heart_os.session_state import SessionState
from eqnet.heart_os.memory_hints import (
    MemoryHint,
    build_memory_hints,
    build_kg_facts_from_rag_docs,
    build_place_memory_hints,
)
from eqnet.memory import MemoryStore, TerrainState
from eqnet.memory.state_vector import TemporalStateVector
from eqnet.memory.moment_knn import MomentKNNIndex
from runtime.config import load_runtime_cfg
from rag.indexer import IndexedDocument, RagIndex
from rag.retriever import RagRetriever, RetrievalHit


def dummy_embed(text: str) -> List[float]:
    """簡易エンコーダ（デモ用）。"""

    length = len(text.encode("utf-8")) or 1
    base = (length % 13) / 12.0
    jitter = ((length % 5) - 2) / 10.0
    return [base, 0.15 + 0.1 * jitter, min(1.0, length / 256.0), 0.35 + 0.02 * (length % 3)]


def resolve_embed_fn(spec: str) -> Callable[[str], List[float]]:
    if spec == "dummy":
        return dummy_embed
    if ":" not in spec:
        raise ValueError("--embed-fn は dummy もしくは module:function 形式で指定してください")
    module_name, func_name = spec.split(":", 1)
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_clock(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def _float_dict(payload: Optional[Dict[str, Any]]) -> Dict[str, float]:
    data = payload or {}
    out: Dict[str, float] = {}
    for key, val in data.items():
        try:
            out[key] = float(val)
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class ScenarioEvent:
    id: str
    clock: str
    stage: str
    talk_mode: str
    awareness_stage: int
    user_text: str
    experience: Dict[str, str]
    share_with: List[str]
    rag: Optional[Dict[str, Any]]
    emotion: Dict[str, float]
    mood: Dict[str, float]
    culture: Dict[str, Any]
    gate_context: Dict[str, Any]
    comments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ViewerSessionInfo:
    viewer_id: str
    viewer_name: str
    first_seen: datetime
    last_comment: datetime
    comment_count: int = 0
    history: List[str] = field(default_factory=list)
    last_emotion: Optional[str] = None
    last_addressed: Optional[datetime] = None


@dataclass
class Scenario:
    session_id: str
    persona_id: str
    title: str
    user_name: str
    timezone: str
    events: List[ScenarioEvent]
    rag_documents: List[Dict[str, Any]] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario yaml は mapping 形式である必要があります")
    required = ["session_id", "persona_id", "title", "user_name", "events"]
    for key in required:
        if key not in data:
            raise ValueError(f"scenario yaml に {key} がありません")
    events_raw = data.get("events") or []
    events: List[ScenarioEvent] = []
    for idx, entry in enumerate(events_raw):
        if not isinstance(entry, dict):
            continue
        event = ScenarioEvent(
            id=str(entry.get("id") or f"event_{idx+1}"),
            clock=str(entry.get("clock") or "00:00"),
            stage=str(entry.get("stage") or ""),
            talk_mode=str(entry.get("talk_mode") or "talk"),
            awareness_stage=int(entry.get("awareness_stage") or 0),
            user_text=str(entry.get("user_text") or ""),
            experience=dict(entry.get("experience") or {}),
            share_with=list(entry.get("share_with") or entry.get("rag", {}).get("persona_targets", []) or []),
            rag=entry.get("rag"),
            emotion=_float_dict(entry.get("emotion")),
            mood=_float_dict(entry.get("mood")),
            culture=dict(entry.get("culture") or {}),
            gate_context=dict(entry.get("gate_context") or {}),
            comments=list(entry.get("comments") or []),
        )
        events.append(event)
    return Scenario(
        session_id=str(data["session_id"]),
        persona_id=str(data["persona_id"]),
        title=str(data.get("title") or data["session_id"]),
        user_name=str(data.get("user_name") or "user"),
        timezone=str(data.get("timezone") or "UTC"),
        events=events,
        rag_documents=list(data.get("rag_documents") or []),
        overrides=dict(data.get("overrides") or {}),
    )


class GraphMemoryRAG:
    """knowledge_graph.parquet を再現する軽量 in-memory RAG。"""

    def __init__(self, documents: Iterable[Dict[str, Any]], embed_fn: Callable[[str], Sequence[float]]) -> None:
        docs = list(documents)
        self.embed_fn = embed_fn
        self.runtime_cfg = load_runtime_cfg()
        if not docs:
            self.index: Optional[RagIndex] = None
            self.retriever: Optional[RagRetriever] = None
            return
        index = RagIndex(normalize=True)
        for doc in docs:
            text = str(doc.get("text") or "")
            embedding = torch.tensor(embed_fn(text), dtype=torch.float32)
            metadata = dict(doc.get("metadata") or {})
            metadata.setdefault("topic", doc.get("topic"))
            metadata.setdefault("persona_ids", list(doc.get("persona_ids") or []))
            metadata.setdefault("consent_tiers", list(doc.get("consent_tiers") or []))
            doc_id = str(doc.get("id") or f"doc_{len(index.documents)+1}")
            index.add(
                IndexedDocument(
                    doc_id=doc_id,
                    text=text,
                    embedding=embedding,
                    metadata=metadata,
                )
            )
        index.build()
        self.index = index
        self.retriever = RagRetriever(index, oversample=2)

    def retrieve(
        self,
        query_text: str,
        *,
        topic: Optional[str],
        top_k: int,
        persona_filter: Optional[Sequence[str]] = None,
    ) -> List[RetrievalHit]:
        if not self.retriever:
            return []
        query_vec = torch.tensor(self.embed_fn(query_text), dtype=torch.float32)
        metadata_filter = {"topic": topic} if topic else None
        temporal_state = TemporalStateVector(
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            value_tags={"topic": 1.0} if topic else {},
            open_loops=0.0,
            event_scale=0.0,
        )
        hits = self.retriever.retrieve_with_assoc(
            runtime_cfg=self.runtime_cfg,
            temporal_state=temporal_state,
            query_embedding=query_vec,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        if persona_filter:
            target = set(persona_filter)
            filtered = []
            for hit in hits:
                personas = set(hit.metadata.get("persona_ids", []))
                if target.issubset(personas):
                    filtered.append(hit)
            hits = filtered
        return hits


def load_viewer_stats(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_viewer_stats(path: Path, stats: Dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


class ConsentGate:
    """persona social preferences を解釈して共有可否を判定。"""

    def __init__(self, persona: PersonaConfig, overrides: Optional[Dict[str, Any]] = None) -> None:
        base = dict(persona.raw.get("social_prefs") or {})
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    merged = dict(base.get(key))
                    merged.update(value)
                    base[key] = merged
                else:
                    base[key] = value
        self.trust_circle = {k: list(v) for k, v in (base.get("trust_circle") or {}).items()}
        self.can_share_topics = {k: list(v) for k, v in (base.get("can_share_topics") or {}).items()}

    def _tier_for(self, persona_id: str) -> Optional[str]:
        for tier, members in self.trust_circle.items():
            if persona_id in members:
                return tier
        return None

    def filter_targets(self, topic: Optional[str], targets: Sequence[str]) -> Tuple[List[str], List[str]]:
        if not targets:
            return [], []
        topic_key = topic or "*"
        allowed_spec = self.can_share_topics.get(topic_key) or self.can_share_topics.get("*")
        if not allowed_spec:
            uniq = list(dict.fromkeys(targets))
            return uniq, []
        allowed: List[str] = []
        blocked: List[str] = []
        for persona_id in targets:
            persona_id = str(persona_id)
            tier = self._tier_for(persona_id)
            if persona_id in allowed_spec or (tier and tier in allowed_spec) or "*" in allowed_spec or "any" in allowed_spec:
                allowed.append(persona_id)
            else:
                blocked.append(persona_id)
        return allowed, blocked


class DemoMoment:
    """EQNetHub.log_moment に渡す簡易 moment entry。"""

    def __init__(self, timestamp: datetime, awareness_stage: int, emotion: Dict[str, float], culture: Dict[str, Any]):
        defaults_emo = {"mask": 0.0, "love": 0.0, "stress": 0.0, "heart_rate_norm": 0.0, "breath_ratio_norm": 0.0}
        defaults_emo.update(emotion)
        defaults_culture = {
            "rho": 0.0,
            "politeness": 0.5,
            "intimacy": 0.0,
            "culture_tag_embed": [],
        }
        defaults_culture.update(culture)
        self.timestamp = timestamp
        self.awareness_stage = awareness_stage
        self.emotion = SimpleNamespace(**defaults_emo)
        self.culture = SimpleNamespace(**defaults_culture)


class HeartOSSessionRunner:
    def __init__(
        self,
        scenario: Scenario,
        manager: EQNetHubManager,
        persona: PersonaConfig,
        *,
        session_date: date,
        embed_fn: Callable[[str], Sequence[float]],
        rag: Optional[GraphMemoryRAG],
        moment_writer: MomentLogWriter,
        viewer_stats_path: Path,
    ) -> None:
        self.scenario = scenario
        self.hub = manager.for_user(scenario.persona_id)
        self.persona = persona
        self.session_date = session_date
        self.embed_fn = embed_fn
        self.rag = rag
        self.moment_writer = moment_writer
        self.viewer_stats_path = viewer_stats_path
        self.viewer_stats = load_viewer_stats(viewer_stats_path)
        overrides = scenario.overrides.get("social_prefs") if scenario.overrides else None
        self.consent_gate = ConsentGate(persona, overrides)
        base_mode = (scenario.overrides or {}).get("conversation_mode", "stream")
        if base_mode not in ("stream", "meet"):
            base_mode = "stream"
        base_voice = (scenario.overrides or {}).get("voice_style", "normal")
        if base_voice not in ("normal", "whisper"):
            base_voice = "normal"
        self.session_config = HeartOSSessionConfig(
            session_id=scenario.session_id,
            persona_id=scenario.persona_id,
            conversation_mode=base_mode,
            default_voice_style=base_voice,
            metadata={"title": scenario.title},
        )
        self.session_state = SessionState.from_config(self.session_config)
        self._qualia_history = self.session_state.qualia_history
        self._records: List[Dict[str, Any]] = []
        self._viewer_session_state = self.session_state.viewer_sessions
        self._viewer_topic_counts: Dict[str, Dict[str, int]] = {}
        self._anchor_limit = 2
        self.knn_index = MomentKNNIndex()
        memory_dir = Path("eqnet_data") / "memory"
        self.memory_store = MemoryStore(memory_dir)
        episodes, monuments = self.memory_store.load_all()
        self.terrain_state = TerrainState(episodes=episodes, monuments=monuments)

    def run(self) -> Dict[str, Any]:
        print(f"=== {self.scenario.title} / persona={self.scenario.persona_id} ===")
        for turn_idx, event in enumerate(self.scenario.events, start=1):
            record = self._run_event(turn_idx, event)
            self._records.append(record)
        future_summary = self._finalize()
        self._update_viewer_stats()
        return {
            "events": self._records,
            "state": self.hub.query_state(),
            "future": future_summary,
        }

    def _run_event(self, turn_idx: int, event: ScenarioEvent) -> Dict[str, Any]:
        self.session_state.tick_voice_style()
        timestamp = self._timestamp(event.clock)
        allowed, blocked = self.consent_gate.filter_targets(event.rag.get("topic") if event.rag else None, event.share_with)
        processed_comments = self._process_event_comments(event, timestamp)
        anchor_comments = self._select_anchor_comments(processed_comments)
        rag_hits: List[RetrievalHit] = []
        context_text: Optional[str] = None
        if event.rag and (allowed or not event.rag.get("require_consent", True)):
            rag_hits = (self.rag.retrieve(
                query_text=event.rag.get("query") or event.user_text,
                topic=event.rag.get("topic"),
                top_k=int(event.rag.get("top_k") or 2),
                persona_filter=allowed or None,
            ) if self.rag else [])
        if rag_hits:
            snippets = [hit.text.strip() for hit in rag_hits]
            context_text = " / ".join(snippets)
        elif event.rag and not allowed and event.rag.get("require_consent", True):
            context_text = None
        else:
            context_text = event.experience.get("base") if event.experience else None
        persona_line = self._render_persona_line(event, context_text, blocked)
        persona_line = self._maybe_add_anchor_mentions(persona_line, anchor_comments)

        rag_doc_ids = [hit.doc_id for hit in rag_hits]
        kg_facts = build_kg_facts_from_rag_docs(
            rag_doc_ids,
            context_text or "",
            self.session_state.latest_emotion_tag,
        )
        embedding = self.embed_fn(event.user_text)
        neighbors = self.knn_index.search(embedding, k=3)
        moment_topic = (event.rag.get("topic") if event.rag else event.culture.get("topic")) if event.culture else None
        memory_hints = build_memory_hints(
            mode=self.session_state.conversation_mode,
            emotion_tag=self.session_state.latest_emotion_tag,
            kg_facts=kg_facts,
            neighbors=neighbors,
            moment_topic=moment_topic,
            voice_style=self.session_state.voice_style,
        )
        place_id = None
        if event.culture and isinstance(event.culture, dict):
            place_id = event.culture.get("place_id")
        if not place_id and event.rag:
            place_id = event.rag.get("place_id")
        if place_id:
            l3_hints = build_place_memory_hints(place_id, getattr(self, "terrain_state", None))
            if l3_hints:
                memory_hints.extend(l3_hints)
        meta_hints = self._build_meta_hints(blocked)
        if meta_hints:
            memory_hints.extend(meta_hints)
        prompt_context = self._build_prompt(event, persona_line, memory_hints)

        moment = DemoMoment(timestamp, event.awareness_stage, event.emotion, event.culture)
        self.hub.log_moment(moment, event.user_text)
        latest_q = self.hub.latest_qualia_state()
        if latest_q is not None:
            emotion_tag = self._infer_emotion_tag(event)
            self.session_state.update_qualia(latest_q, emotion_tag=emotion_tag)

        self._log_moment(
            turn_idx,
            event,
            timestamp,
            persona_line,
            allowed,
            blocked,
            rag_hits,
            anchor_comments,
            memory_hints,
        )
        self.knn_index.add(
            embedding=embedding,
            moment_id=event.id,
            topic=(event.rag.get("topic") if event.rag else None) or event.culture.get("topic"),
            summary=event.user_text,
            emotion_tag=self.session_state.latest_emotion_tag,
        )

        print(f"[{event.clock}] {event.id} ({event.stage})")
        print(f"  ユーザー : {event.user_text}")
        if rag_hits:
            print(f"  RAG文脈 : {context_text}")
        if blocked:
            print(f"  共有ブロック : {', '.join(blocked)}")
        print(f"  キャラ : {persona_line}")

        return {
            "id": event.id,
            "clock": event.clock,
            "stage": event.stage,
            "user_text": event.user_text,
            "persona_line": persona_line,
            "context": context_text,
            "rag_docs": [hit.doc_id for hit in rag_hits],
            "blocked": blocked,
            "memory_hints": [hint.to_payload() for hint in memory_hints],
            "prompt": prompt_context,
        }

    def _render_persona_line(
        self,
        event: ScenarioEvent,
        context_text: Optional[str],
        blocked: Sequence[str],
    ) -> str:
        exp = event.experience or {}
        context_value = (context_text or "").strip()
        if context_value and exp.get("share_allowed"):
            line = exp["share_allowed"].replace("{context}", context_value)
        elif blocked and exp.get("share_blocked"):
            line = exp["share_blocked"].replace("{blocked}", ", ".join(blocked))
        else:
            line = exp.get("base", "")
            if context_value:
                line = line.replace("{context}", context_value)
            if blocked:
                line = line.replace("{blocked}", ", ".join(blocked))
        return self._apply_persona_style(event, line).strip()

    def _apply_persona_style(self, event: ScenarioEvent, line: str) -> str:
        styled = self._maybe_add_filler(event, line)
        return styled

    def _maybe_add_filler(self, event: ScenarioEvent, line: str) -> str:
        speech = (self.persona.speech if self.persona and self.persona.speech else {})
        fillers = speech.get("fillers") or {}
        rules = speech.get("filler_rules") or {}
        rule = self._select_filler_rule(rules, event)
        if not rule:
            return line
        try:
            prob = float(rule.get("prob", 0.5))
        except (TypeError, ValueError):
            prob = 0.5
        if prob <= 0.0 or random.random() > prob:
            return line
        filler_type = str(rule.get("type", "default"))
        options = fillers.get(filler_type) or fillers.get("default") or []
        if not options:
            return line
        filler = str(random.choice(options))
        position = str(rule.get("position", "prefix"))
        if position == "suffix":
            return f"{line}{'' if line.endswith(('。', '！', '!', '？', '?')) else ''} {filler}"
        return f"{filler} {line}"

    def _select_filler_rule(self, rules: Dict[str, Any], event: ScenarioEvent) -> Optional[Dict[str, Any]]:
        if not isinstance(rules, dict):
            return None
        keys: List[Optional[str]] = [getattr(event, "stage", None), getattr(event, "talk_mode", None)]
        topic = None
        if event.rag and isinstance(event.rag, dict):
            topic = event.rag.get("topic")
        keys.append(topic)
        keys.append("default")
        for key in keys:
            if not key:
                continue
            rule = rules.get(str(key))
            if isinstance(rule, dict):
                return rule
        return None

    def _process_event_comments(self, event: ScenarioEvent, timestamp: datetime) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for payload in event.comments or []:
            viewer_id = str(payload.get("viewer_id") or "").strip()
            if not viewer_id:
                continue
            viewer_name = str(payload.get("viewer_name") or viewer_id)
            text = str(payload.get("text") or "")
            info = self._update_viewer_session_entry(viewer_id, viewer_name, text, payload, event, timestamp)
            status = self._classify_viewer_status(viewer_id, info)
            processed.append(
                {
                    "viewer_id": viewer_id,
                    "viewer_name": viewer_name,
                    "text": text,
                    "status": status,
                    "past_snippet": self._recent_history_snippet(info, text),
                    "topic": payload.get("topic") or (event.rag or {}).get("topic"),
                    "highlight": bool(payload.get("highlight")),
                }
            )
        return processed

    def _update_viewer_session_entry(
        self,
        viewer_id: str,
        viewer_name: str,
        text: str,
        payload: Dict[str, Any],
        event: ScenarioEvent,
        timestamp: datetime,
    ) -> ViewerSessionInfo:
        info = self._viewer_session_state.get(viewer_id)
        if info is None:
            info = ViewerSessionInfo(
                viewer_id=viewer_id,
                viewer_name=viewer_name,
                first_seen=timestamp,
                last_comment=timestamp,
            )
            self._viewer_session_state[viewer_id] = info
        info.viewer_name = viewer_name
        info.last_comment = timestamp
        info.comment_count += 1
        snippet = text.strip()
        if snippet:
            info.history.append(snippet)
            info.history = info.history[-5:]
        emotion = payload.get("emotion")
        if emotion:
            info.last_emotion = str(emotion)
        topic = payload.get("topic") or (event.rag or {}).get("topic")
        if topic:
            topics = self._viewer_topic_counts.setdefault(viewer_id, {})
            topics[topic] = topics.get(topic, 0) + 1
        return info

    def _select_anchor_comments(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        preferred = [c for c in candidates if c.get("highlight")]
        ordered = preferred or candidates
        anchors: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for comment in ordered:
            viewer_id = comment.get("viewer_id")
            if viewer_id in seen:
                continue
            anchors.append(comment)
            seen.add(viewer_id)
            if len(anchors) >= self._anchor_limit:
                break
        return anchors

    def _classify_viewer_status(self, viewer_id: str, info: ViewerSessionInfo) -> str:
        stats = self.viewer_stats.get(viewer_id)
        if not stats:
            return "first_time" if info.comment_count <= 1 else "returning"
        try:
            last_seen = date.fromisoformat(stats.get("last_seen", ""))
            delta = (self.session_date - last_seen).days
        except ValueError:
            delta = 0
        if stats.get("visit_count", 0) <= 1 and info.comment_count <= 1:
            return "first_time"
        if delta >= 7:
            return "comeback"
        if int(stats.get("consecutive_days", 1)) >= 3:
            return "streak"
        return "returning"

    def _recent_history_snippet(self, info: ViewerSessionInfo, current_text: str) -> Optional[str]:
        for previous in reversed(info.history[:-1]):
            if previous != current_text.strip():
                return previous
        return None

    def _maybe_add_anchor_mentions(self, base_line: str, anchors: Sequence[Dict[str, Any]]) -> str:
        if not anchors:
            return base_line
        parts = [base_line]
        for anchor in anchors:
            mention = self._build_anchor_line(anchor)
            if mention:
                parts.append(mention)
        return " ".join(part for part in parts if part.strip())

    def _build_anchor_line(self, anchor: Dict[str, Any]) -> str:
        viewer_name = anchor.get("viewer_name") or anchor.get("viewer_id", "")
        greeting = self._lookup_greeting(anchor.get("status", ""), viewer_name)
        snippet = anchor.get("text", "").strip()
        if len(snippet) > 40:
            snippet = f"{snippet[:37]}..."
        pieces: List[str] = []
        if greeting:
            pieces.append(greeting)
        elif viewer_name:
            pieces.append(f"{viewer_name}、")
        past = anchor.get("past_snippet")
        if past:
            pieces.append(f"前に『{past}』って言ってくれてたよね。")
        if snippet:
            pieces.append(f"今の『{snippet}』も心に響いたよ。")
        return " ".join(pieces).strip()

    def _lookup_greeting(self, status: str, viewer_name: str) -> Optional[str]:
        speech = self.persona.speech if self.persona and self.persona.speech else {}
        greetings = speech.get("greetings") or {}
        options = greetings.get(status) or greetings.get("default") or []
        if not options:
            return None
        template = str(random.choice(options))
        return template.replace("{viewer_name}", viewer_name)

    def _update_viewer_stats(self) -> None:
        if not self.viewer_stats_path:
            return
        stats = dict(self.viewer_stats)
        for viewer_id, info in self._viewer_session_state.items():
            entry = dict(stats.get(viewer_id) or {})
            entry["visit_count"] = int(entry.get("visit_count", 0)) + 1
            prev_last = entry.get("last_seen")
            consecutive = int(entry.get("consecutive_days", 0))
            if prev_last:
                try:
                    prev_date = date.fromisoformat(prev_last)
                    delta = (self.session_date - prev_date).days
                except ValueError:
                    delta = 0
                consecutive = consecutive + 1 if delta == 1 else 1
            else:
                consecutive = 1
            entry["consecutive_days"] = consecutive
            entry["last_seen"] = self.session_date.isoformat()
            if info.last_emotion:
                entry["last_emotion"] = info.last_emotion
            topics = self._viewer_topic_counts.get(viewer_id)
            if topics:
                fav = sorted(topics.items(), key=lambda kv: (-kv[1], kv[0]))
                entry["fav_topics"] = [name for name, _ in fav[:3]]
            stats[viewer_id] = entry
        save_viewer_stats(self.viewer_stats_path, stats)
        self.viewer_stats = stats

    def _log_moment(
        self,
        turn_idx: int,
        event: ScenarioEvent,
        timestamp: datetime,
        persona_line: str,
        allowed: Sequence[str],
        blocked: Sequence[str],
        rag_hits: Sequence[RetrievalHit],
        anchor_comments: Sequence[Dict[str, Any]],
        memory_hints: Sequence[MemoryHint],
    ) -> None:
        if not self.moment_writer.enabled:
            return
        mood = event.mood or {
            "valence": event.emotion.get("love", 0.0) - event.emotion.get("stress", 0.0),
            "arousal": event.emotion.get("stress", 0.0),
        }
        metrics = {
            "rho": float(event.culture.get("rho", 0.0)),
            "politeness": float(event.culture.get("politeness", 0.0)),
            "intimacy": float(event.culture.get("intimacy", 0.0)),
        }
        gate_context = dict(event.gate_context)
        gate_context.update(
            {
                "consent_topic": event.rag.get("topic") if event.rag else None,
                "allowed_targets": list(allowed),
                "blocked_targets": list(blocked),
            }
        )
        qualia_vec = None
        if self.session_state.qualia_state is not None:
            qualia_vec = self.session_state.qualia_state.qualia_vec.astype(float).tolist()
        entry = MomentLogEntry(
            ts=timestamp.timestamp(),
            turn_id=turn_idx,
            session_id=self.scenario.session_id,
            talk_mode=event.talk_mode,
            mood=mood,
            metrics=metrics,
            gate_context=gate_context,
            prospective=None,
            heart_rate=event.emotion.get("heart_rate_norm"),
            heart_phase=None,
            culture_tag=event.culture.get("topic"),
            place_id=event.culture.get("place_id"),
            partner_id=event.culture.get("partner_id"),
            object_id=None,
            object_role=None,
            activity_tag=event.culture.get("activity_tag"),
            fast_ack=None,
            persona_meta={
                "share_with": event.share_with,
                "allowed": list(allowed),
                "blocked": list(blocked),
                "viewer_mentions": [comment.get("viewer_id") for comment in anchor_comments if comment.get("viewer_id")],
                "memory_hints": [hint.to_payload() for hint in memory_hints],
            },
            user_text=event.user_text,
            llm_text=persona_line,
            response_meta={"context_docs": [hit.doc_id for hit in rag_hits]},
            behavior_mod=None,
            emotion_tag=self.session_state.latest_emotion_tag,
            qualia_vec=qualia_vec,
        )
        self.moment_writer.write(entry)

    def _build_meta_hints(self, blocked: Sequence[str]) -> List[MemoryHint]:
        hints: List[MemoryHint] = []
        if blocked:
            hints.append(
                MemoryHint(
                    type="blocked_share",
                    source="meta",
                    intent="ack_only",
                    confidence=0.7,
                    emotion_tag=self.session_state.latest_emotion_tag or "love",
                    text=(
                        "ことね本人にはまだすべてをそのまま届けられていないけれど、"
                        "一緒に振り返りたいという願いはちゃんとここにある。"
                    ),
                )
            )
        return hints

    def _build_prompt(
        self,
        event: ScenarioEvent,
        persona_line: str,
        memory_hints: Sequence[MemoryHint],
    ) -> Dict[str, Any]:
        mode = self.session_state.conversation_mode
        voice = self.session_state.voice_style
        emotion_tag = self.session_state.latest_emotion_tag or "neutral"
        system_parts = [
            self.persona.system_prompt if hasattr(self.persona, "system_prompt") else "",
            f"conversation_mode: {mode}",
            f"voice_style: {voice}",
            f"current_emotion: {emotion_tag}",
            "Memory hints are supportive context. Prioritise the user's latest words.",
            "intent=='expand' → elaborate gently; intent=='ack_only' → mirror softly.",
        ]
        system_prompt = "\n".join(part for part in system_parts if part)
        return {
            "system": system_prompt,
            "live_context": {
                "user_text": event.user_text,
                "persona_candidate": persona_line,
                "stage": event.stage,
            },
            "memory_hints": [hint.to_payload() for hint in memory_hints],
        }

    @staticmethod
    def _infer_emotion_tag(event: ScenarioEvent) -> str:
        emotion = event.emotion or {}
        love = float(emotion.get("love", 0.0))
        stress = float(emotion.get("stress", 0.0))
        mask = float(emotion.get("mask", 0.0))
        if love >= stress and love > 0.45:
            return "joy"
        if stress > love and stress > 0.5:
            return "fear"
        if mask > 0.6:
            return "sadness"
        return "calm"

    def _timestamp(self, clock: str) -> datetime:
        hhmm = parse_clock(clock)
        return datetime.combine(self.session_date, hhmm)

    def _finalize(self) -> Dict[str, Any]:
        self.hub.run_nightly(self.session_date)
        if len(self._qualia_history) < 2:
            return {}
        window = min(6, len(self._qualia_history) - 1)
        cfg = FutureReplayConfig(steps=4, window=window)
        predictive = simulate_future(self._qualia_history, ReplayMode.PREDICTIVE, cfg)
        last_vec = self._qualia_history[-1].qualia_vec
        intention = np.zeros_like(last_vec)
        intention[:2] = 0.05
        imagery = simulate_future(
            self._qualia_history,
            ReplayMode.IMAGERY,
            cfg,
            intention_vec=intention,
        )
        avg_potential = float(np.mean([float(np.linalg.norm(vec)) for vec in predictive]))
        state = self.hub.query_state()
        li = state.get("life_indicator") or {}
        li_score = float(li.get("qualia") or 0.5)
        current_prior = self.hub.latest_policy_prior()
        updated_prior = apply_imagery_update(current_prior, imagery, avg_potential, li_score)
        return {
            "predictive_mean_norm": avg_potential,
            "imagery_preview": imagery[1][:4].tolist() if len(imagery) > 1 else [],
            "policy_prior_suggestion": updated_prior.__dict__,
        }


def build_moment_writer(path: Optional[Path]) -> MomentLogWriter:
    if not path:
        return MomentLogWriter(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    return MomentLogWriter(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="EQNet Heart OS フルセッションデモ")
    parser.add_argument("--scenario", type=Path, default=Path("examples/scenarios/heart_os_full_session.yaml"))
    parser.add_argument("--base-dir", type=Path, default=Path("eqnet_data"))
    parser.add_argument("--persona-dir", type=Path, default=Path("personas"))
    parser.add_argument("--session-date", default=date.today().isoformat())
    parser.add_argument("--moment-log", type=Path, default=None)
    parser.add_argument("--embed-fn", default="dummy")
    parser.add_argument("--json", action="store_true", help="結果を JSON で併せて表示")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    session_date = parse_iso_date(args.session_date)
    embed_fn = resolve_embed_fn(args.embed_fn)

    args.base_dir.mkdir(parents=True, exist_ok=True)
    viewer_stats_path = args.base_dir / "viewer_stats.json"

    manager = EQNetHubManager(args.base_dir, embed_text_fn=embed_fn, persona_dir=args.persona_dir)
    persona = load_persona_from_dir(args.persona_dir, scenario.persona_id)
    if persona is None:
        raise RuntimeError(f"persona {scenario.persona_id} が {args.persona_dir} に見つかりません")

    rag = GraphMemoryRAG(scenario.rag_documents, embed_fn)
    moment_log_path = args.moment_log or Path("logs") / f"heart_session_{scenario.session_id}.jsonl"
    writer = build_moment_writer(moment_log_path)

    runner = HeartOSSessionRunner(
        scenario,
        manager,
        persona,
        session_date=session_date,
        embed_fn=embed_fn,
        rag=rag,
        moment_writer=writer,
        viewer_stats_path=viewer_stats_path,
    )
    report = runner.run()

    print("\n=== Nightly 指標 ===")
    state = report["state"]
    print("LifeIndicator:", state.get("life_indicator"))
    print("PolicyPrior:", state.get("policy_prior"))
    if report.get("future"):
        print("Future Replay:", report["future"])

    if args.json:
        payload = {
            "state": state,
            "future": report.get("future"),
            "events": report.get("events"),
        }
        print("\nJSON>>", json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()





