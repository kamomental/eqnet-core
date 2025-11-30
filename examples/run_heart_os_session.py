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
        hits = self.retriever.retrieve(query_vec, top_k=top_k, metadata_filter=metadata_filter)
        if persona_filter:
            target = set(persona_filter)
            filtered = []
            for hit in hits:
                personas = set(hit.metadata.get("persona_ids", []))
                if target.issubset(personas):
                    filtered.append(hit)
            hits = filtered
        return hits


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
    ) -> None:
        self.scenario = scenario
        self.hub = manager.for_user(scenario.persona_id)
        self.persona = persona
        self.session_date = session_date
        self.embed_fn = embed_fn
        self.rag = rag
        self.moment_writer = moment_writer
        overrides = scenario.overrides.get("social_prefs") if scenario.overrides else None
        self.consent_gate = ConsentGate(persona, overrides)
        self._qualia_history: List[Any] = []
        self._records: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        print(f"=== {self.scenario.title} / persona={self.scenario.persona_id} ===")
        for turn_idx, event in enumerate(self.scenario.events, start=1):
            record = self._run_event(turn_idx, event)
            self._records.append(record)
        future_summary = self._finalize()
        return {
            "events": self._records,
            "state": self.hub.query_state(),
            "future": future_summary,
        }

    def _run_event(self, turn_idx: int, event: ScenarioEvent) -> Dict[str, Any]:
        timestamp = self._timestamp(event.clock)
        allowed, blocked = self.consent_gate.filter_targets(event.rag.get("topic") if event.rag else None, event.share_with)
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

        moment = DemoMoment(timestamp, event.awareness_stage, event.emotion, event.culture)
        self.hub.log_moment(moment, event.user_text)
        latest_q = self.hub.latest_qualia_state()
        if latest_q is not None:
            self._qualia_history.append(latest_q)

        self._log_moment(turn_idx, event, timestamp, persona_line, allowed, blocked, rag_hits)

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
    def _log_moment(
        self,
        turn_idx: int,
        event: ScenarioEvent,
        timestamp: datetime,
        persona_line: str,
        allowed: Sequence[str],
        blocked: Sequence[str],
        rag_hits: Sequence[RetrievalHit],
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
            },
            user_text=event.user_text,
            llm_text=persona_line,
            response_meta={"context_docs": [hit.doc_id for hit in rag_hits]},
        )
        self.moment_writer.write(entry)

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





