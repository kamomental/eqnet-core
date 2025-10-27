# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List
from datetime import datetime, timedelta, date
from .emotion import AXES, extract_emotion
from .field import EmotionField, FieldParams
from .membrane import MembraneController
from .memory import (
    EmotionalTerrain, RawExperienceMemory, EpisodicMemory,
    SemanticMemory, save_json, load_json
)
from .diary import DiaryManager
from .narrative import NarrativeProjection, StoryGraph
from .memory_palace import MemoryPalace, MemoryNode
from .ethics import EthicsManager
from .catalyst import CatalystManager, CatalystParams
from .community import CommunityOrchestrator, CommunityProfile, Turn

class EmotionalMemorySystem:
    def __init__(self, state_dir: str, *, metrics_sink: Callable[[Dict[str, float]], None] | None = None, metrics_log_path: str | Path | None = None):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.terrain = EmotionalTerrain()
        self.l1 = RawExperienceMemory()
        self.l2 = EpisodicMemory()
        self.l3 = SemanticMemory(self.terrain)
        self.diary = DiaryManager()
        self.current_emotion = np.zeros(len(AXES), dtype=float)
        locale = os.getenv("CULTURE_LOCALE", "default")
        self.field = EmotionField(FieldParams(locale=locale))
        self.membrane = MembraneController()
        self.narrative = NarrativeProjection()
        self.story_graph = StoryGraph()
        self.memory_palace = MemoryPalace(
            [
                MemoryNode("atelier", locale, [0.3, 0.7]),
                MemoryNode("garden", locale, [0.7, 0.6]),
                MemoryNode("library", locale, [0.5, 0.4]),
            ]
        )
        self.ethics = EthicsManager()
        self.field_metrics_log: list[dict] = []
        self.catalyst = CatalystManager()
        self.community = None
        self.community_last_error = None
        if os.getenv("ENABLE_COMMUNITY_ORCHESTRATOR", "0") not in {"0", "false", "False"}:
            self.community = CommunityOrchestrator(CommunityProfile.default())
        self._load_state()
        self.field.params.locale = locale
        # Skala有効化
        self.use_skala = os.getenv("USE_SKALA", "0") == "1"
        self.skala_alpha = float(os.getenv("SKALA_ALPHA", "0.2"))
        self.skala_win = int(os.getenv("SKALA_WIN", "7"))
        self.skala_scales = tuple(int(x) for x in os.getenv("SKALA_SCALES", "3,5,9,17").split(","))
        self._recent = []  # 直近ベクトルのバッファ
        self.fatigue_entropy_threshold = float(os.getenv("FATIGUE_ENTROPY_THRESHOLD", "8.5"))
        self.fatigue_enthalpy_threshold = float(os.getenv("FATIGUE_ENTHALPY_THRESHOLD", "0.6"))
        self.fatigue_damp_factor = float(os.getenv("FATIGUE_DAMP_FACTOR", "0.5"))
        self._fatigue_active = False
        self.auto_rest_enabled = os.getenv("AUTO_REST_MODE", "1") != "0"
        self.rest_trigger_threshold = int(os.getenv("REST_TRIGGER_THRESHOLD", "3"))
        self.rest_cooldown_minutes = int(os.getenv("REST_MODE_COOLDOWN_MINUTES", "30"))
        self._fatigue_streak = 0
        self._rest_mode_until = None
        self.rest_history_limit = int(os.getenv("REST_HISTORY_LIMIT", "128"))
        self._rest_history = []
        self._field_metrics_sink = metrics_sink
        self._field_metrics_log_path = Path(metrics_log_path) if metrics_log_path else None
        if self._field_metrics_log_path is not None:
            self._field_metrics_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _state_paths(self):
        return {
            "terrain": self.state_dir/"terrain.json",
            "l1": self.state_dir/"l1.json",
            "l2": self.state_dir/"l2.json",
            "l3": self.state_dir/"l3.json",
            "field": self.state_dir/"field.json",
            "membrane": self.state_dir/"membrane.json",
            "narrative": self.state_dir/"narrative.json",
            "story_graph": self.state_dir/"story_graph.json",
            "memory_palace": self.state_dir/"memory_palace.json",
            "consent": self.state_dir/"consent.json",
            "field_metrics": self.state_dir/"field_metrics.json",
            "catalyst": self.state_dir/"catalyst_events.json",
            "diary": self.state_dir/"diary.json",
            "rest": self.state_dir/"rest_state.json",
        }

    def _load_state(self):
        p = self._state_paths()
        if p["terrain"].exists():
            self.terrain = EmotionalTerrain.from_json(load_json(p["terrain"]))
        if p["l1"].exists():
            experiences = json.load(open(p["l1"],"r",encoding="utf-8"))
            self.l1.experiences = [self._resize_vector_fields(rec) for rec in experiences]
        if p["l2"].exists():
            episodes = json.load(open(p["l2"],"r",encoding="utf-8"))
            for ep in episodes:
                ep["emotion_pattern"]["center"] = self._resize_list(ep["emotion_pattern"]["center"])
                ep["emotion_pattern"]["variance"] = self._resize_list(ep["emotion_pattern"]["variance"])
                ep["emotion_pattern"]["trajectory"] = [self._resize_list(vec) for vec in ep["emotion_pattern"]["trajectory"]]
            self.l2.episodes = episodes
        if p["l3"].exists():
            patterns = json.load(open(p["l3"],"r",encoding="utf-8"))
            for pat in patterns:
                pat["emotion_signature"] = self._resize_list(pat["emotion_signature"])
            self.l3.patterns = patterns
        if p["field"].exists():
            self.field = EmotionField.from_json(load_json(p["field"]))
            self.field.set_locale(os.getenv("CULTURE_LOCALE", self.field.params.locale))
        if p["membrane"].exists():
            self.membrane = MembraneController.from_json(load_json(p["membrane"]))
        if p["narrative"].exists():
            self.narrative = NarrativeProjection.from_json(load_json(p["narrative"]))
        if p["story_graph"].exists():
            self.story_graph = StoryGraph.from_json(load_json(p["story_graph"]))
        if p["memory_palace"].exists():
            self.memory_palace = MemoryPalace.from_json(load_json(p["memory_palace"]))
        if p["consent"].exists():
            self.ethics = EthicsManager.from_json(load_json(p["consent"]))
        self.l1.retention_days = getattr(self.ethics.preferences, "retention_days", self.l1.retention_days)
        if p["field_metrics"].exists():
            self.field_metrics_log = load_json(p["field_metrics"])
        if p["catalyst"].exists():
            self.catalyst = CatalystManager.from_json(load_json(p["catalyst"]))
        if p["diary"].exists():
            self.diary = DiaryManager.from_json(load_json(p["diary"]))
        if p["rest"].exists():
            rest_payload = load_json(p["rest"])
            until_val = rest_payload.get("rest_mode_until")
            self._rest_mode_until = datetime.fromisoformat(until_val) if until_val else None
            self._fatigue_streak = int(rest_payload.get("fatigue_streak", 0))
            self._rest_history = list(rest_payload.get("history", []))

    def _resize_list(self, values):
        vec = np.array(values, dtype=float)
        if vec.ndim == 1:
            if vec.shape[0] < len(AXES):
                vec = np.pad(vec, (0, len(AXES) - vec.shape[0]), mode="constant")
            else:
                vec = vec[: len(AXES)]
            return vec.tolist()
        return values

    def _resize_vector_fields(self, record: dict) -> dict:
        if "emotion_vec" in record:
            record["emotion_vec"] = self._resize_list(record["emotion_vec"])
        return record

    def save_state(self):
        self._apply_consent_filters()
        p = self._state_paths()
        save_json(p["terrain"], self.terrain.to_json())
        json.dump(self.l1.experiences, open(p["l1"],"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(self.l2.episodes, open(p["l2"],"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(self.l3.patterns, open(p["l3"],"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        save_json(p["field"], self.field.to_json())
        save_json(p["membrane"], self.membrane.to_json())
        save_json(p["narrative"], self.narrative.to_json())
        save_json(p["story_graph"], self.story_graph.to_json())
        save_json(p["memory_palace"], self.memory_palace.to_json())
        save_json(p["diary"], self.diary.to_json())
        save_json(p["consent"], self.ethics.to_json())
        save_json(p["field_metrics"], self.field_metrics_log)
        save_json(
            p["rest"],
            {
                "rest_mode_until": self._rest_mode_until.isoformat() if self._rest_mode_until else None,
                "fatigue_streak": self._fatigue_streak,
                "history": self._rest_history[-self.rest_history_limit:],
            },
        )

    def _skala_pred(self):
        if not self.use_skala:
            return None
        try:
            from .skala_layer import skala_predict_gradient
        except Exception:
            return None
        if len(self._recent) < max(2, self.skala_win):
            return None
        seq = np.stack(self._recent[-self.skala_win:], axis=0)
        try:
            grad = skala_predict_gradient(seq, z_u_np=None, win=self.skala_win, scales=self.skala_scales)
            return grad
        except Exception:
            return None

    def ingest_dialogue(self, user_id: str, dialogue: str, timestamp: str|None=None):
        ts = datetime.fromisoformat(timestamp) if timestamp else datetime.utcnow()
        rest_mode_active = self._rest_active(ts)
        if self.community:
            try:
                self.community.observe(Turn(speaker_id=user_id, text=dialogue, prosody=None, timestamp=ts))
                self.community_last_error = None
            except Exception as exc:  # pragma: no cover - guard path
                self.community_last_error = str(exc)
        emo = extract_emotion(dialogue)
        if emo.shape[0] != len(AXES):
            if emo.shape[0] < len(AXES):
                emo = np.pad(emo, (0, len(AXES) - emo.shape[0]), mode="constant")
            else:
                emo = emo[: len(AXES)]
        self.current_emotion = emo
        self._recent.append(emo)
        filtered = self.ethics.filter_emotion(emo)
        self.field.inject_emotion(filtered)
        pred_grad = self._skala_pred()
        field_grad = self.field.feedback_vector(filtered)
        combined_grad = field_grad if pred_grad is None else (pred_grad + field_grad)
        self.terrain.update_trajectory(
            [filtered],
            intensity=float(np.linalg.norm(filtered)),
            pred_grad=combined_grad,
            alpha=self.skala_alpha,
        )
        snapshot_before = self.field.snapshot()
        qualia_signature = self.field.qualia_signature(filtered, snapshot_before)
        palace_weight = 1.0 + float(np.clip(qualia_signature.get("magnitude", 0.0), 0.0, 2.0))
        if rest_mode_active:
            palace_weight *= self.fatigue_damp_factor
        self.field.step()
        metrics = self.field.compute_metrics()
        metrics_record = {"timestamp": ts.isoformat(), **metrics}
        self.field_metrics_log.append(metrics_record)
        if len(self.field_metrics_log) > 1000:
            self.field_metrics_log.pop(0)
        self._emit_field_metrics(metrics_record)
        entropy_val = metrics_record.get("entropy", 0.0)
        enthalpy_val = metrics_record.get("enthalpy_mean", 0.0)
        fatigue_active = (
            entropy_val >= self.fatigue_entropy_threshold
            and enthalpy_val >= self.fatigue_enthalpy_threshold
        )
        self._fatigue_active = fatigue_active
        if fatigue_active:
            palace_weight *= self.fatigue_damp_factor
        membrane_state = self.membrane.update(filtered, self.field)
        locale = self.field.params.locale
        projection = self.narrative.project(filtered, locale)
        self.narrative.update(filtered, locale, membrane_state.empowerment)
        loop_flag = False
        if self.ethics.preferences.allow_story_graph and not rest_mode_active:
            qualia_for_story = qualia_signature if self.ethics.preferences.store_field else None
            if qualia_for_story is not None and fatigue_active:
                qualia_for_story = {
                    **qualia_for_story,
                    "fatigue_flag": True,
                    "entropy": entropy_val,
                    "enthalpy": enthalpy_val,
                }
            loop_flag = self.story_graph.log_event(
                ts.isoformat(),
                filtered,
                projection,
                membrane_state.as_dict(),
                dialogue,
                qualia_for_story,
            )
        if loop_flag:
            palace_weight *= self.fatigue_damp_factor
        qualia_for_palace = qualia_signature if self.ethics.preferences.store_field else None
        if qualia_for_palace is not None and fatigue_active:
            qualia_for_palace = {
                **qualia_for_palace,
                "fatigue_flag": True,
                "entropy": entropy_val,
                "enthalpy": enthalpy_val,
            }
        if rest_mode_active:
            palace_overload = False
        else:
            palace_overload = self.memory_palace.update(
                filtered,
                locale,
                dialogue,
                weight=palace_weight,
                qualia=qualia_for_palace,
            )
        catalyst_event = None
        if not rest_mode_active:
            catalyst_event = self.catalyst.evaluate(
                ts.isoformat(),
                locale,
                filtered,
                membrane_state.as_dict(),
                snapshot_before,
                self.field,
                self.memory_palace,
            )
        rest_mode_active, rest_reason = self._update_rest_state(
            ts,
            fatigue_active,
            loop_flag,
            palace_overload,
            rest_mode_active,
            entropy_val,
            enthalpy_val,
        )
        context = {
            "user_id": user_id,
            "membrane": membrane_state.as_dict() if self.ethics.preferences.store_membrane else {},
            "projection": [float(projection[0]), float(projection[1])] if self.ethics.preferences.store_projection else [],
            "locale": locale,
        }
        if catalyst_event and self.ethics.preferences.store_membrane:
            context["catalyst"] = catalyst_event.to_json()
        if self.ethics.preferences.store_field:
            context["qualia"] = qualia_signature
            if fatigue_active:
                context["fatigue"] = {
                    "entropy": entropy_val,
                    "enthalpy": enthalpy_val,
                    "damp_factor": self.fatigue_damp_factor,
                }
            if loop_flag:
                context.setdefault("story_graph", {})["loop_flag"] = True
            if palace_overload:
                context.setdefault("memory_palace", {})["overload_flag"] = True
        if rest_mode_active or self._rest_mode_until:
            rest_info = context.setdefault("rest", {})
            rest_info["active"] = rest_mode_active
            if rest_reason:
                rest_info["reason"] = rest_reason
            rest_info["history_size"] = len(self._rest_history)
            if self._rest_mode_until:
                rest_info["until"] = self._rest_mode_until.isoformat()
        text = dialogue if self.ethics.preferences.store_dialogue else ""
        self.l1.add(text, filtered, ts, context=context)

    def membrane_state(self):
        return self.membrane.state_dict()

    def narrative_state(self):
        return self.narrative.to_json()

    def story_graph_state(self):
        return self.story_graph.to_json()

    def memory_palace_state(self):
        return self.memory_palace.to_json()

    def diary_state(self):
        return self.diary.to_json()

    def ethics_state(self):
        return self.ethics.to_json()

    def field_metrics_state(self):
        return self.field_metrics_log

    def enable_field_metrics_logging(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._field_metrics_log_path = target

    def set_field_metrics_sink(self, sink: Callable[[Dict[str, float]], None] | None) -> None:
        self._field_metrics_sink = sink

    def catalyst_events_state(self):
        return self.catalyst.to_json()

    def community_state(self):
        if not self.community:
            return {}
        snapshot = self.community.summary()
        if self.community_last_error:
            snapshot["last_error"] = self.community_last_error
        return snapshot

    def rest_state(self):
        return {
            "active": self._rest_active(datetime.utcnow()),
            "rest_mode_until": self._rest_mode_until.isoformat() if self._rest_mode_until else None,
            "fatigue_streak": self._fatigue_streak,
            "auto_rest_enabled": self.auto_rest_enabled,
            "history": self._rest_history[-self.rest_history_limit:],
        }

    def bias_report_state(self):
        axis_counts = {axis: 0 for axis in AXES}
        for exp in self.l1.experiences:
            vec = np.array(exp.get("emotion_vec", []), dtype=float)
            if vec.size == 0:
                continue
            idx = int(np.argmax(np.abs(vec)))
            axis_counts[AXES[idx]] += 1
        report = self.ethics.audit_bias(axis_counts)
        return report.to_json()

    def daily_consolidation(self):
        cand = self.l1.distillation_candidates()
        if cand:
            self.l2.distill_from_raw(cand)
        if self.ethics.preferences.allow_forgetting:
            self.l1.retention_days = self.ethics.preferences.retention_days
            self.l1.decay()
        if self.ethics.preferences.store_diary:
            self._record_diary_entry(datetime.utcnow())
        self.save_state()

    def weekly_abstraction(self):
        self.l3.abstract_from_episodes(self.l2.episodes[-8:])
        self.save_state()

    def _rest_active(self, timestamp: datetime) -> bool:
        return self._rest_mode_until is not None and timestamp < self._rest_mode_until

    def _update_rest_state(
        self,
        timestamp: datetime,
        fatigue_active: bool,
        loop_flag: bool,
        palace_overload: bool,
        rest_mode_active: bool,
        entropy: float,
        enthalpy: float,
    ) -> tuple[bool, str | None]:
        if not self.auto_rest_enabled:
            return rest_mode_active, None
        if self._rest_mode_until is not None and timestamp >= self._rest_mode_until:
            self._rest_mode_until = None
            rest_mode_active = False
        trigger = fatigue_active or loop_flag or palace_overload
        if trigger:
            self._fatigue_streak += 1
        else:
            self._fatigue_streak = max(0, self._fatigue_streak - 1)
        reason = None
        if not rest_mode_active and self._fatigue_streak >= self.rest_trigger_threshold:
            reasons = []
            if fatigue_active:
                reasons.append("fatigue")
            if loop_flag:
                reasons.append("loop")
            if palace_overload:
                reasons.append("overload")
            reason = "+".join(reasons) if reasons else "unspecified"
            self._rest_mode_until = timestamp + timedelta(minutes=self.rest_cooldown_minutes)
            rest_mode_active = True
            self._fatigue_streak = 0
            entry = {
                "timestamp": timestamp.isoformat(),
                "rest_until": self._rest_mode_until.isoformat() if self._rest_mode_until else None,
                "reason": reason,
                "triggers": {
                    "fatigue": fatigue_active,
                    "loop": loop_flag,
                    "overload": palace_overload,
                },
                "metrics": {
                    "entropy": float(entropy),
                    "enthalpy": float(enthalpy),
                },
            }
            self._rest_history.append(entry)
            if len(self._rest_history) > self.rest_history_limit:
                self._rest_history = self._rest_history[-self.rest_history_limit :]
        elif rest_mode_active and trigger and self._rest_mode_until is not None:
            self._rest_mode_until = max(
                self._rest_mode_until,
                timestamp + timedelta(minutes=self.rest_cooldown_minutes // 2 or 1),
            )
        return rest_mode_active, reason

    def _record_diary_entry(self, timestamp: datetime) -> None:
        metrics = self._aggregate_metrics_for_day(timestamp.date())
        if not metrics and self.field_metrics_log:
            metrics = self.field_metrics_log[-1].copy()
        elif not metrics:
            metrics = self.field.compute_metrics()
        top_axes = self._top_axes_from_experiences(timestamp.date())
        catalysts = self._catalyst_highlights_for_day(timestamp.date())
        quotes = self._gentle_quotes_for_day(timestamp.date())
        rest_snapshot = self.rest_state()
        use_llm = os.getenv("USE_LLM", "0") != "0"
        self.diary.record_daily_entry(
            timestamp.date(),
            metrics,
            top_axes,
            catalysts,
            quotes,
            rest_snapshot,
            self.story_graph.loop_alert,
            self._fatigue_active,
            use_llm,
        )

    def _aggregate_metrics_for_day(self, day: date) -> Dict[str, float]:
        entries = []
        for record in self.field_metrics_log:
            try:
                ts = datetime.fromisoformat(record.get("timestamp", ""))
            except Exception:
                continue
            if ts.date() == day:
                entries.append(record)
        if not entries:
            return {}
        keys = {key for item in entries for key in item.keys() if key != "timestamp"}
        aggregated: Dict[str, float] = {}
        for key in keys:
            values = [float(item.get(key, 0.0)) for item in entries]
            aggregated[key] = float(np.mean(values))
        return aggregated

    def _experiences_for_day(self, day: date) -> List[dict]:
        matches: List[dict] = []
        for record in self.l1.experiences:
            ts_raw = record.get("timestamp")
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(ts_raw)
            except Exception:
                continue
            if ts.date() == day:
                matches.append(record)
        return matches

    def _top_axes_from_experiences(self, day: date) -> List[str]:
        experiences = self._experiences_for_day(day)
        counts = {axis: 0 for axis in AXES}
        for record in experiences:
            vec = np.array(record.get("emotion_vec", []), dtype=float)
            if vec.size == 0:
                continue
            idx = int(np.argmax(np.abs(vec)))
            counts[AXES[idx]] += 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [axis for axis, count in ordered if count > 0]

    def _gentle_quotes_for_day(self, day: date) -> List[str]:
        if not self.ethics.preferences.store_dialogue:
            return []
        experiences = self._experiences_for_day(day)
        quotes: List[str] = []
        for record in experiences:
            text = (record.get("dialogue") or "").strip()
            if text:
                quotes.append(text[:80])
        return quotes[-3:]

    def _catalyst_highlights_for_day(self, day: date) -> List[str]:
        highlights: List[str] = []
        for event in getattr(self.catalyst, "events", []):
            ts_raw = event.get("timestamp")
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(ts_raw)
            except Exception:
                continue
            if ts.date() != day:
                continue
            node = event.get("node") or "-"
            mode = event.get("mode", "-")
            react = float(event.get("reactiveness", 0.0))
            highlights.append(f"{mode} @ {node} ({react:.2f})")
        return highlights

    def _apply_consent_filters(self):
        mask = np.array([self.ethics.preferences.record_axes.get(axis, True) for axis in AXES], dtype=bool)
        for exp in self.l1.experiences:
            vec = np.array(exp.get("emotion_vec", []), dtype=float)
            if vec.size:
                exp["emotion_vec"] = np.where(mask, vec, 0.0).tolist()
            context = exp.setdefault("context", {})
            if not self.ethics.preferences.store_dialogue:
                exp["dialogue"] = ""
            if not self.ethics.preferences.store_membrane:
                context.pop("membrane", None)
            if not self.ethics.preferences.store_projection:
                context.pop("projection", None)
            if not self.ethics.preferences.store_field:
                context.pop("qualia", None)
        if not self.ethics.preferences.store_diary:
            self.diary.redact()
        for pat in self.l3.patterns:
            sig = np.array(pat.get("emotion_signature", []), dtype=float)
            if sig.size:
                pat["emotion_signature"] = np.where(mask, sig, 0.0).tolist()
            if not self.ethics.preferences.store_field:
                pat.pop("qualia_signature", None)
        if not self.ethics.preferences.store_field:
            for ep in self.l2.episodes:
                ep.pop("qualia_profile", None)
            for node in self.story_graph.nodes.values():
                for example in node.get("examples", []):
                    example.pop("qualia", None)
            self.memory_palace.qualia_state = self.memory_palace._default_qualia_state()

    def _emit_field_metrics(self, metrics_record: Dict[str, float]) -> None:
        if self._field_metrics_sink is not None:
            try:
                self._field_metrics_sink(dict(metrics_record))
            except Exception:
                # Sink failures should not break ingestion but should also not be silently swallowed.
                self.community_last_error = "field_metrics_sink_error"
        if self._field_metrics_log_path is not None:
            try:
                with self._field_metrics_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics_record, ensure_ascii=False) + "\n")
            except Exception:
                self.community_last_error = "field_metrics_log_error"
