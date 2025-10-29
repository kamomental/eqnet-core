# -*- coding: utf-8 -*-
"""
End-to-end runtime scaffold that ties perception, EQNet metrics, policy
controls, and the LLM hub together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import os
import json
import time
import hashlib

import numpy as np
from pathlib import Path

from terrain.emotion import AXES
from .perception import PerceptionBridge, PerceptionConfig, AffectSample
from .policy import PolicyHead, PolicyConfig, AffectControls
from .llm_hub import LLMHub, LLMHubConfig, HubResponse
from .robot_bridge import RobotBridgeConfig, ROS2Bridge
from datetime import datetime
from terrain.system import EmotionalMemorySystem
from emot_terrain_lab.memory.reference_helper import handle_memory_reference
from emot_terrain_lab.perception.text_affect import quick_text_affect_v2

try:
    from runtime.config import load_runtime_cfg
except ImportError:
    load_runtime_cfg = None



@dataclass
class RuntimeConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    llm: LLMHubConfig = field(default_factory=LLMHubConfig)
    use_eqnet_core: bool = False
    eqnet_state_dir: str = "data/state_hub"
    robot: RobotBridgeConfig = field(default_factory=RobotBridgeConfig)


class EmotionalHubRuntime:
    """
    Minimal orchestrator for the EQNet emotional hub.

    The runtime keeps a synthetic ``E`` vector for now (computed from affect),
    making it straightforward to replace with the real EQNet core later.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self.perception = PerceptionBridge(self.config.perception)
        self.policy = PolicyHead(self.config.policy)
        self.llm = LLMHub(self.config.llm)
        self.robot_bridge: Optional[ROS2Bridge] = None
        if self.config.robot.enabled:
            self.robot_bridge = ROS2Bridge(self.config.robot)
            self.robot_bridge.connect()
        self._last_controls: Optional[AffectControls] = None
        self._last_metrics: Dict[str, float] = {}
        self._last_E = np.zeros(len(AXES), dtype=float)
        self._last_snapshot: Optional[Dict[str, np.ndarray]] = None
        self._last_qualia: Dict[str, Any] = {}
        self.eqnet_system: Optional[EmotionalMemorySystem] = None
        if self.config.use_eqnet_core:
            self.eqnet_system = EmotionalMemorySystem(self.config.eqnet_state_dir)
        self._runtime_cfg = None
        if load_runtime_cfg is not None:
            try:
                self._runtime_cfg = load_runtime_cfg()
            except Exception:
                self._runtime_cfg = None
        self._memory_ref_cfg = getattr(self._runtime_cfg, "memory_reference", None)
        self._memory_ref_log_path: Optional[Path] = None
        self.perceived_affect: Dict[str, float] = {"valence": 0.0, "arousal": 0.0, "confidence": 0.0}
        self._pending_text_affect: Optional[AffectSample] = None
        self._memory_ref_cooldown_until = 0.0
        if self._memory_ref_cfg and hasattr(self._memory_ref_cfg, "log_path"):
            try:
                self._memory_ref_log_path = Path(self._memory_ref_cfg.log_path)
                self._memory_ref_log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                self._memory_ref_log_path = None

    # ------------------------------------------------------------------ #
    # Main entrypoints
    # ------------------------------------------------------------------ #

    def observe_video(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        self.perception.ingest_video_frame(frame, timestamp=timestamp)

    def observe_audio(self, chunk: np.ndarray, timestamp: Optional[float] = None) -> None:
        self.perception.ingest_audio_chunk(chunk, timestamp=timestamp)

    def observe_text(
        self,
        user_text: str,
        lang: str = "ja-JP",
        precomputed: Optional[Tuple[Dict[str, float], str]] = None,
    ) -> Dict[str, float]:
        """
        Lightweight textual affect observation. Returns the perceived affect
        dict so that callers can surface it on the UI if desired.
        """
        lang_prefix = (lang or "ja").split("-")[0]
        if precomputed is not None:
            affect_dict, tag = precomputed
        else:
            affect_dict, tag = quick_text_affect_v2(user_text or "", lang_prefix)
        sample = AffectSample(
            valence=affect_dict["valence"],
            arousal=affect_dict["arousal"],
            confidence=affect_dict["confidence"],
            timestamp=time.time(),
        )
        payload = dict(affect_dict)
        payload["tag"] = tag
        self.perceived_affect = payload
        self._pending_text_affect = sample
        # Roughly project onto field metrics so that the UI has non-zero values
        self._last_metrics = self._metrics_from_text_affect(sample)
        self._last_E = self._vector_from_affect(sample)
        return payload

    def step(
        self,
        user_text: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Produce the latest controls/response.

        Parameters
        ----------
        user_text:
            Optional text utterance from the user.
        context:
            Optional contextual text (e.g., RAG snippets).
        intent:
            Intent label for the hub router (qa/chitchat/code…).
        fast_only:
            When true, skips heavy operations (memory reference lookup, LLM call)
            so that the caller can issue a quick acknowledgement before the full
            pipeline completes.
        """
        affect = self.perception.sample_affect()
        if self._pending_text_affect is not None:
            affect = self._pending_text_affect
            self._pending_text_affect = None
        metrics = self._update_metrics(affect)
        # Merge external mood metrics if provided (env or file)
        metrics = self._merge_mood_metrics(metrics)
        self._last_metrics = metrics

        controls = self.policy.affect_to_controls(affect, metrics)
        self._last_controls = controls

        response: Optional[HubResponse] = None
        memory_reference: Optional[Dict[str, Any]] = None
        if not fast_only and user_text is not None:
            memory_reference = self._maybe_memory_reference(user_text)
            if (
                memory_reference
                and memory_reference.get("reply")
                and (
                    not self._memory_ref_cfg
                    or memory_reference.get("fidelity", 0.0)
                    >= float(getattr(self._memory_ref_cfg, "fidelity_low", 0.0))
                )
            ):
                response = self._wrap_memory_response(memory_reference["reply"])
            else:
                response = self.llm.generate(
                    user_text=user_text,
                    context=context,
                    controls={
                        "temperature": controls.temperature,
                        "top_p": controls.top_p,
                        "pause_ms": controls.pause_ms,
                        "directness": controls.directness,
                        "warmth": controls.warmth,
                        "prosody_energy": controls.prosody_energy,
                        "spoiler_mode": "warn",
                    },
                    intent=intent or self.llm.config.default_intent,
                    slos={"p95_ms": 180.0},
                )

        if self.robot_bridge:
            self.robot_bridge.publish(
                {
                    "pause_ms": controls.pause_ms,
                    "gesture_amplitude": controls.gesture_amplitude,
                    "prosody_energy": controls.prosody_energy,
                    "gaze_mode": 1.0 if controls.gaze_mode == "engage" else 0.0,
                }
            )

        return {
            "affect": affect,
            "controls": controls,
            "metrics": metrics,
            "response": response,
            "memory_reference": memory_reference,
            "robot_state": self.robot_bridge.state if self.robot_bridge else None,
            "qualia": self._last_qualia,
        }

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    @property
    def last_controls(self) -> Optional[AffectControls]:
        return self._last_controls

    @property
    def last_metrics(self) -> Dict[str, float]:
        return self._last_metrics

    @property
    def last_E(self) -> np.ndarray:
        return self._last_E

    @property
    def last_snapshot(self) -> Optional[Dict[str, np.ndarray]]:
        return self._last_snapshot

    @property
    def last_qualia(self) -> Dict[str, Any]:
        return self._last_qualia

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _update_metrics(self, affect: AffectSample) -> Dict[str, float]:
        """
        Project affect into a synthetic EQNet state vector. This is a temporary
        shim; real deployments should call into ``EmotionalMemorySystem``.
        """
        if self.eqnet_system:
            entry = self._update_eqnet_system(affect)
            if not entry:
                return self._fallback_metrics(affect)
            entropy = float(entry.get("entropy", 0.0))
            dissipation = float(entry.get("dissipation", 0.0))
            info_flux = float(entry.get("info_flux", 0.0))
            current_emotion = getattr(self.eqnet_system, "current_emotion", None)
            if current_emotion is not None:
                try:
                    self._last_E = np.asarray(current_emotion, dtype=float)
                except Exception:
                    pass
            H = float(np.clip(1.0 - entropy / 10.0, 0.0, 1.0))
            R = float(np.clip(dissipation, 0.0, 1.0))
            kappa = float(np.clip(-0.5 * affect.arousal, -1.0, 1.0))
            metrics = {
                "H": H,
                "R": R,
                "kappa": kappa,
                "entropy": entropy,
                "dissipation": dissipation,
                "info_flux": info_flux,
                "timestamp": entry.get("timestamp", time.time()),
            }
            return metrics

        return self._fallback_metrics(affect)

    def _update_eqnet_system(self, affect: AffectSample) -> Dict[str, float]:
        assert self.eqnet_system is not None
        ts = datetime.utcnow().isoformat()
        dialogue = (
            f"[affect_stream] valence={affect.valence:.3f} "
            f"arousal={affect.arousal:.3f} confidence={affect.confidence:.3f}"
        )
        self.eqnet_system.ingest_dialogue("affect_stream", dialogue, ts)
        metrics_log = self.eqnet_system.field_metrics_state()
        snapshot = self.eqnet_system.field.snapshot()
        self._last_snapshot = snapshot
        current_emotion = getattr(self.eqnet_system, "current_emotion", None)
        qualia = {}
        if current_emotion is not None:
            try:
                qualia = self.eqnet_system.field.qualia_signature(
                    np.asarray(current_emotion, dtype=float),
                    snapshot=snapshot,
                )
                self._last_qualia = qualia
            except Exception:
                self._last_qualia = {}
        else:
            self._last_qualia = {}
        if metrics_log:
            return metrics_log[-1]
        return {}

    def _fallback_metrics(self, affect: AffectSample) -> Dict[str, float]:
        E = np.zeros(len(AXES), dtype=float)
        E[0] = affect.valence
        E[1] = affect.arousal
        E[3] = affect.valence * 0.7 + affect.arousal * 0.3
        E[4] = -abs(affect.arousal) * 0.4
        self._last_E = E

        H = float(np.clip(0.5 - 0.35 * abs(affect.valence), 0.0, 1.0))
        R = float(np.clip(0.5 + 0.4 * abs(affect.arousal), 0.0, 1.0))
        kappa = float(np.clip(-0.5 * affect.arousal, -1.0, 1.0))
        self._last_snapshot = None
        self._last_qualia = {}
        return {"H": H, "R": R, "kappa": kappa, "timestamp": time.time()}

    # ------------------------------------------------------------------ #
    # Mood metrics merge (env/file injection)
    # ------------------------------------------------------------------ #
    def _merge_mood_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        out = dict(metrics)
        try:
            payload = os.getenv("EQNET_MOOD_METRICS")
            path = os.getenv("EQNET_MOOD_METRICS_FILE")
            mood: Dict[str, Any] = {}
            if payload:
                mood = json.loads(payload)
            elif path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    mood = json.load(f)
            for k in ("mood_v", "mood_a", "mood_effort", "mood_uncertainty"):
                if k in mood:
                    out[k] = float(mood.get(k, 0.0))
        except Exception:
            # best-effort merge only
            pass
        return out

    def _metrics_from_text_affect(self, sample: AffectSample) -> Dict[str, float]:
        valence = float(np.clip(sample.valence, -1.0, 1.0))
        arousal = float(np.clip(sample.arousal, -1.0, 1.0))
        H = float(np.clip(0.5 + 0.35 * valence, 0.0, 1.0))
        R = float(np.clip(0.5 + 0.4 * arousal, 0.0, 1.0))
        kappa = float(np.clip(-0.4 * arousal, -1.0, 1.0))
        ignition = float(np.clip(max(arousal, 0.0) ** 1.3, 0.0, 1.0))
        return {"H": H, "R": R, "kappa": kappa, "ignition": ignition, "timestamp": sample.timestamp}

    def _vector_from_affect(self, sample: AffectSample) -> np.ndarray:
        vec = np.zeros(len(AXES), dtype=float)
        if len(vec) >= 2:
            vec[0] = sample.valence
            vec[1] = sample.arousal
        return vec

    # ------------------------------------------------------------------ #
    # Memory reference helpers
    # ------------------------------------------------------------------ #

    def _maybe_memory_reference(self, user_text: str) -> Optional[Dict[str, Any]]:
        cfg = self._memory_ref_cfg
        if (
            cfg is None
            or not getattr(cfg, "enabled", True)
            or self.eqnet_system is None
            or not user_text.strip()
            or time.time() < self._memory_ref_cooldown_until
        ):
            return None
        culture_tag = getattr(self._runtime_cfg.culture, "tag", "ja-JP") if self._runtime_cfg else "ja-JP"
        overrides = {}
        per_culture = getattr(cfg, "per_culture", {}) or {}
        if isinstance(per_culture, dict):
            overrides = per_culture.get(culture_tag.lower(), {}) or {}
        max_reply_chars = int(overrides.get("max_reply_chars", getattr(cfg, "max_reply_chars", 140)))
        result = handle_memory_reference(
            self.eqnet_system,
            user_text,
            tone="support",
            culture=culture_tag,
            k=int(getattr(cfg, "k", 3)),
            max_reply_chars=max_reply_chars,
        )
        cooldown = float(overrides.get("cooldown_s", getattr(cfg, "cooldown_s", 0.0)))
        if cooldown > 0.0:
            self._memory_ref_cooldown_until = time.time() + cooldown
        self._log_memory_reference(result, user_text)
        return result

    def _wrap_memory_response(self, text: str) -> HubResponse:
        return HubResponse(
            text=text,
            model=None,
            trace_id=f"memory-{int(time.time() * 1000)}",
            latency_ms=0.0,
            controls_used={"mode": "memory_recall"},
            safety={"rating": "G", "memory_recall": "true"},
        )

    def _log_memory_reference(self, result: Optional[Dict[str, Any]], user_text: str) -> None:
        if not result or not self._memory_ref_log_path:
            return
        record: Dict[str, Any] = {
            "ts": time.time(),
            "mode": (result.get("meta") or {}).get("mode", "recall"),
            "fidelity": result.get("fidelity"),
            "reply_len": len((result.get("reply") or "")),
            "anchor": (result.get("meta") or {}).get("anchor"),
            "disclaimer_source": (result.get("meta") or {}).get("disclaimer_source"),
        }
        candidate = result.get("candidate") or {}
        if candidate:
            record["node"] = candidate.get("node")
            record["score"] = candidate.get("score")
            record["affective"] = candidate.get("affective")
        topic_hash = hashlib.sha1(user_text.strip().lower().encode("utf-8")).hexdigest()[:12]
        record["topic_hash"] = topic_hash
        try:
            with self._memory_ref_log_path.open("a", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")
        except Exception:
            pass
