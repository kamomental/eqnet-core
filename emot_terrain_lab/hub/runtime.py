# -*- coding: utf-8 -*-
"""
End-to-end runtime scaffold that ties perception, EQNet metrics, policy
controls, and the LLM hub together.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, List, Mapping, Callable
import copy
import math
import os
import json
import time
import hashlib
import logging
import uuid
import platform
import re

import numpy as np
from pathlib import Path

try:
    from eqnet.mask_layer import MaskLayer, MaskPersonaProfile, load_persona_profile
except Exception:  # pragma: no cover - optional dependency
    MaskLayer = None  # type: ignore
    MaskPersonaProfile = None  # type: ignore
    load_persona_profile = None  # type: ignore

from eqnet.culture_model import (
    BehaviorMod,
    CultureContext,
    compute_culture_state,
    culture_to_behavior,
    promote_to_monument_if_needed,
    update_climate_from_event,
)
from eqnet.logs.moment_log import MomentLogEntry, MomentLogWriter
from eqnet.hub.streaming_sensor import StreamingSensorState
from eqnet.hub.runtime_sensors import RuntimeSensors
from emot_terrain_lab.i18n.locale import lookup_text, lookup_value, truncate_text
from eqnet.qualia_model import (
    FutureReplayConfig,
    blend_emotion_axes,
    compute_future_risk,
    compute_future_hopefulness,
    sensor_to_emotion_axes,
)
from eqnet.telemetry.trace_paths import TracePathConfig, trace_output_path
from eqnet.telemetry.trace_writer import append_trace_event
from eqnet.runtime.state import QualiaState
from eqnet.modules.prospective_drive_core import PDCConfig, ProspectiveDriveCore
from emot_terrain_lab.terrain.emotion import AXES
from emot_terrain_lab.mind.shadow_estimator import (
    ShadowEstimator,
    ShadowEstimatorConfig,
)
from eqnet_core.memory.diary import DiaryWriter
from eqnet_core.memory.mosaic import MemoryMosaic
from eqnet_core.models.conscious import (
    ConsciousEpisode,
    ForceMatrix,
    ImplementationContext,
    ResponseRoute,
    BoundarySignal,
    ResetEvent,
    SelfForceSnapshot,
    SelfLayer,
    SelfModel,
    WorldStateSnapshot,
)
from eqnet_core.models.emotion import EmotionVector, ValueGradient
from eqnet_core.models.talk_mode import TalkMode
from eqnet_core.models.runtime_turn import RuntimeTurnResult
from eqnet_core.models.project_atri_2d import ProjectATRI2DState, ProjectATRI2DEvent
from eqnet_core.qualia import AccessGate, AccessGateConfig, MetaMonitor
from .perception import PerceptionBridge, PerceptionConfig, AffectSample
from .policy import PolicyHead, PolicyConfig, AffectControls
from .llm_hub import LLMHub, LLMHubConfig, HubResponse
from emot_terrain_lab.vision.lmstudio_vlm import LMStudioVLMAdapter
from emot_terrain_lab.rag.sse_search import SSESearchAdapter
from emot_terrain_lab.memory.vision_memory_store import VisionMemoryStore
from inner_os.conscious_access import ConsciousAccessCore
from inner_os.expression.content_policy import (
    derive_content_sequence,
    derive_content_skeleton,
    localize_content_sequence,
    render_content_sequence,
)
from inner_os.discussion_thread_registry import update_discussion_thread_registry_snapshot
from inner_os.anchor_normalization import normalize_anchor_hint
from inner_os.expression.hint_bridge import build_expression_hints_from_gate_result
from inner_os.expression.surface_language_profile import shape_surface_language_text
from inner_os.expression.surface_expression_selector import (
    build_surface_expression_candidates,
    choose_surface_expression,
)
from inner_os.expression.surface_context_packet import build_surface_context_packet
from inner_os.headless_runtime import HeadlessInnerOSRuntime
from inner_os.integration_hooks import IntegrationHooks
from inner_os.memory_bridge import collect_runtime_memory_candidates
from inner_os.relational_world import RelationalWorldCore, RelationalWorldState
from inner_os.physiology import (
    BoundaryCore,
    HeartbeatConfig,
    HeartbeatCore,
    HeartbeatState,
    PainStressCore,
    RecoveryCore,
)
from .robot_bridge import RobotBridgeConfig, ROS2Bridge
from datetime import datetime
from emot_terrain_lab.terrain.system import EmotionalMemorySystem
from emot_terrain_lab.memory.reference_helper import handle_memory_reference
from emot_terrain_lab.memory.memory_hint import render_memory_hint
from emot_terrain_lab.memory.inner_os_working_memory_bridge import (
    derive_working_memory_seed_from_signature,
)
from emot_terrain_lab.memory.recall_policy import (
    RarityBudgetState,
    apply_rarity_budget,
    render_recall_cue,
)
from emot_terrain_lab.perception.text_affect import quick_text_affect_v2
from emot_terrain_lab.utils.io import append_jsonl
from inner_os.distillation_record import InnerOSDistillationRecordBuilder
from inner_os.transfer_package import InnerOSTransferPackageBuilder
from inner_os.continuity_summary import ContinuitySummaryBuilder

try:
    from runtime.config import load_runtime_cfg
except ImportError:
    load_runtime_cfg = None

LOGGER = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
LOGGER.setLevel(logging.DEBUG)

INNER_OS_SURFACE_THRESHOLDS = {
    "closing_question_bias": 0.24,
    "probe_question_bias": 0.28,
    "reopening_recovery": 0.24,
}


def _runtime_response_locale(runtime: Any) -> str:
    locale_fn = getattr(runtime, "_response_locale", None)
    if callable(locale_fn):
        try:
            locale = str(locale_fn() or "").strip()
            if locale:
                return locale
        except Exception:
            pass
    runtime_cfg = getattr(runtime, "_runtime_cfg", None)
    if runtime_cfg is not None and hasattr(runtime_cfg, "backchannel"):
        locale = str(getattr(runtime_cfg.backchannel, "culture", "") or "").strip()
        if locale:
            return locale
    return "en-US"


def _runtime_recent_surface_response_history(runtime: Any) -> list[str]:
    history_fn = getattr(runtime, "_recent_surface_response_history", None)
    if callable(history_fn):
        try:
            history = history_fn()
            if isinstance(history, list):
                return [
                    text
                    for text in (
                        str(item or "").strip()
                        for item in history
                    )
                    if text
                ]
        except Exception:
            pass
    return [
        text
        for text in (
            str(item or "").strip()
            for item in getattr(runtime, "_surface_response_history", ())
        )
        if text
    ]


def _choose_compact_phrase(
    runtime: Any,
    *,
    locale: str,
    primary_key: str,
    alternatives_key: str = "",
    surface_profile: Mapping[str, Any] | None = None,
    candidate_profile: str = "",
) -> str:
    primary = str(lookup_text(locale, primary_key) or "").strip()
    alternatives = lookup_value(locale, alternatives_key) if alternatives_key else None
    candidates: list[str] = [primary] if primary else []
    if isinstance(alternatives, list):
        candidates.extend(
            str(item).strip()
            for item in alternatives
            if str(item).strip()
        )
    history = _runtime_recent_surface_response_history(runtime)
    if locale.lower().startswith("ja"):
        selected = choose_surface_expression(
            build_surface_expression_candidates(
                candidates,
                candidate_profile=candidate_profile,
            ),
            cultural_register=str((surface_profile or {}).get("cultural_register") or "").strip(),
            group_register=str((surface_profile or {}).get("group_register") or "").strip(),
            sentence_temperature=str((surface_profile or {}).get("sentence_temperature") or "").strip(),
            mode=str((surface_profile or {}).get("surface_mode") or "").strip(),
            recent_history=history,
        )
        if selected:
            return selected
    for candidate in candidates:
        if candidate and not any(candidate in entry for entry in history):
            return candidate
    return candidates[0] if candidates else ""


def _format_locale_template(locale: str, key: str, **kwargs: str) -> str:
    template = str(lookup_text(locale, key) or "").strip()
    if not template:
        return ""
    try:
        return template.format(**kwargs).strip()
    except Exception:
        return template


_SURFACE_ACT_CANDIDATE_PROFILES: dict[str, str] = {
    "quiet_presence": "quiet_presence",
    "stay_with_present_need": "stay_with_present_need",
    "gentle_question_hidden_need": "light_question",
    "gentle_question_hidden_need_continuing": "light_question",
    "gentle_question_weight": "light_question",
    "gentle_question_weight_continuing": "light_question",
    "gentle_question_fear": "light_question",
    "gentle_question_fear_continuing": "light_question",
    "gentle_question_self_blame": "light_question",
    "gentle_question_self_blame_continuing": "light_question",
}


def _choose_surface_act_text(
    runtime: Any,
    *,
    act: str,
    text: str,
    locale: str,
    surface_profile: Mapping[str, Any] | None = None,
) -> str:
    body = str(text or "").strip()
    if not body or not locale.lower().startswith("ja"):
        return body
    candidate_profile = _SURFACE_ACT_CANDIDATE_PROFILES.get(str(act or "").strip())
    if not candidate_profile:
        return body
    alternatives = lookup_value(locale, f"inner_os.content_policy_segments.{act}_alternatives")
    texts: list[str] = [body]
    if isinstance(alternatives, list):
        texts.extend(
            str(item).strip()
            for item in alternatives
            if str(item).strip()
        )
    selected = choose_surface_expression(
        build_surface_expression_candidates(
            texts,
            candidate_profile=candidate_profile,
        ),
        cultural_register=str((surface_profile or {}).get("cultural_register") or "").strip(),
        group_register=str((surface_profile or {}).get("group_register") or "").strip(),
        sentence_temperature=str((surface_profile or {}).get("sentence_temperature") or "").strip(),
        mode=str((surface_profile or {}).get("surface_mode") or "").strip(),
        recent_history=_runtime_recent_surface_response_history(runtime),
    )
    return selected or body


@dataclass
class MaskLayerConfig:
    """Runtime-level controls for the persona mask layer."""

    enabled: bool = False
    persona: Dict[str, Any] = field(default_factory=dict)
    log_path: str = ""


@dataclass
class ReflexRouteConfig:
    prediction_error_threshold: float = 0.85
    stress_threshold: float = 0.75


@dataclass
class ConsciousThresholdConfig:
    prediction_error_min: float = 0.35
    valence_min: float = 0.3
    love_min: float = 0.65
    stress_min: float = 0.55


@dataclass
class ValueGradientConfig:
    survival_bias: float = 0.5
    physiological_bias: float = 0.5
    social_bias: float = 0.5
    exploration_bias: float = 0.5
    attachment_bias: float = 0.5


@dataclass
class BoundaryConfig:
    enabled: bool = False
    threshold: float = 0.65
    weight_prediction_error: float = 0.35
    weight_force_delta: float = 0.35
    weight_winner_flip: float = 0.15
    weight_raw_gap: float = 0.15
    prediction_error_norm: float = 0.5
    force_delta_norm: float = 1.0
    raw_gap_norm: float = 1.0


@dataclass
class ResetConfig:
    enabled: bool = False
    boundary_threshold: float = 0.75
    targets: List[str] = field(default_factory=lambda: ["scratchpad", "affective_echo"])
    preserve: List[str] = field(default_factory=list)
    # NOTE: reset is intentionally not wired into runtime yet.


@dataclass
class DampingConfig:
    enabled: bool = True
    latency_L0_ms: float = 200.0
    latency_k_narr: float = 0.002
    latency_k_reflex: float = 0.001
    context_bump: float = 0.05
    score_floor: Optional[float] = None
    reflex_tags: Tuple[str, ...] = ("safety", "hazard")
    affective_tags: Tuple[str, ...] = ("support", "counseling")
    narrative_tags: Tuple[str, ...] = ("brainstorm", "creative")
    latency_penalty_cap: float = 0.75
    tag_bumps: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class DevelopmentConfig:
    enabled: bool = False
    mature_turns: int = 800
    narrative_initial_scale: float = 0.2
    reflex_final_scale: float = 0.8
    affective_peak_gain: float = 0.15


@dataclass
class ConsciousnessConfig:
    enabled: bool = True
    reflex: ReflexRouteConfig = field(default_factory=ReflexRouteConfig)
    conscious_threshold: ConsciousThresholdConfig = field(
        default_factory=ConsciousThresholdConfig
    )
    memory_path: str = "logs/conscious_episodes.jsonl"
    diary_path: str = "logs/conscious_diary.jsonl"
    habit_tags: List[str] = field(default_factory=list)
    value_gradient: ValueGradientConfig = field(default_factory=ValueGradientConfig)
    damping: DampingConfig = field(default_factory=DampingConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    force_matrix: Optional[Dict[str, Dict[str, float]]] = None
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    qualia_gate: AccessGateConfig = field(default_factory=AccessGateConfig)


def apply_mask_layer(
    user_text: str,
    context: Optional[str],
    *,
    mask_cfg: MaskLayerConfig,
    prospective: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Pass prompts through the mask layer when available."""

    persona_meta: Dict[str, Any] = {}
    if not getattr(mask_cfg, "enabled", False) or MaskLayer is None:
        return user_text, context, persona_meta

    persona_payload = getattr(mask_cfg, "persona", None)
    profile = None
    if load_persona_profile is not None:
        try:
            profile = load_persona_profile(persona_payload)
        except Exception:
            profile = None
    elif MaskPersonaProfile is not None:
        if isinstance(persona_payload, MaskPersonaProfile):
            profile = persona_payload
        elif isinstance(persona_payload, dict):
            try:
                profile = MaskPersonaProfile(**persona_payload)
            except Exception:
                profile = None
        elif isinstance(persona_payload, str):
            profile = MaskPersonaProfile()

    if profile is None:
        return user_text, context, persona_meta

    layer = MaskLayer(profile)
    inner_spec = {"pdc": prospective or {}, "phi_snapshot": None}
    dialog_context = {"base_context": context or ""}
    try:
        prompt_obj = layer.build_prompt(
            inner_spec=inner_spec, dialog_context=dialog_context
        )
    except Exception:
        return user_text, context, persona_meta

    persona_meta = dict(prompt_obj.persona_meta or {})
    masked_context = _merge_mask_prompt(prompt_obj.system_prompt, context)
    return user_text, masked_context, persona_meta


def _merge_mask_prompt(
    system_prompt: Optional[str], context: Optional[str]
) -> Optional[str]:
    if not system_prompt:
        return context
    snippet = system_prompt.strip()
    if not snippet:
        return context
    if not context:
        return snippet
    return f"{snippet}\n\n---\n\n{context.strip()}"


@dataclass
class RuntimeConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    llm: LLMHubConfig = field(default_factory=LLMHubConfig)
    use_eqnet_core: bool = False
    force_llm_bridge: bool = False
    eqnet_state_dir: str = "data/state_hub"
    robot: RobotBridgeConfig = field(default_factory=RobotBridgeConfig)
    mask_layer: MaskLayerConfig = field(default_factory=MaskLayerConfig)
    moment_log_path: Optional[str] = "logs/moment_log.jsonl"
    distillation_log_path: Optional[str] = None
    distillation_log_include_text: bool = False
    transfer_package_path: Optional[str] = None
    dashboard_snapshot_path: Optional[str] = None
    conscious: ConsciousnessConfig = field(default_factory=ConsciousnessConfig)
    self_model: Optional[SelfModel] = None


@dataclass
class GateContext:
    engaged: bool
    face_motion: float
    blink: float
    voice_energy: float
    delta_m: float
    jerk: float
    text_input: bool
    since_last_user_ms: float
    force_listen: bool = False


@dataclass
class EQFrame:
    """Snapshot of the inner state vs persona mask for a single turn."""

    ts: float
    inner_spec: Dict[str, Any]
    persona_id: Optional[str]
    persona_meta: Dict[str, Any]
    tension_score: float


class EmotionCore:
    """Minimal oscillator keeping a low-dimensional Phi vector alive."""

    def __init__(self, *, dim: int) -> None:
        self.dim = dim
        self._phi = np.zeros(dim, dtype=float)
        self._setpoint = np.zeros(dim, dtype=float)

    def update_setpoint(self, metrics: Dict[str, Any]) -> None:
        mapping = [
            ("valence", 0),
            ("arousal", 1),
            ("rho", 2),
            ("entropy", 3),
        ]
        for key, idx in mapping:
            if idx >= self.dim or key not in metrics:
                continue
            try:
                self._setpoint[idx] = float(metrics[key])
            except (TypeError, ValueError):
                continue

    def seed(self, vec: np.ndarray) -> None:
        if vec.size:
            limit = min(vec.size, self.dim)
            self._phi[:limit] = vec[:limit]

    def step(
        self,
        base_rate: float,
        gain: float,
        noise_scale: float = 0.02,
        damp: float = 0.0,
    ) -> np.ndarray:
        base = float(max(base_rate, 0.0))
        gain = float(max(gain, 0.0))
        noise = np.random.normal(
            scale=noise_scale * max(gain, 1e-3), size=self._phi.shape
        )
        drift = -base * (self._phi - self._setpoint) - float(max(damp, 0.0)) * self._phi
        self._phi = self._phi + drift + noise
        return self._phi

    def get_phi_vec(self) -> np.ndarray:
        return np.array(self._phi, copy=True)

    def get_phi_norm(self) -> float:
        return float(np.linalg.norm(self._phi))


class FastAckState:
    """Minimal state container for fast acknowledgement sampling."""

    def __init__(self) -> None:
        self.last_choice: str = "silence"


class ArousalTracker:
    """Track the previous arousal value to compute delta responses."""

    def __init__(self) -> None:
        self.last_arousal: float = 0.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp_values = np.exp(shifted)
    denom = float(np.sum(exp_values))
    if denom <= 0.0:
        size = logits.size or 1
        return np.full(size, 1.0 / size)
    return exp_values / denom


def sample_fast_ack(
    arousal: float,
    distance: float,
    state: FastAckState,
    tracker: ArousalTracker,
) -> str:
    """Return 'silence' / 'breath' / 'ack' based on continuous inputs."""

    arousal_clamped = float(np.clip(arousal, 0.0, 1.0))
    distance_clamped = float(np.clip(distance, 0.0, 1.0))
    delta = arousal_clamped - tracker.last_arousal
    tracker.last_arousal = arousal_clamped

    logits = np.array([0.0, 0.0, 0.0], dtype=float)
    logits[0] += 2.0 * (1.0 - arousal_clamped)
    logits[2] += 2.0 * arousal_clamped
    logits[1] += 0.5 + 0.5 * (1.0 - distance_clamped)

    if state.last_choice == "ack":
        logits[2] -= 0.5
    elif state.last_choice == "silence":
        logits[0] -= 0.3

    if delta > 0.3:
        logits[2] += 0.5

    probs = _softmax(logits)
    choice = np.random.choice(["silence", "breath", "ack"], p=probs)
    state.last_choice = choice
    return choice


def _decide_talk_mode(ctx: GateContext) -> TalkMode:
    if ctx.force_listen:
        return TalkMode.WATCH
    if not ctx.engaged or ctx.since_last_user_ms < 400.0:
        return TalkMode.WATCH
    if ctx.delta_m > 0.35 or ctx.jerk > 0.2 or ctx.voice_energy < 0.1:
        return TalkMode.SOOTHE
    if ctx.face_motion < 0.02 and ctx.voice_energy < 0.2:
        return TalkMode.ASK
    if ctx.text_input:
        return TalkMode.TALK
    return TalkMode.WATCH


class _NullProspectiveMemory:
    """Fallback memory surface for when EQNet core is unavailable."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _fit(self, vector) -> np.ndarray:
        arr = np.zeros(self.dim, dtype=float)
        if vector is None:
            return arr
        vec = np.asarray(vector, dtype=float).reshape(-1)
        limit = min(arr.size, vec.size)
        if limit:
            arr[:limit] = vec[:limit]
        return arr

    def sample_success_vector(self, phi_t: np.ndarray) -> np.ndarray:
        return self._fit(phi_t)

    def sample_future_template(
        self, phi_t: np.ndarray, psi_t: np.ndarray | None = None
    ) -> np.ndarray:
        phi_vec = self._fit(phi_t)
        psi_vec = self._fit(psi_t) if psi_t is not None else np.zeros_like(phi_vec)
        return 0.6 * phi_vec + 0.4 * psi_vec


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
        self._vision_memory_store = VisionMemoryStore()
        self._inner_os_headless_runtime = HeadlessInnerOSRuntime()
        self.robot_bridge: Optional[ROS2Bridge] = None
        if self.config.robot.enabled:
            self.robot_bridge = ROS2Bridge(self.config.robot)
            self.robot_bridge.connect()
        self._last_controls: Optional[AffectControls] = None
        self._last_metrics: Dict[str, float] = {}
        self._last_E = np.zeros(len(AXES), dtype=float)
        self._last_snapshot: Optional[Dict[str, np.ndarray]] = None
        self._last_qualia: Dict[str, Any] = {}
        self._self_model: SelfModel = (
            self.config.self_model or self._default_self_model()
        )
        self._persona_conscious_overrides: Dict[str, Any] = (
            self._extract_persona_conscious_overrides()
        )
        self._default_value_gradient = self._build_default_value_gradient()
        self._force_matrix: ForceMatrix = self._build_force_matrix()
        self._last_dominant_layer: Optional[SelfLayer] = None
        self._last_impl_context: Optional[ImplementationContext] = None
        self._last_self_force: Optional[SelfForceSnapshot] = None
        self._last_raw_force: Optional[SelfForceSnapshot] = None
        self._last_boundary_signal: Optional[BoundarySignal] = None
        self._prev_boundary_pred_error: Optional[float] = None
        self._prev_boundary_damped: Optional[SelfForceSnapshot] = None
        self._prev_boundary_winner: Optional[SelfLayer] = None
        self._conscious_memory: Optional[MemoryMosaic] = None
        self._diary_writer: Optional[DiaryWriter] = None
        if self.config.conscious.enabled:
            self._conscious_memory = MemoryMosaic(self.config.conscious.memory_path)
            self._diary_writer = DiaryWriter(self.config.conscious.diary_path)
        self._talk_mode: TalkMode = TalkMode.WATCH
        self._force_listen: bool = False
        self._engaged_override: Optional[bool] = None
        self._last_user_ts: float = 0.0
        self._last_surface_user_text: str = ""
        self._surface_user_history = deque(maxlen=4)
        self._surface_response_history = deque(maxlen=3)
        self._prev_affect_vec: Optional[np.ndarray] = None
        self._prev_prev_affect_vec: Optional[np.ndarray] = None
        self._prev_qualia_vec: Optional[np.ndarray] = None
        gate_env = (os.getenv("EQNET_QUALIA_GATE", "1") or "1").lower()
        self._qualia_gate_enabled = gate_env not in {"0", "false", "off"}
        self._qualia_gate = AccessGate(self._build_access_gate_config())
        self._qualia_meta = MetaMonitor()
        self._last_qualia_gate: Dict[str, Any] = {}
        self._last_gate_context: Dict[str, Any] = {}
        self._last_life_indicator: float = 0.0
        self._last_eqframe: Optional[EQFrame] = None
        self._runtime_sensors = RuntimeSensors()
        self._last_sensor_axes: Optional[List[float]] = None
        self._last_blended_axes: Optional[List[float]] = None
        self._future_history = deque(maxlen=16)
        self._future_risk_cfg = FutureReplayConfig(steps=4, noise_scale=0.2, window=3)
        self._future_risk_trigger = float(
            os.getenv("EQNET_FUTURE_RISK_TRIGGER", "0.45")
        )
        self._future_stress_cutoff = float(
            os.getenv("EQNET_FUTURE_STRESS_CUTOFF", "0.6")
        )
        self._future_body_cutoff = float(os.getenv("EQNET_FUTURE_BODY_CUTOFF", "0.4"))
        self._future_risk_log_path = Path("logs/future_risk.jsonl")
        self._last_future_risk: Optional[float] = None
        self._imagery_history = deque(maxlen=16)
        self._future_hope_cfg = FutureReplayConfig(steps=4, noise_scale=0.15, window=3)
        self._future_hope_trigger = float(
            os.getenv("EQNET_FUTURE_HOPE_TRIGGER", "0.25")
        )
        self._future_hope_log_path = Path("logs/future_imagery.jsonl")
        self._last_future_hope: Optional[float] = None
        self.eqnet_system: Optional[EmotionalMemorySystem] = None
        if self.config.use_eqnet_core:
            self.eqnet_system = EmotionalMemorySystem(
                self.config.eqnet_state_dir,
                moment_log_path=self.config.moment_log_path,
            )
        self._emotion_core = EmotionCore(dim=len(AXES))
        self._pdc = ProspectiveDriveCore(PDCConfig(dim=len(AXES)))
        self._shadow_estimator = ShadowEstimator()
        self._pdc_memory_fallback = _NullProspectiveMemory(len(AXES))
        self._last_prospective: Optional[Dict[str, Any]] = None
        self._last_shadow_estimate: Optional[Dict[str, Any]] = None
        self._runtime_cfg = None
        if load_runtime_cfg is not None:
            try:
                self._runtime_cfg = load_runtime_cfg()
            except Exception:
                self._runtime_cfg = None
        self._presence_cfg = (
            getattr(self._runtime_cfg, "presence", None) if self._runtime_cfg else None
        )
        self._ack_cfg = (
            getattr(self._runtime_cfg, "ack", None) if self._runtime_cfg else None
        )
        self._memory_hint_cfg = (
            getattr(self._runtime_cfg, "memory_hint_policy", None) if self._runtime_cfg else None
        )
        self._apply_runtime_gate_overrides()
        self._model_cfg = (
            getattr(self._runtime_cfg, "model", None) if self._runtime_cfg else None
        )
        self._assoc_kernel_cfg = (
            getattr(self._model_cfg, "assoc_kernel", None)
            if self._model_cfg is not None
            else None
        )
        self._memory_ref_cfg = getattr(self._runtime_cfg, "memory_reference", None)
        self._memory_ref_log_path: Optional[Path] = None
        self._think_log_path: Optional[Path] = None
        self._distillation_log_path: Optional[Path] = None
        self._transfer_package_path: Optional[Path] = None
        self._distillation_log_include_text = bool(
            getattr(self.config, "distillation_log_include_text", False)
        )
        self._distillation_record_builder = InnerOSDistillationRecordBuilder()
        self._transfer_package_builder = InnerOSTransferPackageBuilder()
        self.perceived_affect: Dict[str, float] = {
            "valence": 0.0,
            "arousal": 0.0,
            "confidence": 0.0,
        }
        self._pending_text_affect: Optional[AffectSample] = None
        self._memory_ref_cooldown_until = 0.0
        self._recall_budget_state = RarityBudgetState()
        if self._memory_ref_cfg and hasattr(self._memory_ref_cfg, "log_path"):
            try:
                self._memory_ref_log_path = Path(self._memory_ref_cfg.log_path)
                self._memory_ref_log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                self._memory_ref_log_path = None
        try:
            think_path = (
                getattr(self._memory_ref_cfg, "think_log_path", None)
                if self._memory_ref_cfg is not None
                else None
            )
            self._think_log_path = Path(think_path or "logs/think_log.jsonl")
            self._think_log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._think_log_path = None
        try:
            distillation_path = getattr(self.config, "distillation_log_path", None) or os.getenv(
                "EQNET_DISTILLATION_LOG_PATH",
                "",
            )
            if distillation_path:
                self._distillation_log_path = Path(distillation_path)
                self._distillation_log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._distillation_log_path = None
        try:
            transfer_package_path = getattr(self.config, "transfer_package_path", None) or os.getenv(
                "EQNET_TRANSFER_PACKAGE_PATH",
                "",
            )
            if transfer_package_path:
                self._transfer_package_path = Path(transfer_package_path)
                self._transfer_package_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._transfer_package_path = None
        try:
            dashboard_snapshot_path = getattr(self.config, "dashboard_snapshot_path", None) or os.getenv(
                "EQNET_DASHBOARD_SNAPSHOT_PATH",
                "",
            )
            if dashboard_snapshot_path:
                self._dashboard_snapshot_path = Path(dashboard_snapshot_path)
                self._dashboard_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._dashboard_snapshot_path = None
        self._load_inner_os_transfer_package_from_disk()

        self._fast_ack_state = FastAckState()
        self._arousal_tracker = ArousalTracker()
        self._last_persona_meta: Dict[str, Any] = {}
        self._last_fast_ack_sample: Optional[Dict[str, Any]] = None
        self._last_presence_ack_ts = 0.0
        self._last_memory_hint_meta: Optional[Dict[str, Any]] = None
        self._last_speaker: Optional[str] = None
        self._memory_hint_pressure: float = 0.0
        self._memory_hint_pressure_ts: float = time.time()
        self._memory_hint_prev_blocked: bool = False
        self._heart_rate = 0.85
        self._heart_phase = 0.0
        self._heart_last_ts = time.time()
        self._heart_base_rate = 0.85
        self._heart_gain = 0.45
        self._heartbeat_core = HeartbeatCore(
            HeartbeatConfig(base_rate=self._heart_base_rate, gain=self._heart_gain),
            HeartbeatState(
                rate=self._heart_rate,
                phase=self._heart_phase,
                last_ts=self._heart_last_ts,
            ),
        )
        self._pain_stress_core = PainStressCore()
        self._recovery_core = RecoveryCore()
        self._boundary_core = BoundaryCore()
        self._moment_log_writer = MomentLogWriter(self.config.moment_log_path)
        self._turn_id = 0
        self._session_id = None
        runtime_session = (
            getattr(self._runtime_cfg, "session", None) if self._runtime_cfg else None
        )
        if runtime_session is not None:
            self._session_id = getattr(runtime_session, "session_id", None)
        self._relational_world_core = RelationalWorldCore(
            RelationalWorldState(
                mode="reality",
                world_id="harbor_town",
                world_type="infrastructure",
                zone_id="market",
                time_phase="day",
                weather="clear",
                simulation_enabled=False,
                simulation_episode_id=None,
                simulation_transfer_pending=False,
                world_source="runtime",
            )
        )
        self._surface_world_state: Dict[str, Any] = self._relational_world_core.snapshot()
        self._conscious_access_core = ConsciousAccessCore()
        self._integration_hooks = IntegrationHooks(
            pain_stress_core=self._pain_stress_core,
            recovery_core=self._recovery_core,
            boundary_core=self._boundary_core,
            conscious_access_core=self._conscious_access_core,
        )
        self._last_2d_event: Optional[Dict[str, Any]] = None
        self._last_observed_vision_entry: Optional[Dict[str, Any]] = None
        self._culture_recurrence: Dict[Tuple[str, str, str, str], int] = defaultdict(
            int
        )
        self._default_persona_meta = self._build_default_persona_meta()

    @staticmethod
    def _ensure_interaction_gate_defaults(
        gate_context: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Phase 1: expose interaction gate fields with safe defaults."""
        payload = dict(gate_context or {})
        payload.setdefault("interaction_decision", "IGNORE")
        # Keep list creation local to avoid shared mutable defaults.
        payload.setdefault("interaction_reason_tags", [])
        return payload

    def _default_self_model(self) -> SelfModel:
        return SelfModel(
            role_labels=["eqnet", "companion"],
            long_term_traits={"warmth": 0.6, "stability": 0.4},
            current_mode=TalkMode.WATCH,
            current_energy=0.85,
            attachment_to_user=0.5,
        )

    def _extract_persona_conscious_overrides(self) -> Dict[str, Any]:
        persona_cfg = getattr(self.config.mask_layer, "persona", None)
        if isinstance(persona_cfg, Mapping):
            payload = persona_cfg.get("conscious")
            if isinstance(payload, Mapping):
                return copy.deepcopy(payload)
        return {}

    def _persona_conscious_value(self, key: str) -> Optional[Mapping[str, Any]]:
        payload = self._persona_conscious_overrides.get(key)
        if isinstance(payload, Mapping):
            return copy.deepcopy(payload)
        return None

    @staticmethod
    def _merge_access_gate_config(
        base: AccessGateConfig, overrides: Mapping[str, Any]
    ) -> AccessGateConfig:
        data = base.__dict__.copy()
        for key, value in overrides.items():
            if key not in data:
                continue
            current = data[key]
            try:
                if isinstance(current, bool):
                    if isinstance(value, str):
                        v = value.strip().lower()
                        if v in {"true", "yes", "1", "on"}:
                            data[key] = True
                        elif v in {"false", "no", "0", "off"}:
                            data[key] = False
                        else:
                            continue
                    else:
                        data[key] = bool(value)
                elif isinstance(current, (int, float)):
                    data[key] = float(value)
                else:
                    data[key] = value
            except (TypeError, ValueError):
                continue
        return AccessGateConfig(**data)

    def _build_access_gate_config(self) -> AccessGateConfig:
        cfg = getattr(self.config.conscious, "qualia_gate", None)
        if isinstance(cfg, AccessGateConfig):
            base = AccessGateConfig(**cfg.__dict__)
        elif isinstance(cfg, Mapping):
            base = self._merge_access_gate_config(AccessGateConfig(), cfg)
        else:
            base = AccessGateConfig()
        overrides = self._persona_conscious_value("qualia_gate")
        if overrides:
            base = self._merge_access_gate_config(base, overrides)
        return base

    def _apply_runtime_gate_overrides(self) -> None:
        if self._runtime_cfg is None:
            return
        gate_cfg = getattr(self._runtime_cfg, "qualia_access_gate", None)
        if gate_cfg is None:
            return
        overrides = gate_cfg.__dict__ if hasattr(gate_cfg, "__dict__") else gate_cfg
        if not isinstance(overrides, Mapping):
            return
        updated = self._merge_access_gate_config(self._qualia_gate.config, overrides)
        self._qualia_gate.config = updated
        self._qualia_gate.reset()

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _apply_value_gradient_overrides(
        self, base: ValueGradient, overrides: Mapping[str, Any]
    ) -> ValueGradient:
        def pick(primary: str, alias: str, current: float) -> float:
            if primary in overrides:
                return self._safe_float(overrides[primary], current)
            if alias in overrides:
                return self._safe_float(overrides[alias], current)
            return current

        return ValueGradient(
            survival_bias=pick("survival_bias", "survival", base.survival_bias),
            physiological_bias=pick(
                "physiological_bias", "physiological", base.physiological_bias
            ),
            social_bias=pick("social_bias", "social", base.social_bias),
            exploration_bias=pick(
                "exploration_bias", "exploration", base.exploration_bias
            ),
            attachment_bias=pick("attachment_bias", "attachment", base.attachment_bias),
        )

    def _build_default_value_gradient(self) -> ValueGradient:
        vg_cfg = self.config.conscious.value_gradient
        base = ValueGradient(
            survival_bias=float(np.clip(vg_cfg.survival_bias, 0.0, 1.0)),
            physiological_bias=float(np.clip(vg_cfg.physiological_bias, 0.0, 1.0)),
            social_bias=float(np.clip(vg_cfg.social_bias, 0.0, 1.0)),
            exploration_bias=float(np.clip(vg_cfg.exploration_bias, 0.0, 1.0)),
            attachment_bias=float(np.clip(vg_cfg.attachment_bias, 0.0, 1.0)),
        )
        overrides = self._persona_conscious_value("value_gradient")
        if overrides:
            return self._apply_value_gradient_overrides(base, overrides)
        return base

    def _build_force_matrix(self) -> ForceMatrix:
        cfg_matrix = getattr(self.config.conscious, "force_matrix", None)
        if isinstance(cfg_matrix, ForceMatrix):
            matrix = cfg_matrix
        elif isinstance(cfg_matrix, Mapping):
            matrix = ForceMatrix.from_mapping(cfg_matrix)
        else:
            matrix = ForceMatrix()
        overrides = self._persona_conscious_value("force_matrix")
        if overrides:
            return matrix.merge(overrides)
        return matrix

    def _current_force_matrix(self) -> ForceMatrix:
        dev_cfg = getattr(self.config.conscious, "development", None)
        if not dev_cfg or not dev_cfg.enabled:
            return self._force_matrix
        turns = max(int(dev_cfg.mature_turns or 0), 1)
        maturity = float(np.clip(self._turn_id / turns, 0.0, 1.0))
        narrative_scale = max(
            0.0,
            dev_cfg.narrative_initial_scale
            + (1.0 - dev_cfg.narrative_initial_scale) * maturity,
        )
        reflex_scale = max(
            0.0,
            1.0 - (1.0 - dev_cfg.reflex_final_scale) * maturity,
        )
        affective_scale = max(
            0.0, 1.0 + dev_cfg.affective_peak_gain * math.sin(math.pi * maturity)
        )
        return ForceMatrix(
            reflex=self._force_matrix.reflex.scaled(reflex_scale),
            affective=self._force_matrix.affective.scaled(affective_scale),
            narrative=self._force_matrix.narrative.scaled(narrative_scale),
        )

    # ------------------------------------------------------------------ #
    # Main entrypoints
    # ------------------------------------------------------------------ #

    def set_listening(self, wait: bool) -> None:
        """Force the runtime into a passive listening mode."""
        self._force_listen = bool(wait)

    def set_engaged(self, engaged: Optional[bool]) -> None:
        """Hint whether the user is actively engaged (e.g., eye contact)."""
        if engaged is None:
            self._engaged_override = None
        else:
            self._engaged_override = bool(engaged)

    def set_heart_params(
        self, *, base_rate: Optional[float] = None, gain: Optional[float] = None
    ) -> None:
        """Allow external callers (e.g., UI sliders) to tune the heart loop."""

        try:
            self._heartbeat_core.set_params(base_rate=base_rate, gain=gain)
        except (TypeError, ValueError):
            pass
        self._heart_base_rate = float(self._heartbeat_core.config.base_rate)
        self._heart_gain = float(self._heartbeat_core.config.gain)

    def _build_default_persona_meta(self) -> Dict[str, Any]:
        persona_cfg = getattr(self.config.mask_layer, "persona", None)
        profile = None
        if load_persona_profile is not None:
            try:
                profile = load_persona_profile(persona_cfg)
            except Exception:
                profile = None
        elif MaskPersonaProfile is not None and isinstance(persona_cfg, dict):
            try:
                profile = MaskPersonaProfile(**persona_cfg)
            except Exception:
                profile = None
        if profile is None:
            return {}
        payload: Dict[str, Any] = {
            "persona_id": getattr(profile, "persona_id", None),
            "display_name": getattr(profile, "display_name", None),
        }
        tone = getattr(profile, "tone_vec", ()) or ()
        if tone:
            payload["tone_vec"] = [float(v) for v in tone]
        return payload

    def on_sensor_tick(self, raw_frame: Dict[str, Any]) -> None:
        """Ingest a raw sensor frame and update the fused snapshot."""
        if not isinstance(raw_frame, dict):
            return
        try:
            self._runtime_sensors.tick(raw_frame)
        except Exception:
            return

    def current_talk_mode(self) -> TalkMode:
        return self._talk_mode

    def load_state(self) -> None:
        if self.eqnet_system is None:
            return
        loader = getattr(self.eqnet_system, "_load_state", None)
        if callable(loader):
            loader()

    def save_state(self) -> None:
        if self.eqnet_system is None:
            return
        self.eqnet_system.save_state()

    def _inner_os_relational_context(self, sensor_metrics: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        sensor_metrics = dict(sensor_metrics or {})
        world_state = dict(self._surface_world_state or {})
        object_counts = sensor_metrics.get("object_counts") or {}
        nearby_objects: List[str] = []
        if isinstance(object_counts, Mapping):
            nearby_objects = [str(key).strip() for key in object_counts.keys() if str(key).strip()][:6]
        place_anchor = self._dominant_memory_anchor() or str(world_state.get("zone_id") or "").strip() or None
        return {
            "culture_id": str(world_state.get("culture_id") or "default"),
            "community_id": str(world_state.get("community_id") or "local"),
            "social_role": str(world_state.get("social_role") or "companion"),
            "place_memory_anchor": place_anchor,
            "nearby_objects": nearby_objects,
        }

    def _inner_os_working_memory_seed(self) -> Dict[str, Any]:
        world_state = dict(self._surface_world_state or {})
        seed = world_state.get("working_memory_seed") or {}
        normalized: Dict[str, Any] = {}
        if isinstance(seed, Mapping):
            for key in (
                "semantic_seed_focus",
                "semantic_seed_anchor",
                "semantic_seed_strength",
                "semantic_seed_recurrence",
                "long_term_theme_focus",
                "long_term_theme_anchor",
                "long_term_theme_strength",
                "long_term_theme_kind",
                "long_term_theme_summary",
                "conscious_residue_focus",
                "conscious_residue_anchor",
                "conscious_residue_summary",
                "conscious_residue_strength",
                "related_person_id",
                "related_person_ids",
                "attachment",
                "familiarity",
                "trust_memory",
                "relation_seed_summary",
                "relation_seed_strength",
                "identity_arc_kind",
                "identity_arc_phase",
                "identity_arc_summary",
                "identity_arc_open_tension",
                "identity_arc_stability",
                "identity_arc_registry_summary",
                "relation_arc_kind",
                "relation_arc_phase",
                "relation_arc_summary",
                "relation_arc_open_tension",
                "relation_arc_stability",
                "relation_arc_registry_summary",
                "group_relation_arc_kind",
                "group_relation_arc_phase",
                "group_relation_arc_summary",
                "group_relation_arc_boundary_mode",
                "group_relation_arc_reentry_window_focus",
                "group_relation_arc_group_thread_id",
                "group_relation_arc_topology_focus",
                "group_relation_arc_dominant_person_id",
                "group_relation_arc_stability",
            ):
                if key in seed:
                    normalized[key] = seed[key]
        if normalized:
            return normalized
        conscious_seed: Dict[str, Any] = {}
        conscious_theme: Dict[str, Any] = {}
        conscious_memory = getattr(self, "_conscious_memory", None)
        if conscious_memory is not None and hasattr(conscious_memory, "latest_working_memory_seed"):
            try:
                conscious_seed = dict(conscious_memory.latest_working_memory_seed(12) or {})
            except Exception:
                conscious_seed = {}
        if conscious_memory is not None and hasattr(conscious_memory, "latest_long_term_theme"):
            try:
                conscious_theme = dict(conscious_memory.latest_long_term_theme(12) or {})
            except Exception:
                conscious_theme = {}
        nightly_theme = self._latest_nightly_long_term_theme_summary()
        nightly_identity_arc = self._latest_nightly_identity_arc_summary()
        nightly_identity_arc_registry = self._latest_nightly_identity_arc_registry_summary()
        nightly_relation_arc = self._latest_nightly_relation_arc_summary()
        nightly_relation_arc_registry = self._latest_nightly_relation_arc_registry_summary()
        nightly_group_relation_arc = self._latest_nightly_group_relation_arc_summary()
        nightly_partner = self._latest_nightly_partner_relation_summary()
        nightly_partner_registry = self._latest_nightly_partner_relation_registry_summary()
        nightly_theme_seed: Dict[str, Any] = {}
        nightly_identity_arc_seed: Dict[str, Any] = {}
        nightly_relation_arc_seed: Dict[str, Any] = {}
        nightly_group_relation_arc_seed: Dict[str, Any] = {}
        nightly_partner_seed: Dict[str, Any] = {}
        if isinstance(nightly_theme, Mapping):
            theme_focus = str(nightly_theme.get("focus") or "").strip()
            theme_anchor = str(nightly_theme.get("anchor") or "").strip()
            theme_strength = round(float(nightly_theme.get("strength") or 0.0), 4)
            if theme_focus or theme_anchor:
                nightly_theme_seed = {
                    "semantic_seed_focus": theme_focus,
                    "semantic_seed_anchor": theme_anchor,
                    "semantic_seed_strength": round(min(theme_strength * 0.82, 1.0), 4),
                    "semantic_seed_recurrence": 0.0,
                    "long_term_theme_focus": theme_focus,
                    "long_term_theme_anchor": theme_anchor,
                    "long_term_theme_strength": theme_strength,
                    "long_term_theme_kind": str(nightly_theme.get("kind") or "ambient").strip(),
                    "long_term_theme_summary": str(nightly_theme.get("summary") or "").strip(),
                }
        if isinstance(nightly_partner, Mapping):
            partner_id = str(nightly_partner.get("person_id") or "").strip()
            partner_strength = round(float(nightly_partner.get("strength") or 0.0), 4)
            if partner_id and partner_strength > 0.0:
                nightly_partner_seed = {
                    "related_person_id": partner_id,
                    "attachment": round(float(nightly_partner.get("attachment") or 0.0), 4),
                    "familiarity": round(float(nightly_partner.get("familiarity") or 0.0), 4),
                    "trust_memory": round(float(nightly_partner.get("trust_memory") or 0.0), 4),
                    "relation_seed_summary": str(nightly_partner.get("summary") or "").strip(),
                    "relation_seed_strength": partner_strength,
                    "partner_address_hint": str(nightly_partner.get("address_hint") or "").strip(),
                    "partner_timing_hint": str(nightly_partner.get("timing_hint") or "").strip(),
                    "partner_stance_hint": str(nightly_partner.get("stance_hint") or "").strip(),
                    "partner_social_interpretation": str(nightly_partner.get("social_interpretation") or "").strip(),
                }
        if isinstance(nightly_partner_registry, Mapping):
            related_person_ids = [
                str(item).strip()
                for item in list(nightly_partner_registry.get("top_person_ids") or [])
                if str(item).strip()
            ]
            if related_person_ids:
                nightly_partner_seed.setdefault("related_person_ids", related_person_ids)
            dominant_person_id = str(nightly_partner_registry.get("dominant_person_id") or "").strip()
            if dominant_person_id and not nightly_partner_seed.get("related_person_id"):
                nightly_partner_seed["related_person_id"] = dominant_person_id
        if isinstance(nightly_identity_arc, Mapping):
            arc_kind = str(nightly_identity_arc.get("arc_kind") or "").strip()
            arc_summary = str(nightly_identity_arc.get("summary") or "").strip()
            if arc_kind or arc_summary:
                nightly_identity_arc_seed = {
                    "identity_arc_kind": arc_kind,
                    "identity_arc_phase": str(nightly_identity_arc.get("phase") or "").strip(),
                    "identity_arc_summary": arc_summary,
                    "identity_arc_open_tension": str(nightly_identity_arc.get("open_tension") or "").strip(),
                    "identity_arc_stability": round(float(nightly_identity_arc.get("stability") or 0.0), 4),
                }
        if isinstance(nightly_identity_arc_registry, Mapping) and int(nightly_identity_arc_registry.get("total_arcs") or 0) > 0:
            nightly_identity_arc_seed["identity_arc_registry_summary"] = dict(nightly_identity_arc_registry)
        if isinstance(nightly_relation_arc, Mapping):
            arc_kind = str(nightly_relation_arc.get("arc_kind") or "").strip()
            arc_summary = str(nightly_relation_arc.get("summary") or "").strip()
            if arc_kind or arc_summary:
                nightly_relation_arc_seed = {
                    "relation_arc_kind": arc_kind,
                    "relation_arc_phase": str(nightly_relation_arc.get("phase") or "").strip(),
                    "relation_arc_summary": arc_summary,
                    "relation_arc_open_tension": str(nightly_relation_arc.get("open_tension") or "").strip(),
                    "relation_arc_stability": round(float(nightly_relation_arc.get("stability") or 0.0), 4),
                }
        if isinstance(nightly_relation_arc_registry, Mapping) and int(nightly_relation_arc_registry.get("total_arcs") or 0) > 0:
            nightly_relation_arc_seed["relation_arc_registry_summary"] = dict(nightly_relation_arc_registry)
        if isinstance(nightly_group_relation_arc, Mapping):
            arc_kind = str(nightly_group_relation_arc.get("arc_kind") or "").strip()
            arc_summary = str(nightly_group_relation_arc.get("summary") or "").strip()
            if arc_kind or arc_summary:
                nightly_group_relation_arc_seed = {
                    "group_relation_arc_kind": arc_kind,
                    "group_relation_arc_phase": str(nightly_group_relation_arc.get("phase") or "").strip(),
                    "group_relation_arc_summary": arc_summary,
                    "group_relation_arc_boundary_mode": str(nightly_group_relation_arc.get("boundary_mode") or "").strip(),
                    "group_relation_arc_reentry_window_focus": str(nightly_group_relation_arc.get("reentry_window_focus") or "").strip(),
                    "group_relation_arc_group_thread_id": str(nightly_group_relation_arc.get("group_thread_id") or "").strip(),
                    "group_relation_arc_topology_focus": str(nightly_group_relation_arc.get("topology_focus") or "").strip(),
                    "group_relation_arc_dominant_person_id": str(nightly_group_relation_arc.get("dominant_person_id") or "").strip(),
                    "group_relation_arc_stability": round(float(nightly_group_relation_arc.get("stability") or 0.0), 4),
                }
        eqnet_system = getattr(self, "eqnet_system", None)
        derived: Dict[str, Any] = {}
        if eqnet_system is not None:
            signature_getter = getattr(eqnet_system, "_working_memory_signature_summary", None)
            replay_getter = getattr(eqnet_system, "_latest_nightly_working_memory_replay_summary", None)
            signature_summary = signature_getter() if callable(signature_getter) else None
            replay_summary = replay_getter() if callable(replay_getter) else None
            payload = derive_working_memory_seed_from_signature(signature_summary, replay_summary)
            derived = dict(payload) if isinstance(payload, Mapping) else {}
        if nightly_theme_seed:
            if not derived:
                derived = dict(nightly_theme_seed)
            else:
                if not derived.get("long_term_theme_focus"):
                    derived["long_term_theme_focus"] = nightly_theme_seed.get("long_term_theme_focus")
                if not derived.get("long_term_theme_anchor"):
                    derived["long_term_theme_anchor"] = nightly_theme_seed.get("long_term_theme_anchor")
                if float(derived.get("long_term_theme_strength") or 0.0) <= 0.0:
                    derived["long_term_theme_strength"] = nightly_theme_seed.get("long_term_theme_strength")
                if not derived.get("long_term_theme_kind"):
                    derived["long_term_theme_kind"] = nightly_theme_seed.get("long_term_theme_kind")
                if not derived.get("long_term_theme_summary"):
                    derived["long_term_theme_summary"] = nightly_theme_seed.get("long_term_theme_summary")
        derived = self._merge_conscious_long_term_theme(derived, conscious_theme)
        if nightly_partner_seed:
            derived.update({k: v for k, v in nightly_partner_seed.items() if k not in derived or not derived.get(k)})
        if nightly_identity_arc_seed:
            derived.update({k: v for k, v in nightly_identity_arc_seed.items() if k not in derived or not derived.get(k)})
        if nightly_relation_arc_seed:
            derived.update({k: v for k, v in nightly_relation_arc_seed.items() if k not in derived or not derived.get(k)})
        if nightly_group_relation_arc_seed:
            derived.update({k: v for k, v in nightly_group_relation_arc_seed.items() if k not in derived or not derived.get(k)})
        if not conscious_seed:
            return derived
        if not derived:
            return {
                "semantic_seed_focus": str(conscious_seed.get("focus") or "").strip(),
                "semantic_seed_anchor": str(conscious_seed.get("anchor") or "").strip(),
                "semantic_seed_strength": round(float(conscious_seed.get("strength") or 0.0), 4),
                "semantic_seed_recurrence": 0.0,
                "conscious_residue_focus": str(conscious_theme.get("focus") or conscious_seed.get("focus") or "").strip(),
                "conscious_residue_anchor": str(conscious_theme.get("anchor") or conscious_seed.get("anchor") or "").strip(),
                "conscious_residue_summary": str(conscious_theme.get("summary") or "").strip(),
                "conscious_residue_strength": round(
                    max(float(conscious_theme.get("strength") or 0.0), float(conscious_seed.get("strength") or 0.0)),
                    4,
                ),
            }
        conscious_strength = float(conscious_seed.get("strength") or 0.0)
        merged_focus = str(derived.get("semantic_seed_focus") or conscious_seed.get("focus") or "").strip()
        merged_anchor = str(derived.get("semantic_seed_anchor") or conscious_seed.get("anchor") or "").strip()
        merged_strength = min(
            1.0,
            max(float(derived.get("semantic_seed_strength") or 0.0), conscious_strength * 0.8),
        )
        merged = dict(derived)
        merged["semantic_seed_focus"] = merged_focus
        merged["semantic_seed_anchor"] = merged_anchor
        merged["semantic_seed_strength"] = round(merged_strength, 4)
        if not merged.get("long_term_theme_focus"):
            merged["long_term_theme_focus"] = merged_focus
        if not merged.get("long_term_theme_anchor"):
            merged["long_term_theme_anchor"] = merged_anchor
        if float(merged.get("long_term_theme_strength") or 0.0) <= 0.0:
            merged["long_term_theme_strength"] = round(merged_strength, 4)
        return merged

    def _merge_conscious_long_term_theme(
        self,
        seed: Mapping[str, Any] | None,
        conscious_theme: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        merged = dict(seed or {})
        theme = dict(conscious_theme or {})
        if not theme:
            return merged
        theme_focus = str(theme.get("focus") or "").strip()
        theme_anchor = str(theme.get("anchor") or "").strip()
        theme_kind = str(theme.get("kind") or "").strip()
        theme_summary = str(theme.get("summary") or "").strip()
        theme_strength = round(float(theme.get("strength") or 0.0), 4)
        if not theme_focus and not theme_anchor and not theme_summary:
            return merged
        seed_focus = str(merged.get("long_term_theme_focus") or merged.get("semantic_seed_focus") or "").strip()
        seed_anchor = str(merged.get("long_term_theme_anchor") or merged.get("semantic_seed_anchor") or "").strip()
        corroborated = bool(
            (theme_focus and seed_focus and theme_focus == seed_focus)
            or (theme_anchor and seed_anchor and theme_anchor == seed_anchor)
        )
        if corroborated:
            if not merged.get("long_term_theme_focus"):
                merged["long_term_theme_focus"] = theme_focus
            if not merged.get("long_term_theme_anchor"):
                merged["long_term_theme_anchor"] = theme_anchor
            if not merged.get("long_term_theme_kind"):
                merged["long_term_theme_kind"] = theme_kind
            if not merged.get("long_term_theme_summary"):
                merged["long_term_theme_summary"] = theme_summary
            merged["long_term_theme_strength"] = round(
                max(float(merged.get("long_term_theme_strength") or 0.0), theme_strength),
                4,
            )
            merged["conscious_residue_focus"] = theme_focus
            merged["conscious_residue_anchor"] = theme_anchor
            merged["conscious_residue_summary"] = theme_summary
            merged["conscious_residue_strength"] = round(theme_strength, 4)
            merged["conscious_residue_corroborated"] = True
            return merged
        merged["conscious_residue_focus"] = theme_focus
        merged["conscious_residue_anchor"] = theme_anchor
        merged["conscious_residue_summary"] = theme_summary
        merged["conscious_residue_strength"] = round(theme_strength, 4)
        merged["conscious_residue_corroborated"] = False
        return merged

    def _latest_nightly_long_term_theme_summary(self) -> Dict[str, Any]:
        for candidate in (Path("reports/nightly.json"), Path("reports/nightly") / "nightly.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("inner_os_long_term_theme_summary")
            if isinstance(summary, Mapping) and (
                str(summary.get("focus") or "").strip()
                or str(summary.get("anchor") or "").strip()
                or str(summary.get("summary") or "").strip()
            ):
                return dict(summary)
        return {}

    def _latest_nightly_identity_arc_summary(self) -> Dict[str, Any]:
        for candidate in (Path("reports/nightly.json"), Path("reports/nightly") / "nightly.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("inner_os_identity_arc_summary")
            if isinstance(summary, Mapping) and (
                str(summary.get("arc_kind") or "").strip()
                or str(summary.get("summary") or "").strip()
                or str(summary.get("memory_anchor") or "").strip()
            ):
                return dict(summary)
        return {}

    def _latest_nightly_identity_arc_registry_summary(self) -> Dict[str, Any]:
        for candidate in (Path("reports/nightly.json"), Path("reports/nightly") / "nightly.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("inner_os_identity_arc_registry_summary")
            if isinstance(summary, Mapping) and int(summary.get("total_arcs") or 0) > 0:
                return dict(summary)
        return {}

    def _latest_nightly_relation_arc_summary(self) -> Dict[str, Any]:
        for candidate in (Path("reports/nightly.json"), Path("reports/nightly") / "nightly.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("inner_os_relation_arc_summary")
            if isinstance(summary, Mapping) and (
                str(summary.get("arc_kind") or "").strip()
                or str(summary.get("summary") or "").strip()
                or str(summary.get("related_person_id") or "").strip()
                or str(summary.get("group_thread_id") or "").strip()
            ):
                return dict(summary)
        return {}

    def _latest_nightly_relation_arc_registry_summary(self) -> Dict[str, Any]:
        for candidate in (Path("reports/nightly.json"), Path("reports/nightly") / "nightly.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("inner_os_relation_arc_registry_summary")
            if isinstance(summary, Mapping) and int(summary.get("total_arcs") or 0) > 0:
                return dict(summary)
        return {}

    def _latest_nightly_group_relation_arc_summary(self) -> Dict[str, Any]:
        for candidate in (Path("reports/nightly.json"), Path("reports/nightly") / "nightly.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("inner_os_group_relation_arc_summary")
            if isinstance(summary, Mapping) and (
                str(summary.get("arc_kind") or "").strip()
                or str(summary.get("summary") or "").strip()
                or str(summary.get("group_thread_id") or "").strip()
            ):
                return dict(summary)
        return {}

    def _latest_nightly_partner_relation_summary(self) -> Dict[str, Any]:
        for candidate in (Path("reports/nightly.json"), Path("reports/nightly") / "nightly.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("inner_os_partner_relation_summary")
            if isinstance(summary, Mapping) and str(summary.get("person_id") or "").strip():
                return dict(summary)
        return {}

    def _latest_nightly_partner_relation_registry_summary(self) -> Dict[str, Any]:
        for candidate in (Path("reports/nightly.json"), Path("reports/nightly") / "nightly.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("inner_os_partner_relation_registry_summary")
            if isinstance(summary, Mapping) and dict(summary.get("persons") or {}):
                return dict(summary)
        return {}

    def process_turn(
        self,
        user_text: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        image_path: Optional[str] = None,
        transferred_lessons: Optional[list[Mapping[str, Any]]] = None,
    ) -> RuntimeTurnResult:
        self._last_surface_user_text = str(user_text or "").strip()
        sensor_snapshot = getattr(self._runtime_sensors, "snapshot", None)
        sensor_metrics = (
            getattr(sensor_snapshot, "metrics", {}) if sensor_snapshot is not None else {}
        )
        working_memory_seed = self._inner_os_working_memory_seed()
        try:
            safety_bias = float(self._surface_safety_bias())
        except Exception:
            safety_bias = 0.0
        pre_hook = self._integration_hooks.pre_turn_update(
            user_input={"text": user_text or ""},
            sensor_input=sensor_metrics,
            local_context={
                "last_shadow_estimate": self._last_shadow_estimate,
                "last_gate_context": self._last_gate_context,
                "relational_world": self._inner_os_relational_context(sensor_metrics),
                "working_memory_seed": working_memory_seed,
                "person_registry": dict(self._last_gate_context.get("inner_os_person_registry_snapshot") or {}),
            },
            current_state={
                "current_energy": float(getattr(self._self_model, "current_energy", 0.7) or 0.7),
                "temporal_pressure": float(self._last_gate_context.get("inner_os_temporal_pressure") or 0.0),
                "social_update_strength": float(self._last_gate_context.get("inner_os_social_update_strength") or 1.0),
                "identity_update_strength": float(self._last_gate_context.get("inner_os_identity_update_strength") or 1.0),
                "interaction_afterglow": float(self._last_gate_context.get("inner_os_interaction_afterglow") or 0.0),
                "interaction_afterglow_intent": self._last_gate_context.get("inner_os_interaction_afterglow_intent"),
                "replay_intensity": float(self._last_gate_context.get("inner_os_replay_intensity") or 0.0),
                "anticipation_tension": float(self._last_gate_context.get("inner_os_anticipation_tension") or 0.0),
                "stabilization_drive": float(self._last_gate_context.get("inner_os_stabilization_drive") or 0.0),
                "relational_clarity": float(self._last_gate_context.get("inner_os_relational_clarity") or 0.0),
                "meaning_inertia": float(self._last_gate_context.get("inner_os_meaning_inertia") or 0.0),
                "recovery_reopening": float(self._last_gate_context.get("inner_os_recovery_reopening") or 0.0),
                "conscious_workspace": dict(self._last_gate_context.get("inner_os_conscious_workspace") or {}),
                "conscious_residue_focus": self._last_gate_context.get("inner_os_conscious_residue_focus"),
                "conscious_residue_anchor": self._last_gate_context.get("inner_os_conscious_residue_anchor"),
                "conscious_residue_summary": self._last_gate_context.get("inner_os_conscious_residue_summary"),
                "conscious_residue_strength": float(self._last_gate_context.get("inner_os_conscious_residue_strength") or 0.0),
                "future_signal": max(float(getattr(self, "_last_future_risk", 0.0) or 0.0) - float(getattr(self, "_last_future_hope", 0.0) or 0.0) * 0.35, 0.0),
            },
            safety_bias=safety_bias,
        )
        expression_hint = pre_hook.interaction_hints.get("expression_hint") if isinstance(pre_hook.interaction_hints, Mapping) else {}
        context_for_step = context
        if isinstance(expression_hint, Mapping):
            inner_os_context_hint = _inner_os_expression_context_line(expression_hint)
            if inner_os_context_hint:
                context_for_step = f"{inner_os_context_hint}\n\n{context.strip()}" if context else inner_os_context_hint
        self._last_gate_context["inner_os_expression_hint"] = dict(expression_hint) if isinstance(expression_hint, Mapping) else {}
        self._last_gate_context["inner_os_working_memory_seed"] = dict(working_memory_seed)
        payload = self.step(
            user_text=user_text,
            intent=intent,
            context=context_for_step,
            fast_only=fast_only,
            image_path=image_path,
        )
        result = RuntimeTurnResult.from_payload(payload)
        result.metrics.setdefault("inner_os/stress", round(float(pre_hook.state.stress), 4))
        result.metrics.setdefault("inner_os/recovery_need", round(float(pre_hook.state.recovery_need), 4))
        result.metrics.setdefault(
            "inner_os/temporal_pressure_before",
            round(float(pre_hook.state.temporal_pressure), 4),
        )
        result.metrics.setdefault(
            "inner_os/semantic_seed_strength",
            round(float(working_memory_seed.get("semantic_seed_strength") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/semantic_seed_recurrence",
            round(float(working_memory_seed.get("semantic_seed_recurrence") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/long_term_theme_strength",
            round(float(working_memory_seed.get("long_term_theme_strength") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/identity_arc_stability",
            round(float(working_memory_seed.get("identity_arc_stability") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/identity_arc_visible",
            1.0
            if str(working_memory_seed.get("identity_arc_kind") or "").strip()
            or str(working_memory_seed.get("identity_arc_summary") or "").strip()
            else 0.0,
        )
        llm_raw_text = str(payload.get("llm_bridge_raw_text") or "").strip()
        llm_raw_model = str(payload.get("llm_bridge_raw_model") or "").strip()
        llm_raw_model_source = str(payload.get("llm_bridge_raw_model_source") or "").strip()
        llm_final_text = (
            str(getattr(result.response, "text", None) or "").strip()
            if result.response is not None
            else ""
        )
        qualia_gate_snapshot = dict(result.qualia_gate or {})

        visual_cue = ""
        retrieval_summary = None
        if result.response is not None:
            retrieval_summary = dict(result.response.retrieval_summary or {})
            if isinstance(result.response.perception_summary, dict):
                visual_cue = str(result.response.perception_summary.get("text") or "").strip()
        recall_state = pre_hook.state.to_dict()
        conscious_memory = getattr(self, "_conscious_memory", None)
        if conscious_memory is not None and hasattr(conscious_memory, "tail"):
            try:
                tail = list(conscious_memory.tail(6))
            except Exception:
                tail = []
            recall_state["conscious_mosaic_density"] = round(min(len(tail) / 6.0, 1.0), 4)
            recall_state["conscious_mosaic_recentness"] = round(1.0 if tail else 0.0, 4)
        conscious_seed = {}
        if conscious_memory is not None and hasattr(conscious_memory, "latest_working_memory_seed"):
            try:
                conscious_seed = dict(conscious_memory.latest_working_memory_seed(12) or {})
            except Exception:
                conscious_seed = {}
        if conscious_seed:
            recall_state["working_memory_replay_focus"] = str(conscious_seed.get("focus") or "").strip()
            recall_state["working_memory_replay_anchor"] = str(conscious_seed.get("anchor") or "").strip()
            recall_state["working_memory_replay_strength"] = float(conscious_seed.get("strength") or 0.0)
            result.metrics.setdefault(
                "inner_os/conscious_seed_strength",
                round(float(conscious_seed.get("strength") or 0.0), 4),
            )
        recall_hook = self._integration_hooks.memory_recall(
            text_cue=str(user_text or "").strip(),
            visual_cue=visual_cue,
            world_cue=str(self._surface_world_state.get("zone_id") or "").strip(),
            current_state=recall_state,
            retrieval_summary=retrieval_summary,
        )
        current_state = pre_hook.state.to_dict()
        current_state.update(self._inner_os_relational_context(sensor_metrics))
        current_state["memory_anchor"] = recall_hook.recall_payload.get("memory_anchor")
        current_state["mode"] = self._surface_mode()
        current_state["recalled_tentative_bias"] = float(recall_hook.recall_payload.get("tentative_bias") or 0.0)
        current_state["recalled_reinterpretation_mode"] = str(recall_hook.recall_payload.get("reinterpretation_mode") or "").strip() or None
        current_state["route"] = str(result.response_route or pre_hook.state.route)
        current_state["talk_mode"] = str(result.talk_mode or pre_hook.state.talk_mode)
        current_state["surface_user_text"] = str(user_text or "").strip()
        recent_dialogue_history_fn = getattr(self, "_recent_dialogue_thread_history", None)
        if callable(recent_dialogue_history_fn):
            current_state["recent_dialogue_history"] = recent_dialogue_history_fn()
        else:
            current_state["recent_dialogue_history"] = self._recent_surface_user_history()
        current_state["contact_dynamics"] = dict(self._last_gate_context.get("inner_os_contact_dynamics") or {})
        current_state["access_dynamics"] = dict(self._last_gate_context.get("inner_os_access_dynamics") or {})
        current_state["conscious_workspace"] = dict(self._last_gate_context.get("inner_os_conscious_workspace") or {})
        current_state["conversational_objects"] = dict(self._last_gate_context.get("inner_os_conversational_objects") or {})
        current_state["object_operations"] = dict(self._last_gate_context.get("inner_os_object_operations") or {})
        current_state["interaction_effects"] = dict(self._last_gate_context.get("inner_os_interaction_effects") or {})
        current_state["interaction_judgement_view"] = dict(self._last_gate_context.get("inner_os_interaction_judgement_view") or {})
        current_state["interaction_judgement_summary"] = dict(self._last_gate_context.get("inner_os_interaction_judgement_summary") or {})
        current_state["interaction_condition_report"] = dict(self._last_gate_context.get("inner_os_interaction_condition_report") or {})
        current_state["interaction_inspection_report"] = dict(self._last_gate_context.get("inner_os_interaction_inspection_report") or {})
        current_state["interaction_audit_bundle"] = dict(self._last_gate_context.get("inner_os_interaction_audit_bundle") or {})
        current_state["interaction_audit_casebook"] = dict(self._last_gate_context.get("inner_os_interaction_audit_casebook") or {})
        current_state["interaction_audit_report"] = dict(self._last_gate_context.get("inner_os_interaction_audit_report") or {})
        current_state["prev_qualia"] = list(self._last_gate_context.get("inner_os_prev_qualia") or [])
        current_state["prev_qualia_habituation"] = list(self._last_gate_context.get("inner_os_prev_qualia_habituation") or [])
        current_state["prev_protection_grad_x"] = list(self._last_gate_context.get("inner_os_prev_protection_grad_x") or [])
        current_state["prev_affective_position"] = dict(self._last_gate_context.get("inner_os_prev_affective_position") or {})
        current_state["affective_terrain_state"] = dict(self._last_gate_context.get("inner_os_affective_terrain_state") or {})
        current_state["association_graph_state"] = dict(self._last_gate_context.get("inner_os_association_graph_state") or {})
        current_state["person_registry_snapshot"] = dict(self._last_gate_context.get("inner_os_person_registry_snapshot") or {})
        current_state["group_thread_registry_snapshot"] = dict(self._last_gate_context.get("inner_os_group_thread_registry_snapshot") or {})
        current_state["social_topology_state"] = dict(
            (recall_state or {}).get("social_topology_state")
            or self._last_gate_context.get("inner_os_social_topology_state")
            or {}
        )
        recalled_terrain_bias = (recall_state or {}).get("terrain_reweighting_bias")
        recalled_association_bias = (recall_state or {}).get("association_reweighting_bias")
        recalled_association_focus = (recall_state or {}).get("association_reweighting_focus")
        recalled_association_reason = (recall_state or {}).get("association_reweighting_reason")
        recalled_insight_reframing_bias = (recall_state or {}).get("insight_reframing_bias")
        recalled_insight_class_focus = (recall_state or {}).get("insight_class_focus")
        recalled_insight_terrain_shape_target = (recall_state or {}).get("insight_terrain_shape_target")
        recalled_insight_link_counts = (recall_state or {}).get("insight_link_counts")
        recalled_insight_class_counts = (recall_state or {}).get("insight_class_counts")
        recalled_initiative_followup_bias = (recall_state or {}).get("initiative_followup_bias")
        recalled_initiative_followup_state = (recall_state or {}).get("initiative_followup_state")
        recalled_commitment_target_focus = (recall_state or {}).get("commitment_target_focus")
        recalled_commitment_state_focus = (recall_state or {}).get("commitment_state_focus")
        recalled_commitment_carry_bias = (recall_state or {}).get("commitment_carry_bias")
        recalled_commitment_followup_focus = (recall_state or {}).get("commitment_followup_focus")
        recalled_commitment_mode_focus = (recall_state or {}).get("commitment_mode_focus")
        recalled_commitment_carry_reason = (recall_state or {}).get("commitment_carry_reason")
        recalled_agenda_focus = (recall_state or {}).get("agenda_focus")
        recalled_agenda_bias = (recall_state or {}).get("agenda_bias")
        recalled_agenda_reason = (recall_state or {}).get("agenda_reason")
        recalled_agenda_window_focus = (recall_state or {}).get("agenda_window_focus")
        recalled_agenda_window_bias = (recall_state or {}).get("agenda_window_bias")
        recalled_agenda_window_reason = (recall_state or {}).get("agenda_window_reason")
        recalled_learning_mode_focus = (recall_state or {}).get("learning_mode_focus")
        recalled_learning_mode_carry_bias = (recall_state or {}).get("learning_mode_carry_bias")
        recalled_social_experiment_focus = (recall_state or {}).get("social_experiment_focus")
        recalled_social_experiment_carry_bias = (recall_state or {}).get("social_experiment_carry_bias")
        recalled_temporal_membrane_focus = (recall_state or {}).get("temporal_membrane_focus")
        recalled_temporal_timeline_bias = (recall_state or {}).get("temporal_timeline_bias")
        recalled_temporal_reentry_bias = (recall_state or {}).get("temporal_reentry_bias")
        recalled_temporal_supersession_bias = (recall_state or {}).get("temporal_supersession_bias")
        recalled_temporal_continuity_bias = (recall_state or {}).get("temporal_continuity_bias")
        recalled_temporal_relation_reentry_bias = (recall_state or {}).get("temporal_relation_reentry_bias")
        recalled_temperament_trace = (recall_state or {}).get("temperament_trace")
        recalled_body_homeostasis_focus = (recall_state or {}).get("body_homeostasis_focus")
        recalled_body_homeostasis_carry_bias = (recall_state or {}).get("body_homeostasis_carry_bias")
        recalled_homeostasis_budget_state = (recall_state or {}).get("homeostasis_budget_state")
        recalled_homeostasis_budget_focus = (recall_state or {}).get("homeostasis_budget_focus")
        recalled_homeostasis_budget_bias = (recall_state or {}).get("homeostasis_budget_bias")
        recalled_relational_continuity_focus = (recall_state or {}).get("relational_continuity_focus")
        recalled_relational_continuity_carry_bias = (recall_state or {}).get("relational_continuity_carry_bias")
        recalled_group_thread_registry_snapshot = (recall_state or {}).get("group_thread_registry_snapshot")
        recalled_discussion_thread_registry_snapshot = (recall_state or {}).get("discussion_thread_registry_snapshot")
        recalled_group_thread_focus = (recall_state or {}).get("group_thread_focus")
        recalled_group_thread_carry_bias = (recall_state or {}).get("group_thread_carry_bias")
        recalled_expressive_style_focus = (recall_state or {}).get("expressive_style_focus")
        recalled_expressive_style_carry_bias = (recall_state or {}).get("expressive_style_carry_bias")
        recalled_expressive_style_history_focus = (recall_state or {}).get("expressive_style_history_focus")
        recalled_expressive_style_history_bias = (recall_state or {}).get("expressive_style_history_bias")
        recalled_banter_style_focus = (recall_state or {}).get("banter_style_focus")
        recalled_lexical_variation_carry_bias = (recall_state or {}).get("lexical_variation_carry_bias")
        current_state["terrain_reweighting_bias"] = float(
            recalled_terrain_bias
            if recalled_terrain_bias is not None
            else self._last_gate_context.get("inner_os_terrain_reweighting_bias") or 0.0
        )
        current_state["association_reweighting_bias"] = float(
            recalled_association_bias
            if recalled_association_bias is not None
            else self._last_gate_context.get("inner_os_association_reweighting_bias") or 0.0
        )
        current_state["association_reweighting_focus"] = str(
            recalled_association_focus
            if recalled_association_focus is not None
            else self._last_gate_context.get("inner_os_association_reweighting_focus") or ""
        ).strip()
        current_state["association_reweighting_reason"] = str(
            recalled_association_reason
            if recalled_association_reason is not None
            else self._last_gate_context.get("inner_os_association_reweighting_reason") or ""
        ).strip()
        current_state["insight_reframing_bias"] = float(
            recalled_insight_reframing_bias
            if recalled_insight_reframing_bias is not None
            else self._last_gate_context.get("inner_os_insight_reframing_bias") or 0.0
        )
        current_state["insight_class_focus"] = str(
            recalled_insight_class_focus
            if recalled_insight_class_focus is not None
            else self._last_gate_context.get("inner_os_insight_class_focus") or ""
        ).strip()
        current_state["insight_terrain_shape_target"] = str(
            recalled_insight_terrain_shape_target
            if recalled_insight_terrain_shape_target is not None
            else self._last_gate_context.get("inner_os_insight_terrain_shape_target") or ""
        ).strip()
        current_state["insight_link_counts"] = dict(
            recalled_insight_link_counts
            if isinstance(recalled_insight_link_counts, dict)
            else self._last_gate_context.get("inner_os_insight_link_counts") or {}
        )
        current_state["insight_class_counts"] = dict(
            recalled_insight_class_counts
            if isinstance(recalled_insight_class_counts, dict)
            else self._last_gate_context.get("inner_os_insight_class_counts") or {}
        )
        current_state["initiative_followup_bias"] = float(
            recalled_initiative_followup_bias
            if recalled_initiative_followup_bias is not None
            else self._last_gate_context.get("inner_os_initiative_followup_bias") or 0.0
        )
        current_state["initiative_followup_state"] = str(
            recalled_initiative_followup_state
            if recalled_initiative_followup_state is not None
            else self._last_gate_context.get("inner_os_initiative_followup_state") or "hold"
        ).strip() or "hold"
        current_state["commitment_target_focus"] = str(
            recalled_commitment_target_focus
            if recalled_commitment_target_focus is not None
            else self._last_gate_context.get("inner_os_commitment_target_focus") or ""
        ).strip()
        current_state["commitment_state_focus"] = str(
            recalled_commitment_state_focus
            if recalled_commitment_state_focus is not None
            else self._last_gate_context.get("inner_os_commitment_state_focus") or "waver"
        ).strip() or "waver"
        current_state["commitment_carry_bias"] = float(
            recalled_commitment_carry_bias
            if recalled_commitment_carry_bias is not None
            else self._last_gate_context.get("inner_os_commitment_carry_bias") or 0.0
        )
        current_state["commitment_followup_focus"] = str(
            recalled_commitment_followup_focus
            if recalled_commitment_followup_focus is not None
            else self._last_gate_context.get("inner_os_commitment_followup_focus") or ""
        ).strip()
        current_state["commitment_mode_focus"] = str(
            recalled_commitment_mode_focus
            if recalled_commitment_mode_focus is not None
            else self._last_gate_context.get("inner_os_commitment_mode_focus") or ""
        ).strip()
        current_state["commitment_carry_reason"] = str(
            recalled_commitment_carry_reason
            if recalled_commitment_carry_reason is not None
            else self._last_gate_context.get("inner_os_commitment_carry_reason") or ""
        ).strip()
        current_state["agenda_focus"] = str(
            recalled_agenda_focus
            if recalled_agenda_focus is not None
            else self._last_gate_context.get("inner_os_agenda_focus") or ""
        ).strip()
        current_state["agenda_bias"] = float(
            recalled_agenda_bias
            if recalled_agenda_bias is not None
            else self._last_gate_context.get("inner_os_agenda_bias") or 0.0
        )
        current_state["agenda_reason"] = str(
            recalled_agenda_reason
            if recalled_agenda_reason is not None
            else self._last_gate_context.get("inner_os_agenda_reason") or ""
        ).strip()
        current_state["agenda_window_focus"] = str(
            recalled_agenda_window_focus
            if recalled_agenda_window_focus is not None
            else self._last_gate_context.get("inner_os_agenda_window_focus") or ""
        ).strip()
        current_state["agenda_window_bias"] = float(
            recalled_agenda_window_bias
            if recalled_agenda_window_bias is not None
            else self._last_gate_context.get("inner_os_agenda_window_bias") or 0.0
        )
        current_state["agenda_window_reason"] = str(
            recalled_agenda_window_reason
            if recalled_agenda_window_reason is not None
            else self._last_gate_context.get("inner_os_agenda_window_reason") or ""
        ).strip()
        current_state["learning_mode_focus"] = str(
            recalled_learning_mode_focus
            if recalled_learning_mode_focus is not None
            else self._last_gate_context.get("inner_os_learning_mode_focus") or ""
        ).strip()
        current_state["learning_mode_carry_bias"] = float(
            recalled_learning_mode_carry_bias
            if recalled_learning_mode_carry_bias is not None
            else self._last_gate_context.get("inner_os_learning_mode_carry_bias") or 0.0
        )
        current_state["social_experiment_focus"] = str(
            recalled_social_experiment_focus
            if recalled_social_experiment_focus is not None
            else self._last_gate_context.get("inner_os_social_experiment_focus") or ""
        ).strip()
        current_state["social_experiment_carry_bias"] = float(
            recalled_social_experiment_carry_bias
            if recalled_social_experiment_carry_bias is not None
            else self._last_gate_context.get("inner_os_social_experiment_carry_bias") or 0.0
        )
        current_state["temporal_membrane_focus"] = str(
            recalled_temporal_membrane_focus
            if recalled_temporal_membrane_focus is not None
            else self._last_gate_context.get("inner_os_temporal_membrane_focus") or ""
        ).strip()
        current_state["temporal_timeline_bias"] = float(
            recalled_temporal_timeline_bias
            if recalled_temporal_timeline_bias is not None
            else self._last_gate_context.get("inner_os_temporal_timeline_bias") or 0.0
        )
        current_state["temporal_reentry_bias"] = float(
            recalled_temporal_reentry_bias
            if recalled_temporal_reentry_bias is not None
            else self._last_gate_context.get("inner_os_temporal_reentry_bias") or 0.0
        )
        current_state["temporal_supersession_bias"] = float(
            recalled_temporal_supersession_bias
            if recalled_temporal_supersession_bias is not None
            else self._last_gate_context.get("inner_os_temporal_supersession_bias") or 0.0
        )
        current_state["temporal_continuity_bias"] = float(
            recalled_temporal_continuity_bias
            if recalled_temporal_continuity_bias is not None
            else self._last_gate_context.get("inner_os_temporal_continuity_bias") or 0.0
        )
        current_state["temporal_relation_reentry_bias"] = float(
            recalled_temporal_relation_reentry_bias
            if recalled_temporal_relation_reentry_bias is not None
            else self._last_gate_context.get("inner_os_temporal_relation_reentry_bias") or 0.0
        )
        current_state["body_homeostasis_focus"] = str(
            recalled_body_homeostasis_focus
            if recalled_body_homeostasis_focus is not None
            else self._last_gate_context.get("inner_os_body_homeostasis_focus") or ""
        ).strip()
        current_state["body_homeostasis_carry_bias"] = float(
            recalled_body_homeostasis_carry_bias
            if recalled_body_homeostasis_carry_bias is not None
            else self._last_gate_context.get("inner_os_body_homeostasis_carry_bias") or 0.0
        )
        current_state["homeostasis_budget_focus"] = str(
            recalled_homeostasis_budget_focus
            if recalled_homeostasis_budget_focus is not None
            else self._last_gate_context.get("inner_os_homeostasis_budget_focus") or ""
        ).strip()
        current_state["homeostasis_budget_bias"] = float(
            recalled_homeostasis_budget_bias
            if recalled_homeostasis_budget_bias is not None
            else self._last_gate_context.get("inner_os_homeostasis_budget_bias") or 0.0
        )
        current_state["homeostasis_budget_state"] = dict(
            recalled_homeostasis_budget_state
            if isinstance(recalled_homeostasis_budget_state, Mapping)
            else self._last_gate_context.get("inner_os_homeostasis_budget_state") or {}
        )
        current_state["relational_continuity_focus"] = str(
            recalled_relational_continuity_focus
            if recalled_relational_continuity_focus is not None
            else self._last_gate_context.get("inner_os_relational_continuity_focus") or ""
        ).strip()
        current_state["relational_continuity_carry_bias"] = float(
            recalled_relational_continuity_carry_bias
            if recalled_relational_continuity_carry_bias is not None
            else self._last_gate_context.get("inner_os_relational_continuity_carry_bias") or 0.0
        )
        current_state["group_thread_registry_snapshot"] = dict(
            recalled_group_thread_registry_snapshot
            if isinstance(recalled_group_thread_registry_snapshot, Mapping)
            else self._last_gate_context.get("inner_os_group_thread_registry_snapshot") or {}
        )
        current_state["discussion_thread_registry_snapshot"] = dict(
            recalled_discussion_thread_registry_snapshot
            if isinstance(recalled_discussion_thread_registry_snapshot, Mapping)
            else self._last_gate_context.get("inner_os_discussion_thread_registry_snapshot") or {}
        )
        current_state["autobiographical_thread_mode"] = str(
            current_state.get("autobiographical_thread_mode")
            or self._last_gate_context.get("inner_os_autobiographical_thread_mode")
            or ""
        ).strip()
        current_state["autobiographical_thread_anchor"] = str(
            current_state.get("autobiographical_thread_anchor")
            or self._last_gate_context.get("inner_os_autobiographical_thread_anchor")
            or ""
        ).strip()
        current_state["autobiographical_thread_focus"] = str(
            current_state.get("autobiographical_thread_focus")
            or self._last_gate_context.get("inner_os_autobiographical_thread_focus")
            or ""
        ).strip()
        current_state["autobiographical_thread_strength"] = float(
            current_state.get("autobiographical_thread_strength")
            or self._last_gate_context.get("inner_os_autobiographical_thread_strength")
            or 0.0
        )
        current_state["group_thread_focus"] = str(
            recalled_group_thread_focus
            if recalled_group_thread_focus is not None
            else self._last_gate_context.get("inner_os_group_thread_focus") or ""
        ).strip()
        current_state["group_thread_carry_bias"] = float(
            recalled_group_thread_carry_bias
            if recalled_group_thread_carry_bias is not None
            else self._last_gate_context.get("inner_os_group_thread_carry_bias") or 0.0
        )
        current_state["expressive_style_focus"] = str(
            recalled_expressive_style_focus
            if recalled_expressive_style_focus is not None
            else self._last_gate_context.get("inner_os_expressive_style_focus") or ""
        ).strip()
        current_state["expressive_style_carry_bias"] = float(
            recalled_expressive_style_carry_bias
            if recalled_expressive_style_carry_bias is not None
            else self._last_gate_context.get("inner_os_expressive_style_carry_bias") or 0.0
        )
        current_state["expressive_style_history_focus"] = str(
            recalled_expressive_style_history_focus
            if recalled_expressive_style_history_focus is not None
            else self._last_gate_context.get("inner_os_expressive_style_history_focus") or ""
        ).strip()
        current_state["expressive_style_history_bias"] = float(
            recalled_expressive_style_history_bias
            if recalled_expressive_style_history_bias is not None
            else self._last_gate_context.get("inner_os_expressive_style_history_bias") or 0.0
        )
        current_state["banter_style_focus"] = str(
            recalled_banter_style_focus
            if recalled_banter_style_focus is not None
            else self._last_gate_context.get("inner_os_banter_style_focus") or ""
        ).strip()
        current_state["lexical_variation_carry_bias"] = float(
            recalled_lexical_variation_carry_bias
            if recalled_lexical_variation_carry_bias is not None
            else self._last_gate_context.get("inner_os_lexical_variation_carry_bias") or 0.0
        )
        temperament_trace = (
            recalled_temperament_trace
            if isinstance(recalled_temperament_trace, dict)
            else self._last_gate_context.get("inner_os_temperament_trace") or {}
        )
        for key in (
            "temperament_forward_trace",
            "temperament_guard_trace",
            "temperament_bond_trace",
            "temperament_recovery_trace",
        ):
            current_state[key] = float(dict(temperament_trace).get(key) or 0.0)
        response_hook = self._integration_hooks.response_gate(
            draft={"text": getattr(result.response, "text", None)},
            current_state=current_state,
            safety_signals={"safety_bias": safety_bias},
        )
        response_hook.expression_hints = build_expression_hints_from_gate_result(
            response_hook,
            existing_hints=response_hook.expression_hints,
            expected_source="shared",
        )
        llm_raw_text_hint = str(
            getattr(self, "_last_guarded_narrative_bridge_text", "") or ""
        ).strip()
        if llm_raw_text_hint:
            response_hook.expression_hints["llm_raw_text"] = llm_raw_text_hint
            response_hook.expression_hints["allow_guarded_narrative_bridge"] = bool(
                getattr(self, "_last_guarded_narrative_bridge_allowed", False)
            )
        current_state["conscious_workspace"] = dict(response_hook.expression_hints.get("conscious_workspace") or {})
        headless_actuation = self._inner_os_headless_runtime.step(
            actuation_plan=response_hook.expression_hints.get("actuation_plan"),
        )
        if result.response is not None:
            result.response, response_hook, live_response_steps = self._run_inner_os_live_response_loop(
                response=result.response,
                initial_hook=response_hook,
                current_state=current_state,
                safety_bias=safety_bias,
            )
            merged_retrieval = dict(result.response.retrieval_summary or {})
            merged_retrieval.setdefault("inner_os", {})
            merged_retrieval["inner_os"].update(recall_hook.ignition_hints)
            result.response.retrieval_summary = merged_retrieval
            merged_controls = dict(getattr(result.response, "controls", {}) or {})
            merged_controls_used = dict(
                getattr(result.response, "controls_used", {}) or {}
            )
            for key, value in merged_controls_used.items():
                merged_controls.setdefault(key, value)
            inner_os_controls = response_hook.to_dict()
            inner_os_controls["live_response_steps"] = live_response_steps
            inner_os_controls["surface_policy_level"] = self._inner_os_surface_policy_level(getattr(result.response, "controls_used", {}) or {})
            inner_os_controls["surface_policy_intent"] = self._inner_os_surface_policy_intent(getattr(result.response, "controls_used", {}) or {})
            inner_os_controls["headless_actuation"] = headless_actuation.to_dict()
            merged_controls["inner_os"] = inner_os_controls
            result.response.controls = merged_controls
        result.qualia_gate.setdefault("inner_os", response_hook.to_dict())
        result.metrics.setdefault(
            "inner_os/allowed_surface_intensity",
            round(float(response_hook.allowed_surface_intensity), 4),
        )
        result.metrics.setdefault(
            "inner_os/hesitation_bias",
            round(float(response_hook.hesitation_bias), 4),
        )
        result.metrics.setdefault(
            "inner_os/stream_update_count",
            float(response_hook.expression_hints.get("stream_update_count") or 0.0),
        )
        result.metrics.setdefault(
            "inner_os/headless_wait_required",
            1.0 if headless_actuation.wait_before_action in {"extended", "held"} else 0.0,
        )
        surface_policy_level = self._inner_os_surface_policy_level(getattr(result.response, "controls_used", {}) if result.response is not None else {})
        surface_policy_intent = self._inner_os_surface_policy_intent(getattr(result.response, "controls_used", {}) if result.response is not None else {}) or ""
        result.metrics.setdefault("inner_os/surface_policy_active", 0.0 if surface_policy_level == "none" else 1.0)
        result.metrics.setdefault("inner_os/surface_policy_layered", 1.0 if surface_policy_level == "layered" else 0.0)
        result.metrics.setdefault("inner_os/surface_policy_intent_clarify", 1.0 if surface_policy_intent == "clarify" else 0.0)
        result.metrics.setdefault("inner_os/surface_policy_intent_check_in", 1.0 if surface_policy_intent == "check_in" else 0.0)
        boundary_transform = dict(response_hook.expression_hints.get("boundary_transform") or {})
        residual_reflection = dict(response_hook.expression_hints.get("residual_reflection") or {})
        current_state["surface_policy_active"] = 0.0 if surface_policy_level == "none" else 1.0
        current_state["surface_policy_level"] = surface_policy_level
        current_state["surface_policy_intent"] = surface_policy_intent or None
        current_state["boundary_gate_mode"] = str(boundary_transform.get("gate_mode") or "").strip()
        current_state["boundary_transform_mode"] = str(boundary_transform.get("transformation_mode") or "").strip()
        current_state["boundary_authority_scope"] = str(boundary_transform.get("authority_scope") or "").strip()
        current_state["boundary_softened_acts"] = list(boundary_transform.get("softened_acts") or [])
        current_state["boundary_withheld_acts"] = list(boundary_transform.get("withheld_acts") or [])
        current_state["boundary_deferred_topics"] = list(boundary_transform.get("deferred_topics") or [])
        current_state["boundary_residual_pressure"] = round(float(boundary_transform.get("residual_pressure") or 0.0), 4)
        current_state["residual_reflection_mode"] = str(residual_reflection.get("mode") or "").strip()
        current_state["residual_reflection_focus"] = str(residual_reflection.get("focus") or "").strip()
        current_state["residual_reflection_strength"] = round(float(residual_reflection.get("strength") or 0.0), 4)
        current_state["residual_reflection_reasons"] = list(residual_reflection.get("reason_tokens") or [])
        current_state["autobiographical_thread_mode"] = str(current_state.get("autobiographical_thread_mode") or "").strip()
        current_state["autobiographical_thread_anchor"] = str(current_state.get("autobiographical_thread_anchor") or "").strip()
        current_state["autobiographical_thread_focus"] = str(current_state.get("autobiographical_thread_focus") or "").strip()
        current_state["autobiographical_thread_strength"] = round(
            float(current_state.get("autobiographical_thread_strength") or 0.0),
            4,
        )
        current_state["autobiographical_thread_reasons"] = list(current_state.get("autobiographical_thread_reasons") or [])
        current_state["recent_dialogue_state"] = dict(response_hook.expression_hints.get("recent_dialogue_state") or {})
        current_state["discussion_thread_state"] = dict(response_hook.expression_hints.get("discussion_thread_state") or {})
        current_state["issue_state"] = dict(response_hook.expression_hints.get("issue_state") or {})
        current_state["discussion_thread_registry_snapshot"] = update_discussion_thread_registry_snapshot(
            existing_snapshot=current_state.get("discussion_thread_registry_snapshot"),
            recent_dialogue_state=current_state.get("recent_dialogue_state"),
            discussion_thread_state=current_state.get("discussion_thread_state"),
            issue_state=current_state.get("issue_state"),
        )
        current_state["opening_pace_windowed"] = response_hook.expression_hints.get("opening_pace_windowed")
        current_state["return_gaze_expectation"] = response_hook.expression_hints.get("return_gaze_expectation")
        current_state["contact_dynamics"] = dict(response_hook.expression_hints.get("contact_dynamics") or {})
        current_state["access_dynamics"] = dict(response_hook.expression_hints.get("access_dynamics") or {})
        current_state["conscious_workspace"] = dict(response_hook.expression_hints.get("conscious_workspace") or {})
        current_state["resonance_evaluation"] = dict(response_hook.expression_hints.get("resonance_evaluation") or {})
        current_state["interaction_audit_bundle"] = dict(response_hook.expression_hints.get("interaction_audit_bundle") or {})
        current_state["interaction_audit_report"] = dict(response_hook.expression_hints.get("interaction_audit_report") or {})
        current_state["prev_qualia"] = list(((response_hook.expression_hints.get("qualia_state") or {}).get("qualia")) or [])
        current_state["prev_qualia_habituation"] = list(((response_hook.expression_hints.get("qualia_state") or {}).get("habituation")) or [])
        current_state["prev_protection_grad_x"] = list(response_hook.expression_hints.get("qualia_protection_grad_x") or [])
        current_state["prev_affective_position"] = dict(response_hook.expression_hints.get("affective_position") or {})
        current_state["affective_terrain_state"] = dict(response_hook.expression_hints.get("affective_terrain_state") or {})
        current_state["terrain_readout"] = dict(response_hook.expression_hints.get("terrain_readout") or {})
        current_state["protection_mode"] = dict(response_hook.expression_hints.get("protection_mode") or {})
        current_state["association_graph_state"] = dict(response_hook.expression_hints.get("association_graph") or {}).get("state_hint") or {}
        current_state["insight_event"] = dict(response_hook.expression_hints.get("insight_event") or {})
        current_state["qualia_planner_view"] = dict(response_hook.expression_hints.get("qualia_planner_view") or {})
        current_state["memory_write_class"] = str(
            response_hook.expression_hints.get("interaction_policy_memory_write_class") or ""
        ).strip()
        current_state["memory_write_class_reason"] = str(
            response_hook.expression_hints.get("interaction_policy_memory_write_class_reason") or ""
        ).strip()
        current_state["body_recovery_guard"] = dict(
            response_hook.expression_hints.get("interaction_policy_body_recovery_guard") or {}
        )
        current_state["body_homeostasis_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_body_homeostasis_state") or {}
        )
        current_state["homeostasis_budget_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_homeostasis_budget_state") or {}
        )
        current_state["initiative_readiness"] = dict(
            response_hook.expression_hints.get("interaction_policy_initiative_readiness") or {}
        )
        current_state["agenda_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_agenda_state") or {}
        )
        current_state["agenda_window_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_agenda_window_state") or {}
        )
        current_state["commitment_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_commitment_state") or {}
        )
        current_state["cultural_conversation_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_cultural_conversation_state") or {}
        )
        current_state["expressive_style_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_expressive_style_state") or {}
        )
        current_state["relational_continuity_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_relational_continuity_state") or {}
        )
        current_state["relation_competition_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_relation_competition_state") or {}
        )
        current_state["social_topology_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_social_topology_state") or {}
        )
        current_state["active_relation_table"] = dict(
            response_hook.expression_hints.get("interaction_policy_active_relation_table") or {}
        )
        current_state["conscious_residue_focus"] = (
            (response_hook.expression_hints.get("conscious_workspace_withheld_slice") or [None])[0]
            or (response_hook.expression_hints.get("conscious_workspace_actionable_slice") or [None])[0]
            or (response_hook.expression_hints.get("conscious_workspace_reportable_slice") or [None])[0]
        )
        current_state["conscious_residue_anchor"] = recall_hook.recall_payload.get("memory_anchor")
        current_state["conscious_residue_summary"] = " / ".join(
            [
                item
                for item in (
                    (response_hook.expression_hints.get("conscious_workspace_withheld_slice") or [None])[0],
                    (response_hook.expression_hints.get("conscious_workspace_reportable_slice") or [None])[0],
                )
                if item
            ]
        )
        current_state["conscious_residue_strength"] = float(
            ((response_hook.expression_hints.get("conscious_workspace") or {}).get("recurrent_residue") or 0.0)
        )
        memory_write_candidates = collect_runtime_memory_candidates(
            recall_payload=recall_hook.recall_payload,
            memory_reference=result.memory_reference,
            vision_entry=getattr(self, "_last_observed_vision_entry", None),
            relational_context=current_state,
        )
        post_hook = self._integration_hooks.post_turn_update(
            user_input={"text": user_text or ""},
            output={
                "reply_text": getattr(result.response, "text", None),
                "observed_shared_attention_window_mean": response_hook.expression_hints.get("stream_shared_attention_window_mean"),
                "observed_strained_pause_window_mean": response_hook.expression_hints.get("stream_strained_pause_window_mean"),
                "observed_repair_window_hold": response_hook.expression_hints.get("stream_repair_window_hold"),
            },
            current_state=current_state,
            memory_write_candidates=memory_write_candidates or None,
            recall_payload=recall_hook.recall_payload,
            transferred_lessons=transferred_lessons or None,
        )
        self._last_gate_context["inner_os_temporal_pressure"] = float(post_hook.state.temporal_pressure)
        self._last_gate_context["inner_os_route"] = str(post_hook.state.route)
        self._last_gate_context["inner_os_social_update_strength"] = float(post_hook.state.social_update_strength)
        self._last_gate_context["inner_os_identity_update_strength"] = float(post_hook.state.identity_update_strength)
        self._last_gate_context["inner_os_interaction_afterglow"] = float(post_hook.state.interaction_afterglow)
        self._last_gate_context["inner_os_interaction_afterglow_intent"] = post_hook.state.interaction_afterglow_intent
        self._last_gate_context["inner_os_replay_intensity"] = float(post_hook.state.replay_intensity)
        self._last_gate_context["inner_os_anticipation_tension"] = float(post_hook.state.anticipation_tension)
        self._last_gate_context["inner_os_stabilization_drive"] = float(post_hook.state.stabilization_drive)
        self._last_gate_context["inner_os_relational_clarity"] = float(post_hook.state.relational_clarity)
        self._last_gate_context["inner_os_meaning_inertia"] = float(post_hook.state.meaning_inertia)
        self._last_gate_context["inner_os_recovery_reopening"] = float(post_hook.state.recovery_reopening)
        self._last_gate_context["inner_os_contact_dynamics"] = dict(response_hook.expression_hints.get("contact_dynamics") or {})
        self._last_gate_context["inner_os_contact_dynamics_mode"] = response_hook.expression_hints.get("contact_dynamics_mode")
        self._last_gate_context["inner_os_access_dynamics"] = dict(response_hook.expression_hints.get("access_dynamics") or {})
        self._last_gate_context["inner_os_access_dynamics_mode"] = response_hook.expression_hints.get("access_dynamics_mode")
        self._last_gate_context["inner_os_prev_qualia"] = list(((response_hook.expression_hints.get("qualia_state") or {}).get("qualia")) or [])
        self._last_gate_context["inner_os_prev_qualia_habituation"] = list(((response_hook.expression_hints.get("qualia_state") or {}).get("habituation")) or [])
        self._last_gate_context["inner_os_prev_protection_grad_x"] = list(response_hook.expression_hints.get("qualia_protection_grad_x") or [])
        self._last_gate_context["inner_os_prev_affective_position"] = dict(response_hook.expression_hints.get("affective_position") or {})
        self._last_gate_context["inner_os_affective_terrain_state"] = dict(
            (post_hook.audit_record or {}).get("affective_terrain_state")
            or response_hook.expression_hints.get("affective_terrain_state")
            or {}
        )
        self._last_gate_context["inner_os_terrain_readout"] = dict(response_hook.expression_hints.get("terrain_readout") or {})
        self._last_gate_context["inner_os_protection_mode"] = dict(response_hook.expression_hints.get("protection_mode") or {})
        self._last_gate_context["inner_os_association_graph_state"] = dict(
            (post_hook.audit_record or {}).get("association_graph_state")
            or dict(response_hook.expression_hints.get("association_graph") or {}).get("state_hint")
            or {}
        )
        self._last_gate_context["inner_os_insight_event"] = dict(response_hook.expression_hints.get("insight_event") or {})
        self._last_gate_context["inner_os_terrain_reweighting_bias"] = float(current_state.get("terrain_reweighting_bias") or 0.0)
        self._last_gate_context["inner_os_association_reweighting_bias"] = float(current_state.get("association_reweighting_bias") or 0.0)
        self._last_gate_context["inner_os_association_reweighting_focus"] = str(current_state.get("association_reweighting_focus") or "")
        self._last_gate_context["inner_os_association_reweighting_reason"] = str(current_state.get("association_reweighting_reason") or "")
        self._last_gate_context["inner_os_insight_reframing_bias"] = float(current_state.get("insight_reframing_bias") or 0.0)
        self._last_gate_context["inner_os_insight_class_focus"] = str(current_state.get("insight_class_focus") or "")
        self._last_gate_context["inner_os_insight_terrain_shape_target"] = str(current_state.get("insight_terrain_shape_target") or "")
        self._last_gate_context["inner_os_insight_link_counts"] = dict(current_state.get("insight_link_counts") or {})
        self._last_gate_context["inner_os_insight_class_counts"] = dict(current_state.get("insight_class_counts") or {})
        self._last_gate_context["inner_os_relation_competition_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_relation_competition_state") or {}
        )
        self._last_gate_context["inner_os_social_topology_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_social_topology_state") or {}
        )
        self._last_gate_context["inner_os_active_relation_table"] = dict(
            response_hook.expression_hints.get("interaction_policy_active_relation_table") or {}
        )
        self._last_gate_context["inner_os_overnight_bias_roles"] = dict(
            response_hook.expression_hints.get("interaction_policy_overnight_bias_roles") or {}
        )
        self._last_gate_context["inner_os_reaction_vs_overnight_bias"] = dict(
            response_hook.expression_hints.get("interaction_policy_reaction_vs_overnight_bias") or {}
        )
        self._last_gate_context["inner_os_body_homeostasis_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_body_homeostasis_state") or {}
        )
        self._last_gate_context["inner_os_body_homeostasis_focus"] = str(
            ((response_hook.expression_hints.get("interaction_policy_body_homeostasis_state") or {}).get("state") or "")
        ).strip()
        self._last_gate_context["inner_os_body_homeostasis_carry_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(((response_hook.expression_hints.get("interaction_policy_body_homeostasis_state") or {}).get("score") or 0.0)) * 0.28
                    + float(((response_hook.expression_hints.get("interaction_policy_body_homeostasis_state") or {}).get("winner_margin") or 0.0)) * 0.12,
                ),
            ),
            4,
        )
        self._last_gate_context["inner_os_homeostasis_budget_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_homeostasis_budget_state") or {}
        )
        self._last_gate_context["inner_os_relational_continuity_state"] = dict(
            response_hook.expression_hints.get("interaction_policy_relational_continuity_state") or {}
        )
        self._last_gate_context["inner_os_relational_continuity_focus"] = str(
            ((response_hook.expression_hints.get("interaction_policy_relational_continuity_state") or {}).get("state") or "")
        ).strip()
        self._last_gate_context["inner_os_relational_continuity_carry_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(((response_hook.expression_hints.get("interaction_policy_relational_continuity_state") or {}).get("score") or 0.0)) * 0.26
                    + float(((response_hook.expression_hints.get("interaction_policy_relational_continuity_state") or {}).get("winner_margin") or 0.0)) * 0.12,
                ),
            ),
            4,
        )
        self._last_gate_context["inner_os_group_thread_focus"] = str(
            ((response_hook.expression_hints.get("interaction_policy_social_topology_state") or {}).get("state") or "")
        ).strip()
        self._last_gate_context["inner_os_group_thread_carry_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(((response_hook.expression_hints.get("interaction_policy_social_topology_state") or {}).get("score") or 0.0)) * 0.18
                    + float(((response_hook.expression_hints.get("interaction_policy_social_topology_state") or {}).get("winner_margin") or 0.0)) * 0.08,
                ),
            ),
            4,
        )
        self._last_gate_context["inner_os_expressive_style_focus"] = str(
            ((response_hook.expression_hints.get("interaction_policy_expressive_style_state") or {}).get("state") or "")
        ).strip()
        self._last_gate_context["inner_os_expressive_style_carry_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(((response_hook.expression_hints.get("interaction_policy_expressive_style_state") or {}).get("score") or 0.0)) * 0.22
                    + float(((response_hook.expression_hints.get("interaction_policy_expressive_style_state") or {}).get("winner_margin") or 0.0)) * 0.1,
                ),
            ),
            4,
        )
        interaction_policy_packet = dict(response_hook.expression_hints.get("interaction_policy_packet") or {})
        expressive_style_state = dict(response_hook.expression_hints.get("interaction_policy_expressive_style_state") or {})
        relational_style_memory_state = dict(response_hook.expression_hints.get("interaction_policy_relational_style_memory_state") or {})
        self._last_gate_context["inner_os_expressive_style_history_focus"] = str(
            interaction_policy_packet.get("expressive_style_history_focus")
            or expressive_style_state.get("state")
            or ""
        ).strip()
        self._last_gate_context["inner_os_expressive_style_history_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(interaction_policy_packet.get("expressive_style_history_bias") or 0.0)
                    or (
                        float(expressive_style_state.get("score") or 0.0) * 0.18
                        + float(expressive_style_state.get("winner_margin") or 0.0) * 0.08
                    ),
                ),
            ),
            4,
        )
        self._last_gate_context["inner_os_banter_style_focus"] = str(
            interaction_policy_packet.get("banter_style_focus")
            or relational_style_memory_state.get("banter_style")
            or ""
        ).strip()
        self._last_gate_context["inner_os_lexical_variation_carry_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(interaction_policy_packet.get("lexical_variation_carry_bias") or 0.0)
                    or float(relational_style_memory_state.get("lexical_variation_bias") or 0.0) * 0.22,
                ),
            ),
            4,
        )
        self._last_gate_context["inner_os_person_registry_snapshot"] = dict(post_hook.person_registry_snapshot or {})
        self._last_gate_context["inner_os_group_thread_registry_snapshot"] = dict(post_hook.group_thread_registry_snapshot or {})
        self._last_gate_context["inner_os_discussion_thread_registry_snapshot"] = dict(
            current_state.get("discussion_thread_registry_snapshot") or {}
        )
        previous_budget_focus = str(current_state.get("homeostasis_budget_focus") or "").strip()
        previous_budget_bias = float(current_state.get("homeostasis_budget_bias") or 0.0)
        next_budget_state = dict(
            response_hook.expression_hints.get("interaction_policy_homeostasis_budget_state") or {}
        )
        next_budget_focus = str(next_budget_state.get("state") or previous_budget_focus or "").strip()
        next_budget_signal = max(
            0.0,
            min(
                1.0,
                float(next_budget_state.get("score") or 0.0) * 0.18
                + float(next_budget_state.get("winner_margin") or 0.0) * 0.08,
            ),
        )
        carried_budget_bias = max(0.0, min(1.0, previous_budget_bias * 0.84))
        if next_budget_focus and next_budget_focus != "steady":
            carried_budget_bias = max(carried_budget_bias, min(1.0, carried_budget_bias + next_budget_signal))
        elif not next_budget_focus:
            next_budget_focus = previous_budget_focus or "steady"
        self._last_gate_context["inner_os_homeostasis_budget_focus"] = next_budget_focus
        self._last_gate_context["inner_os_homeostasis_budget_bias"] = round(carried_budget_bias, 4)
        initiative_followup_bias = dict((post_hook.audit_record or {}).get("initiative_followup_bias") or {})
        self._last_gate_context["inner_os_initiative_followup_bias"] = float(
            initiative_followup_bias.get("score") or 0.0
        )
        self._last_gate_context["inner_os_initiative_followup_state"] = str(
            initiative_followup_bias.get("state") or "hold"
        )
        agenda_state = dict(response_hook.expression_hints.get("interaction_policy_agenda_state") or {})
        self._last_gate_context["inner_os_agenda_focus"] = str(
            agenda_state.get("state") or current_state.get("agenda_focus") or ""
        ).strip()
        self._last_gate_context["inner_os_agenda_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(agenda_state.get("score") or 0.0) * 0.22
                    + float(agenda_state.get("winner_margin") or 0.0) * 0.1,
                ),
            ),
            4,
        )
        self._last_gate_context["inner_os_agenda_reason"] = str(
            agenda_state.get("reason") or current_state.get("agenda_reason") or ""
        ).strip()
        agenda_window_state = dict(
            response_hook.expression_hints.get("interaction_policy_agenda_window_state") or {}
        )
        self._last_gate_context["inner_os_agenda_window_focus"] = str(
            agenda_window_state.get("state") or current_state.get("agenda_window_focus") or ""
        ).strip()
        self._last_gate_context["inner_os_agenda_window_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(agenda_window_state.get("score") or 0.0) * 0.2
                    + float(agenda_window_state.get("winner_margin") or 0.0) * 0.08,
                ),
            ),
            4,
        )
        self._last_gate_context["inner_os_agenda_window_reason"] = str(
            agenda_window_state.get("reason") or current_state.get("agenda_window_reason") or ""
        ).strip()
        learning_mode_state = dict(
            response_hook.expression_hints.get("interaction_policy_learning_mode_state") or {}
        )
        self._last_gate_context["inner_os_learning_mode_focus"] = str(
            learning_mode_state.get("state") or current_state.get("learning_mode_focus") or ""
        ).strip()
        self._last_gate_context["inner_os_learning_mode_carry_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(learning_mode_state.get("score") or 0.0) * 0.18
                    + float(learning_mode_state.get("winner_margin") or 0.0) * 0.08,
                ),
            ),
            4,
        )
        social_experiment_loop_state = dict(
            response_hook.expression_hints.get("interaction_policy_social_experiment_loop_state") or {}
        )
        self._last_gate_context["inner_os_social_experiment_focus"] = str(
            social_experiment_loop_state.get("state") or current_state.get("social_experiment_focus") or ""
        ).strip()
        self._last_gate_context["inner_os_social_experiment_carry_bias"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(social_experiment_loop_state.get("score") or 0.0) * 0.16
                    + float(social_experiment_loop_state.get("winner_margin") or 0.0) * 0.08
                    + float(social_experiment_loop_state.get("probe_intensity") or 0.0) * 0.06,
                ),
            ),
            4,
        )
        commitment_carry = dict(
            ((response_hook.expression_hints.get("interaction_policy_packet") or {}).get("commitment_carry") or {})
        )
        self._last_gate_context["inner_os_commitment_target_focus"] = str(
            current_state.get("commitment_target_focus")
            or commitment_carry.get("target_focus")
            or ""
        )
        self._last_gate_context["inner_os_commitment_state_focus"] = str(
            current_state.get("commitment_state_focus")
            or commitment_carry.get("state_focus")
            or "waver"
        )
        self._last_gate_context["inner_os_commitment_carry_bias"] = float(
            current_state.get("commitment_carry_bias")
            or commitment_carry.get("carry_bias")
            or 0.0
        )
        self._last_gate_context["inner_os_commitment_followup_focus"] = str(
            current_state.get("commitment_followup_focus")
            or commitment_carry.get("followup_focus")
            or ""
        )
        self._last_gate_context["inner_os_commitment_mode_focus"] = str(
            current_state.get("commitment_mode_focus")
            or commitment_carry.get("mode_focus")
            or ""
        )
        self._last_gate_context["inner_os_commitment_carry_reason"] = str(
            current_state.get("commitment_carry_reason")
            or commitment_carry.get("carry_reason")
            or ""
        )
        self._last_gate_context["inner_os_temperament_trace"] = dict(
            (post_hook.audit_record or {}).get("temperament_trace") or {}
        )
        self._last_gate_context["inner_os_terrain_plasticity_update"] = dict(
            (post_hook.audit_record or {}).get("terrain_plasticity_update") or {}
        )
        self._last_gate_context["inner_os_qualia_state"] = dict(response_hook.expression_hints.get("qualia_state") or {})
        self._last_gate_context["inner_os_qualia_estimator_health"] = dict(response_hook.expression_hints.get("qualia_estimator_health") or {})
        self._last_gate_context["inner_os_qualia_planner_view"] = dict(response_hook.expression_hints.get("qualia_planner_view") or {})
        self._last_gate_context["inner_os_qualia_hint_source"] = str(response_hook.expression_hints.get("qualia_hint_source") or "none")
        self._last_gate_context["inner_os_qualia_hint_fallback_reason"] = str(response_hook.expression_hints.get("qualia_hint_fallback_reason") or "")
        self._last_gate_context["inner_os_qualia_hint_expected_source"] = str(response_hook.expression_hints.get("qualia_hint_expected_source") or "")
        self._last_gate_context["inner_os_qualia_hint_expected_mismatch"] = bool(response_hook.expression_hints.get("qualia_hint_expected_mismatch", False))
        self._last_gate_context["inner_os_conscious_workspace"] = dict(response_hook.expression_hints.get("conscious_workspace") or {})
        self._last_gate_context["inner_os_conscious_workspace_mode"] = response_hook.expression_hints.get("conscious_workspace_mode")
        self._last_gate_context["inner_os_conversational_objects"] = dict(response_hook.expression_hints.get("conversational_objects") or {})
        self._last_gate_context["inner_os_object_operations"] = dict(response_hook.expression_hints.get("object_operations") or {})
        self._last_gate_context["inner_os_interaction_effects"] = dict(response_hook.expression_hints.get("interaction_effects") or {})
        self._last_gate_context["inner_os_interaction_judgement_view"] = dict(response_hook.expression_hints.get("interaction_judgement_view") or {})
        self._last_gate_context["inner_os_interaction_judgement_summary"] = dict(response_hook.expression_hints.get("interaction_judgement_summary") or {})
        self._last_gate_context["inner_os_interaction_condition_report"] = dict(response_hook.expression_hints.get("interaction_condition_report") or {})
        self._last_gate_context["inner_os_interaction_inspection_report"] = dict(response_hook.expression_hints.get("interaction_inspection_report") or {})
        self._last_gate_context["inner_os_interaction_audit_bundle"] = dict(response_hook.expression_hints.get("interaction_audit_bundle") or {})
        self._last_gate_context["inner_os_interaction_audit_casebook"] = dict(response_hook.expression_hints.get("interaction_audit_casebook") or {})
        self._last_gate_context["inner_os_interaction_audit_report"] = dict(response_hook.expression_hints.get("interaction_audit_report") or {})
        self._last_gate_context["inner_os_interaction_audit_reference_case_ids"] = list(response_hook.expression_hints.get("interaction_audit_reference_case_ids") or [])
        self._last_gate_context["inner_os_interaction_audit_reference_case_meta"] = dict(response_hook.expression_hints.get("interaction_audit_reference_case_meta") or {})
        self._last_gate_context["inner_os_resonance_evaluation"] = dict(response_hook.expression_hints.get("resonance_evaluation") or {})
        self._last_gate_context["inner_os_conscious_residue_focus"] = post_hook.state.conscious_residue_focus
        self._last_gate_context["inner_os_conscious_residue_anchor"] = post_hook.state.conscious_residue_anchor
        self._last_gate_context["inner_os_conscious_residue_summary"] = post_hook.state.conscious_residue_summary
        self._last_gate_context["inner_os_conscious_residue_strength"] = float(post_hook.state.conscious_residue_strength)
        self._last_gate_context["inner_os_autobiographical_thread_mode"] = post_hook.state.autobiographical_thread_mode
        self._last_gate_context["inner_os_autobiographical_thread_anchor"] = post_hook.state.autobiographical_thread_anchor
        self._last_gate_context["inner_os_autobiographical_thread_focus"] = post_hook.state.autobiographical_thread_focus
        self._last_gate_context["inner_os_autobiographical_thread_strength"] = float(
            post_hook.state.autobiographical_thread_strength
        )
        result.metrics.setdefault(
            "inner_os/temporal_pressure_after",
            round(float(post_hook.state.temporal_pressure), 4),
        )
        result.metrics.setdefault(
            "inner_os/transferred_lessons_used",
            float(len(transferred_lessons or [])),
        )
        result.metrics.setdefault(
            "inner_os/interaction_afterglow",
            round(float(post_hook.state.interaction_afterglow), 4),
        )
        result.metrics.setdefault(
            "inner_os/replay_intensity",
            round(float(post_hook.state.replay_intensity), 4),
        )
        result.metrics.setdefault(
            "inner_os/anticipation_tension",
            round(float(post_hook.state.anticipation_tension), 4),
        )
        result.metrics.setdefault(
            "inner_os/recovery_reopening",
            round(float(post_hook.state.recovery_reopening), 4),
        )
        result.metrics.setdefault(
            "inner_os/object_affordance_bias",
            round(float(post_hook.state.object_affordance_bias), 4),
        )
        result.metrics.setdefault(
            "inner_os/defensive_salience",
            round(float(post_hook.state.defensive_salience), 4),
        )
        result.metrics.setdefault(
            "inner_os/reachability",
            round(float(post_hook.state.reachability), 4),
        )
        result.metrics.setdefault(
            "inner_os/consolidation_priority",
            round(float(post_hook.state.consolidation_priority), 4),
        )
        result.metrics.setdefault(
            "inner_os/interference_pressure",
            round(float(post_hook.state.interference_pressure), 4),
        )
        result.metrics.setdefault(
            "inner_os/prospective_memory_pull",
            round(float(post_hook.state.prospective_memory_pull), 4),
        )
        result.metrics.setdefault(
            "inner_os/reuse_trajectory",
            round(float(post_hook.state.reuse_trajectory), 4),
        )
        result.metrics.setdefault(
            "inner_os/contact_reentry_bias",
            round(float(((response_hook.expression_hints.get("contact_dynamics") or {}).get("reentry_bias") or 0.0)), 4),
        )
        result.metrics.setdefault(
            "inner_os/access_membrane_inertia",
            round(float(((response_hook.expression_hints.get("access_dynamics") or {}).get("membrane_inertia") or 0.0)), 4),
        )
        result.metrics.setdefault(
            "inner_os/resonance_score",
            round(
                float(
                    (((response_hook.expression_hints.get("resonance_evaluation") or {}).get("assessments") or [{}])[0].get("resonance_score") or 0.0)
                ),
                4,
            ),
        )
        qualia_hint_source = str(response_hook.expression_hints.get("qualia_hint_source") or "none")
        qualia_hint_expected_mismatch = bool(
            response_hook.expression_hints.get("qualia_hint_expected_mismatch", False)
        )
        result.metrics.setdefault(
            "inner_os/qualia_hint_shared",
            1.0 if qualia_hint_source == "shared" else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/qualia_hint_fallback",
            1.0 if qualia_hint_source == "fallback" else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/qualia_hint_none",
            1.0 if qualia_hint_source == "none" else 0.0,
        )
        overnight_association_focus = str(current_state.get("association_reweighting_focus") or "").strip()
        overnight_terrain_shape_target = str(current_state.get("insight_terrain_shape_target") or "").strip()
        overnight_commitment_focus = str(current_state.get("commitment_target_focus") or "").strip()
        overnight_agenda_focus = str(current_state.get("agenda_focus") or "").strip()
        overnight_agenda_window_focus = str(current_state.get("agenda_window_focus") or "").strip()
        overnight_learning_mode_focus = str(current_state.get("learning_mode_focus") or "").strip()
        overnight_social_experiment_focus = str(current_state.get("social_experiment_focus") or "").strip()
        result.metrics.setdefault(
            "inner_os/overnight_association_bias_active",
            1.0 if overnight_association_focus else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_terrain_shape_bias_active",
            1.0 if overnight_terrain_shape_target else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_bias_alignment_visible",
            1.0 if overnight_association_focus or overnight_terrain_shape_target or overnight_commitment_focus else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_commitment_bias_active",
            1.0 if overnight_commitment_focus or float(current_state.get("commitment_carry_bias") or 0.0) > 0.0 else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_agenda_bias_active",
            1.0 if overnight_agenda_focus or float(current_state.get("agenda_bias") or 0.0) > 0.0 else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_agenda_window_bias_active",
            1.0 if overnight_agenda_window_focus or float(current_state.get("agenda_window_bias") or 0.0) > 0.0 else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_learning_mode_bias_active",
            1.0 if overnight_learning_mode_focus or float(current_state.get("learning_mode_carry_bias") or 0.0) > 0.0 else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_social_experiment_bias_active",
            1.0 if overnight_social_experiment_focus or float(current_state.get("social_experiment_carry_bias") or 0.0) > 0.0 else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_body_homeostasis_bias_active",
            1.0 if str(current_state.get("body_homeostasis_focus") or "").strip() and float(current_state.get("body_homeostasis_carry_bias") or 0.0) > 0.0 else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_homeostasis_budget_active",
            1.0 if str(current_state.get("homeostasis_budget_focus") or "").strip() and float(current_state.get("homeostasis_budget_bias") or 0.0) > 0.0 else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_relational_continuity_bias_active",
            1.0 if str(current_state.get("relational_continuity_focus") or "").strip() and float(current_state.get("relational_continuity_carry_bias") or 0.0) > 0.0 else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_expressive_style_active",
            1.0 if str(current_state.get("expressive_style_focus") or "").strip() and float(current_state.get("expressive_style_carry_bias") or 0.0) > 0.0 else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_expressive_style_history_active",
            1.0
            if str(current_state.get("expressive_style_history_focus") or "").strip()
            and float(current_state.get("expressive_style_history_bias") or 0.0) > 0.0
            else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/overnight_lexical_variation_carry_active",
            1.0
            if str(current_state.get("banter_style_focus") or "").strip()
            and float(current_state.get("lexical_variation_carry_bias") or 0.0) > 0.0
            else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/person_registry_total_people",
            float(int((current_state.get("person_registry_snapshot") or {}).get("total_people") or 0)),
        )
        result.metrics.setdefault(
            "inner_os/group_thread_total_threads",
            float(int((current_state.get("group_thread_registry_snapshot") or {}).get("total_threads") or 0)),
        )
        result.metrics.setdefault(
            "inner_os/discussion_thread_total_threads",
            float(int((current_state.get("discussion_thread_registry_snapshot") or {}).get("total_threads") or 0)),
        )
        result.metrics.setdefault(
            "inner_os/overnight_group_thread_bias_active",
            1.0
            if str(current_state.get("group_thread_focus") or "").strip()
            and float(current_state.get("group_thread_carry_bias") or 0.0) > 0.0
            else 0.0,
        )
        interaction_policy_packet = dict(response_hook.expression_hints.get("interaction_policy_packet") or {})
        memory_write_class_bias = dict(interaction_policy_packet.get("memory_write_class_bias") or {})
        protection_mode_decision = dict(interaction_policy_packet.get("protection_mode_decision") or {})
        body_recovery_guard = dict(interaction_policy_packet.get("body_recovery_guard") or {})
        body_homeostasis_state = dict(interaction_policy_packet.get("body_homeostasis_state") or {})
        homeostasis_budget_state = dict(interaction_policy_packet.get("homeostasis_budget_state") or {})
        initiative_readiness = dict(interaction_policy_packet.get("initiative_readiness") or {})
        agenda_state = dict(interaction_policy_packet.get("agenda_state") or {})
        agenda_window_state = dict(interaction_policy_packet.get("agenda_window_state") or {})
        commitment_state = dict(interaction_policy_packet.get("commitment_state") or {})
        relational_style_memory_state = dict(interaction_policy_packet.get("relational_style_memory_state") or {})
        cultural_conversation_state = dict(interaction_policy_packet.get("cultural_conversation_state") or {})
        expressive_style_state = dict(interaction_policy_packet.get("expressive_style_state") or {})
        lightness_budget_state = dict(interaction_policy_packet.get("lightness_budget_state") or {})
        relational_continuity_state = dict(interaction_policy_packet.get("relational_continuity_state") or {})
        relation_competition_state = dict(interaction_policy_packet.get("relation_competition_state") or {})
        social_topology_state = dict(interaction_policy_packet.get("social_topology_state") or {})
        result.metrics.setdefault(
            "inner_os/memory_write_winner_margin",
            round(float(memory_write_class_bias.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/protection_mode_winner_margin",
            round(float(protection_mode_decision.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/memory_write_mode_prior_active",
            1.0 if any(float(value or 0.0) > 0.0 for value in (memory_write_class_bias.get("mode_prior") or {}).values()) else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/memory_write_insight_prior_active",
            1.0 if any(float(value or 0.0) > 0.0 for value in (memory_write_class_bias.get("insight_prior") or {}).values()) else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/workspace_winner_margin",
            round(float(((response_hook.expression_hints.get("conscious_workspace") or {}).get("winner_margin") or 0.0)), 4),
        )
        result.metrics.setdefault(
            "inner_os/association_graph_winner_margin",
            round(float(((response_hook.expression_hints.get("association_graph") or {}).get("winner_margin") or 0.0)), 4),
        )
        result.metrics.setdefault(
            "inner_os/body_recovery_guard_winner_margin",
            round(float(body_recovery_guard.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/body_homeostasis_score",
            round(float(body_homeostasis_state.get("score") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/body_homeostasis_winner_margin",
            round(float(body_homeostasis_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/homeostasis_budget_score",
            round(float(homeostasis_budget_state.get("score") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/homeostasis_budget_winner_margin",
            round(float(homeostasis_budget_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/initiative_readiness",
            round(float(initiative_readiness.get("score") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/commitment_winner_margin",
            round(float(commitment_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/agenda_score",
            round(float(agenda_state.get("score") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/agenda_winner_margin",
            round(float(agenda_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/agenda_window_score",
            round(float(agenda_window_state.get("score") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/agenda_window_deferral_budget",
            round(float(agenda_window_state.get("deferral_budget") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/learning_mode_winner_margin",
            round(float(learning_mode_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/social_experiment_winner_margin",
            round(float(social_experiment_loop_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/expressive_style_winner_margin",
            round(float(expressive_style_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/relational_style_playful_ceiling",
            round(float(relational_style_memory_state.get("playful_ceiling") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/relational_style_advice_tolerance",
            round(float(relational_style_memory_state.get("advice_tolerance") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/relational_style_lexical_variation_bias",
            round(float(relational_style_memory_state.get("lexical_variation_bias") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/relational_style_banter_room",
            round(float(relational_style_memory_state.get("banter_room") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/cultural_directness_ceiling",
            round(float(cultural_conversation_state.get("directness_ceiling") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/cultural_joke_ratio_ceiling",
            round(float(cultural_conversation_state.get("joke_ratio_ceiling") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/expressive_lightness_room",
            round(float(expressive_style_state.get("lightness_room") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/lightness_budget_banter_room",
            round(float(lightness_budget_state.get("banter_room") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/lightness_budget_suppression",
            round(float(lightness_budget_state.get("suppression") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/relational_continuity_score",
            round(float(relational_continuity_state.get("score") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/relational_continuity_winner_margin",
            round(float(relational_continuity_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/relation_competition_level",
            round(float(relation_competition_state.get("competition_level") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/relation_competition_winner_margin",
            round(float(relation_competition_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/social_topology_score",
            round(float(social_topology_state.get("score") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/social_topology_winner_margin",
            round(float(social_topology_state.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/active_relation_total_people",
            float(int((interaction_policy_packet.get("active_relation_table") or {}).get("total_people") or 0)),
        )
        result.metrics.setdefault(
            "inner_os/commitment_accepted_cost",
            round(float(commitment_state.get("accepted_cost") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/initiative_followup_bias",
            round(float(initiative_followup_bias.get("score") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/initiative_followup_winner_margin",
            round(float(initiative_followup_bias.get("winner_margin") or 0.0), 4),
        )
        temperament_trace = dict((post_hook.audit_record or {}).get("temperament_trace") or {})
        result.metrics.setdefault(
            "inner_os/temperament_forward_trace",
            round(float(temperament_trace.get("temperament_forward_trace") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/temperament_guard_trace",
            round(float(temperament_trace.get("temperament_guard_trace") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/qualia_hint_expected_mismatch",
            1.0 if qualia_hint_expected_mismatch else 0.0,
        )
        terrain_plasticity_applied = bool((post_hook.audit_record or {}).get("terrain_plasticity_applied", False))
        terrain_plasticity_update = (post_hook.audit_record or {}).get("terrain_plasticity_update") or {}
        association_graph_state = (post_hook.audit_record or {}).get("association_graph_state") or {}
        raw_response = payload.get("response") if isinstance(payload, Mapping) else None
        resolved_llm_model = str(
            getattr(result.response, "model", "")
            or getattr(raw_response, "model", "")
            or llm_raw_model
            or ""
        )
        resolved_llm_model_source = str(
            getattr(result.response, "model_source", "")
            or getattr(raw_response, "model_source", "")
            or llm_raw_model_source
            or ""
        )
        result.metrics.setdefault(
            "inner_os/terrain_plasticity_applied",
            1.0 if terrain_plasticity_applied else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/terrain_plasticity_confidence",
            round(float(terrain_plasticity_update.get("confidence") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/terrain_plasticity_winner_margin",
            round(float(terrain_plasticity_update.get("winner_margin") or 0.0), 4),
        )
        result.metrics.setdefault(
            "inner_os/association_reinforcement_winner_margin",
            round(float(association_graph_state.get("winner_margin") or 0.0), 4),
        )
        result.persona_meta.setdefault("inner_os", {})
        result.persona_meta["inner_os"].update(
            {
                "route": post_hook.state.route,
                "talk_mode": post_hook.state.talk_mode,
                "memory_anchor": post_hook.state.memory_anchor,
                "culture_id": current_state.get("culture_id"),
                "community_id": current_state.get("community_id"),
                "social_role": current_state.get("social_role"),
                "continuity_score": round(float(post_hook.state.continuity_score), 4),
                "social_grounding": round(float(post_hook.state.social_grounding), 4),
                "recent_strain": round(float(post_hook.state.recent_strain), 4),
                "culture_resonance": round(float(post_hook.state.culture_resonance), 4),
                "community_resonance": round(float(post_hook.state.community_resonance), 4),
                "meaning_pacing": response_hook.expression_hints.get("meaning_pacing"),
                "interaction_pacing": response_hook.expression_hints.get("interaction_pacing"),
                "recalled_tentative_bias": round(float(post_hook.state.recalled_tentative_bias), 4),
                "tentative_bias": response_hook.expression_hints.get("tentative_bias"),
                "question_bias": response_hook.expression_hints.get("question_bias"),
                "surface_policy_level": self._inner_os_surface_policy_level(getattr(result.response, "controls_used", {}) if result.response is not None else {}),
                "surface_policy_intent": self._inner_os_surface_policy_intent(getattr(result.response, "controls_used", {}) if result.response is not None else {}),
                "boundary_transform": dict(response_hook.expression_hints.get("boundary_transform") or {}),
                "residual_reflection": dict(response_hook.expression_hints.get("residual_reflection") or {}),
                "actuation_execution_mode": headless_actuation.execution_mode,
                "actuation_primary_action": headless_actuation.primary_action,
                "actuation_reply_permission": headless_actuation.reply_permission,
                "actuation_wait_before_action": headless_actuation.wait_before_action,
                "social_update_strength": round(float(post_hook.state.social_update_strength), 4),
                "identity_update_strength": round(float(post_hook.state.identity_update_strength), 4),
                "interaction_afterglow": round(float(post_hook.state.interaction_afterglow), 4),
                "interaction_afterglow_intent": post_hook.state.interaction_afterglow_intent,
                "replay_intensity": round(float(post_hook.state.replay_intensity), 4),
                "anticipation_tension": round(float(post_hook.state.anticipation_tension), 4),
                "stabilization_drive": round(float(post_hook.state.stabilization_drive), 4),
                "relational_clarity": round(float(post_hook.state.relational_clarity), 4),
                "meaning_inertia": round(float(post_hook.state.meaning_inertia), 4),
                "recovery_reopening": round(float(post_hook.state.recovery_reopening), 4),
                "terrain_transition_roughness": round(float(post_hook.state.terrain_transition_roughness), 4),
                "object_affordance_bias": round(float(post_hook.state.object_affordance_bias), 4),
                "fragility_guard": round(float(post_hook.state.fragility_guard), 4),
                "defensive_salience": round(float(post_hook.state.defensive_salience), 4),
                "reachability": round(float(post_hook.state.reachability), 4),
                "current_focus": post_hook.state.current_focus,
                "pending_meaning": round(float(post_hook.state.pending_meaning), 4),
                "working_memory_pressure": round(float(post_hook.state.working_memory_pressure), 4),
                "autobiographical_thread_mode": post_hook.state.autobiographical_thread_mode,
                "autobiographical_thread_anchor": post_hook.state.autobiographical_thread_anchor,
                "autobiographical_thread_focus": post_hook.state.autobiographical_thread_focus,
                "autobiographical_thread_strength": round(float(post_hook.state.autobiographical_thread_strength), 4),
                "autobiographical_thread_reasons": list(post_hook.state.autobiographical_thread_reasons or []),
                "contact_dynamics_mode": response_hook.expression_hints.get("contact_dynamics_mode"),
                "access_dynamics_mode": response_hook.expression_hints.get("access_dynamics_mode"),
                "conscious_workspace_mode": response_hook.expression_hints.get("conscious_workspace_mode"),
                "conscious_workspace_reportable_slice": list(response_hook.expression_hints.get("conscious_workspace_reportable_slice") or []),
                "conscious_workspace_withheld_slice": list(response_hook.expression_hints.get("conscious_workspace_withheld_slice") or []),
                "conscious_workspace_actionable_slice": list(response_hook.expression_hints.get("conscious_workspace_actionable_slice") or []),
                "conscious_workspace_winner_margin": round(float(response_hook.expression_hints.get("conscious_workspace_winner_margin") or 0.0), 4),
                "conscious_workspace_slot_scores": dict(response_hook.expression_hints.get("conscious_workspace_slot_scores") or {}),
                "conscious_workspace_dominant_inputs": list(response_hook.expression_hints.get("conscious_workspace_dominant_inputs") or []),
                "association_graph_winner_margin": round(float(response_hook.expression_hints.get("association_graph_winner_margin") or 0.0), 4),
                "association_graph_dominant_inputs": list(response_hook.expression_hints.get("association_graph_dominant_inputs") or []),
                "qualia_hint_source": str(response_hook.expression_hints.get("qualia_hint_source") or "none"),
                "qualia_hint_fallback_reason": str(response_hook.expression_hints.get("qualia_hint_fallback_reason") or ""),
                "qualia_hint_expected_source": str(response_hook.expression_hints.get("qualia_hint_expected_source") or ""),
                "qualia_hint_expected_mismatch": bool(response_hook.expression_hints.get("qualia_hint_expected_mismatch", False)),
                "association_reweighting_focus": str(current_state.get("association_reweighting_focus") or ""),
                "association_reweighting_reason": str(current_state.get("association_reweighting_reason") or ""),
                "insight_terrain_shape_target": str(current_state.get("insight_terrain_shape_target") or ""),
                "commitment_target_focus": str(current_state.get("commitment_target_focus") or ""),
                "commitment_state_focus": str(current_state.get("commitment_state_focus") or "waver"),
                "commitment_carry_bias": round(float(current_state.get("commitment_carry_bias") or 0.0), 4),
                "commitment_followup_focus": str(current_state.get("commitment_followup_focus") or ""),
                "commitment_mode_focus": str(current_state.get("commitment_mode_focus") or ""),
                "commitment_carry_reason": str(current_state.get("commitment_carry_reason") or ""),
                "agenda_focus": str(current_state.get("agenda_focus") or ""),
                "agenda_bias": round(float(current_state.get("agenda_bias") or 0.0), 4),
                "agenda_reason": str(current_state.get("agenda_reason") or ""),
                "agenda_window_focus": str(current_state.get("agenda_window_focus") or ""),
                "agenda_window_bias": round(float(current_state.get("agenda_window_bias") or 0.0), 4),
                "agenda_window_reason": str(current_state.get("agenda_window_reason") or ""),
                "body_homeostasis_focus": str(current_state.get("body_homeostasis_focus") or ""),
                "body_homeostasis_carry_bias": round(float(current_state.get("body_homeostasis_carry_bias") or 0.0), 4),
                "homeostasis_budget_focus": str(current_state.get("homeostasis_budget_focus") or ""),
                "homeostasis_budget_bias": round(float(current_state.get("homeostasis_budget_bias") or 0.0), 4),
                "relational_continuity_focus": str(current_state.get("relational_continuity_focus") or ""),
                "relational_continuity_carry_bias": round(float(current_state.get("relational_continuity_carry_bias") or 0.0), 4),
                "expressive_style_focus": str(current_state.get("expressive_style_focus") or ""),
                "expressive_style_carry_bias": round(float(current_state.get("expressive_style_carry_bias") or 0.0), 4),
                "expressive_style_history_focus": str(current_state.get("expressive_style_history_focus") or ""),
                "expressive_style_history_bias": round(float(current_state.get("expressive_style_history_bias") or 0.0), 4),
                "banter_style_focus": str(current_state.get("banter_style_focus") or ""),
                "lexical_variation_carry_bias": round(float(current_state.get("lexical_variation_carry_bias") or 0.0), 4),
                "person_registry_summary": {
                    "dominant_person_id": str((current_state.get("person_registry_snapshot") or {}).get("dominant_person_id") or ""),
                    "total_people": int((current_state.get("person_registry_snapshot") or {}).get("total_people") or 0),
                    "top_person_ids": list((current_state.get("person_registry_snapshot") or {}).get("top_person_ids") or []),
                },
                "group_thread_registry_summary": {
                    "dominant_thread_id": str((current_state.get("group_thread_registry_snapshot") or {}).get("dominant_thread_id") or ""),
                    "total_threads": int((current_state.get("group_thread_registry_snapshot") or {}).get("total_threads") or 0),
                    "top_thread_ids": list((current_state.get("group_thread_registry_snapshot") or {}).get("top_thread_ids") or []),
                },
                "discussion_thread_registry_summary": {
                    "dominant_thread_id": str((current_state.get("discussion_thread_registry_snapshot") or {}).get("dominant_thread_id") or ""),
                    "dominant_anchor": str((current_state.get("discussion_thread_registry_snapshot") or {}).get("dominant_anchor") or ""),
                    "dominant_issue_state": str((current_state.get("discussion_thread_registry_snapshot") or {}).get("dominant_issue_state") or ""),
                    "total_threads": int((current_state.get("discussion_thread_registry_snapshot") or {}).get("total_threads") or 0),
                    "top_thread_ids": list((current_state.get("discussion_thread_registry_snapshot") or {}).get("top_thread_ids") or []),
                },
                "group_thread_focus": str(current_state.get("group_thread_focus") or ""),
                "group_thread_carry_bias": round(float(current_state.get("group_thread_carry_bias") or 0.0), 4),
                "overnight_bias_roles": dict(response_hook.expression_hints.get("interaction_policy_overnight_bias_roles") or {}),
                "reaction_vs_overnight_bias": dict(response_hook.expression_hints.get("interaction_policy_reaction_vs_overnight_bias") or {}),
                "workspace_decision": dict((response_hook.expression_hints.get("conscious_workspace") or {})),
                "interaction_policy_packet": interaction_policy_packet,
                "memory_write_class_bias": memory_write_class_bias,
                "protection_mode_decision": protection_mode_decision,
                "body_recovery_guard": body_recovery_guard,
                "body_homeostasis_state": body_homeostasis_state,
                "homeostasis_budget_state": homeostasis_budget_state,
                "initiative_readiness": initiative_readiness,
                "agenda_state": agenda_state,
                "agenda_window_state": agenda_window_state,
                "commitment_state": commitment_state,
                "relational_style_memory_state": relational_style_memory_state,
                "cultural_conversation_state": cultural_conversation_state,
                "expressive_style_state": expressive_style_state,
                "lightness_budget_state": lightness_budget_state,
                "relational_continuity_state": relational_continuity_state,
                "relation_competition_state": dict(interaction_policy_packet.get("relation_competition_state") or {}),
                "social_topology_state": social_topology_state,
                "active_relation_table": dict(interaction_policy_packet.get("active_relation_table") or {}),
                "initiative_followup_bias": initiative_followup_bias,
                "temperament_trace": temperament_trace,
                "terrain_plasticity_update": dict(terrain_plasticity_update),
                "association_graph_state": dict(association_graph_state),
                "conversational_object_labels": list(response_hook.expression_hints.get("conversational_object_labels") or []),
                "conversational_object_pressure_balance": round(float(response_hook.expression_hints.get("conversational_object_pressure_balance") or 0.0), 4),
                "object_operation_question_budget": int(response_hook.expression_hints.get("object_operation_question_budget") or 0),
                "object_operation_question_pressure": round(float(response_hook.expression_hints.get("object_operation_question_pressure") or 0.0), 4),
                "object_operation_defer_dominance": round(float(response_hook.expression_hints.get("object_operation_defer_dominance") or 0.0), 4),
                "judgement_observed_count": int(len((response_hook.expression_hints.get("interaction_judgement_view") or {}).get("observed_signals") or [])),
                "judgement_inferred_count": int(len((response_hook.expression_hints.get("interaction_judgement_view") or {}).get("inferred_signals") or [])),
                "judgement_selected_object_labels": list((response_hook.expression_hints.get("interaction_judgement_view") or {}).get("selected_object_labels") or []),
                "judgement_deferred_object_labels": list((response_hook.expression_hints.get("interaction_judgement_view") or {}).get("deferred_object_labels") or []),
                "judgement_active_operation_labels": list((response_hook.expression_hints.get("interaction_judgement_view") or {}).get("active_operation_labels") or []),
                "judgement_intended_effect_labels": list((response_hook.expression_hints.get("interaction_judgement_view") or {}).get("intended_effect_labels") or []),
                "judgement_summary_observed": list((response_hook.expression_hints.get("interaction_judgement_summary") or {}).get("observed_lines") or []),
                "judgement_summary_inferred": list((response_hook.expression_hints.get("interaction_judgement_summary") or {}).get("inferred_lines") or []),
                "judgement_summary_objects": list((response_hook.expression_hints.get("interaction_judgement_summary") or {}).get("selected_object_lines") or []),
                "judgement_summary_deferred": list((response_hook.expression_hints.get("interaction_judgement_summary") or {}).get("deferred_object_lines") or []),
                "judgement_summary_operations": list((response_hook.expression_hints.get("interaction_judgement_summary") or {}).get("operation_lines") or []),
                "judgement_summary_effects": list((response_hook.expression_hints.get("interaction_judgement_summary") or {}).get("intended_effect_lines") or []),
                "condition_scene_lines": list((response_hook.expression_hints.get("interaction_condition_report") or {}).get("scene_lines") or []),
                "condition_relation_lines": list((response_hook.expression_hints.get("interaction_condition_report") or {}).get("relation_lines") or []),
                "condition_memory_lines": list((response_hook.expression_hints.get("interaction_condition_report") or {}).get("memory_lines") or []),
                "condition_integration_lines": list((response_hook.expression_hints.get("interaction_condition_report") or {}).get("integration_lines") or []),
                "inspection_report_lines": list((response_hook.expression_hints.get("interaction_inspection_report") or {}).get("report_lines") or []),
                "inspection_changed_sections": list((response_hook.expression_hints.get("interaction_inspection_report") or {}).get("changed_sections") or []),
                "audit_report_lines": list((response_hook.expression_hints.get("interaction_audit_bundle") or {}).get("report_lines") or []),
                "audit_key_metrics": dict((response_hook.expression_hints.get("interaction_audit_bundle") or {}).get("key_metrics") or {}),
                "audit_casebook_case_ids": [
                    f"case_{index}"
                    for index, _ in enumerate(
                        ((response_hook.expression_hints.get("interaction_audit_casebook") or {}).get("cases") or []),
                        start=1,
                    )
                ],
                "audit_reference_case_ids": list(response_hook.expression_hints.get("interaction_audit_reference_case_ids") or []),
                "audit_reference_case_meta": dict(response_hook.expression_hints.get("interaction_audit_reference_case_meta") or {}),
                "audit_comparison_report_lines": list((response_hook.expression_hints.get("interaction_audit_report") or {}).get("report_lines") or []),
                "audit_comparison_changed_sections": list((response_hook.expression_hints.get("interaction_audit_report") or {}).get("changed_sections") or []),
                "resonance_recommended_family": (response_hook.expression_hints.get("resonance_evaluation") or {}).get("recommended_family_id"),
                "other_person_detail_room": ((response_hook.expression_hints.get("resonance_evaluation") or {}).get("estimated_other_person_state") or {}).get("detail_room_level"),
                "other_person_acknowledgement_need": ((response_hook.expression_hints.get("resonance_evaluation") or {}).get("estimated_other_person_state") or {}).get("acknowledgement_need_level"),
                "other_person_pressure_sensitivity": ((response_hook.expression_hints.get("resonance_evaluation") or {}).get("estimated_other_person_state") or {}).get("pressure_sensitivity_level"),
                "related_person_id": post_hook.state.related_person_id,
                "attachment": round(float(post_hook.state.attachment), 4),
                "familiarity": round(float(post_hook.state.familiarity), 4),
                "trust_memory": round(float(post_hook.state.trust_memory), 4),
                "semantic_seed_focus": working_memory_seed.get("semantic_seed_focus"),
                "semantic_seed_anchor": working_memory_seed.get("semantic_seed_anchor"),
                "semantic_seed_strength": round(float(working_memory_seed.get("semantic_seed_strength") or 0.0), 4),
                "semantic_seed_recurrence": round(float(working_memory_seed.get("semantic_seed_recurrence") or 0.0), 4),
                "long_term_theme_focus": working_memory_seed.get("long_term_theme_focus"),
                "long_term_theme_anchor": working_memory_seed.get("long_term_theme_anchor"),
                "long_term_theme_strength": round(float(working_memory_seed.get("long_term_theme_strength") or 0.0), 4),
                "long_term_theme_kind": working_memory_seed.get("long_term_theme_kind"),
                "long_term_theme_summary": working_memory_seed.get("long_term_theme_summary"),
                "identity_arc_kind": working_memory_seed.get("identity_arc_kind"),
                "identity_arc_phase": working_memory_seed.get("identity_arc_phase"),
                "identity_arc_summary": working_memory_seed.get("identity_arc_summary"),
                "identity_arc_open_tension": working_memory_seed.get("identity_arc_open_tension"),
                "identity_arc_stability": round(float(working_memory_seed.get("identity_arc_stability") or 0.0), 4),
                "identity_arc_registry_summary": dict(working_memory_seed.get("identity_arc_registry_summary") or {}),
                "relation_arc_kind": working_memory_seed.get("relation_arc_kind"),
                "relation_arc_phase": working_memory_seed.get("relation_arc_phase"),
                "relation_arc_summary": working_memory_seed.get("relation_arc_summary"),
                "relation_arc_open_tension": working_memory_seed.get("relation_arc_open_tension"),
                "relation_arc_stability": round(float(working_memory_seed.get("relation_arc_stability") or 0.0), 4),
                "relation_arc_registry_summary": dict(working_memory_seed.get("relation_arc_registry_summary") or {}),
                "group_relation_arc_kind": working_memory_seed.get("group_relation_arc_kind"),
                "group_relation_arc_phase": working_memory_seed.get("group_relation_arc_phase"),
                "group_relation_arc_summary": working_memory_seed.get("group_relation_arc_summary"),
                "group_relation_arc_boundary_mode": working_memory_seed.get("group_relation_arc_boundary_mode"),
                "group_relation_arc_reentry_window_focus": working_memory_seed.get("group_relation_arc_reentry_window_focus"),
                "group_relation_arc_group_thread_id": working_memory_seed.get("group_relation_arc_group_thread_id"),
                "group_relation_arc_topology_focus": working_memory_seed.get("group_relation_arc_topology_focus"),
                "group_relation_arc_dominant_person_id": working_memory_seed.get("group_relation_arc_dominant_person_id"),
                "group_relation_arc_stability": round(float(working_memory_seed.get("group_relation_arc_stability") or 0.0), 4),
                "identity_arc": {
                    "arc_kind": working_memory_seed.get("identity_arc_kind"),
                    "phase": working_memory_seed.get("identity_arc_phase"),
                    "summary": working_memory_seed.get("identity_arc_summary"),
                    "open_tension": working_memory_seed.get("identity_arc_open_tension"),
                    "stability": round(float(working_memory_seed.get("identity_arc_stability") or 0.0), 4),
                    "memory_anchor": working_memory_seed.get("semantic_seed_anchor"),
                    "long_term_theme_focus": working_memory_seed.get("long_term_theme_focus"),
                    "long_term_theme_kind": working_memory_seed.get("long_term_theme_kind"),
                },
                "relation_arc": {
                    "arc_kind": working_memory_seed.get("relation_arc_kind"),
                    "phase": working_memory_seed.get("relation_arc_phase"),
                    "summary": working_memory_seed.get("relation_arc_summary"),
                    "open_tension": working_memory_seed.get("relation_arc_open_tension"),
                    "stability": round(float(working_memory_seed.get("relation_arc_stability") or 0.0), 4),
                    "related_person_id": working_memory_seed.get("related_person_id"),
                    "group_thread_id": working_memory_seed.get("group_thread_id"),
                },
                "group_relation_arc": {
                    "arc_kind": working_memory_seed.get("group_relation_arc_kind"),
                    "phase": working_memory_seed.get("group_relation_arc_phase"),
                    "summary": working_memory_seed.get("group_relation_arc_summary"),
                    "boundary_mode": working_memory_seed.get("group_relation_arc_boundary_mode"),
                    "reentry_window_focus": working_memory_seed.get("group_relation_arc_reentry_window_focus"),
                    "group_thread_id": working_memory_seed.get("group_relation_arc_group_thread_id"),
                    "topology_focus": working_memory_seed.get("group_relation_arc_topology_focus"),
                    "dominant_person_id": working_memory_seed.get("group_relation_arc_dominant_person_id"),
                    "stability": round(float(working_memory_seed.get("group_relation_arc_stability") or 0.0), 4),
                },
                "relation_seed_summary": working_memory_seed.get("relation_seed_summary"),
                "relation_seed_strength": round(float(working_memory_seed.get("relation_seed_strength") or 0.0), 4),
                "partner_address_hint": working_memory_seed.get("partner_address_hint"),
                "partner_timing_hint": working_memory_seed.get("partner_timing_hint"),
                "partner_stance_hint": working_memory_seed.get("partner_stance_hint"),
                "partner_social_interpretation": working_memory_seed.get("partner_social_interpretation"),
                "transfer_summary": dict(getattr(self, "_last_gate_context", {}).get("inner_os_transfer_summary") or {}),
                "reuse_trajectory": round(float(post_hook.state.reuse_trajectory), 4),
                "interference_pressure": round(float(post_hook.state.interference_pressure), 4),
                "consolidation_priority": round(float(post_hook.state.consolidation_priority), 4),
                "prospective_memory_pull": round(float(post_hook.state.prospective_memory_pull), 4),
                "llm_model": resolved_llm_model,
                "llm_model_source": resolved_llm_model_source,
                "force_llm_bridge": bool(
                    getattr(getattr(self, "config", None), "force_llm_bridge", False)
                ),
                "llm_bridge_called": bool(result.metrics.get("inner_os/llm_bridge_called") or 0.0),
                "llm_raw_text": llm_raw_text,
                "llm_raw_model": llm_raw_model,
                "llm_raw_model_source": llm_raw_model_source,
                "llm_final_text": llm_final_text,
                "llm_raw_differs_from_final": bool(
                    llm_raw_text and llm_final_text and llm_raw_text != llm_final_text
                ),
                "qualia_gate_reason": str(qualia_gate_snapshot.get("reason") or "").strip(),
            }
        )
        result.metrics.setdefault(
            "inner_os/llm_model_source_live_list",
            1.0 if resolved_llm_model_source == "live_list" else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/llm_model_source_cache",
            1.0 if resolved_llm_model_source == "cache" else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/llm_model_source_forced",
            1.0 if resolved_llm_model_source == "forced" else 0.0,
        )
        transfer_summary = dict(result.persona_meta["inner_os"].get("transfer_summary") or {})
        migration = dict(transfer_summary.get("migration") or {})
        result.metrics.setdefault(
            "inner_os/transfer_migration_active",
            1.0 if transfer_summary else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/transfer_migration_from_legacy",
            1.0 if migration.get("applied") else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/transfer_semantic_seed_visible",
            1.0 if transfer_summary.get("semantic_seed_anchor") else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/transfer_commitment_carry_visible",
            1.0 if transfer_summary.get("commitment_target_focus") else 0.0,
        )
        result.metrics.setdefault(
            "inner_os/transfer_target_model_requested",
            1.0 if transfer_summary.get("target_model") else 0.0,
        )
        current_state.setdefault("identity_arc_kind", working_memory_seed.get("identity_arc_kind"))
        current_state.setdefault("identity_arc_phase", working_memory_seed.get("identity_arc_phase"))
        current_state.setdefault("identity_arc_summary", working_memory_seed.get("identity_arc_summary"))
        current_state.setdefault("identity_arc_open_tension", working_memory_seed.get("identity_arc_open_tension"))
        current_state.setdefault("identity_arc_stability", working_memory_seed.get("identity_arc_stability"))
        current_state.setdefault("identity_arc_registry_summary", working_memory_seed.get("identity_arc_registry_summary"))
        current_state.setdefault("relation_arc_kind", working_memory_seed.get("relation_arc_kind"))
        current_state.setdefault("relation_arc_phase", working_memory_seed.get("relation_arc_phase"))
        current_state.setdefault("relation_arc_summary", working_memory_seed.get("relation_arc_summary"))
        current_state.setdefault("relation_arc_open_tension", working_memory_seed.get("relation_arc_open_tension"))
        current_state.setdefault("relation_arc_stability", working_memory_seed.get("relation_arc_stability"))
        current_state.setdefault("relation_arc_registry_summary", working_memory_seed.get("relation_arc_registry_summary"))
        current_state.setdefault("group_relation_arc_kind", working_memory_seed.get("group_relation_arc_kind"))
        current_state.setdefault("group_relation_arc_phase", working_memory_seed.get("group_relation_arc_phase"))
        current_state.setdefault("group_relation_arc_summary", working_memory_seed.get("group_relation_arc_summary"))
        current_state.setdefault("group_relation_arc_boundary_mode", working_memory_seed.get("group_relation_arc_boundary_mode"))
        current_state.setdefault("group_relation_arc_reentry_window_focus", working_memory_seed.get("group_relation_arc_reentry_window_focus"))
        current_state.setdefault("group_relation_arc_group_thread_id", working_memory_seed.get("group_relation_arc_group_thread_id"))
        current_state.setdefault("group_relation_arc_topology_focus", working_memory_seed.get("group_relation_arc_topology_focus"))
        current_state.setdefault("group_relation_arc_dominant_person_id", working_memory_seed.get("group_relation_arc_dominant_person_id"))
        current_state.setdefault("group_relation_arc_stability", working_memory_seed.get("group_relation_arc_stability"))
        continuity_summary = ContinuitySummaryBuilder().build(
            interaction_policy_packet=interaction_policy_packet,
            current_state=current_state,
            transfer_summary=transfer_summary,
        ).to_dict()
        result.persona_meta["inner_os"]["continuity_summary"] = continuity_summary
        result.metrics.setdefault("inner_os/continuity_summary_ready", 1.0)
        transfer_package = self.export_inner_os_transfer_package(
            result=result,
        )
        result.persona_meta["inner_os"]["transfer_package"] = transfer_package
        result.metrics.setdefault("inner_os/transfer_package_ready", 1.0)
        result.metrics.setdefault(
            "inner_os/transfer_package_written",
            1.0 if self._persist_inner_os_transfer_package(transfer_package) else 0.0,
        )
        dashboard_snapshot = self._build_inner_os_dashboard_snapshot(
            result=result,
            continuity_summary=continuity_summary,
            transfer_summary=transfer_summary,
        )
        result.persona_meta["inner_os"]["dashboard_snapshot_summary"] = {
            "schema": str(dashboard_snapshot.get("schema") or ""),
            "dominant_carry_channel": str(dashboard_snapshot.get("dominant_carry_channel") or ""),
            "transfer_target_model_requested": str(
                ((dashboard_snapshot.get("transfer") or {}).get("target_model_requested") or "")
            ),
            "social_topology_state": str(
                ((dashboard_snapshot.get("same_turn") or {}).get("social_topology_state") or "")
            ),
        }
        result.metrics.setdefault("inner_os/dashboard_snapshot_ready", 1.0)
        result.metrics.setdefault(
            "inner_os/dashboard_snapshot_written",
            1.0 if self._persist_inner_os_dashboard_snapshot(dashboard_snapshot) else 0.0,
        )
        final_surface_text = str(getattr(result.response, "text", "") or "").strip()
        final_user_text = str(user_text or "").strip()
        if final_user_text:
            if not hasattr(self, "_surface_user_history"):
                self._surface_user_history = deque(maxlen=4)
            self._surface_user_history.append(final_user_text)
        if final_surface_text:
            if not hasattr(self, "_surface_response_history"):
                self._surface_response_history = deque(maxlen=3)
            self._surface_response_history.append(final_surface_text)
        self._emit_inner_os_distillation_record(
            user_text=user_text,
            context_text=context,
            result=result,
        )
        return result

    def _emit_inner_os_distillation_record(
        self,
        *,
        user_text: Optional[str],
        context_text: Optional[str],
        result: RuntimeTurnResult,
    ) -> None:
        log_path = getattr(self, "_distillation_log_path", None)
        if log_path is None:
            return
        try:
            response = getattr(result, "response", None)
            persona_meta = dict(getattr(result, "persona_meta", {}) or {})
            inner_os_meta = dict(persona_meta.get("inner_os") or {})
            builder = getattr(self, "_distillation_record_builder", None) or InnerOSDistillationRecordBuilder()
            record = builder.build(
                turn_id=str(getattr(response, "trace_id", "") or ""),
                session_id=str(getattr(self, "_session_id", "") or ""),
                timestamp_ms=int(time.time() * 1000),
                user_text=user_text,
                context_text=context_text,
                response_text=getattr(response, "text", None),
                response_meta=self._serialize_response_meta(response),
                interaction_policy_packet=inner_os_meta.get("interaction_policy_packet"),
                persona_meta_inner_os=inner_os_meta,
                include_text=bool(getattr(self, "_distillation_log_include_text", False)),
            )
            append_jsonl(log_path, record.to_dict())
        except Exception:
            LOGGER.exception("Failed to emit inner_os distillation record")

    def export_inner_os_transfer_package(
        self,
        *,
        result: RuntimeTurnResult | None = None,
        nightly_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        builder = getattr(self, "_transfer_package_builder", None) or InnerOSTransferPackageBuilder()
        response = getattr(result, "response", None) if result is not None else None
        persona_meta = dict(getattr(result, "persona_meta", {}) or {}) if result is not None else {}
        inner_os_meta = dict(persona_meta.get("inner_os") or {})
        return builder.build(
            session_id=str(getattr(self, "_session_id", "") or ""),
            turn_id=str(getattr(response, "trace_id", "") or ""),
            timestamp_ms=int(time.time() * 1000),
            current_state={},
            last_gate_context=dict(getattr(self, "_last_gate_context", {}) or {}),
            persona_meta_inner_os=inner_os_meta,
            response_meta=self._serialize_response_meta(response),
            nightly_summary=nightly_summary,
        ).to_dict()

    def build_inner_os_model_swap_bundle(
        self,
        *,
        target_model: str,
        target_base_url: str = "",
        result: RuntimeTurnResult | None = None,
        nightly_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from emot_terrain_lab.terrain import llm as terrain_llm

        transfer_package = self.export_inner_os_transfer_package(
            result=result,
            nightly_summary=nightly_summary,
        )
        resolved_base = str(target_base_url or terrain_llm.LM_BASE or terrain_llm.CUSTOM_BASE or "").strip()
        target_model_text = str(target_model or "").strip()
        if resolved_base and target_model_text:
            terrain_llm.prefer_cached_model(resolved_base, target_model_text)
        return {
            "schema": "inner_os_model_swap_bundle/v1",
            "target_model": target_model_text,
            "target_base_url": resolved_base,
            "selection_mode": "cache_preferred" if resolved_base and target_model_text else "none",
            "transfer_package": transfer_package,
        }

    def load_inner_os_transfer_package(
        self,
        package_or_path: Mapping[str, Any] | str | Path,
        *,
        persist_normalized: bool = False,
    ) -> dict[str, Any]:
        builder = getattr(self, "_transfer_package_builder", None) or InnerOSTransferPackageBuilder()
        if isinstance(package_or_path, (str, Path)):
            payload = json.loads(Path(package_or_path).read_text(encoding="utf-8"))
        else:
            payload = dict(package_or_path or {})
        normalized = builder.normalize(payload)
        restored_seed = self.apply_inner_os_transfer_package(normalized)
        working_memory_seed = builder.to_working_memory_seed(normalized)
        if persist_normalized:
            self._persist_inner_os_transfer_package(normalized)
        migration = dict(normalized.get("migration") or {})
        summary = {
            "schema": normalized.get("schema"),
            "package_version": normalized.get("package_version"),
            "migration": migration,
            "source_model": dict(normalized.get("source_model") or {}),
            "restored_seed": restored_seed,
            "working_memory_seed": working_memory_seed,
            "commitment_target_focus": restored_seed.get("commitment_target_focus", ""),
            "agenda_focus": restored_seed.get("agenda_focus", ""),
            "agenda_bias": restored_seed.get("agenda_bias", 0.0),
            "agenda_window_focus": restored_seed.get("agenda_window_focus", ""),
            "agenda_window_bias": restored_seed.get("agenda_window_bias", 0.0),
            "learning_mode_focus": restored_seed.get("learning_mode_focus", ""),
            "learning_mode_carry_bias": restored_seed.get("learning_mode_carry_bias", 0.0),
            "social_experiment_focus": restored_seed.get("social_experiment_focus", ""),
            "social_experiment_carry_bias": restored_seed.get("social_experiment_carry_bias", 0.0),
            "temporal_membrane_focus": restored_seed.get("temporal_membrane_focus", ""),
            "temporal_timeline_bias": restored_seed.get("temporal_timeline_bias", 0.0),
            "temporal_reentry_bias": restored_seed.get("temporal_reentry_bias", 0.0),
            "temporal_supersession_bias": restored_seed.get("temporal_supersession_bias", 0.0),
            "temporal_continuity_bias": restored_seed.get("temporal_continuity_bias", 0.0),
            "temporal_relation_reentry_bias": restored_seed.get("temporal_relation_reentry_bias", 0.0),
            "identity_arc_kind": restored_seed.get("identity_arc_kind", ""),
            "identity_arc_phase": restored_seed.get("identity_arc_phase", ""),
            "identity_arc_summary": restored_seed.get("identity_arc_summary", ""),
            "identity_arc_registry_summary": dict(restored_seed.get("identity_arc_registry_summary") or {}),
            "relation_arc_kind": restored_seed.get("relation_arc_kind", ""),
            "relation_arc_phase": restored_seed.get("relation_arc_phase", ""),
            "relation_arc_summary": restored_seed.get("relation_arc_summary", ""),
            "relation_arc_registry_summary": dict(restored_seed.get("relation_arc_registry_summary") or {}),
            "group_relation_arc_kind": restored_seed.get("group_relation_arc_kind", ""),
            "group_relation_arc_phase": restored_seed.get("group_relation_arc_phase", ""),
            "group_relation_arc_summary": restored_seed.get("group_relation_arc_summary", ""),
            "group_relation_arc_boundary_mode": restored_seed.get("group_relation_arc_boundary_mode", ""),
            "group_relation_arc_reentry_window_focus": restored_seed.get("group_relation_arc_reentry_window_focus", ""),
            "group_relation_arc_group_thread_id": restored_seed.get("group_relation_arc_group_thread_id", ""),
            "group_relation_arc_stability": restored_seed.get("group_relation_arc_stability", 0.0),
            "initiative_followup_state": restored_seed.get("initiative_followup_state", "hold"),
            "homeostasis_budget_focus": restored_seed.get("homeostasis_budget_focus", ""),
            "homeostasis_budget_bias": restored_seed.get("homeostasis_budget_bias", 0.0),
            "person_registry_total_people": int(
                ((restored_seed.get("person_registry_snapshot") or {}).get("total_people") or 0)
            ),
            "group_thread_total_threads": int(
                ((restored_seed.get("group_thread_registry_snapshot") or {}).get("total_threads") or 0)
            ),
            "discussion_thread_total_threads": int(
                ((restored_seed.get("discussion_thread_registry_snapshot") or {}).get("total_threads") or 0)
            ),
            "discussion_thread_dominant_anchor": str(
                ((restored_seed.get("discussion_thread_registry_snapshot") or {}).get("dominant_anchor") or "")
            ),
            "discussion_thread_dominant_issue_state": str(
                ((restored_seed.get("discussion_thread_registry_snapshot") or {}).get("dominant_issue_state") or "")
            ),
            "autobiographical_thread_mode": restored_seed.get("autobiographical_thread_mode", ""),
            "autobiographical_thread_anchor": restored_seed.get("autobiographical_thread_anchor", ""),
            "autobiographical_thread_focus": restored_seed.get("autobiographical_thread_focus", ""),
            "autobiographical_thread_strength": restored_seed.get("autobiographical_thread_strength", 0.0),
            "group_thread_focus": restored_seed.get("group_thread_focus", ""),
            "group_thread_carry_bias": restored_seed.get("group_thread_carry_bias", 0.0),
            "semantic_seed_focus": working_memory_seed.get("semantic_seed_focus", ""),
            "semantic_seed_anchor": working_memory_seed.get("semantic_seed_anchor", ""),
        }
        gate = getattr(self, "_last_gate_context", None)
        if isinstance(gate, dict):
            gate["inner_os_transfer_summary"] = dict(summary)
        return summary

    def warm_start_from_transfer_package(
        self,
        package_or_path: Mapping[str, Any] | str | Path,
        *,
        target_model: str = "",
        target_base_url: str = "",
        persist_normalized: bool = False,
        prefer_target_model: bool = True,
    ) -> dict[str, Any]:
        from emot_terrain_lab.terrain import llm as terrain_llm

        target_model_text = str(target_model or "").strip()
        resolved_base = str(target_base_url or terrain_llm.LM_BASE or terrain_llm.CUSTOM_BASE or "").strip()
        if prefer_target_model and resolved_base and target_model_text:
            terrain_llm.prefer_cached_model(resolved_base, target_model_text)
        summary = self.load_inner_os_transfer_package(
            package_or_path,
            persist_normalized=persist_normalized,
        )
        gate = getattr(self, "_last_gate_context", None)
        if isinstance(gate, dict):
            gate["inner_os_transfer_target_model"] = target_model_text
            gate["inner_os_transfer_target_base_url"] = resolved_base
        summary["target_model"] = target_model_text
        summary["target_base_url"] = resolved_base
        summary["selected_model_after_warm_start"] = (
            terrain_llm.get_cached_selected_model(resolved_base) if resolved_base else None
        )
        if isinstance(gate, dict):
            gate["inner_os_transfer_summary"] = dict(summary)
        return summary

    def build_inner_os_state_seed_from_transfer_package(
        self,
        package: Mapping[str, Any],
    ) -> dict[str, Any]:
        builder = getattr(self, "_transfer_package_builder", None) or InnerOSTransferPackageBuilder()
        normalized = builder.normalize(package)
        return builder.to_runtime_seed(normalized)

    def apply_inner_os_transfer_package(
        self,
        package: Mapping[str, Any],
    ) -> dict[str, Any]:
        builder = getattr(self, "_transfer_package_builder", None) or InnerOSTransferPackageBuilder()
        normalized_package = builder.normalize(package)
        seed = self.build_inner_os_state_seed_from_transfer_package(normalized_package)
        working_memory_seed = builder.to_working_memory_seed(normalized_package)
        gate = getattr(self, "_last_gate_context", None)
        if not isinstance(gate, dict):
            self._last_gate_context = {}
            gate = self._last_gate_context

        gate["inner_os_prev_qualia"] = list(seed.get("prev_qualia") or [])
        gate["inner_os_prev_qualia_habituation"] = list(seed.get("prev_qualia_habituation") or [])
        gate["inner_os_prev_protection_grad_x"] = list(seed.get("prev_protection_grad_x") or [])
        gate["inner_os_prev_affective_position"] = dict(seed.get("prev_affective_position") or {})
        gate["inner_os_affective_terrain_state"] = dict(seed.get("affective_terrain_state") or {})
        gate["inner_os_association_graph_state"] = dict(seed.get("association_graph_state") or {})
        gate["inner_os_terrain_reweighting_bias"] = float(seed.get("terrain_reweighting_bias") or 0.0)
        gate["inner_os_association_reweighting_bias"] = float(seed.get("association_reweighting_bias") or 0.0)
        gate["inner_os_association_reweighting_focus"] = str(seed.get("association_reweighting_focus") or "")
        gate["inner_os_association_reweighting_reason"] = str(seed.get("association_reweighting_reason") or "")
        gate["inner_os_insight_reframing_bias"] = float(seed.get("insight_reframing_bias") or 0.0)
        gate["inner_os_insight_class_focus"] = str(seed.get("insight_class_focus") or "")
        gate["inner_os_insight_terrain_shape_target"] = str(seed.get("insight_terrain_shape_target") or "")
        gate["inner_os_insight_link_counts"] = dict(seed.get("insight_link_counts") or {})
        gate["inner_os_insight_class_counts"] = dict(seed.get("insight_class_counts") or {})
        gate["inner_os_initiative_followup_bias"] = float(seed.get("initiative_followup_bias") or 0.0)
        gate["inner_os_initiative_followup_state"] = str(seed.get("initiative_followup_state") or "hold")
        gate["inner_os_commitment_target_focus"] = str(seed.get("commitment_target_focus") or "")
        gate["inner_os_commitment_state_focus"] = str(seed.get("commitment_state_focus") or "waver")
        gate["inner_os_commitment_carry_bias"] = float(seed.get("commitment_carry_bias") or 0.0)
        gate["inner_os_commitment_followup_focus"] = str(seed.get("commitment_followup_focus") or "")
        gate["inner_os_commitment_mode_focus"] = str(seed.get("commitment_mode_focus") or "")
        gate["inner_os_commitment_carry_reason"] = str(seed.get("commitment_carry_reason") or "")
        gate["inner_os_agenda_focus"] = str(seed.get("agenda_focus") or "")
        gate["inner_os_agenda_bias"] = float(seed.get("agenda_bias") or 0.0)
        gate["inner_os_agenda_reason"] = str(seed.get("agenda_reason") or "")
        gate["inner_os_agenda_window_focus"] = str(seed.get("agenda_window_focus") or "")
        gate["inner_os_agenda_window_bias"] = float(seed.get("agenda_window_bias") or 0.0)
        gate["inner_os_agenda_window_reason"] = str(seed.get("agenda_window_reason") or "")
        gate["inner_os_learning_mode_focus"] = str(seed.get("learning_mode_focus") or "")
        gate["inner_os_learning_mode_carry_bias"] = float(seed.get("learning_mode_carry_bias") or 0.0)
        gate["inner_os_social_experiment_focus"] = str(seed.get("social_experiment_focus") or "")
        gate["inner_os_social_experiment_carry_bias"] = float(seed.get("social_experiment_carry_bias") or 0.0)
        gate["inner_os_temporal_membrane_focus"] = str(seed.get("temporal_membrane_focus") or "")
        gate["inner_os_temporal_timeline_bias"] = float(seed.get("temporal_timeline_bias") or 0.0)
        gate["inner_os_temporal_reentry_bias"] = float(seed.get("temporal_reentry_bias") or 0.0)
        gate["inner_os_temporal_supersession_bias"] = float(seed.get("temporal_supersession_bias") or 0.0)
        gate["inner_os_temporal_continuity_bias"] = float(seed.get("temporal_continuity_bias") or 0.0)
        gate["inner_os_temporal_relation_reentry_bias"] = float(seed.get("temporal_relation_reentry_bias") or 0.0)
        gate["inner_os_identity_arc_kind"] = str(seed.get("identity_arc_kind") or "")
        gate["inner_os_identity_arc_phase"] = str(seed.get("identity_arc_phase") or "")
        gate["inner_os_identity_arc_summary"] = str(seed.get("identity_arc_summary") or "")
        gate["inner_os_identity_arc_open_tension"] = str(seed.get("identity_arc_open_tension") or "")
        gate["inner_os_identity_arc_stability"] = float(seed.get("identity_arc_stability") or 0.0)
        gate["inner_os_identity_arc_registry_summary"] = dict(seed.get("identity_arc_registry_summary") or {})
        gate["inner_os_relation_arc_kind"] = str(seed.get("relation_arc_kind") or "")
        gate["inner_os_relation_arc_phase"] = str(seed.get("relation_arc_phase") or "")
        gate["inner_os_relation_arc_summary"] = str(seed.get("relation_arc_summary") or "")
        gate["inner_os_relation_arc_open_tension"] = str(seed.get("relation_arc_open_tension") or "")
        gate["inner_os_relation_arc_stability"] = float(seed.get("relation_arc_stability") or 0.0)
        gate["inner_os_relation_arc_registry_summary"] = dict(seed.get("relation_arc_registry_summary") or {})
        gate["inner_os_group_relation_arc_kind"] = str(seed.get("group_relation_arc_kind") or "")
        gate["inner_os_group_relation_arc_phase"] = str(seed.get("group_relation_arc_phase") or "")
        gate["inner_os_group_relation_arc_summary"] = str(seed.get("group_relation_arc_summary") or "")
        gate["inner_os_group_relation_arc_boundary_mode"] = str(seed.get("group_relation_arc_boundary_mode") or "")
        gate["inner_os_group_relation_arc_reentry_window_focus"] = str(seed.get("group_relation_arc_reentry_window_focus") or "")
        gate["inner_os_group_relation_arc_group_thread_id"] = str(seed.get("group_relation_arc_group_thread_id") or "")
        gate["inner_os_group_relation_arc_topology_focus"] = str(seed.get("group_relation_arc_topology_focus") or "")
        gate["inner_os_group_relation_arc_dominant_person_id"] = str(seed.get("group_relation_arc_dominant_person_id") or "")
        gate["inner_os_group_relation_arc_stability"] = float(seed.get("group_relation_arc_stability") or 0.0)
        gate["inner_os_body_homeostasis_focus"] = str(seed.get("body_homeostasis_focus") or "")
        gate["inner_os_body_homeostasis_carry_bias"] = float(seed.get("body_homeostasis_carry_bias") or 0.0)
        gate["inner_os_homeostasis_budget_focus"] = str(seed.get("homeostasis_budget_focus") or "")
        gate["inner_os_homeostasis_budget_bias"] = float(seed.get("homeostasis_budget_bias") or 0.0)
        gate["inner_os_relational_continuity_focus"] = str(seed.get("relational_continuity_focus") or "")
        gate["inner_os_relational_continuity_carry_bias"] = float(seed.get("relational_continuity_carry_bias") or 0.0)
        gate["inner_os_expressive_style_focus"] = str(seed.get("expressive_style_focus") or "")
        gate["inner_os_expressive_style_carry_bias"] = float(seed.get("expressive_style_carry_bias") or 0.0)
        gate["inner_os_expressive_style_history_focus"] = str(seed.get("expressive_style_history_focus") or "")
        gate["inner_os_expressive_style_history_bias"] = float(seed.get("expressive_style_history_bias") or 0.0)
        gate["inner_os_banter_style_focus"] = str(seed.get("banter_style_focus") or "")
        gate["inner_os_lexical_variation_carry_bias"] = float(seed.get("lexical_variation_carry_bias") or 0.0)
        gate["inner_os_person_registry_snapshot"] = dict(seed.get("person_registry_snapshot") or {})
        gate["inner_os_group_thread_registry_snapshot"] = dict(seed.get("group_thread_registry_snapshot") or {})
        gate["inner_os_discussion_thread_registry_snapshot"] = dict(seed.get("discussion_thread_registry_snapshot") or {})
        gate["inner_os_autobiographical_thread_mode"] = str(seed.get("autobiographical_thread_mode") or "")
        gate["inner_os_autobiographical_thread_anchor"] = str(seed.get("autobiographical_thread_anchor") or "")
        gate["inner_os_autobiographical_thread_focus"] = str(seed.get("autobiographical_thread_focus") or "")
        gate["inner_os_autobiographical_thread_strength"] = float(seed.get("autobiographical_thread_strength") or 0.0)
        gate["inner_os_group_thread_focus"] = str(seed.get("group_thread_focus") or "")
        gate["inner_os_group_thread_carry_bias"] = float(seed.get("group_thread_carry_bias") or 0.0)
        gate["inner_os_temperament_trace"] = {
            "temperament_forward_trace": float(seed.get("temperament_forward_trace") or 0.0),
            "temperament_guard_trace": float(seed.get("temperament_guard_trace") or 0.0),
            "temperament_bond_trace": float(seed.get("temperament_bond_trace") or 0.0),
            "temperament_recovery_trace": float(seed.get("temperament_recovery_trace") or 0.0),
        }
        gate["inner_os_transfer_package_migration"] = dict(normalized_package.get("migration") or {})
        surface_world = getattr(self, "_surface_world_state", None)
        if not isinstance(surface_world, dict):
            self._surface_world_state = {}
            surface_world = self._surface_world_state
        existing_seed = dict(surface_world.get("working_memory_seed") or {})
        merged_seed = {
            **existing_seed,
            **{
                key: value
                for key, value in working_memory_seed.items()
                if value is not None
                and not (isinstance(value, str) and not value.strip())
                and not (isinstance(value, (int, float)) and float(value) == 0.0)
                and not (isinstance(value, (list, tuple, dict)) and len(value) == 0)
            },
        }
        if merged_seed:
            surface_world["working_memory_seed"] = merged_seed
        return seed

    def _load_inner_os_transfer_package_from_disk(self) -> bool:
        path = getattr(self, "_transfer_package_path", None)
        if path is None or not path.exists():
            return False
        try:
            summary = self.load_inner_os_transfer_package(path, persist_normalized=True)
            migration = dict(summary.get("migration") or {})
            if migration.get("applied"):
                LOGGER.info(
                    "Loaded legacy inner_os transfer package and normalized it (source_schema=%s)",
                    migration.get("source_schema", "legacy"),
                )
            return True
        except Exception:
            LOGGER.exception("Failed to load inner_os transfer package")
            return False

    def _build_inner_os_dashboard_snapshot(
        self,
        *,
        result: RuntimeTurnResult,
        continuity_summary: Mapping[str, Any],
        transfer_summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        response = getattr(result, "response", None)
        response_meta = self._serialize_response_meta(response) or {}
        metrics = dict(getattr(result, "metrics", {}) or {})
        same_turn = dict((continuity_summary or {}).get("same_turn") or {})
        overnight = dict((continuity_summary or {}).get("overnight") or {})
        carry_strengths = dict((continuity_summary or {}).get("carry_strengths") or {})
        transfer = dict((continuity_summary or {}).get("transfer") or {})
        same_turn_temporal_mode = str(same_turn.get("temporal_membrane_mode") or "")
        same_turn_temporal_reentry_pull = round(float(same_turn.get("temporal_reentry_pull") or 0.0), 4)
        overnight_temporal_focus = str(overnight.get("temporal_membrane_focus") or "")
        overnight_temporal_reentry_bias = round(float(overnight.get("temporal_reentry_bias") or 0.0), 4)
        transfer_temporal_focus = str(transfer_summary.get("temporal_membrane_focus") or "")
        transfer_temporal_reentry_bias = round(float(transfer_summary.get("temporal_reentry_bias") or 0.0), 4)
        temporal_reentry_carry_strength = round(float(carry_strengths.get("temporal_reentry") or 0.0), 4)
        temporal_focus_alignment = bool(
            overnight_temporal_focus
            and transfer_temporal_focus
            and overnight_temporal_focus == transfer_temporal_focus
        )
        same_turn_boundary_gate_mode = str(same_turn.get("boundary_gate_mode") or "")
        same_turn_boundary_transform_mode = str(same_turn.get("boundary_transform_mode") or "")
        same_turn_boundary_softened = list(same_turn.get("boundary_softened_acts") or [])
        same_turn_boundary_withheld = list(same_turn.get("boundary_withheld_acts") or [])
        same_turn_boundary_deferred = list(same_turn.get("boundary_deferred_topics") or [])
        same_turn_boundary_residual_pressure = round(float(same_turn.get("boundary_residual_pressure") or 0.0), 4)
        same_turn_residual_mode = str(same_turn.get("residual_reflection_mode") or "")
        same_turn_residual_focus = str(same_turn.get("residual_reflection_focus") or "")
        same_turn_residual_strength = round(float(same_turn.get("residual_reflection_strength") or 0.0), 4)
        same_turn_contact_state = str(same_turn.get("contact_reflection_state") or "")
        same_turn_contact_style = str(same_turn.get("contact_reflection_style") or "")
        same_turn_contact_transmit = round(float(same_turn.get("contact_transmit_share") or 0.0), 4)
        same_turn_contact_reflect = round(float(same_turn.get("contact_reflect_share") or 0.0), 4)
        same_turn_contact_absorb = round(float(same_turn.get("contact_absorb_share") or 0.0), 4)
        same_turn_contact_block = round(float(same_turn.get("contact_block_share") or 0.0), 4)
        return {
            "schema": "inner_os_dashboard_snapshot/v1",
            "timestamp_ms": int(time.time() * 1000),
            "session_id": str(getattr(self, "_session_id", "") or ""),
            "turn_id": str(response_meta.get("trace_id") or ""),
            "model": {
                "name": str(response_meta.get("model") or ""),
                "source": str(response_meta.get("model_source") or ""),
            },
            "same_turn": {
                "protection_mode": str(same_turn.get("protection_mode") or ""),
                "memory_write_class": str(same_turn.get("memory_write_class") or ""),
                "agenda_state": str(same_turn.get("agenda_state") or ""),
                "agenda_window_state": str(same_turn.get("agenda_window_state") or ""),
                "agenda_window_carry_target": str(same_turn.get("agenda_window_carry_target") or ""),
                "commitment_target": str(same_turn.get("commitment_target") or ""),
                "temporal_membrane_mode": same_turn_temporal_mode,
                "temporal_timeline_coherence": round(float(same_turn.get("temporal_timeline_coherence") or 0.0), 4),
                "temporal_reentry_pull": same_turn_temporal_reentry_pull,
                "temporal_supersession_pressure": round(float(same_turn.get("temporal_supersession_pressure") or 0.0), 4),
                "temporal_continuity_pressure": round(float(same_turn.get("temporal_continuity_pressure") or 0.0), 4),
                "temporal_relation_reentry_pull": round(float(same_turn.get("temporal_relation_reentry_pull") or 0.0), 4),
                "boundary_gate_mode": same_turn_boundary_gate_mode,
                "boundary_transform_mode": same_turn_boundary_transform_mode,
                "boundary_softened_count": len(same_turn_boundary_softened),
                "boundary_withheld_count": len(same_turn_boundary_withheld),
                "boundary_deferred_count": len(same_turn_boundary_deferred),
                "boundary_residual_pressure": same_turn_boundary_residual_pressure,
                "residual_reflection_mode": same_turn_residual_mode,
                "residual_reflection_focus": same_turn_residual_focus,
                "residual_reflection_strength": same_turn_residual_strength,
                "contact_reflection_state": same_turn_contact_state,
                "contact_reflection_style": same_turn_contact_style,
                "contact_transmit_share": same_turn_contact_transmit,
                "contact_reflect_share": same_turn_contact_reflect,
                "contact_absorb_share": same_turn_contact_absorb,
                "contact_block_share": same_turn_contact_block,
                "body_homeostasis_state": str(same_turn.get("body_homeostasis_state") or ""),
                "homeostasis_budget_state": str(same_turn.get("homeostasis_budget_state") or ""),
                "relational_continuity_state": str(same_turn.get("relational_continuity_state") or ""),
                "relation_competition_state": str(same_turn.get("relation_competition_state") or ""),
                "social_topology_state": str(same_turn.get("social_topology_state") or ""),
                "dominant_person_id": str(same_turn.get("dominant_person_id") or ""),
                "active_relation_total_people": int(same_turn.get("active_relation_total_people") or 0),
            },
            "overnight": dict(overnight),
            "carry_strengths": {
                str(key): round(float(value), 4) for key, value in carry_strengths.items()
            },
            "dominant_carry_channel": str((continuity_summary or {}).get("dominant_carry_channel") or ""),
            "transfer": {
                "migration_active": bool(transfer.get("migration_active", False)),
                "from_legacy": bool(transfer.get("from_legacy", False)),
                "semantic_seed_visible": bool(transfer.get("semantic_seed_visible", False)),
                "commitment_carry_visible": bool(transfer.get("commitment_carry_visible", False)),
                "target_model_requested": str(transfer.get("target_model_requested") or ""),
                "temporal_membrane_focus": transfer_temporal_focus,
                "temporal_timeline_bias": round(float(transfer_summary.get("temporal_timeline_bias") or 0.0), 4),
                "temporal_reentry_bias": transfer_temporal_reentry_bias,
                "temporal_supersession_bias": round(float(transfer_summary.get("temporal_supersession_bias") or 0.0), 4),
                "temporal_continuity_bias": round(float(transfer_summary.get("temporal_continuity_bias") or 0.0), 4),
                "temporal_relation_reentry_bias": round(float(transfer_summary.get("temporal_relation_reentry_bias") or 0.0), 4),
            },
            "temporal_alignment": {
                "same_turn_mode": same_turn_temporal_mode,
                "overnight_focus": overnight_temporal_focus,
                "transfer_focus": transfer_temporal_focus,
                "focus_alignment": temporal_focus_alignment,
                "same_to_overnight_reentry_delta": round(
                    overnight_temporal_reentry_bias - same_turn_temporal_reentry_pull,
                    4,
                ),
                "overnight_to_transfer_reentry_delta": round(
                    transfer_temporal_reentry_bias - overnight_temporal_reentry_bias,
                    4,
                ),
                "reentry_carry_visible": temporal_reentry_carry_strength > 0.0,
                "reentry_carry_strength": temporal_reentry_carry_strength,
            },
            "boundary_alignment": {
                "gate_mode": same_turn_boundary_gate_mode,
                "transform_mode": same_turn_boundary_transform_mode,
                "softened_count": len(same_turn_boundary_softened),
                "withheld_count": len(same_turn_boundary_withheld),
                "deferred_count": len(same_turn_boundary_deferred),
                "residual_pressure": same_turn_boundary_residual_pressure,
                "residual_mode": same_turn_residual_mode,
                "residual_focus": same_turn_residual_focus,
                "residual_strength": same_turn_residual_strength,
                "unsaid_pressure_visible": bool(
                    same_turn_boundary_gate_mode
                    or same_turn_boundary_transform_mode
                    or same_turn_boundary_softened
                    or same_turn_boundary_withheld
                    or same_turn_boundary_deferred
                    or same_turn_residual_strength > 0.0
                ),
            },
            "contact_alignment": {
                "state": same_turn_contact_state,
                "style": same_turn_contact_style,
                "transmit_share": same_turn_contact_transmit,
                "reflect_share": same_turn_contact_reflect,
                "absorb_share": same_turn_contact_absorb,
                "block_share": same_turn_contact_block,
                "reflection_visible": bool(
                    same_turn_contact_state
                    or same_turn_contact_style
                    or same_turn_contact_reflect > 0.0
                    or same_turn_contact_absorb > 0.0
                    or same_turn_contact_block > 0.0
                ),
            },
            "metrics": {
                "social_topology_score": round(float(metrics.get("inner_os/social_topology_score") or 0.0), 4),
                "social_topology_winner_margin": round(
                    float(metrics.get("inner_os/social_topology_winner_margin") or 0.0), 4
                ),
                "relation_competition_level": round(
                    float(metrics.get("inner_os/relation_competition_level") or 0.0), 4
                ),
                "continuity_summary_ready": round(
                    float(metrics.get("inner_os/continuity_summary_ready") or 0.0), 4
                ),
                "transfer_package_ready": round(
                    float(metrics.get("inner_os/transfer_package_ready") or 0.0), 4
                ),
            },
            "continuity_summary": dict(continuity_summary or {}),
            "transfer_summary": dict(transfer_summary or {}),
        }

    def _persist_inner_os_transfer_package(
        self,
        package: Mapping[str, Any],
    ) -> bool:
        path = getattr(self, "_transfer_package_path", None)
        if path is None:
            return False
        try:
            builder = getattr(self, "_transfer_package_builder", None) or InnerOSTransferPackageBuilder()
            normalized = builder.normalize(package)
            path.write_text(
                json.dumps(dict(normalized or {}), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except Exception:
            LOGGER.exception("Failed to persist inner_os transfer package")
            return False

    def _persist_inner_os_dashboard_snapshot(
        self,
        snapshot: Mapping[str, Any],
    ) -> bool:
        path = getattr(self, "_dashboard_snapshot_path", None)
        if path is None:
            return False
        try:
            path.write_text(
                json.dumps(dict(snapshot or {}), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except Exception:
            LOGGER.exception("Failed to persist inner_os dashboard snapshot")
            return False

    def _run_inner_os_live_response_loop(
        self,
        *,
        response: HubResponse,
        initial_hook,
        current_state: Mapping[str, Any],
        safety_bias: float,
    ) -> tuple[HubResponse, Any, int]:
        hook = initial_hook
        shaped_response = response
        steps = 0
        min_steps = 2
        max_steps = 4
        loop_state = dict(current_state)
        previous_stream = self._inner_os_stream_state_from_hints(hook.expression_hints)
        last_style_feedback: Dict[str, Any] = {}
        persistent_planning_hints = {
            key: value
            for key, value in (
                (
                    "interaction_policy_packet",
                    dict(hook.expression_hints.get("interaction_policy_packet") or {}),
                ),
                (
                    "interaction_policy_dialogue_act",
                    str(hook.expression_hints.get("interaction_policy_dialogue_act") or ""),
                ),
                (
                    "actuation_plan",
                    dict(hook.expression_hints.get("actuation_plan") or {}),
                ),
                (
                    "action_posture",
                    dict(hook.expression_hints.get("action_posture") or {}),
                ),
                (
                    "planned_content_sequence",
                    list(
                        (getattr(response, "controls_used", {}) or {}).get(
                            "inner_os_planned_content_sequence"
                        )
                        or getattr(self, "_last_planned_content_sequence", [])
                        or []
                    ),
                ),
            )
            if value
        }
        bridge_allowed = getattr(response, "controls_used", {}).get(
            "inner_os_allow_guarded_narrative_bridge"
        )
        if any(
            str(item.get("act") or "").strip()
            in {"offer_small_opening_line", "offer_small_opening_frame"}
            for item in (persistent_planning_hints.get("planned_content_sequence") or [])
            if isinstance(item, Mapping)
        ):
            bridge_allowed = False
        if bridge_allowed is None:
            bridge_allowed = hook.expression_hints.get("allow_guarded_narrative_bridge")
        persistent_bridge_hints = {
            "llm_raw_text": str(
                hook.expression_hints.get("llm_raw_text")
                or getattr(response, "controls_used", {}).get("inner_os_llm_raw_text")
                or ""
            ).strip(),
            "allow_guarded_narrative_bridge": bool(bridge_allowed),
        }
        if persistent_bridge_hints["llm_raw_text"]:
            hook.expression_hints.update(persistent_bridge_hints)
        if persistent_planning_hints:
            hook.expression_hints.update(persistent_planning_hints)

        def _stamp_persistent_inner_os_controls(
            target: Optional[HubResponse],
        ) -> Optional[HubResponse]:
            if target is None:
                return target
            controls_used = dict(getattr(target, "controls_used", {}) or {})
            planned_sequence = persistent_planning_hints.get("planned_content_sequence")
            if isinstance(planned_sequence, list) and planned_sequence:
                controls_used["inner_os_planned_content_sequence"] = [
                    dict(item)
                    for item in planned_sequence
                    if isinstance(item, Mapping)
                ]
            raw_bridge_text = str(persistent_bridge_hints.get("llm_raw_text") or "").strip()
            if raw_bridge_text:
                controls_used["inner_os_llm_raw_text"] = raw_bridge_text
            controls_used["inner_os_allow_guarded_narrative_bridge"] = bool(
                persistent_bridge_hints.get("allow_guarded_narrative_bridge", False)
            )
            target.controls_used = controls_used
            return target

        while steps < max_steps:
            shaped_response = _stamp_persistent_inner_os_controls(shaped_response)
            shaped_response = self._apply_inner_os_surface_policy(
                shaped_response,
                hook.expression_hints,
                hook.conscious_access,
            )
            shaped_response = self._apply_inner_os_surface_profile(
                shaped_response,
                hook.expression_hints,
            )
            steps += 1
            if steps >= max_steps:
                break
            loop_state = dict(loop_state)
            loop_state["interaction_stream_state"] = self._inner_os_stream_state_from_hints(
                hook.expression_hints
            )
            if last_style_feedback:
                loop_state["recent_strain"] = max(
                    0.0,
                    min(
                        1.0,
                        float(loop_state.get("recent_strain") or 0.0)
                        + float(last_style_feedback.get("strain_delta") or 0.0),
                    ),
                )
                loop_state["social_grounding"] = max(
                    0.0,
                    min(
                        1.0,
                        float(loop_state.get("social_grounding") or 0.0)
                        + float(last_style_feedback.get("grounding_delta") or 0.0),
                    ),
                )
                loop_state["contact_readiness"] = max(
                    0.0,
                    min(
                        1.0,
                        float(loop_state.get("contact_readiness") or 0.0)
                        + float(last_style_feedback.get("contact_delta") or 0.0),
                    ),
                )
                loop_state["opening_pace_windowed"] = last_style_feedback.get("next_opening_pace") or loop_state.get("opening_pace_windowed")
                loop_state["return_gaze_expectation"] = last_style_feedback.get("next_return_gaze") or loop_state.get("return_gaze_expectation")
                stream_state = dict(loop_state.get("interaction_stream_state") or {})
                stream_state["shared_attention_level"] = max(
                    0.0,
                    min(
                        1.0,
                        float(stream_state.get("shared_attention_level") or 0.0)
                        + float(last_style_feedback.get("shared_attention_delta") or 0.0),
                    ),
                )
                stream_state["contact_readiness"] = max(
                    0.0,
                    min(
                        1.0,
                        float(stream_state.get("contact_readiness") or 0.0)
                        + float(last_style_feedback.get("contact_delta") or 0.0),
                    ),
                )
                stream_state["repair_window_open"] = bool(
                    stream_state.get("repair_window_open", False)
                    or last_style_feedback.get("repair_reopen", False)
                )
                loop_state["interaction_stream_state"] = stream_state
            loop_state["prev_qualia"] = list(((hook.expression_hints.get("qualia_state") or {}).get("qualia")) or loop_state.get("prev_qualia") or [])
            loop_state["prev_qualia_habituation"] = list(((hook.expression_hints.get("qualia_state") or {}).get("habituation")) or loop_state.get("prev_qualia_habituation") or [])
            loop_state["prev_protection_grad_x"] = list(hook.expression_hints.get("qualia_protection_grad_x") or loop_state.get("prev_protection_grad_x") or [])
            loop_state["prev_affective_position"] = dict(hook.expression_hints.get("affective_position") or loop_state.get("prev_affective_position") or {})
            loop_state["affective_terrain_state"] = dict(hook.expression_hints.get("affective_terrain_state") or loop_state.get("affective_terrain_state") or {})
            next_hook = self._integration_hooks.response_gate(
                draft={"text": getattr(shaped_response, "text", None)},
                current_state=loop_state,
                safety_signals=self._inner_os_live_followup_signals(
                    hook.expression_hints,
                    safety_bias=safety_bias,
                ),
            )
            last_style_feedback = self._inner_os_live_style_feedback(
                hook.expression_hints,
                next_hook.expression_hints,
            )
            next_hook.expression_hints["live_opening_pace_mismatch"] = float(last_style_feedback.get("opening_pace_mismatch") or 0.0)
            next_hook.expression_hints["live_return_gaze_mismatch"] = float(last_style_feedback.get("return_gaze_mismatch") or 0.0)
            next_hook.expression_hints["live_style_alignment"] = float(last_style_feedback.get("style_alignment") or 0.0)
            next_hook.expression_hints = build_expression_hints_from_gate_result(
                next_hook,
                existing_hints=next_hook.expression_hints,
                expected_source="shared",
            )
            if persistent_planning_hints:
                next_hook.expression_hints.update(persistent_planning_hints)
            if persistent_bridge_hints["llm_raw_text"]:
                next_hook.expression_hints.update(persistent_bridge_hints)
            next_stream = self._inner_os_stream_state_from_hints(next_hook.expression_hints)
            hook = next_hook
            if steps >= min_steps and self._inner_os_stream_converged(previous_stream, next_stream):
                break
            previous_stream = next_stream
        if last_style_feedback:
            hook.expression_hints["live_opening_pace_mismatch"] = float(last_style_feedback.get("opening_pace_mismatch") or 0.0)
            hook.expression_hints["live_return_gaze_mismatch"] = float(last_style_feedback.get("return_gaze_mismatch") or 0.0)
            hook.expression_hints["live_style_alignment"] = float(last_style_feedback.get("style_alignment") or 0.0)
            hook.expression_hints = build_expression_hints_from_gate_result(
                hook,
                existing_hints=hook.expression_hints,
                expected_source="shared",
            )
        if persistent_planning_hints:
            hook.expression_hints.update(persistent_planning_hints)
        if persistent_bridge_hints["llm_raw_text"]:
            hook.expression_hints.update(persistent_bridge_hints)
        shaped_response = _stamp_persistent_inner_os_controls(shaped_response)
        shaped_response = self._apply_inner_os_surface_policy(
            shaped_response,
            hook.expression_hints,
            hook.conscious_access,
        )
        shaped_response = self._apply_inner_os_surface_profile(
            shaped_response,
            hook.expression_hints,
        )
        return shaped_response, hook, steps

    def _inner_os_stream_state_from_hints(self, expression_hint: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not isinstance(expression_hint, Mapping):
            return {}
        return {
            "shared_attention_level": float(expression_hint.get("stream_shared_attention_level") or 0.0),
            "strained_pause_level": float(expression_hint.get("stream_strained_pause_level") or 0.0),
            "repair_window_open": bool(expression_hint.get("stream_repair_window_open", False)),
            "repair_window_hold": float(expression_hint.get("stream_repair_window_hold") or 0.0),
            "contact_readiness": float(expression_hint.get("stream_contact_readiness") or 0.0),
            "human_presence_signal": float(expression_hint.get("stream_human_presence_signal") or 0.0),
            "shared_attention_window": [
                float(expression_hint.get("stream_shared_attention_window_mean") or 0.0)
            ],
            "strained_pause_window": [
                float(expression_hint.get("stream_strained_pause_window_mean") or 0.0)
            ],
            "update_count": int(expression_hint.get("stream_update_count") or 0),
        }

    def _inner_os_live_followup_signals(
        self,
        expression_hint: Optional[Mapping[str, Any]],
        *,
        safety_bias: float,
    ) -> Dict[str, Any]:
        if not isinstance(expression_hint, Mapping):
            return {"safety_bias": safety_bias}
        return {
            "safety_bias": safety_bias,
            "mutual_attention_score": float(expression_hint.get("stream_shared_attention_level") or 0.0),
            "pause_latency": float(expression_hint.get("stream_strained_pause_level") or 0.0),
            "repair_signal": 1.0 if bool(expression_hint.get("stream_repair_window_open", False)) else 0.0,
            "hesitation_signal": float(expression_hint.get("strained_pause") or 0.0),
        }

    def _inner_os_stream_converged(
        self,
        previous_stream: Optional[Mapping[str, Any]],
        next_stream: Optional[Mapping[str, Any]],
    ) -> bool:
        previous = dict(previous_stream or {})
        current = dict(next_stream or {})
        deltas = [
            abs(float(current.get("shared_attention_level") or 0.0) - float(previous.get("shared_attention_level") or 0.0)),
            abs(float(current.get("strained_pause_level") or 0.0) - float(previous.get("strained_pause_level") or 0.0)),
            abs(float(current.get("contact_readiness") or 0.0) - float(previous.get("contact_readiness") or 0.0)),
            abs(float(current.get("human_presence_signal") or 0.0) - float(previous.get("human_presence_signal") or 0.0)),
            abs(float(current.get("repair_window_hold") or 0.0) - float(previous.get("repair_window_hold") or 0.0)),
        ]
        repair_changed = bool(current.get("repair_window_open", False)) != bool(previous.get("repair_window_open", False))
        return (max(deltas) if deltas else 0.0) < 0.03 and not repair_changed

    def _inner_os_live_style_feedback(
        self,
        previous_hint: Optional[Mapping[str, Any]],
        next_hint: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        previous = dict(previous_hint or {})
        current = dict(next_hint or {})
        opening_prev = str(previous.get("opening_pace_windowed") or "").strip().lower()
        opening_next = str(current.get("opening_pace_windowed") or "").strip().lower()
        return_prev = str(previous.get("return_gaze_expectation") or "").strip().lower()
        return_next = str(current.get("return_gaze_expectation") or "").strip().lower()
        opening_alignment = type(self)._inner_os_style_alignment(
            opening_prev,
            opening_next,
            similar_groups=({"held", "measured"}, {"ready", "measured"}),
        )
        return_alignment = type(self)._inner_os_style_alignment(
            return_prev,
            return_next,
            similar_groups=({"soft_return", "steady_return"}, {"careful_return", "defer_return"}),
        )
        opening_mismatch = round(1.0 - opening_alignment, 4)
        return_mismatch = round(1.0 - return_alignment, 4)
        style_alignment = round(max(0.0, min((opening_alignment + return_alignment) / 2.0, 1.0)), 4)
        mismatch_pressure = opening_mismatch * 0.55 + return_mismatch * 0.45
        return {
            "opening_pace_mismatch": opening_mismatch,
            "return_gaze_mismatch": return_mismatch,
            "style_alignment": style_alignment,
            "strain_delta": round(mismatch_pressure * 0.04, 4),
            "grounding_delta": round(-mismatch_pressure * 0.025, 4),
            "contact_delta": round(-mismatch_pressure * 0.03, 4),
            "shared_attention_delta": round(-mismatch_pressure * 0.04, 4),
            "repair_reopen": bool(mismatch_pressure >= 0.42),
            "next_opening_pace": opening_next or opening_prev,
            "next_return_gaze": return_next or return_prev,
        }

    @staticmethod
    def _inner_os_style_alignment(
        predicted: str,
        observed: str,
        *,
        similar_groups: tuple[set[str], ...] = (),
    ) -> float:
        if not predicted or not observed:
            return 0.5
        if predicted == observed:
            return 1.0
        for group in similar_groups:
            if predicted in group and observed in group:
                return 0.68
        return 0.0

    def serialize_2d_state(self) -> Dict[str, Any]:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        sensor_snapshot = getattr(self._runtime_sensors, "snapshot", None)
        sensor_metrics = getattr(sensor_snapshot, "metrics", {}) if sensor_snapshot is not None else {}
        world_state = self._relational_world_core.snapshot()
        self._surface_world_state = dict(world_state)
        memory_anchor = self._dominant_memory_anchor()
        nearby_entities = self._nearby_entities_from_sensor(sensor_metrics)
        access = self._conscious_access_core.snapshot(
            talk_mode=self._talk_mode.name.lower(),
            route=str(self._last_gate_context.get("response_route") or "watch"),
            mode=self._relational_world_core.mode(),
            memory_anchor=memory_anchor,
            replay_active=bool(self._last_future_risk or self._last_future_hope),
        )
        state = ProjectATRI2DState(
            schema="project_atri_2d_state/v1",
            timestamp=timestamp,
            identity={
                "entity_id": self._session_id or "atri_core_01",
                "mode": self._relational_world_core.mode(),
                "talk_mode": self._talk_mode.name.lower(),
            },
            world={
                "world_id": world_state.get("world_id") or "harbor_town",
                "world_type": world_state.get("world_type") or "infrastructure",
                "zone_id": world_state.get("zone_id") or "market",
                "time_phase": world_state.get("time_phase") or "day",
                "weather": world_state.get("weather") or "clear",
                "culture_id": world_state.get("culture_id") or "default",
                "community_id": world_state.get("community_id") or "local",
                "social_role": world_state.get("social_role") or "companion",
                "place_memory_anchor": world_state.get("place_memory_anchor") or memory_anchor,
            },
            body={
                "energy": round(float(getattr(self._self_model, "current_energy", 0.0) or 0.0), 4),
                "stress": round(float(self._surface_stress(sensor_metrics)), 4),
                "love": round(float(getattr(self._self_model, "attachment_to_user", 0.0) or 0.0), 4),
                "arousal": round(float(self.perceived_affect.get("arousal", 0.0) or 0.0), 4),
                "recovery_need": round(float(self._surface_recovery_need(sensor_metrics)), 4),
                "attention_density": round(float(self._surface_attention_density(sensor_metrics)), 4),
            },
            sensing={
                "person_count": int(sensor_metrics.get("person_count", 0) or 0),
                "voice_level": round(float(sensor_metrics.get("voice_level", 0.0) or 0.0), 4),
                "breath_rate": round(float(sensor_metrics.get("breath_rate", 0.0) or 0.0), 4),
                "body_stress_index": round(float(sensor_metrics.get("body_stress_index", 0.0) or 0.0), 4),
                "autonomic_balance": round(float(sensor_metrics.get("autonomic_balance", 0.0) or 0.0), 4),
                "place_id": str(sensor_metrics.get("place_id") or world_state.get("zone_id") or ""),
                "privacy_tags": [str(tag) for tag in (sensor_metrics.get("privacy_tags") or [])[:4]],
                "body_state_flag": str(sensor_metrics.get("body_state_flag") or "normal"),
                "scene_density": round(float(sensor_metrics.get("motion_score", 0.0) or 0.0), 4),
            },
            activity={
                "state": access.surface_state,
                "target": "user" if self._last_speaker == "user" else "world",
                "intent": access.intent,
                "route": access.route,
                "streaming": self._relational_world_core.mode() == "streaming",
                "replay_active": access.replay_active,
                "recall_active": access.recall_active,
            },
            social={
                "nearby_entities": nearby_entities,
                "bond_strength": {"user": round(float(getattr(self._self_model, "attachment_to_user", 0.0) or 0.0), 4)},
                "safety_bias": round(float(self._surface_safety_bias()), 4),
                "nearby_objects": self._inner_os_relational_context(sensor_metrics).get("nearby_objects") or [],
            },
            memory={
                "dominant_anchor": memory_anchor,
                "recent_recall_ids": self._surface_recent_recall_ids(),
                "perception_available": bool(self._last_observed_vision_entry),
                "retrieval_hit_count": int(self._last_gate_context.get("retrieval_hit_count") or 0),
            },
            simulation={
                "enabled": bool(world_state.get("simulation_enabled", False)),
                "episode_id": world_state.get("simulation_episode_id"),
                "transfer_pending": bool(world_state.get("simulation_transfer_pending", False)),
                "world_source": world_state.get("world_source") or "runtime",
            },
            last_event=dict(self._last_2d_event) if isinstance(self._last_2d_event, dict) else None,
        )
        return state.to_dict()

    def ingest_2d_event(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        event = ProjectATRI2DEvent.from_mapping(payload)
        event_payload = dict(event.payload)
        world_state = self._relational_world_core.ingest_surface_event(
            event_type=event.event_type,
            payload=event_payload,
            world_id=event.world_id,
        )
        self._surface_world_state = dict(world_state)
        self._last_2d_event = event.to_dict()
        return {
            "status": "accepted",
            "event_type": event.event_type,
            "mode": self._relational_world_core.mode(),
            "world_id": world_state.get("world_id"),
            "zone_id": world_state.get("zone_id"),
        }

    def shutdown(self) -> None:
        self.save_state()

    def run_forever(
        self,
        *,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        image_path: Optional[str] = None,
        prompt: str = "> ",
        render_fn: Optional[Callable[[RuntimeTurnResult], None]] = None,
        stop_commands: Tuple[str, ...] = ("/quit", "/exit"),
    ) -> int:
        renderer = render_fn or (lambda result: print(result.to_dict()))
        try:
            while True:
                try:
                    user_text = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if not user_text:
                    continue
                if user_text.lower() in stop_commands:
                    return 0
                result = self.process_turn(
                    user_text=user_text,
                    context=context,
                    intent=intent,
                    fast_only=fast_only,
                    image_path=image_path,
                )
                renderer(result)
        finally:
            self.shutdown()

    def _surface_mode(self) -> str:
        mode = str(self._surface_world_state.get("mode") or "reality").strip().lower()
        if mode not in {"reality", "streaming", "simulation"}:
            return "reality"
        return mode

    def _surface_activity_state(self) -> str:
        mapping = {
            TalkMode.WATCH: "watch",
            TalkMode.ASK: "attend",
            TalkMode.TALK: "talk",
            TalkMode.SOOTHE: "sync",
            TalkMode.PRESENCE: "rest",
        }
        base = mapping.get(self._talk_mode, "idle")
        if self._surface_mode() == "streaming":
            return "stream"
        if self._surface_mode() == "simulation":
            return "simulate"
        if self._last_future_risk or self._last_future_hope:
            return "replay"
        if self._dominant_memory_anchor():
            return "recall"
        return base

    def _surface_intent_label(self) -> str:
        route = str(self._last_gate_context.get("response_route") or "watch")
        if route == "conscious":
            return "engage"
        if route == "reflex":
            return "guard"
        if self._dominant_memory_anchor():
            return "remember"
        return "listen"

    def _surface_stress(self, sensor_metrics: Mapping[str, Any]) -> float:
        return self._pain_stress_core.stress(
            sensor_metrics=sensor_metrics,
            last_shadow_estimate=self._last_shadow_estimate,
            last_gate_context=self._last_gate_context,
        )

    def _surface_recovery_need(self, sensor_metrics: Mapping[str, Any]) -> float:
        stress = self._surface_stress(sensor_metrics)
        energy = float(getattr(self._self_model, "current_energy", 0.0) or 0.0)
        return self._recovery_core.recovery_need(
            stress=stress,
            current_energy=energy,
        )

    def _surface_attention_density(self, sensor_metrics: Mapping[str, Any]) -> float:
        return self._recovery_core.attention_density(
            sensor_metrics=sensor_metrics,
            last_gate_context=self._last_gate_context,
        )

    def _surface_safety_bias(self) -> float:
        vg = self._value_gradient_snapshot()
        return self._boundary_core.safety_bias(
            value_gradient=vg,
            safety_lens=self._safety_lens,
        )

    def _dominant_memory_anchor(self) -> Optional[str]:
        if isinstance(self._last_memory_hint_meta, Mapping):
            anchor = self._last_memory_hint_meta.get("anchor")
            if isinstance(anchor, str) and anchor.strip():
                return anchor.strip()
        if isinstance(self._last_2d_event, Mapping):
            payload = self._last_2d_event.get("payload")
            if isinstance(payload, Mapping):
                anchor = payload.get("anchor") or payload.get("memory_anchor")
                if isinstance(anchor, str) and anchor.strip():
                    return anchor.strip()
        return None

    def _surface_recent_recall_ids(self) -> List[str]:
        ids: List[str] = []
        if isinstance(self._last_observed_vision_entry, Mapping):
            identifier = self._last_observed_vision_entry.get("id")
            if isinstance(identifier, str) and identifier.strip():
                ids.append(identifier.strip())
        return ids

    def _nearby_entities_from_sensor(self, sensor_metrics: Mapping[str, Any]) -> List[str]:
        entities: List[str] = []
        person_count = int(sensor_metrics.get("person_count", 0) or 0)
        if person_count > 0:
            entities.append("user")
            for idx in range(1, min(person_count, 4)):
                entities.append(f"person_{idx}")
        object_counts = sensor_metrics.get("object_counts")
        if isinstance(object_counts, Mapping):
            for key in sorted(object_counts.keys()):
                if len(entities) >= 6:
                    break
                entities.append(str(key))
        if not entities:
            entities.append("world")
        return entities

    def _describe_image_if_available(
        self,
        image_path: Optional[str],
        user_text: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not image_path:
            return None
        adapter = getattr(self, "_vlm_adapter", None)
        if adapter is None:
            adapter = LMStudioVLMAdapter()
            self._vlm_adapter = adapter
        if not adapter.enabled:
            return None
        try:
            payload = adapter.describe_image(image_path, user_text=user_text)
            return payload or None
        except Exception as exc:
            return {
                "backend": "lmstudio_vlm",
                "image_path": image_path,
                "error": "vision_error",
                "detail": str(exc),
            }

    def _append_vision_memory(
        self,
        perception_summary: Optional[Dict[str, Any]],
        *,
        user_text: Optional[str],
        image_path: Optional[str],
        route_value: str,
    ) -> None:
        self._last_observed_vision_entry = self._vision_memory_store.append_observed(
            perception_summary=perception_summary,
            turn_id=self._turn_id,
            session_id=self._session_id,
            talk_mode=self._talk_mode.name.lower(),
            response_route=route_value,
            user_text=user_text,
            image_path=image_path,
        )

    def observe_video(
        self, frame: np.ndarray, timestamp: Optional[float] = None
    ) -> None:
        self.perception.ingest_video_frame(frame, timestamp=timestamp)

    def observe_audio(
        self, chunk: np.ndarray, timestamp: Optional[float] = None
    ) -> None:
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
        image_path: Optional[str] = None,
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
            Intent label for the hub router (qa/chitchat/code鬩包ｽｯ繝ｻ・ｶ郢晢ｽｻ繝ｻ・ｽE郢晢ｽｻ繝ｻ・ｽ).
        fast_only:
            When true, skips heavy operations (memory reference lookup, LLM call)
            so that the caller can issue a quick acknowledgement before the full
        pipeline completes.
        """
        start_ts = time.time()
        self._last_dominant_layer = None
        self._last_self_force = None
        self._last_raw_force = None
        self._last_boundary_signal = None
        affect = self.perception.sample_affect()
        if self._pending_text_affect is not None:
            affect = self._pending_text_affect
            self._pending_text_affect = None
        heart_rate, heart_phase = self._update_heart_state(
            float(getattr(affect, "arousal", 0.0))
        )
        heart_snapshot = {"rate": heart_rate, "phase": heart_phase}
        metrics = self._update_metrics(affect, fast_only=fast_only)
        # Merge external mood metrics if provided (env or file)
        metrics = self._merge_mood_metrics(metrics)
        metrics = self._merge_sensor_metrics(metrics)
        culture_ctx = self._current_culture_context()
        prospective: Optional[Dict[str, Any]] = None
        shadow_estimate = None
        if self._pdc is not None:
            phi_vec = np.array(self._last_E, dtype=float)
            psi_vec = self._psi_vector_from_metrics(metrics)
            memory_iface = self.eqnet_system or self._pdc_memory_fallback
            try:
                prospective = self._pdc.step(phi_vec, psi_vec, memory_iface)
            except Exception:
                prospective = self._pdc.step(
                    phi_vec, psi_vec, self._pdc_memory_fallback
                )
            prospective["jerk_p95"] = self._pdc.compute_jerk_p95()
            metrics = dict(metrics)
            metrics.setdefault("pdc_story", prospective["E_story"])
            metrics.setdefault("pdc_temperature", prospective["T"])
            self._last_prospective = prospective
        shadow_estimate = self._compute_shadow_estimate(
            affect=affect,
            prospective=prospective,
            culture_ctx=culture_ctx,
        )
        if shadow_estimate is not None:
            metrics = dict(metrics)
            metrics.setdefault(
                "shadow_uncertainty", float(shadow_estimate.mood_uncertainty)
            )
            metrics.setdefault("shadow_residual", float(shadow_estimate.residual))
        self._last_shadow_estimate = self._serialize_shadow(shadow_estimate)
        self._last_metrics = metrics
        self._update_emotion_axes_from_sensors(affect, metrics)

        affect_vec = np.array([affect.valence, affect.arousal], dtype=float)
        delta_m = 0.0
        jerk = 0.0
        if self._prev_affect_vec is not None:
            delta_m = float(np.linalg.norm(affect_vec - self._prev_affect_vec))
        if self._prev_prev_affect_vec is not None and self._prev_affect_vec is not None:
            jerk = float(
                np.linalg.norm(
                    affect_vec
                    - 2.0 * self._prev_affect_vec
                    + self._prev_prev_affect_vec
                )
            )
        self._prev_prev_affect_vec = self._prev_affect_vec
        self._prev_affect_vec = affect_vec

        now_ts = time.time()
        text_input = bool(user_text and user_text.strip())
        if text_input:
            self._last_user_ts = now_ts
            self._last_speaker = "user"
        since_last_user_ms = float("inf")
        if self._last_user_ts > 0.0:
            since_last_user_ms = max((now_ts - self._last_user_ts) * 1000.0, 0.0)

        stats = self.perception.stats()
        video_metrics = stats.last_video_metrics or {}
        audio_metrics = stats.last_audio_metrics or {}
        face_motion = float(video_metrics.get("motion", 0.0))
        blink = float(video_metrics.get("blink", 0.0))
        voice_energy = float(audio_metrics.get("rms", 0.0))

        if self._engaged_override is not None:
            engaged = self._engaged_override
        else:
            engaged = bool(video_metrics) and (
                face_motion > 0.01 or blink > 0.001 or voice_energy > 0.05
            )

        gate_ctx = GateContext(
            engaged=engaged,
            face_motion=face_motion,
            blink=blink,
            voice_energy=voice_energy,
            delta_m=delta_m,
            jerk=jerk,
            text_input=text_input,
            since_last_user_ms=since_last_user_ms,
            force_listen=self._force_listen,
        )
        self._talk_mode = _decide_talk_mode(gate_ctx)
        last_gate_allow: Optional[bool] = None
        if isinstance(self._last_qualia_gate, dict) and "allow" in self._last_qualia_gate:
            last_gate_allow = bool(self._last_qualia_gate.get("allow"))
        self._last_gate_context = self._ensure_interaction_gate_defaults({
            "engaged": engaged,
            "face_motion": face_motion,
            "blink": blink,
            "voice_energy": voice_energy,
            "delta_m": delta_m,
            "jerk": jerk,
            "since_last_user_ms": since_last_user_ms,
            "text_input": text_input,
            "mode": self._talk_mode.name.lower(),
            "force_listen": self._force_listen,
        })
        shadow_uncertainty = None
        shadow_mode = None
        if shadow_estimate is not None:
            shadow_uncertainty = float(shadow_estimate.mood_uncertainty)
            self._last_gate_context["shadow_uncertainty"] = shadow_uncertainty
            shadow_mode = self._shadow_mode_label(shadow_uncertainty)
            self._last_gate_context["shadow_mode"] = shadow_mode
            metrics = dict(metrics)
            metrics.setdefault("shadow_mode", shadow_mode)
            metrics.setdefault("shadow_uncertainty", shadow_uncertainty)
            if shadow_uncertainty >= 0.85:
                self._talk_mode = TalkMode.WATCH
                self._last_gate_context["mode"] = self._talk_mode.name.lower()
            elif shadow_uncertainty >= 0.65 and self._talk_mode == TalkMode.TALK:
                self._talk_mode = TalkMode.ASK
                self._last_gate_context["mode"] = self._talk_mode.name.lower()
        else:
            shadow_uncertainty = None
        if prospective:
            self._last_gate_context["pdc_story"] = float(
                prospective.get("E_story", 0.0)
            )
        sensor_snapshot = self._runtime_sensors.snapshot
        if sensor_snapshot is not None:
            flag = sensor_snapshot.metrics.get("body_state_flag")
            if flag:
                self._last_gate_context["body_state_flag"] = flag
        updated_mode = self._apply_future_risk(metrics, self._talk_mode)
        if updated_mode is not self._talk_mode:
            self._talk_mode = updated_mode
            self._last_gate_context["mode"] = self._talk_mode.name.lower()
        updated_mode = self._apply_future_imagery(metrics, self._talk_mode)
        if updated_mode is not self._talk_mode:
            self._talk_mode = updated_mode
            self._last_gate_context["mode"] = self._talk_mode.name.lower()
        if self._should_presence(gate_ctx, shadow_uncertainty, last_gate_allow, now_ts):
            self._talk_mode = TalkMode.PRESENCE
            self._last_gate_context["mode"] = self._talk_mode.name.lower()
            self._last_gate_context["presence"] = True

        qualia_vec = self._build_emotion_vector(metrics, affect)
        if not self.config.use_eqnet_core:
            self._last_qualia = qualia_vec.to_dict()
        prediction_error = self._compute_prediction_error(delta_m, metrics)
        route = self._decide_response_route(
            qualia_vec, prediction_error, text_input, gate_ctx
        )
        metrics = dict(metrics)
        metrics["conscious_prediction_error"] = prediction_error
        metrics["response_route"] = route.value
        self._last_gate_context["response_route"] = route.value
        self._last_gate_context["prediction_error"] = prediction_error
        self._update_self_model_state(qualia_vec)
        self._last_metrics = metrics

        controls = self.policy.affect_to_controls(
            affect, metrics, prospective=prospective
        )
        if self._talk_mode == TalkMode.SOOTHE:
            controls.temperature = float(np.clip(controls.temperature, 0.25, 0.55))
            controls.pause_ms = int(
                np.clip(controls.pause_ms + 150, *self.policy.config.pause_bounds)
            )
            controls.prosody_energy = float(min(controls.prosody_energy, -0.05))
            controls.gesture_amplitude = float(min(controls.gesture_amplitude, 0.35))
        elif self._talk_mode == TalkMode.ASK:
            controls.pause_ms = int(
                np.clip(controls.pause_ms + 120, *self.policy.config.pause_bounds)
            )
            controls.prosody_energy = float(max(controls.prosody_energy, 0.05))
        elif self._talk_mode in (TalkMode.WATCH, TalkMode.PRESENCE):
            controls.gesture_amplitude = float(min(controls.gesture_amplitude, 0.25))
        if shadow_estimate is not None:
            self._apply_shadow_controls(controls, shadow_estimate)

        self._last_controls = controls

        ack_for_fast: Optional[str] = None
        if fast_only and text_input:
            ack_for_fast = self._sample_fast_ack_text(affect, gate_ctx)

        ack_text: Optional[str] = None
        if self._talk_mode in (TalkMode.SOOTHE, TalkMode.ASK):
            ack_text = self._render_ack_for_mode(self._talk_mode, gate_ctx)
        elif self._talk_mode == TalkMode.PRESENCE:
            ack_text = self._render_presence_ack(gate_ctx)
        elif text_input and self._talk_mode == TalkMode.WATCH:
            ack_text = self._render_ack_for_mode(TalkMode.WATCH, gate_ctx)

        force_llm_bridge = bool(self.config.force_llm_bridge and text_input and not fast_only)
        metrics["inner_os/force_llm_bridge"] = 1.0 if force_llm_bridge else 0.0

        route_response: Optional[str] = None
        if route == ResponseRoute.REFLEX and not force_llm_bridge:
            route_response = self._reflex_prompt(prediction_error)
        elif route == ResponseRoute.HABIT and text_input and not force_llm_bridge:
            route_response = self._habit_prompt(user_text)

        response: Optional[HubResponse] = None
        llm_bridge_called = False
        llm_bridge_raw_text = ""
        llm_bridge_raw_model = ""
        llm_bridge_raw_model_source = ""
        self._last_guarded_narrative_bridge_text = ""
        self._last_guarded_narrative_bridge_allowed = False
        self._last_planned_content_sequence = []
        memory_reference: Optional[Dict[str, Any]] = None
        retrieval_summary: Optional[Dict[str, Any]] = None
        persona_meta: Dict[str, Any] = {}
        perception_summary = self._normalize_perception_summary(self._describe_image_if_available(image_path, user_text))
        self._append_vision_memory(
            perception_summary,
            user_text=user_text,
            image_path=image_path,
            route_value=route.value,
        )
        metrics["perception/available"] = 1.0 if perception_summary and isinstance(perception_summary.get("text"), str) else 0.0
        metrics["perception/degraded"] = 1.0 if perception_summary and str(perception_summary.get("status") or "") == "degraded" else 0.0
        if perception_summary and isinstance(perception_summary.get("text"), str):
            visual_text = perception_summary["text"].strip()
            if visual_text:
                visual_context = f"[vision]\n{visual_text}"
                context = f"{visual_context}\n\n{context.strip()}" if context else visual_context
        if fast_only:
            ack_payload = ack_for_fast or route_response or ack_text
            if ack_payload:
                response = self._wrap_ack_response(ack_payload, self._talk_mode)
            if response is not None:
                response = self._attach_visual_reflection(response, perception_summary)
                response.retrieval_summary = retrieval_summary
            if response and response.text:
                self._last_speaker = "ai"
            eqframe = self._record_eqframe(None, persona_meta)
            metrics = dict(metrics)
            metrics["phi_norm"] = float(
                eqframe.inner_spec.get("phi", {}).get("norm", 0.0)
            )
            metrics["tension_score"] = eqframe.tension_score
            self._last_metrics = metrics
            self._update_emotion_axes_from_sensors(affect, metrics)
            impl_ctx = self._current_impl_context((time.time() - start_ts) * 1000.0)
            self._last_impl_context = impl_ctx
            return {
                "affect": affect,
                "controls": controls,
                "metrics": metrics,
                "response": response,
                "memory_reference": None,
                "robot_state": self.robot_bridge.state if self.robot_bridge else None,
                "qualia": self._last_qualia,
                "talk_mode": self._talk_mode.name.lower(),
                "gate_context": self._last_gate_context,
                "prospective": self._serialize_prospective(prospective),
                "heart": heart_snapshot,
                "persona_meta": persona_meta,
                "response_route": route.value,
                "shadow": self._last_shadow_estimate,
                "perception_summary": perception_summary,
                "retrieval_summary": retrieval_summary,
            }

        if route_response and response is None:
            response = self._wrap_ack_response(route_response, self._talk_mode)
            response = self._attach_visual_reflection(response, perception_summary)
            response.retrieval_summary = retrieval_summary

        behavior_mod: Optional[BehaviorMod] = None
        memory_cue_text = self._memory_cue_text(user_text, perception_summary)
        retrieval_summary = self._retrieve_from_memory_cue(memory_cue_text)
        metrics["memory/visual_cue"] = 1.0 if memory_cue_text != (user_text or "").strip() else 0.0
        if isinstance(retrieval_summary, dict):
            metrics["retrieval/sse_hit_count"] = float(len(retrieval_summary.get("hits") or []))
            self._last_gate_context["retrieval_hit_count"] = int(len(retrieval_summary.get("hits") or []))
            if not context and retrieval_summary.get("hits"):
                hit_lines = []
                for hit in retrieval_summary.get("hits") or []:
                    if not isinstance(hit, dict):
                        continue
                    hit_text = str(hit.get("text") or "").strip()
                    if hit_text:
                        hit_lines.append(f"- {hit_text}")
                if hit_lines:
                    context = "[sse-recall]\n" + "\n".join(hit_lines)
        masked_user_text = user_text or ""
        masked_context = context
        should_call_llm = (
            text_input
            and not route_response
            and (
                force_llm_bridge
                or (
                    self._talk_mode == TalkMode.TALK
                    and route == ResponseRoute.CONSCIOUS
                )
            )
        )
        if (
            not should_call_llm
            and user_text
            and self._memory_hint_cfg
            and getattr(self._memory_hint_cfg, "enable", False)
        ):
            _ = self._maybe_memory_reference(memory_cue_text)
        if should_call_llm:
            memory_reference = self._maybe_memory_reference(memory_cue_text)
            if (
                memory_reference
                and memory_reference.get("reply")
                and (
                    not self._memory_ref_cfg
                    or memory_reference.get("fidelity", 0.0)
                    >= float(getattr(self._memory_ref_cfg, "fidelity_low", 0.0))
                )
            ):
                response = self._wrap_memory_response(
                    memory_reference["reply"],
                    controls_used=memory_reference.get("controls_used"),
                )
            else:
                llm_controls = {
                    "temperature": controls.temperature,
                    "top_p": controls.top_p,
                    "pause_ms": controls.pause_ms,
                    "directness": controls.directness,
                    "warmth": controls.warmth,
                    "prosody_energy": controls.prosody_energy,
                    "spoiler_mode": "warn",
                }
                inner_os_expression_hint = self._last_gate_context.get("inner_os_expression_hint")
                if isinstance(inner_os_expression_hint, Mapping):
                    llm_controls = _apply_inner_os_expression_controls(llm_controls, inner_os_expression_hint)
                inner_os_llm_guidance = self._build_inner_os_llm_guidance(
                    expression_hint=inner_os_expression_hint if isinstance(inner_os_expression_hint, Mapping) else None,
                    user_text=masked_user_text,
                    intent=intent or self.llm.config.default_intent,
                )
                behavior_mod = None
                try:
                    culture_state = compute_culture_state(
                        self._current_culture_context()
                    )
                    behavior_mod = culture_to_behavior(culture_state)
                except Exception:
                    behavior_mod = None
                if behavior_mod:
                    llm_controls["culture_tone"] = behavior_mod.tone
                    llm_controls["culture_empathy"] = behavior_mod.empathy_level
                    llm_controls["culture_directness"] = behavior_mod.directness
                    llm_controls["culture_joke_ratio"] = behavior_mod.joke_ratio

                if prospective:
                    llm_controls["pdc_story"] = float(prospective.get("E_story", 0.0))
                    llm_controls["pdc_temperature"] = float(
                        prospective.get("T", controls.temperature)
                    )
                    for key in ("m_t", "m_past_hat", "m_future_hat"):
                        value = prospective.get(key)
                        if isinstance(value, np.ndarray):
                            llm_controls[f"pdc_{key}"] = value.tolist()
                masked_user_text, masked_context, persona_meta = apply_mask_layer(
                    user_text or "",
                    context,
                    mask_cfg=self.config.mask_layer,
                    prospective=prospective,
                )
                if behavior_mod:
                    persona_meta.setdefault(
                        "culture_behavior",
                        {
                            "tone": behavior_mod.tone,
                            "empathy": behavior_mod.empathy_level,
                            "directness": behavior_mod.directness,
                            "joke_ratio": behavior_mod.joke_ratio,
                        },
                    )
                    hint_line = self._culture_behavior_hint(behavior_mod)
                    masked_context = (
                        f"{hint_line}\n\n{masked_context.strip()}"
                        if masked_context
                        else hint_line
                    )

                persona_cfg = getattr(self.config.mask_layer, "persona", None)
                persona_id = persona_meta.get("persona_id") or persona_meta.get("id")
                if persona_id is None and isinstance(persona_cfg, dict):
                    persona_id = persona_cfg.get("id")
                eqframe = self._record_eqframe(persona_id, persona_meta)
                metrics = dict(metrics)
                metrics["phi_norm"] = float(
                    eqframe.inner_spec.get("phi", {}).get("norm", 0.0)
                )
                metrics["tension_score"] = eqframe.tension_score
                self._last_metrics = metrics
                self._update_emotion_axes_from_sensors(affect, metrics)
                llm_bridge_called = True
                response = self.llm.generate(
                    user_text=masked_user_text,
                    context=masked_context,
                    controls=llm_controls,
                    intent=intent or self.llm.config.default_intent,
                    slos={"p95_ms": 180.0},
                    image_path=None,
                    interaction_policy=inner_os_llm_guidance.get("interaction_policy"),
                    action_posture=inner_os_llm_guidance.get("action_posture"),
                    actuation_plan=inner_os_llm_guidance.get("actuation_plan"),
                    conversational_objects=inner_os_llm_guidance.get("conversational_objects"),
                    object_operations=inner_os_llm_guidance.get("object_operations"),
                    interaction_effects=inner_os_llm_guidance.get("interaction_effects"),
                    interaction_judgement_summary=inner_os_llm_guidance.get("interaction_judgement_summary"),
                    interaction_condition_report=inner_os_llm_guidance.get("interaction_condition_report"),
                    content_sequence=inner_os_llm_guidance.get("content_sequence"),
                    surface_context_packet=inner_os_llm_guidance.get("surface_context_packet"),
                    surface_profile=inner_os_llm_guidance.get("surface_profile"),
                    utterance_stance=inner_os_llm_guidance.get("utterance_stance"),
                )
                llm_bridge_raw_text = str(getattr(response, "text", "") or "")
                llm_bridge_raw_model = str(getattr(response, "model", "") or "")
                llm_bridge_raw_model_source = str(
                    getattr(response, "model_source", "") or ""
                )
                self._last_planned_content_sequence = list(
                    inner_os_llm_guidance.get("content_sequence") or []
                )
                response_controls = dict(getattr(response, "controls_used", {}) or {})
                response_controls["inner_os_planned_content_sequence"] = list(
                    self._last_planned_content_sequence
                )
                response.controls_used = response_controls
                response = self._attach_visual_reflection(response, perception_summary)
        elif response is None and ack_text:
            response = self._wrap_ack_response(ack_text, self._talk_mode)
        metrics["inner_os/llm_bridge_called"] = 1.0 if llm_bridge_called else 0.0
        if self.robot_bridge:
            self.robot_bridge.publish(
                {
                    "pause_ms": controls.pause_ms,
                    "gesture_amplitude": controls.gesture_amplitude,
                    "prosody_energy": controls.prosody_energy,
                    "gaze_mode": 1.0 if controls.gaze_mode == "engage" else 0.0,
                }
            )

        impl_ctx = self._current_impl_context((time.time() - start_ts) * 1000.0)
        self._last_impl_context = impl_ctx
        boundary_signal = self._compute_boundary_signal(prediction_error)
        response, episode_narrative = self._apply_qualia_gate(
            response=response,
            narrative_text=getattr(response, "text", None),
            qualia_vec=qualia_vec,
            gate_ctx=gate_ctx,
            boundary_signal=boundary_signal,
            ack_text=ack_text,
            ack_for_fast=ack_for_fast,
            metrics=metrics,
            prediction_error=prediction_error,
        )
        if response is not None:
            response = self._attach_visual_reflection(response, perception_summary)
            response.retrieval_summary = retrieval_summary
        self._last_persona_meta = dict(persona_meta)
        if not fast_only:
            context_snapshot = self._build_context_payload(
                user_text=user_text,
                context_text=context,
                metrics=metrics,
                gate_ctx=gate_ctx,
                route=route,
            )
            self._maybe_store_conscious_episode(
                qualia=qualia_vec,
                prediction_error=prediction_error,
                narrative=episode_narrative,
                route=route,
                context=context_snapshot,
                implementation=impl_ctx,
                boundary_signal=boundary_signal,
            )
            self._log_moment_entry(
                affect=affect,
                metrics=metrics,
                heart_snapshot=heart_snapshot,
                prospective=prospective,
                response=response,
                user_text=user_text,
                persona_meta=persona_meta,
                behavior_mod=behavior_mod,
            )

        return {
            "affect": affect,
            "controls": controls,
            "metrics": metrics,
            "response": response,
            "llm_bridge_raw_text": llm_bridge_raw_text,
            "llm_bridge_raw_model": llm_bridge_raw_model,
            "llm_bridge_raw_model_source": llm_bridge_raw_model_source,
            "memory_reference": memory_reference,
            "robot_state": self.robot_bridge.state if self.robot_bridge else None,
            "qualia": self._last_qualia,
            "talk_mode": self._talk_mode.name.lower(),
            "gate_context": self._last_gate_context,
            "prospective": self._serialize_prospective(prospective),
            "heart": heart_snapshot,
            "persona_meta": persona_meta,
            "response_route": route.value,
            "shadow": self._last_shadow_estimate,
            "qualia_gate": dict(self._last_qualia_gate),
            "perception_summary": perception_summary,
            "retrieval_summary": retrieval_summary,
        }

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    @property
    def last_persona_meta(self) -> Dict[str, Any]:

        return self._last_persona_meta

    @property
    def last_fast_ack_sample(self) -> Optional[Dict[str, Any]]:
        return self._last_fast_ack_sample

    @property
    def heart_state(self) -> Dict[str, float]:
        return {"rate": self._heart_rate, "phase": self._heart_phase}

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

    @property
    def emotion_core(self) -> EmotionCore:
        return self._emotion_core

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _update_heart_state(
        self,
        arousal: float,
        *,
        noise_scale: Optional[float] = None,
        damp: float = 0.02,
    ) -> Tuple[float, float]:
        """Update the synthetic heart oscillator from the latest arousal."""

        rate, phase, phi_vec = self._heartbeat_core.update(
            arousal,
            emotion_step=self._emotion_core.step,
            noise_scale=noise_scale,
            damp=damp,
        )
        self._heart_rate = rate
        self._heart_phase = phase
        self._heart_last_ts = self._heartbeat_core.state.last_ts
        self._heart_base_rate = float(self._heartbeat_core.config.base_rate)
        self._heart_gain = float(self._heartbeat_core.config.gain)
        self._last_E = np.array(phi_vec, copy=True)
        return rate, phase

    def _apply_qualia_gate(
        self,
        *,
        response: Optional[HubResponse],
        narrative_text: Optional[str],
        qualia_vec: EmotionVector,
        gate_ctx: GateContext,
        boundary_signal: Optional[BoundarySignal],
        ack_text: Optional[str],
        ack_for_fast: Optional[str],
        metrics: Dict[str, float],
        prediction_error: float,
    ) -> Tuple[Optional[HubResponse], Optional[str]]:
        """Decide whether to surface the narrative output."""
        current_vec = self._qualia_vector_to_array(qualia_vec)
        pred_vec = self._prev_qualia_vec if self._prev_qualia_vec is not None else current_vec
        self._prev_qualia_vec = current_vec
        boundary_score = float(getattr(boundary_signal, "score", 0.0) or 0.0)
        u_t = float(metrics.get("shadow_uncertainty", prediction_error) or 0.0)
        if not math.isfinite(u_t):
            u_t = 0.0
        load_t = float(
            max(
                0.0,
                gate_ctx.delta_m + gate_ctx.jerk + (1.0 if gate_ctx.text_input else 0.0),
            )
        )
        payload: Dict[str, Any] = {"enabled": self._qualia_gate_enabled, "timestamp": time.time(), "boundary_score": boundary_score, "load_t": load_t}
        if not self._qualia_gate_enabled:
            payload.update({"u_t": u_t, "m_t": 0.0, "allow": True})
            self._last_qualia_gate = payload
            return response, narrative_text
        m_t = float(self._qualia_meta.compute(pred_vec, current_vec))
        gate_result = self._qualia_gate.decide(
            u_t=u_t,
            m_t=m_t,
            load_t=load_t,
            override=bool(gate_ctx.force_listen),
            reason="safety" if gate_ctx.force_listen else "normal",
            boundary_score=boundary_score,
        )
        payload.update(gate_result)
        payload["m_kind"] = "cosine"
        allow = bool(gate_result.get("allow", True))
        if self._talk_mode == TalkMode.PRESENCE:
            allow = False
            payload["allow"] = False
            payload["presence_override"] = True
        metrics["qualia/u_t"] = u_t
        metrics["qualia/m_t"] = m_t
        metrics["qualia/load"] = load_t
        metrics["qualia/gate_allow"] = 1.0 if allow else 0.0
        if not allow:
            planned_sequence = localize_content_sequence(
                [
                    dict(item)
                    for item in (getattr(self, "_last_planned_content_sequence", []) or [])
                    if isinstance(item, Mapping) and str(item.get("text") or "").strip()
                ],
                locale=_runtime_response_locale(self),
            )
            planned_text = render_content_sequence(planned_sequence).strip()
            minimal = planned_text or ack_text or ack_for_fast
            if not minimal and self._talk_mode == TalkMode.PRESENCE:
                minimal = self._render_presence_ack(gate_ctx)
            if not minimal:
                minimal = self._render_ack_for_mode(TalkMode.WATCH, gate_ctx)
            response_controls = dict(getattr(response, "controls_used", {}) or {})
            raw_narrative = str(narrative_text or "").strip()
            self._last_guarded_narrative_bridge_text = raw_narrative
            if raw_narrative:
                response_controls["inner_os_llm_raw_text"] = raw_narrative
            response_controls["inner_os_qualia_gate_suppressed"] = True
            response_controls["inner_os_qualia_gate_reason"] = str(
                gate_result.get("reason") or ""
            ).strip()
            self._last_guarded_narrative_bridge_allowed = bool(
                raw_narrative
                and not bool(gate_result.get("override"))
                and str(gate_result.get("reason") or "").strip() == "normal"
                and boundary_score < 0.18
                and self._talk_mode != TalkMode.PRESENCE
            )
            if any(
                str(item.get("act") or "").strip() == "offer_small_opening_line"
                for item in (getattr(self, "_last_planned_content_sequence", []) or [])
                if isinstance(item, Mapping)
            ):
                self._last_guarded_narrative_bridge_allowed = False
            response_controls["inner_os_allow_guarded_narrative_bridge"] = bool(
                self._last_guarded_narrative_bridge_allowed
            )
            if minimal:
                response = self._wrap_ack_response(
                    minimal,
                    self._talk_mode,
                    controls_used=response_controls,
                )
            narrative_text = None
            payload["suppress_narrative"] = True
            payload["guarded_narrative_bridge"] = bool(
                response_controls.get("inner_os_allow_guarded_narrative_bridge")
            )
            payload["unconscious_success"] = 1
        else:
            payload["unconscious_success"] = 0
        self._last_qualia_gate = payload
        return response, narrative_text

    def _qualia_vector_to_array(self, qualia: EmotionVector) -> np.ndarray:
        return np.array(
            [
                float(qualia.valence),
                float(qualia.arousal),
                float(qualia.love),
                float(qualia.stress),
                float(qualia.mask),
            ],
            dtype=float,
        )

    def _build_inner_spec(self) -> Dict[str, Any]:
        """Collect a lightweight snapshot of the inner (Phi/Psi/K/M/PDC) state."""

        inner: Dict[str, Any] = {}
        try:
            vec = self._emotion_core.get_phi_vec()
        except Exception:
            vec = np.zeros(len(AXES), dtype=float)
        inner["phi"] = {
            "vec": vec.tolist(),
            "norm": float(np.linalg.norm(vec)),
        }
        inner["psi"] = {}
        inner["k"] = {}
        inner["m"] = {}
        serialized = self._serialize_prospective(self._last_prospective)
        inner["pdc"] = serialized or {}
        return inner

    def _compute_tension(
        self, inner_spec: Dict[str, Any], persona_meta: Dict[str, Any]
    ) -> float:
        """Simple L2 distance between inner Phi and persona tone hints."""

        phi_payload = inner_spec.get("phi", {})
        phi_vec = np.array(phi_payload.get("vec") or [], dtype=float)
        if phi_vec.size == 0:
            return 0.0
        tone_vec = persona_meta.get("tone_vec")
        if tone_vec is None:
            tone_vec = persona_meta.get("tone_vector")
        if tone_vec is None:
            return 0.0
        tone_arr = np.array(tone_vec, dtype=float)
        if tone_arr.shape != phi_vec.shape:
            return 0.0
        return float(np.linalg.norm(phi_vec - tone_arr))

    def _update_life_indicator(
        self, phi_norm: float, metrics: Dict[str, float]
    ) -> float:
        heart_rate = float(np.clip(getattr(self, "_heart_rate", 0.0), 0.0, 3.0))
        rho = float(metrics.get("rho", 0.0)) if metrics else 0.0
        arousal = float(metrics.get("arousal", 0.0)) if metrics else 0.0
        story_energy = 0.0
        if isinstance(self._last_prospective, dict):
            story_energy = float(self._last_prospective.get("E_story", 0.0))
        calm_factor = 1.0 - min(1.0, abs(rho))
        vitality = 0.5 * phi_norm + 0.2 * calm_factor
        vitality += 0.2 * (heart_rate / 3.0)
        vitality += 0.05 * min(1.0, abs(arousal))
        vitality += 0.05 * min(1.0, max(0.0, story_energy))
        life_val = float(np.clip(vitality, 0.0, 1.0))
        self._last_life_indicator = life_val
        return life_val

    def _record_eqframe(
        self, persona_id: Optional[str], persona_meta: Dict[str, Any]
    ) -> EQFrame:
        """Build and store the latest EQFrame, updating gate context as well."""

        persona_meta = persona_meta or dict(self._default_persona_meta)
        if persona_id is None:
            persona_id = persona_meta.get("persona_id")
        inner_spec = self._build_inner_spec()
        tension = self._compute_tension(inner_spec, persona_meta)
        LOGGER.debug(
            "[persona_meta] id=%s tone_vec=%s tension=%.4f",
            persona_id,
            persona_meta.get("tone_vec"),
            tension,
        )
        phi_norm = float(inner_spec.get("phi", {}).get("norm", 0.0))
        self._last_gate_context["phi_norm"] = phi_norm
        life_val = self._update_life_indicator(phi_norm, self._last_metrics or {})
        self._last_gate_context["tension_score"] = tension
        self._last_gate_context["life_indicator"] = life_val
        if persona_id:
            self._last_gate_context["persona_id"] = persona_id
        eqframe = EQFrame(
            ts=time.time(),
            inner_spec=inner_spec,
            persona_id=persona_id,
            persona_meta=dict(persona_meta or {}),
            tension_score=tension,
        )
        self._last_eqframe = eqframe
        return eqframe

    def _update_metrics(
        self, affect: AffectSample, fast_only: bool = False
    ) -> Dict[str, float]:
        """
        Project affect into a synthetic EQNet state vector. This is a temporary
        shim; real deployments should call into ``EmotionalMemorySystem``.
        """
        metrics: Dict[str, float]
        if self.eqnet_system:
            if fast_only:
                metrics = self._fallback_metrics(affect)
            else:
                entry = self._update_eqnet_system(affect)
                if not entry:
                    metrics = self._fallback_metrics(affect)
                else:
                    entropy = float(entry.get("entropy", 0.0))
                    dissipation = float(entry.get("dissipation", 0.0))
                    info_flux = float(entry.get("info_flux", 0.0))
                    current_emotion = getattr(
                        self.eqnet_system, "current_emotion", None
                    )
                    if current_emotion is not None:
                        try:
                            vec = np.asarray(current_emotion, dtype=float)
                            self._last_E = vec
                            self._emotion_core.seed(vec)
                        except Exception:
                            pass
                    LOGGER.debug(
                        "[eqnet_field] rho=%s entropy=%s",
                        entry.get("rho"),
                        entry.get("entropy"),
                    )
                    metrics = {
                        "H": float(np.clip(1.0 - entropy / 10.0, 0.0, 1.0)),
                        "R": float(np.clip(dissipation, 0.0, 1.0)),
                        "kappa": float(np.clip(-0.5 * affect.arousal, -1.0, 1.0)),
                        "entropy": entropy,
                        "dissipation": dissipation,
                        "info_flux": info_flux,
                        "timestamp": entry.get("timestamp", time.time()),
                    }
                    rho_val = entry.get("rho")
                    if rho_val is not None:
                        try:
                            metrics["rho"] = float(rho_val)
                        except (TypeError, ValueError):
                            pass
        else:
            metrics = self._fallback_metrics(affect)

        metrics.setdefault("valence", float(getattr(affect, "valence", 0.0)))
        metrics.setdefault("arousal", float(getattr(affect, "arousal", 0.0)))
        metrics.setdefault("rho", float(metrics.get("rho", 0.0)))

        self._emotion_core.update_setpoint(metrics)
        LOGGER.debug("[_last_E vec]%s", self._last_E)
        return metrics

    def idle_tick(self) -> Dict[str, Any]:
        """Advance the core by one idle beat, injecting light affect noise."""

        noise_val = float(np.random.normal(0.0, 0.02))
        noise_aro = float(np.random.normal(0.0, 0.02))
        sample = AffectSample(
            valence=noise_val,
            arousal=noise_aro,
            confidence=0.05,
            timestamp=time.time(),
        )
        metrics = self._update_metrics(sample, fast_only=False)
        try:
            self._update_heart_state(noise_aro, noise_scale=0.05, damp=0.02)
        except Exception:
            pass
        persona_meta = dict(self._last_persona_meta or self._default_persona_meta)
        persona_id = persona_meta.get("persona_id")
        try:
            eqframe = self._record_eqframe(
                persona_id=persona_id, persona_meta=persona_meta
            )
        except Exception:
            eqframe = None
        return {"metrics": metrics, "eqframe": eqframe}

    def _update_eqnet_system(self, affect: AffectSample) -> Dict[str, float]:
        assert self.eqnet_system is not None
        ts = datetime.utcnow().isoformat()
        dialogue = (
            f"[affect_stream] valence={affect.valence:.3f} "
            f"arousal={affect.arousal:.3f} confidence={affect.confidence:.3f}"
        )
        entry_payload: Dict[str, Any] = {
            "timestamp": ts,
            "user_id": "affect_stream",
            "dialogue": dialogue,
            "mood": {
                "valence": float(affect.valence),
                "arousal": float(affect.arousal),
            },
            "metrics": {
                "rho": self._estimate_rho_from_affect(affect),
                "confidence": float(getattr(affect, "confidence", 0.0)),
            },
        }
        metrics_record: Dict[str, float] = {}
        updater = getattr(self.eqnet_system, "update", None)
        if callable(updater):
            try:
                metrics_record = updater(entry_payload) or {}
            except Exception:
                metrics_record = {}
        else:
            self.eqnet_system.ingest_dialogue("affect_stream", dialogue, ts)
        metrics_log = self.eqnet_system.field_metrics_state()
        snapshot = self.eqnet_system.field.snapshot()
        self._last_snapshot = snapshot
        if not metrics_record and metrics_log:
            metrics_record = metrics_log[-1]
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
        return metrics_record

    def _fallback_metrics(self, affect: AffectSample) -> Dict[str, float]:
        E = np.zeros(len(AXES), dtype=float)
        E[0] = affect.valence
        E[1] = affect.arousal
        E[3] = affect.valence * 0.7 + affect.arousal * 0.3
        E[4] = -abs(affect.arousal) * 0.4
        self._last_E = E
        self._emotion_core.seed(E)

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

    def _merge_sensor_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        snapshot = self._runtime_sensors.snapshot
        if snapshot is None:
            return metrics
        out = dict(metrics)
        for key, value in snapshot.metrics.items():
            if isinstance(value, (int, float, np.floating)):
                out[key] = float(value)
        return out

    def _evaluate_future_risk(self, metrics: Mapping[str, Any]) -> Optional[float]:
        body_r = float(np.clip(metrics.get("R", metrics.get("rho", 0.5)), 0.0, 1.0))
        stress_val = float(np.clip(abs(metrics.get("kappa", 0.0)), 0.0, 1.0))
        vec = np.array([body_r, stress_val], dtype=float)
        state = QualiaState(timestamp=datetime.utcnow(), qualia_vec=vec)
        self._future_history.append(state)
        if len(self._future_history) < 2:
            return None
        history = list(self._future_history)
        return compute_future_risk(
            history,
            self._future_risk_cfg,
            stress_index=1,
            stress_threshold=self._future_stress_cutoff,
            body_index=0,
            body_threshold=self._future_body_cutoff,
        )

    def _apply_future_risk(
        self, metrics: Dict[str, float], talk_mode: TalkMode
    ) -> TalkMode:
        risk = self._evaluate_future_risk(metrics)
        if risk is None:
            return talk_mode
        metrics.setdefault("future_risk_stress", risk)
        self._last_future_risk = risk
        new_mode = talk_mode
        if risk >= self._future_risk_trigger:
            new_mode = TalkMode.SOOTHE
        self._last_gate_context.setdefault("future_risk_stress", risk)
        if new_mode is not talk_mode:
            self._last_gate_context["future_risk_triggered"] = True
        self._log_future_risk(risk, talk_mode, new_mode)
        return new_mode

    def _log_future_risk(self, risk: float, before: TalkMode, after: TalkMode) -> None:
        payload = {
            "ts": time.time(),
            "session_id": self._session_id,
            "episode_id": self._session_id,
            "turn_id": self._turn_id,
            "future_risk_stress": float(risk),
            "threshold": self._future_risk_trigger,
            "stress_cutoff": self._future_stress_cutoff,
            "body_cutoff": self._future_body_cutoff,
            "talk_mode_before": before.name.lower(),
            "talk_mode_after": after.name.lower(),
        }
        try:
            self._future_risk_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._future_risk_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            LOGGER.debug("[future-risk] log write failed")

    def _evaluate_future_hope(self, metrics: Mapping[str, Any]) -> Optional[float]:
        valence = float(metrics.get("valence", 0.0))
        love = float(metrics.get("affect.love", metrics.get("love", 0.0)))
        vec = np.array([valence, love], dtype=float)
        state = QualiaState(timestamp=datetime.utcnow(), qualia_vec=vec)
        self._imagery_history.append(state)
        if len(self._imagery_history) < 1:
            return None
        intention = self._imagery_intention_vector(valence, love)
        return compute_future_hopefulness(
            list(self._imagery_history),
            self._future_hope_cfg,
            intention_vec=intention,
            valence_index=0,
            love_index=1,
        )

    def _imagery_intention_vector(self, valence: float, love: float) -> np.ndarray:
        target_valence = 0.6
        target_love = 0.7
        return np.array([target_valence - valence, target_love - love], dtype=float)

    def _apply_future_imagery(
        self, metrics: Dict[str, float], talk_mode: TalkMode
    ) -> TalkMode:
        hope = self._evaluate_future_hope(metrics)
        if hope is None:
            return talk_mode
        metrics.setdefault("future_hopefulness", hope)
        imagery_positive = hope >= self._future_hope_trigger
        metrics.setdefault("imagery_positive", imagery_positive)
        self._last_future_hope = hope
        new_mode = talk_mode
        if imagery_positive and talk_mode != TalkMode.TALK:
            new_mode = TalkMode.TALK
        self._last_gate_context.setdefault("future_hopefulness", hope)
        if imagery_positive:
            self._last_gate_context["imagery_positive"] = True
        self._log_future_hope(hope, talk_mode, new_mode, imagery_positive)
        return new_mode

    def _log_future_hope(
        self, hope: float, before: TalkMode, after: TalkMode, imagery_positive: bool
    ) -> None:
        payload = {
            "ts": time.time(),
            "session_id": self._session_id,
            "episode_id": self._session_id,
            "turn_id": self._turn_id,
            "future_hopefulness": float(hope),
            "threshold": self._future_hope_trigger,
            "talk_mode_before": before.name.lower(),
            "talk_mode_after": after.name.lower(),
            "imagery_positive": imagery_positive,
        }
        try:
            self._future_hope_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._future_hope_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            LOGGER.debug("[future-hope] log write failed")

    def _update_emotion_axes_from_sensors(
        self,
        affect: AffectSample,
        metrics: Mapping[str, Any],
    ) -> None:
        text_axes = self._vector_from_affect(affect)
        snapshot = self._runtime_sensors.snapshot
        sensor_axes = sensor_to_emotion_axes(snapshot)
        activity = 0.0
        privacy = 0.0
        if snapshot is not None:
            activity = float(snapshot.metrics.get("activity_level", 0.0))
            privacy = float(snapshot.metrics.get("body_flag_private", 0.0))
        fog = float(metrics.get("fog_level", metrics.get("fog", 0.5)))
        blended = blend_emotion_axes(
            text_axes,
            sensor_axes,
            activity_level=activity,
            privacy_level=privacy,
            fog_level=fog,
        )
        self._last_sensor_axes = sensor_axes.tolist()
        self._last_blended_axes = blended.tolist()
        self._last_E = blended

    def _metrics_from_text_affect(self, sample: AffectSample) -> Dict[str, float]:
        valence = float(np.clip(sample.valence, -1.0, 1.0))
        arousal = float(np.clip(sample.arousal, -1.0, 1.0))
        H = float(np.clip(0.5 + 0.35 * valence, 0.0, 1.0))
        R = float(np.clip(0.5 + 0.4 * arousal, 0.0, 1.0))
        kappa = float(np.clip(-0.4 * arousal, -1.0, 1.0))
        ignition = float(np.clip(max(arousal, 0.0) ** 1.3, 0.0, 1.0))
        return {
            "H": H,
            "R": R,
            "kappa": kappa,
            "ignition": ignition,
            "timestamp": sample.timestamp,
        }

    def _vector_from_affect(self, sample: AffectSample) -> np.ndarray:
        vec = np.zeros(len(AXES), dtype=float)
        if len(vec) >= 2:
            vec[0] = sample.valence
            vec[1] = sample.arousal
        return vec

    def _estimate_rho_from_affect(self, sample: AffectSample) -> float:
        calm = 1.0 - float(np.clip(abs(sample.arousal), 0.0, 1.0))
        confidence = float(np.clip(getattr(sample, "confidence", 0.0), 0.0, 1.0))
        rho = 0.6 * calm + 0.4 * confidence
        return float(np.clip(2.0 * rho - 1.0, -1.0, 1.0))

    def _psi_vector_from_metrics(self, metrics: Dict[str, float]) -> np.ndarray:
        dim = len(self._last_E)
        vec = np.zeros(dim, dtype=float)
        keys = ("H", "R", "kappa", "ignition", "rho", "novelty", "coherence")
        idx = 0
        for key in keys:
            if idx >= dim:
                break
            val = metrics.get(key)
            if val is None:
                continue
            vec[idx] = float(val)
            idx += 1
        return vec

    def _serialize_shadow(self, estimate) -> Optional[Dict[str, Any]]:
        if estimate is None:
            return None
        return {
            "completed_valence": float(estimate.completed_valence),
            "completed_arousal": float(estimate.completed_arousal),
            "pred_valence": float(estimate.pred_valence),
            "pred_arousal": float(estimate.pred_arousal),
            "uncertainty": float(estimate.mood_uncertainty),
            "residual": float(estimate.residual),
            "alpha": float(estimate.alpha),
            "evidence": dict(estimate.evidence),
        }

    def _shadow_mode_label(self, u: float) -> str:
        if u >= 0.85:
            return "yield"
        if u >= 0.65:
            return "confirm"
        if u >= 0.4:
            return "explore"
        return "commit"

    def _compute_shadow_estimate(
        self,
        affect: AffectSample,
        prospective: Optional[Dict[str, Any]],
        culture_ctx: CultureContext,
    ):
        estimator = getattr(self, "_shadow_estimator", None)
        if estimator is None or prospective is None:
            return None
        try:
            confidence = float(getattr(affect, "confidence", 0.5))
            culture_val = getattr(culture_ctx, "valence", None)
            return estimator.estimate(
                affect,
                prospective=prospective,
                replay_stats=None,
                sensor_confidence=confidence,
                culture_bias=culture_val,
            )
        except Exception:
            return None

    def _apply_shadow_controls(self, controls: AffectControls, estimate) -> None:
        u = float(getattr(estimate, "mood_uncertainty", 0.0))
        cfg = self.policy.config
        pause = float(controls.pause_ms) + 200.0 * u
        controls.pause_ms = int(np.clip(pause, *cfg.pause_bounds))
        temp = float(controls.temperature) - 0.15 * u
        controls.temperature = float(np.clip(temp, *cfg.temp_bounds))
        top_p = float(controls.top_p) - 0.05 * u
        controls.top_p = float(np.clip(top_p, *cfg.top_p_bounds))

    def _serialize_prospective(
        self, prospective: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not prospective:
            return None
        payload: Dict[str, Any] = {
            "E_story": float(prospective.get("E_story", 0.0)),
            "T": float(prospective.get("T", 0.0)),
        }
        for key in ("m_t", "m_past_hat", "m_future_hat"):
            value = prospective.get(key)
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
        if "jerk_p95" in prospective:
            payload["jerk_p95"] = float(prospective["jerk_p95"])
        return payload

    def _numeric_metrics_snapshot(self, metrics: Dict[str, float]) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating)):
                snapshot[key] = float(value)
        return snapshot

    def _snapshot_fast_ack(self) -> Optional[Dict[str, Any]]:
        if not self._last_fast_ack_sample:
            return None
        snapshot: Dict[str, Any] = {}
        for key, value in self._last_fast_ack_sample.items():
            if isinstance(value, (int, float, np.floating)):
                snapshot[key] = float(value)
            else:
                snapshot[key] = value
        return snapshot

    def _serialize_response_meta(
        self, response: Optional[HubResponse]
    ) -> Optional[Dict[str, Any]]:
        if not response:
            return None
        controls_used = {
            key: float(val)
            for key, val in dict(getattr(response, "controls_used", {}) or {}).items()
            if isinstance(val, (int, float, np.floating))
        }
        return {
            "model": getattr(response, "model", None),
            "model_source": str(getattr(response, "model_source", "") or ""),
            "trace_id": str(getattr(response, "trace_id", "") or ""),
            "latency_ms": float(getattr(response, "latency_ms", 0.0) or 0.0),
            "controls_used": controls_used,
            "safety": dict(getattr(response, "safety", {}) or {}),
            "confidence": float(getattr(response, "confidence", 0.0) or 0.0),
            "uncertainty_reason": list(getattr(response, "uncertainty_reason", ()) or ()),
            "perception_summary": dict(getattr(response, "perception_summary", {}) or {}) if isinstance(getattr(response, "perception_summary", None), dict) else None,
            "retrieval_summary": dict(getattr(response, "retrieval_summary", {}) or {}) if isinstance(getattr(response, "retrieval_summary", None), dict) else None,
        }

    def _current_culture_tag(self) -> str:
        return self._current_culture_context().culture_tag

    def _current_culture_context(self) -> CultureContext:
        culture_cfg = (
            getattr(self._runtime_cfg, "culture", None) if self._runtime_cfg else None
        )
        ctx = CultureContext(
            culture_tag=getattr(culture_cfg, "tag", None) if culture_cfg else None,
            place_id=getattr(culture_cfg, "place_id", None) if culture_cfg else None,
            partner_id=(
                getattr(culture_cfg, "partner_id", None) if culture_cfg else None
            ),
            object_id=getattr(culture_cfg, "object_id", None) if culture_cfg else None,
            object_role=(
                getattr(culture_cfg, "object_role", None) if culture_cfg else None
            ),
            activity_tag=(
                getattr(culture_cfg, "activity_tag", None) if culture_cfg else None
            ),
        )
        return ctx.normalized()

    def _bump_culture_recurrence(self, ctx: CultureContext) -> int:
        key = (
            ctx.culture_tag or "",
            ctx.place_id or "",
            ctx.partner_id or "",
            ctx.object_id or "",
        )
        self._culture_recurrence[key] += 1
        return self._culture_recurrence[key]

    def _feed_culture_models(
        self,
        entry: MomentLogEntry,
        ctx: CultureContext,
        user_text: Optional[str],
    ) -> None:
        metrics = entry.metrics or {}
        culture_cfg = (
            getattr(self._runtime_cfg, "culture", None) if self._runtime_cfg else None
        )

        def _coerce(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _pick_metric(keys: Tuple[str, ...], default: float) -> float:
            for key in keys:
                if key in metrics and metrics[key] is not None:
                    return _coerce(metrics[key], default)
            return default

        baseline_intimacy = (
            float(getattr(culture_cfg, "intimacy", 0.5)) if culture_cfg else 0.5
        )
        baseline_politeness = (
            float(getattr(culture_cfg, "politeness", 0.5)) if culture_cfg else 0.5
        )

        event = {
            "ts": entry.ts,
            "culture_tag": ctx.culture_tag,
            "place_id": ctx.place_id,
            "partner_id": ctx.partner_id,
            "object_id": ctx.object_id,
            "object_role": ctx.object_role,
            "activity_tag": ctx.activity_tag,
            "valence": _coerce(entry.mood.get("valence"), 0.0),
            "arousal": _coerce(entry.mood.get("arousal"), 0.0),
            "rho": _pick_metric(("rho", "rho_norm", "R"), 0.5),
            "intimacy": _coerce(metrics.get("intimacy"), baseline_intimacy),
            "politeness": _coerce(metrics.get("politeness"), baseline_politeness),
        }
        entry.metrics.setdefault("rho", event["rho"])
        entry.metrics.setdefault("politeness", event["politeness"])
        entry.metrics.setdefault("intimacy", event["intimacy"])
        recurrence_count = self._bump_culture_recurrence(ctx)
        text_bits = [chunk for chunk in (user_text, entry.llm_text) if chunk]
        text_blob = " ".join(text_bits)
        try:
            update_climate_from_event(event, context=ctx)
            promote_to_monument_if_needed(
                event,
                text=text_blob,
                recurrence_count=recurrence_count,
                context=ctx,
            )
        except Exception:
            pass

    def _infer_emotion_tag(self, affect: AffectSample) -> str:
        valence = float(getattr(affect, "valence", 0.0))
        arousal = float(getattr(affect, "arousal", 0.0))
        if valence > 0.45 and arousal < 0.4:
            return "joy"
        if valence > 0.35 and arousal >= 0.4:
            return "anticipation"
        if valence < -0.35 and arousal > 0.2:
            return "anger"
        if valence < -0.35:
            return "sadness"
        if arousal > 0.6:
            return "surprise"
        return "calm"

    def _qualia_vec_snapshot(self) -> Optional[List[float]]:
        try:
            vec = np.asarray(self._last_E, dtype=float).reshape(-1)
        except Exception:
            return None
        if vec.size == 0:
            return None
        return [float(v) for v in vec.tolist()]

    def _culture_behavior_hint(self, behavior: BehaviorMod) -> str:
        tone_hints = {
            "polite": "Use a gentle, respectful tone and avoid abrupt wording.",
            "casual": "Keep the atmosphere warm and relaxed while staying considerate.",
            "neutral": "Use a steady, readable tone with balanced emotional distance.",
        }
        tone_line = tone_hints.get(behavior.tone, tone_hints["neutral"])

        empathy_line = (
            "Prioritize emotional attunement and acknowledge the other person's state."
            if behavior.empathy_level >= 0.65
            else "Keep empathy present, but avoid over-reading or excessive reassurance."
        )

        joke_line = (
            "A small amount of light humor is acceptable if the moment supports it."
            if behavior.joke_ratio >= 0.4
            else "Keep humor restrained and preserve clarity over playfulness."
        )

        return (
            f"[culture-field] {tone_line} {empathy_line} "
            f"Directness={behavior.directness:.2f}, "
            f"joke_ratio={behavior.joke_ratio:.2f}. {joke_line}"
        )

    def _log_moment_entry(
        self,
        *,
        affect: AffectSample,
        metrics: Dict[str, float],
        heart_snapshot: Dict[str, float],
        prospective: Optional[Dict[str, Any]],
        response: Optional[HubResponse],
        user_text: Optional[str],
        persona_meta: Dict[str, Any],
        behavior_mod: Optional[BehaviorMod],
    ) -> None:
        if not self._moment_log_writer.enabled:
            return
        heart_rate = heart_snapshot.get("rate")
        heart_phase = heart_snapshot.get("phase")
        persona_payload = dict(persona_meta) if persona_meta else None
        culture_ctx = self._current_culture_context()
        behavior_payload = None
        if behavior_mod:
            behavior_payload = {
                "tone": behavior_mod.tone,
                "empathy": float(behavior_mod.empathy_level),
                "directness": float(behavior_mod.directness),
                "joke_ratio": float(behavior_mod.joke_ratio),
            }
        if response and response.text:
            self._last_speaker = "ai"
        emotion_tag = self._infer_emotion_tag(affect)
        qualia_vec = self._qualia_vec_snapshot()
        trace_observations: Dict[str, Any] = {}
        if self._last_memory_hint_meta:
            trace_observations.setdefault("policy", {})["memory_hint"] = dict(
                self._last_memory_hint_meta
            )
        entry = MomentLogEntry(
            ts=time.time(),
            turn_id=self._turn_id,
            session_id=self._session_id,
            talk_mode=self._talk_mode.name.lower(),
            mood={
                "valence": float(getattr(affect, "valence", 0.0)),
                "arousal": float(getattr(affect, "arousal", 0.0)),
            },
            metrics=self._numeric_metrics_snapshot(metrics),
            gate_context=self._ensure_interaction_gate_defaults(self._last_gate_context),
            prospective=self._serialize_prospective(prospective),
            heart_rate=float(heart_rate) if heart_rate is not None else None,
            heart_phase=float(heart_phase) if heart_phase is not None else None,
            culture_tag=culture_ctx.culture_tag,
            place_id=culture_ctx.place_id,
            partner_id=culture_ctx.partner_id,
            object_id=culture_ctx.object_id,
            object_role=culture_ctx.object_role,
            activity_tag=culture_ctx.activity_tag,
            fast_ack=self._snapshot_fast_ack(),
            persona_meta=persona_payload,
            user_text=user_text if user_text else None,
            llm_text=response.text if response else None,
            response_meta=self._serialize_response_meta(response),
            trace_observations=trace_observations or None,
            behavior_mod=behavior_payload,
            emotion_tag=emotion_tag,
            shadow=self._last_shadow_estimate,
            emotion_axes_sensor=(
                list(self._last_sensor_axes) if self._last_sensor_axes else None
            ),
            emotion_axes_blended=(
                list(self._last_blended_axes) if self._last_blended_axes else None
            ),
            qualia_vec=qualia_vec,
        )
        self._feed_culture_models(entry, culture_ctx, user_text)
        try:
            self._moment_log_writer.write(entry)
            self._emit_trace_v1_observations(entry, trace_observations)
            self._turn_id += 1
        except Exception:
            pass

    def _emit_trace_v1_observations(
        self, entry: MomentLogEntry, trace_observations: Dict[str, Any]
    ) -> None:
        if not trace_observations:
            return
        flag = (os.getenv("EQNET_TRACE_V1") or "").strip().lower()
        if flag not in {"1", "true", "yes", "on"}:
            return
        trace_root = os.getenv("EQNET_TRACE_V1_DIR")
        if not trace_root:
            return
        try:
            timestamp_ms = int(entry.ts * 1000)
        except Exception:
            timestamp_ms = int(time.time() * 1000)
        record = {
            "schema_version": "trace_v1",
            "source_loop": "runtime",
            "timestamp_ms": timestamp_ms,
            "turn_id": str(entry.turn_id),
            "scenario_id": entry.session_id or "runtime",
            "trace_observations": trace_observations,
        }
        target = trace_output_path(
            TracePathConfig(base_dir=Path(trace_root), source_loop="runtime"),
            timestamp_ms=timestamp_ms,
        )
        append_trace_event(target, record)

    # ------------------------------------------------------------------ #
    # Memory reference helpers
    # ------------------------------------------------------------------ #

    def _sample_fast_ack_text(
        self, affect: AffectSample, gate_ctx: GateContext
    ) -> Optional[str]:
        arousal = float(np.clip(getattr(affect, "arousal", 0.0), 0.0, 1.0))
        distance = self._estimate_interpersonal_distance(gate_ctx)
        previous_arousal = self._arousal_tracker.last_arousal
        choice = sample_fast_ack(
            arousal, distance, self._fast_ack_state, self._arousal_tracker
        )
        self._last_fast_ack_sample = {
            "arousal": arousal,
            "delta": arousal - previous_arousal,
            "distance": distance,
            "last_choice": self._fast_ack_state.last_choice,
            "choice": choice,
        }
        if choice == "silence":
            return None
        if choice == "breath":
            return self._render_fast_ack("breath")
        fast_ack = self._render_fast_ack("ack")
        if fast_ack:
            return fast_ack
        if self._talk_mode == TalkMode.PRESENCE:
            presence_ack = self._render_presence_ack(gate_ctx)
            if presence_ack:
                return presence_ack
        return self._render_ack_for_mode(self._talk_mode, gate_ctx)

    def _should_presence(
        self,
        gate_ctx: GateContext,
        shadow_uncertainty: Optional[float],
        last_gate_allow: Optional[bool],
        now_ts: float,
    ) -> bool:
        cfg = self._presence_cfg
        if not cfg or not getattr(cfg, "enable", False):
            return False
        if gate_ctx.force_listen or gate_ctx.text_input:
            return False
        if gate_ctx.since_last_user_ms < float(getattr(cfg, "min_silence_ms", 0)):
            return False
        if (now_ts - self._last_presence_ack_ts) < float(
            getattr(cfg, "max_ack_interval_s", 0.0)
        ):
            return False
        if last_gate_allow is False:
            return True
        if shadow_uncertainty is None:
            return False
        return shadow_uncertainty >= float(getattr(cfg, "shadow_threshold", 1.0))

    def _render_presence_ack(self, gate_ctx: GateContext) -> Optional[str]:
        cfg = self._presence_cfg
        if not cfg or not getattr(cfg, "enable", False):
            return None
        locale = "ja-JP"
        if self._runtime_cfg and hasattr(self._runtime_cfg, "backchannel"):
            locale = getattr(self._runtime_cfg.backchannel, "culture", locale)
        key = cfg.ack_key_minimal
        if gate_ctx.since_last_user_ms >= float(
            getattr(cfg, "short_after_silence_ms", 0.0)
        ):
            key = cfg.ack_key_short
        text = lookup_text(locale, str(key))
        if not text:
            return None
        return truncate_text(text, int(getattr(cfg, "ack_max_chars", 0)) or None)

    def _render_ack_for_mode(
        self, mode: TalkMode, gate_ctx: GateContext
    ) -> Optional[str]:
        cfg = self._ack_cfg
        if not cfg:
            return None
        locale = "ja-JP"
        if self._runtime_cfg and hasattr(self._runtime_cfg, "backchannel"):
            locale = getattr(self._runtime_cfg.backchannel, "culture", locale)
        key_map = {
            TalkMode.WATCH: cfg.watch_key,
            TalkMode.SOOTHE: cfg.soothe_key,
            TalkMode.ASK: cfg.ask_key,
            TalkMode.TALK: cfg.talk_key,
        }
        key = key_map.get(mode)
        if not key:
            return None
        text = lookup_text(locale, str(key))
        if not text:
            return None
        return truncate_text(text, int(getattr(cfg, "max_chars", 0)) or None)

    def _render_fast_ack(self, key: str) -> Optional[str]:
        cfg = self._ack_cfg
        if not cfg:
            return None
        locale = "ja-JP"
        if self._runtime_cfg and hasattr(self._runtime_cfg, "backchannel"):
            locale = getattr(self._runtime_cfg.backchannel, "culture", locale)
        lookup_key = cfg.fast_ack_key if key == "ack" else cfg.fast_breath_key
        text = lookup_text(locale, str(lookup_key))
        if not text:
            return None
        return truncate_text(text, int(getattr(cfg, "max_chars", 0)) or None)

    def _response_locale(self) -> str:
        locale = "ja-JP"
        if self._runtime_cfg and hasattr(self._runtime_cfg, "backchannel"):
            locale = str(
                getattr(self._runtime_cfg.backchannel, "culture", locale) or locale
            )
        return locale

    def _estimate_interpersonal_distance(self, gate_ctx: GateContext) -> float:
        distance = 0.7
        if gate_ctx.engaged:
            distance -= 0.2
        distance -= min(0.25, gate_ctx.face_motion * 2.5)
        distance -= min(0.25, gate_ctx.voice_energy * 1.5)
        if gate_ctx.force_listen:
            distance += 0.2
        return float(np.clip(distance, 0.0, 1.0))

    def _maybe_memory_reference(self, user_text: str) -> Optional[Dict[str, Any]]:
        cfg = self._memory_ref_cfg
        hint_cfg = self._memory_hint_cfg
        if not user_text.strip() or time.time() < self._memory_ref_cooldown_until:
            return None
        allow_reference = bool(
            cfg is not None and getattr(cfg, "enabled", True) and self.eqnet_system is not None
        )
        culture_tag = (
            getattr(self._runtime_cfg.culture, "tag", "ja-JP")
            if self._runtime_cfg
            else "ja-JP"
        )
        overrides = {}
        per_culture = getattr(cfg, "per_culture", {}) or {}
        if isinstance(per_culture, dict):
            overrides = per_culture.get(culture_tag.lower(), {}) or {}
        max_reply_chars = int(
            overrides.get("max_reply_chars", getattr(cfg, "max_reply_chars", 140))
        )
        result = None
        if allow_reference:
            result = handle_memory_reference(
                self.eqnet_system,
                user_text,
                tone="support",
                culture=culture_tag,
                k=int(getattr(cfg, "k", 3)),
                max_reply_chars=max_reply_chars,
            )
            if result is not None:
                cue_label_chars = int(
                    overrides.get("cue_label_chars", getattr(cfg, "cue_label_chars", 24))
                )
                result = render_recall_cue(
                    result,
                    culture=culture_tag,
                    max_reply_chars=max_reply_chars,
                    cue_label_chars=cue_label_chars,
                )
                daily_limit = int(
                    overrides.get("rarity_budget_day", getattr(cfg, "rarity_budget_day", 3))
                )
                weekly_limit = int(
                    overrides.get("rarity_budget_week", getattr(cfg, "rarity_budget_week", 12))
                )
                result = apply_rarity_budget(
                    result,
                    state=self._recall_budget_state,
                    daily_limit=daily_limit,
                    weekly_limit=weekly_limit,
                )
        self._last_memory_hint_meta = None
        hint_payload: Dict[str, Any] = result or {"reply": None, "fidelity": 0.0}
        if hint_cfg and getattr(hint_cfg, "enable", False):
            gate_ctx = {
                "since_last_user_ms": float(
                    self._last_gate_context.get("since_last_user_ms", 0.0) or 0.0
                ),
                "text_input": bool(self._last_gate_context.get("text_input", False)),
            }
            shadow_uncertainty = None
            if self._last_shadow_estimate:
                shadow_uncertainty = self._last_shadow_estimate.get("mood_uncertainty")
            now_ts = time.time()
            dt_s = max(0.0, now_ts - self._memory_hint_pressure_ts)
            hint = render_memory_hint(
                hint_payload.get("label"),
                float(hint_payload.get("fidelity", 0.0) or 0.0),
                locale=culture_tag,
                cfg=hint_cfg,
                gate_ctx=gate_ctx,
                shadow_uncertainty=shadow_uncertainty,
                last_speaker=self._last_speaker,
                prev_pressure=self._memory_hint_pressure,
                dt_s=dt_s,
                prev_blocked=self._memory_hint_prev_blocked,
            )
            controls: Dict[str, Any] = {"memory_hint": None}
            if hint:
                controls["memory_hint"] = {
                    "enabled": bool(hint.get("enabled", True)),
                    "shown": bool(hint.get("shown", False)),
                    "blocked": bool(hint.get("blocked", False)),
                    "reason": hint.get("reason"),
                    "key": hint.get("key"),
                    "style": hint.get("style"),
                    "interrupt_cost": hint.get("interrupt_cost"),
                    "confidence": hint.get("confidence"),
                    "social_mode": getattr(hint_cfg, "social_mode", None),
                    "pressure": hint.get("pressure"),
                    "pressure_delta": hint.get("pressure_delta"),
                }
                self._last_memory_hint_meta = dict(controls["memory_hint"])
                self._memory_hint_pressure = float(
                    hint.get("pressure", self._memory_hint_pressure)
                )
                self._memory_hint_pressure_ts = now_ts
                self._memory_hint_prev_blocked = bool(hint.get("blocked", False))
                if hint.get("shown"):
                    if not getattr(hint_cfg, "allow_verbatim", False):
                        hint_payload["reply"] = hint.get("text")
                elif hint.get("blocked") and not getattr(hint_cfg, "allow_verbatim", False):
                    hint_payload["reply"] = None
                hint_payload["memory_hint"] = hint
            hint_payload["controls_used"] = controls
        elif hint_cfg:
            now_ts = time.time()
            dt_s = max(0.0, now_ts - self._memory_hint_pressure_ts)
            decay = float(getattr(hint_cfg, "pressure_decay_per_s", 1.0) or 1.0)
            decay = max(0.0, min(1.0, decay))
            self._memory_hint_pressure *= decay**dt_s
            self._memory_hint_pressure_ts = now_ts
        if allow_reference and result is not None:
            cooldown = float(
                overrides.get("cooldown_s", getattr(cfg, "cooldown_s", 0.0))
            )
            if cooldown > 0.0:
                self._memory_ref_cooldown_until = time.time() + cooldown
            self._log_memory_reference(result, user_text)
            return result
        return None

    def _wrap_ack_response(
        self,
        text: str,
        mode: TalkMode,
        *,
        controls_used: Optional[Mapping[str, Any]] = None,
    ) -> HubResponse:
        if mode == TalkMode.PRESENCE:
            self._last_presence_ack_ts = time.time()
        merged_controls: Dict[str, Any] = {"mode": mode.name.lower()}
        if isinstance(controls_used, Mapping):
            merged_controls.update(dict(controls_used))
        return HubResponse(
            text=text,
            model=None,
            trace_id=f"ack-{int(time.time() * 1000)}",
            latency_ms=0.0,
            controls_used=merged_controls,
            safety={"rating": "G", "ack": "true"},
        )

    def _normalize_perception_summary(
        self, perception_summary: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(perception_summary, dict):
            return None
        normalized = dict(perception_summary)
        text = normalized.get("text")
        if isinstance(text, str) and text.strip():
            normalized["status"] = str(normalized.get("status") or "observed")
            normalized["memory_eligible"] = True
            return normalized

        error_code = str(normalized.get("error") or "vision_unavailable")
        detail = str(normalized.get("detail") or "").strip().lower()
        error_kind = "unavailable"
        surface_text = "I cannot read the scene clearly yet, so I am staying tentative."
        if "timeout" in detail:
            error_kind = "timeout"
            surface_text = "The scene is still fuzzy to me, so I am holding back from guessing."
        elif "404" in detail or "not found" in detail:
            error_kind = "missing_image"
            surface_text = "The image did not arrive clearly, so I am waiting before I describe it."
        elif "500" in detail or "connection" in detail or "refused" in detail:
            error_kind = "backend"
            surface_text = "My visual read did not settle, so I am keeping the scene uncertain for now."

        normalized["status"] = "degraded"
        normalized["error"] = error_code
        normalized["error_kind"] = error_kind
        normalized["memory_eligible"] = False
        normalized["surface_text"] = surface_text
        return normalized

    def _visual_reflection_line(
        self, perception_summary: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        if not isinstance(perception_summary, dict):
            return None
        text = perception_summary.get("text")
        if isinstance(text, str) and text.strip():
            snippet = text.strip().splitlines()[0].strip()
            if not snippet:
                return None
            if len(snippet) > 96:
                snippet = snippet[:93].rstrip() + "..."
            return f"I am also seeing this: {snippet}"
        if str(perception_summary.get("status") or "") == "degraded":
            surface_text = perception_summary.get("surface_text")
            if isinstance(surface_text, str) and surface_text.strip():
                return surface_text.strip()
        return None

    def _apply_inner_os_surface_policy(
        self,
        response: Optional[HubResponse],
        expression_hint: Optional[Mapping[str, Any]],
        conscious_access: Optional[Mapping[str, Any]] = None,
    ) -> Optional[HubResponse]:
        if response is None or not isinstance(expression_hint, Mapping):
            return response
        if not bool(expression_hint.get("clarify_first", False)):
            return response
        existing = (response.text or "").strip()
        intent = str((conscious_access or {}).get("intent") or "").strip().lower()
        interaction_policy = expression_hint.get("interaction_policy_packet")
        prefix = self._inner_os_surface_prefix(intent=intent, interaction_policy=interaction_policy)
        question_bias = max(0.0, min(1.0, float(expression_hint.get("question_bias", 0.0) or 0.0)))
        closing = self._inner_os_surface_closing(intent=intent, question_bias=question_bias, interaction_policy=interaction_policy)
        if prefix and existing.startswith(prefix):
            existing = existing[len(prefix):].strip()
        if closing and existing.endswith(closing):
            existing = existing[: -len(closing)].strip()
        existing_controls_used = dict(getattr(response, "controls_used", {}) or {})
        planned_content_sequence = existing_controls_used.get(
            "inner_os_planned_content_sequence"
        )
        if not isinstance(planned_content_sequence, list):
            planned_content_sequence = expression_hint.get("planned_content_sequence")
        if not isinstance(planned_content_sequence, list) or not planned_content_sequence:
            runtime_planned = getattr(self, "_last_planned_content_sequence", [])
            planned_content_sequence = runtime_planned if isinstance(runtime_planned, list) and runtime_planned else None
        if isinstance(planned_content_sequence, list):
            localized_sequence = localize_content_sequence(
                [
                    dict(item)
                    for item in planned_content_sequence
                    if isinstance(item, Mapping) and str(item.get("text") or "").strip()
                ],
                locale=_runtime_response_locale(self),
            )
            existing = render_content_sequence(localized_sequence)
        else:
            fallback_user_text = str(
                getattr(self, "_last_surface_user_text", "") or ""
            ).strip()
            if fallback_user_text:
                fallback_sequence = derive_content_sequence(
                    current_text=fallback_user_text,
                    interaction_policy=interaction_policy
                    if isinstance(interaction_policy, Mapping)
                    else None,
                    conscious_access=conscious_access
                    if isinstance(conscious_access, Mapping)
                    else None,
                    history=_runtime_recent_surface_response_history(self),
                    locale=_runtime_response_locale(self),
                )
                if fallback_sequence:
                    existing = render_content_sequence(fallback_sequence)
                else:
                    existing = derive_content_skeleton(
                        current_text=existing,
                        interaction_policy=interaction_policy
                        if isinstance(interaction_policy, Mapping)
                        else None,
                        conscious_access=conscious_access
                        if isinstance(conscious_access, Mapping)
                        else None,
                    )
            else:
                existing = derive_content_skeleton(
                    current_text=existing,
                    interaction_policy=interaction_policy if isinstance(interaction_policy, Mapping) else None,
                    conscious_access=conscious_access if isinstance(conscious_access, Mapping) else None,
                )
        shaped_existing = self._shape_inner_os_surface_text(existing, intent=intent)
        probe = self._inner_os_surface_probe(intent=intent, expression_hint=expression_hint, question_bias=question_bias)
        if probe:
            shaped_existing = self._compose_inner_os_surface_text(probe, shaped_existing, "")
        reopening_line = self._inner_os_surface_reopening_line(intent=intent, expression_hint=expression_hint)
        if reopening_line:
            shaped_existing = self._compose_inner_os_surface_text(reopening_line, shaped_existing, "")
        response.text = self._compose_inner_os_surface_text(prefix, shaped_existing, closing)
        controls_used = dict(getattr(response, "controls_used", {}) or {})
        controls_used["inner_os_surface_policy"] = f"clarify_first_prefix:{intent or 'generic'}"
        response.controls_used = controls_used
        return response

    def _build_inner_os_llm_guidance(
        self,
        *,
        expression_hint: Optional[Mapping[str, Any]],
        user_text: str = "",
        intent: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(expression_hint, Mapping):
            return {}
        interaction_policy = expression_hint.get("interaction_policy_packet")
        if not isinstance(interaction_policy, Mapping):
            return {}
        dialogue_act = str(
            interaction_policy.get("dialogue_act")
            or expression_hint.get("interaction_policy_dialogue_act")
            or intent
            or ""
        ).strip()
        current_text = str(user_text or "").strip()
        content_sequence = []
        locale = _runtime_response_locale(self)
        if current_text:
            content_sequence = derive_content_sequence(
                current_text=current_text,
                interaction_policy=interaction_policy,
                conscious_access={"intent": dialogue_act},
                history=_runtime_recent_surface_response_history(self),
                locale=locale,
            )
        surface_profile = {
            "opening_delay": str(expression_hint.get("surface_opening_delay") or "").strip(),
            "response_length": str(expression_hint.get("surface_response_length") or "").strip(),
            "sentence_temperature": str(expression_hint.get("surface_sentence_temperature") or "").strip(),
            "pause_insertion": str(expression_hint.get("surface_pause_insertion") or "").strip(),
            "certainty_style": str(expression_hint.get("surface_certainty_style") or "").strip(),
            "opening_pace_windowed": str(expression_hint.get("opening_pace_windowed") or "").strip(),
            "return_gaze_expectation": str(expression_hint.get("return_gaze_expectation") or "").strip(),
            "voice_texture": str(expression_hint.get("surface_voice_texture") or "").strip(),
            "cultural_register": str(expression_hint.get("surface_cultural_register") or "").strip(),
            "lightness_room": float(expression_hint.get("surface_lightness_room") or 0.0),
            "continuity_weight": float(expression_hint.get("surface_continuity_weight") or 0.0),
            "banter_move": str(expression_hint.get("surface_banter_move") or "").strip(),
            "lexical_variation_mode": str(expression_hint.get("surface_lexical_variation_mode") or "").strip(),
            "group_register": str(expression_hint.get("surface_group_register") or "").strip(),
        }
        surface_context_packet = dict(expression_hint.get("surface_context_packet") or {})
        if not surface_context_packet:
            surface_context_packet = build_surface_context_packet(
                recent_dialogue_state=interaction_policy.get("recent_dialogue_state"),
                discussion_thread_state=interaction_policy.get("discussion_thread_state"),
                issue_state=interaction_policy.get("issue_state"),
                turn_delta=expression_hint.get("turn_delta"),
                interaction_constraints=expression_hint.get("interaction_constraints"),
                boundary_transform=expression_hint.get("boundary_transform"),
                residual_reflection=expression_hint.get("residual_reflection"),
                surface_profile=surface_profile,
                contact_reflection_state=expression_hint.get("contact_reflection_state"),
                green_kernel_composition=expression_hint.get("green_kernel_composition"),
                dialogue_context={"user_text": current_text},
            ).to_dict()
        if not str(surface_context_packet.get("conversation_phase") or "").strip():
            surface_context_packet["conversation_phase"] = dialogue_act
        content_sequence = self._apply_surface_context_packet_to_content_sequence(
            content_sequence,
            surface_context_packet=surface_context_packet,
            surface_profile=surface_profile,
        )
        return {
            "interaction_policy": dict(interaction_policy),
            "action_posture": dict(expression_hint.get("action_posture") or {}),
            "actuation_plan": dict(expression_hint.get("actuation_plan") or {}),
            "conversation_contract": dict(expression_hint.get("conversation_contract") or {}),
            "conversational_objects": dict(expression_hint.get("conversational_objects") or {}),
            "object_operations": dict(expression_hint.get("object_operations") or {}),
            "interaction_effects": dict(expression_hint.get("interaction_effects") or {}),
            "interaction_judgement_view": dict(expression_hint.get("interaction_judgement_view") or {}),
            "interaction_judgement_summary": dict(expression_hint.get("interaction_judgement_summary") or {}),
            "interaction_condition_report": dict(expression_hint.get("interaction_condition_report") or {}),
            "interaction_inspection_report": dict(expression_hint.get("interaction_inspection_report") or {}),
            "interaction_audit_bundle": dict(expression_hint.get("interaction_audit_bundle") or {}),
            "interaction_audit_casebook": dict(expression_hint.get("interaction_audit_casebook") or {}),
            "interaction_audit_report": dict(expression_hint.get("interaction_audit_report") or {}),
            "interaction_audit_reference_case_ids": list(expression_hint.get("interaction_audit_reference_case_ids") or []),
            "interaction_audit_reference_case_meta": dict(expression_hint.get("interaction_audit_reference_case_meta") or {}),
            "content_sequence": content_sequence,
            "surface_context_packet": surface_context_packet,
            "surface_profile": surface_profile,
            "utterance_stance": str(expression_hint.get("partner_utterance_stance") or "").strip(),
        }

    def _apply_surface_context_packet_to_content_sequence(
        self,
        content_sequence: Sequence[Mapping[str, Any]] | None,
        *,
        surface_context_packet: Optional[Mapping[str, Any]] = None,
        surface_profile: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        sequence = [
            dict(item)
            for item in (content_sequence or [])
            if isinstance(item, Mapping) and str(item.get("text") or "").strip()
        ]
        if not sequence:
            return []
        packet = dict(surface_context_packet or {})
        phase = str(packet.get("conversation_phase") or "").strip()
        response_role = dict(packet.get("response_role") or {})
        constraints = dict(packet.get("constraints") or {})
        shared_core = dict(packet.get("shared_core") or {})
        response_length = str((surface_profile or {}).get("response_length") or "").strip()
        primary_role = str(response_role.get("primary") or "").strip()
        anchor = str(shared_core.get("anchor") or "").strip()
        max_questions = int(constraints.get("max_questions") or 0)
        locale = _runtime_response_locale(self)

        def _pick_by_acts(preferred_acts: Sequence[str], *, limit: int = 2) -> List[Dict[str, Any]]:
            picked: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for act_name in preferred_acts:
                for item in sequence:
                    act = str(item.get("act") or "").strip()
                    if act != act_name or act in seen:
                        continue
                    picked.append(dict(item))
                    seen.add(act)
                    break
                if len(picked) >= limit:
                    break
            return picked

        if response_length == "short" and anchor and (
            phase in {"issue_pause", "reopening_thread", "continuing_thread", "thread_reopening"}
            or primary_role in {"leave_return_point_from_anchor", "reopen_from_anchor"}
        ):
            if len(sequence) == 1 and str(sequence[0].get("act") or "").strip() == "carry_text":
                if primary_role == "leave_return_point_from_anchor":
                    synthetic_text = _format_locale_template(
                        locale,
                        "inner_os.content_policy_templates.leave_return_point_from_anchor",
                        anchor=anchor,
                    )
                    if synthetic_text:
                        return [{"act": "leave_return_point_from_anchor", "text": synthetic_text}]
                synthetic_text = _format_locale_template(
                    locale,
                    "inner_os.content_policy_templates.reopen_from_anchor",
                    anchor=anchor,
                )
                if synthetic_text:
                    return [{"act": "reopen_from_anchor", "text": synthetic_text}]
            preferred = _pick_by_acts(
                (
                    "reopen_from_anchor_soft",
                    "reopen_from_anchor",
                    "leave_return_point_from_anchor_soft",
                    "leave_return_point_from_anchor",
                    "leave_return_point",
                    "quiet_presence",
                    "keep_shared_thread_visible",
                ),
                limit=2,
            )
            if preferred:
                return preferred

        if response_length == "short" and (
            primary_role == "stay_with_present_need"
            or phase in {"discussion_unresolved", "deep_disclosure", "exploring_issue"}
        ):
            preferred = _pick_by_acts(
                (
                    "reflect_hidden_need",
                    "reflect_self_blame",
                    "reflect_fear_of_being_seen",
                    "reflect_unspoken_weight",
                    "gentle_question_hidden_need_continuing",
                    "gentle_question_hidden_need",
                    "gentle_question_self_blame_continuing",
                    "gentle_question_self_blame",
                    "gentle_question_fear_continuing",
                    "gentle_question_fear",
                    "gentle_question_weight_continuing",
                    "gentle_question_weight",
                    "stay_with_present_need",
                    "quiet_presence",
                ),
                limit=2 if max_questions <= 0 else 3,
            )
            if preferred:
                if max_questions <= 0:
                    preferred = [
                        item
                        for item in preferred
                        if not str(item.get("act") or "").strip().startswith("gentle_question")
                    ][:2]
                    if (
                        primary_role == "stay_with_present_need"
                        and len(preferred) == 1
                        and str(preferred[0].get("act") or "").strip().startswith("reflect_")
                    ):
                        stay_text = str(
                            lookup_text(locale, "inner_os.content_policy_segments.stay_with_present_need") or ""
                        ).strip()
                        if stay_text:
                            preferred.append(
                                {"act": "stay_with_present_need", "text": stay_text}
                            )
                return preferred

        return sequence

    def _inner_os_surface_prefix(
        self,
        *,
        intent: str = "",
        interaction_policy: Optional[Mapping[str, Any]] = None,
    ) -> str:
        packet = dict(interaction_policy or {})
        contract = self._inner_os_surface_contract_state(packet)
        strategy = contract["response_strategy"]
        opening_move = contract["opening_move"]
        operation_kind = contract["primary_operation_kind"]
        operation_kinds = contract["operation_kinds"]
        effect_kinds = contract["effect_kinds"]
        deferred_object_labels = contract["deferred_object_labels"]
        question_budget = contract["question_budget"]
        question_pressure = contract["question_pressure"]
        defer_dominance = contract["defer_dominance"]

        if operation_kind == "narrow_clarify" or (
            intent == "clarify"
            and opening_move in {"narrow_scope_first", "narrow_scope_before_extend", "ask_one_bounded_part"}
        ):
            return "Let me check one small thing before I go further."
        if opening_move == "name_overreach_and_reduce_force":
            return "Let me slow down and meet the strain before I go further."
        if (
            operation_kind == "offer_small_next_step"
            or opening_move in {"anchor_visible_part", "anchor_shared_thread", "offer_one_small_next_step", "synchronize_then_propose"}
            or "keep_next_step_connected" in effect_kinds
            or "enable_small_next_step" in effect_kinds
        ):
            return "Let me line up one small next step with what is already shared."
        if (
            opening_move == "reduce_force_and_secure_boundary"
            or "protect_boundary" in effect_kinds
            or (
                operation_kind in {"defer_detail", "hold_without_probe"}
                and "protect_boundary" in effect_kinds
                and defer_dominance >= 0.56
            )
        ):
            return "Let me keep this inside what feels stable first."
        if (
            opening_move in {"acknowledge_and_wait", "acknowledge_without_probe"}
            or (
                operation_kind == "hold_without_probe"
                and (
                    bool(deferred_object_labels)
                    or question_budget <= 0
                    or question_pressure >= 0.52
                    or defer_dominance >= 0.52
                    or "avoid_forced_reopening" in effect_kinds
                    or "protect_unfinished_part" in operation_kinds
                    or "preserve_self_pacing" in effect_kinds
                )
            )
        ):
            return "Let me give this a little more room before I press further."
        if operation_kind == "acknowledge" and (
            "preserve_continuity" in effect_kinds
            or "feel_received" in effect_kinds
            or "keep_connection_open" in effect_kinds
        ):
            return "Let me stay with what is already here between us before I go further."
        if intent == "check_in":
            return "Let me check in gently before I go further."
        if intent == "clarify":
            return "Let me check one small thing before I go further."
        if strategy == "repair_then_attune":
            return "Let me slow down and meet the strain before I go further."
        if strategy == "respectful_wait":
            return "Let me give this a little more room before I press further."
        if strategy == "shared_world_next_step":
            return "Let me line up the next step with what is already shared."
        if strategy == "contain_then_stabilize":
            return "Let me keep this inside what feels stable first."
        if strategy == "reflect_without_settling":
            return "Let me stay close to what is here before I settle the meaning."
        return "Let me pause for one brief check before I go further."

    def _apply_inner_os_surface_profile(
        self,
        response: Optional[HubResponse],
        expression_hint: Optional[Mapping[str, Any]],
    ) -> Optional[HubResponse]:
        if response is None or not isinstance(expression_hint, Mapping):
            return response
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            return response
        fallback_user_text = str(
            getattr(self, "_last_surface_user_text", "") or ""
        ).strip()
        existing_controls_used = dict(getattr(response, "controls_used", {}) or {})
        interaction_policy = expression_hint.get("interaction_policy_packet")
        planned_content_sequence = existing_controls_used.get(
            "inner_os_planned_content_sequence"
        )
        if not isinstance(planned_content_sequence, list):
            planned_content_sequence = expression_hint.get("planned_content_sequence")
        if not isinstance(planned_content_sequence, list) or not planned_content_sequence:
            runtime_planned = getattr(self, "_last_planned_content_sequence", [])
            planned_content_sequence = runtime_planned if isinstance(runtime_planned, list) and runtime_planned else None
        if isinstance(planned_content_sequence, list):
            content_sequence = [
                dict(item)
                for item in planned_content_sequence
                if isinstance(item, Mapping) and str(item.get("text") or "").strip()
            ]
            content_sequence = localize_content_sequence(
                content_sequence,
                locale=_runtime_response_locale(self),
            )
        else:
            content_seed_text = fallback_user_text or text
            content_sequence = derive_content_sequence(
                current_text=content_seed_text,
                interaction_policy=interaction_policy if isinstance(interaction_policy, Mapping) else None,
                conscious_access={
                    "intent": str(expression_hint.get("interaction_policy_dialogue_act") or "")
                },
                history=_runtime_recent_surface_response_history(self),
                locale=_runtime_response_locale(self),
            )
        surface_context_packet = dict(expression_hint.get("surface_context_packet") or {})
        text = render_content_sequence(content_sequence)
        emergency_surface_acts = {
            "emergency_deescalate_boundary",
            "emergency_create_distance",
            "emergency_exit_now",
            "emergency_seek_help_now",
            "emergency_protect_now",
        }
        emergency_surface_active = any(
            str(item.get("act") or "").strip() in emergency_surface_acts
            for item in content_sequence
            if isinstance(item, Mapping)
        )
        surface_profile = {
            "opening_delay": str(expression_hint.get("surface_opening_delay") or "").strip(),
            "response_length": str(expression_hint.get("surface_response_length") or "").strip(),
            "sentence_temperature": str(expression_hint.get("surface_sentence_temperature") or "").strip(),
            "pause_insertion": str(expression_hint.get("surface_pause_insertion") or "").strip(),
            "certainty_style": str(expression_hint.get("surface_certainty_style") or "").strip(),
            "opening_pace_windowed": str(expression_hint.get("opening_pace_windowed") or "").strip(),
            "return_gaze_expectation": str(expression_hint.get("return_gaze_expectation") or "").strip(),
            "voice_texture": str(expression_hint.get("surface_voice_texture") or "").strip(),
            "cultural_register": str(expression_hint.get("surface_cultural_register") or "").strip(),
            "lightness_room": float(expression_hint.get("surface_lightness_room") or 0.0),
            "continuity_weight": float(expression_hint.get("surface_continuity_weight") or 0.0),
            "banter_move": str(expression_hint.get("surface_banter_move") or "").strip(),
            "lexical_variation_mode": str(expression_hint.get("surface_lexical_variation_mode") or "").strip(),
            "group_register": str(expression_hint.get("surface_group_register") or "").strip(),
            "live_opening_pace_mismatch": float(expression_hint.get("live_opening_pace_mismatch") or 0.0),
            "live_return_gaze_mismatch": float(expression_hint.get("live_return_gaze_mismatch") or 0.0),
            "stream_shared_attention_window_mean": float(expression_hint.get("stream_shared_attention_window_mean") or 0.0),
        }
        if not surface_context_packet:
            interaction_policy = (
                interaction_policy if isinstance(interaction_policy, Mapping) else {}
            )
            surface_context_packet = build_surface_context_packet(
                recent_dialogue_state=interaction_policy.get("recent_dialogue_state"),
                discussion_thread_state=interaction_policy.get("discussion_thread_state"),
                issue_state=interaction_policy.get("issue_state"),
                turn_delta=expression_hint.get("turn_delta"),
                interaction_constraints=expression_hint.get("interaction_constraints"),
                boundary_transform=expression_hint.get("boundary_transform"),
                residual_reflection=expression_hint.get("residual_reflection"),
                surface_profile=surface_profile,
                contact_reflection_state=expression_hint.get("contact_reflection_state"),
                green_kernel_composition=expression_hint.get("green_kernel_composition"),
                dialogue_context={"user_text": fallback_user_text},
            ).to_dict()
        content_sequence = self._apply_surface_context_packet_to_content_sequence(
            content_sequence,
            surface_context_packet=surface_context_packet,
            surface_profile=surface_profile,
        )
        actuation_plan = expression_hint.get("actuation_plan")
        if isinstance(actuation_plan, Mapping):
            execution_mode = str(actuation_plan.get("execution_mode") or "").strip()
            reply_permission = str(actuation_plan.get("reply_permission") or "").strip()
            if execution_mode == "shared_progression" and surface_profile["response_length"] == "balanced":
                surface_profile["response_length"] = "forward_leaning"
            elif execution_mode == "open_reflection" and surface_profile["response_length"] == "balanced":
                surface_profile["response_length"] = "reflective"
            elif execution_mode in {"defer_with_presence", "stabilize_boundary"}:
                surface_profile["certainty_style"] = "careful"
            if reply_permission in {"hold_or_brief", "speak_minimal"}:
                surface_profile["response_length"] = "short"
            elif reply_permission == "speak_reflective":
                surface_profile["response_length"] = "reflective"
            surface_profile["actuation_execution_mode"] = execution_mode
            surface_profile["actuation_reply_permission"] = reply_permission
            surface_profile["actuation_primary_action"] = str(actuation_plan.get("primary_action") or "").strip()
        live_mismatch = max(
            float(surface_profile["live_opening_pace_mismatch"]),
            float(surface_profile["live_return_gaze_mismatch"]),
        )
        shared_attention_window_mean = float(surface_profile["stream_shared_attention_window_mean"])
        if live_mismatch >= 0.42 and surface_profile["pause_insertion"] == "none":
            surface_profile["pause_insertion"] = "soft_pause"
        if live_mismatch >= 0.64:
            surface_profile["pause_insertion"] = "visible_pause"
            if surface_profile["return_gaze_expectation"] == "soft_return":
                surface_profile["return_gaze_expectation"] = "careful_return"
        if live_mismatch >= 0.48 and shared_attention_window_mean <= 0.34:
            surface_profile["certainty_style"] = "careful"
        if live_mismatch >= 0.58 and shared_attention_window_mean <= 0.28:
            surface_profile["response_length"] = "short"
        if emergency_surface_active:
            surface_profile["response_length"] = "short"
            surface_profile["certainty_style"] = "careful"
            surface_profile["pause_insertion"] = "none"
            surface_profile["opening_pace_windowed"] = ""
            surface_profile["return_gaze_expectation"] = ""
        narrative_bridge_text = ""
        allow_guarded_narrative_bridge_value = existing_controls_used.get(
            "inner_os_allow_guarded_narrative_bridge"
        )
        if allow_guarded_narrative_bridge_value is None:
            allow_guarded_narrative_bridge_value = expression_hint.get(
                "allow_guarded_narrative_bridge"
            )
        allow_guarded_narrative_bridge = bool(allow_guarded_narrative_bridge_value)
        if any(
            str(item.get("act") or "").strip()
            in {"offer_small_opening_line", "offer_small_opening_frame"}
            for item in content_sequence
        ):
            allow_guarded_narrative_bridge = False
        if (
            str(surface_profile.get("response_length") or "").strip() == "short"
            and len(content_sequence) >= 3
        ):
            allow_guarded_narrative_bridge = False
        if emergency_surface_active:
            allow_guarded_narrative_bridge = False
        if allow_guarded_narrative_bridge:
            narrative_bridge_text = self._extract_guarded_narrative_bridge_text(
                str(
                    existing_controls_used.get("inner_os_llm_raw_text")
                    or expression_hint.get("llm_raw_text")
                    or ""
                ),
                locale=_runtime_response_locale(self),
            )
        shaped = self._shape_inner_os_content_sequence(
            content_sequence,
            surface_profile=surface_profile,
            fallback_text=text,
            narrative_bridge_text=narrative_bridge_text,
        )
        response.text = shaped
        controls_used = dict(existing_controls_used)
        controls_used["inner_os_planned_content_sequence"] = [
            dict(item) for item in content_sequence
        ]
        controls_used["inner_os_surface_profile"] = dict(surface_profile)
        controls_used["inner_os_surface_profile"]["content_sequence_length"] = len(content_sequence)
        controls_used["inner_os_boundary_transform"] = dict(
            expression_hint.get("boundary_transform") or {}
        )
        controls_used["inner_os_residual_reflection"] = dict(
            expression_hint.get("residual_reflection") or {}
        )
        controls_used["inner_os_allow_guarded_narrative_bridge"] = bool(
            allow_guarded_narrative_bridge
        )
        controls_used["inner_os_guarded_narrative_bridge_used"] = bool(
            narrative_bridge_text
        )
        response.controls_used = controls_used
        return response

    def _recent_surface_response_history(self) -> list[str]:
        return [
            text
            for text in (
                str(item or "").strip()
                for item in getattr(self, "_surface_response_history", ())
            )
            if text
        ]

    def _recent_surface_user_history(self) -> list[str]:
        return [
            text
            for text in (
                str(item or "").strip()
                for item in getattr(self, "_surface_user_history", ())
            )
            if text
        ]

    def _recent_dialogue_thread_history(self) -> list[str]:
        merged: list[str] = []
        for item in (
            *self._recent_surface_user_history(),
            *self._recent_surface_response_history(),
        ):
            text = str(item or "").strip()
            if text and text not in merged:
                merged.append(text)
        return merged[-6:]

    def _inner_os_surface_probe(self, *, intent: str = "", expression_hint: Optional[Mapping[str, Any]] = None, question_bias: float = 0.0) -> str:
        if intent != "check_in" or not isinstance(expression_hint, Mapping):
            return ""
        if not bool(expression_hint.get("carry_gentleness", False)):
            return ""
        if str(expression_hint.get("interaction_afterglow_intent") or "").strip().lower() != "check_in":
            return ""
        if question_bias < INNER_OS_SURFACE_THRESHOLDS["probe_question_bias"]:
            return ""
        return "Would it help if I stay close to what is visible first?"

    def _inner_os_surface_reopening_line(self, *, intent: str = "", expression_hint: Optional[Mapping[str, Any]] = None) -> str:
        if not isinstance(expression_hint, Mapping):
            return ""
        if not bool(expression_hint.get("allow_reopening", False)):
            return ""
        if float(expression_hint.get("recovery_reopening", 0.0) or 0.0) < INNER_OS_SURFACE_THRESHOLDS["reopening_recovery"]:
            return ""
        if intent == "check_in":
            return "I think we can open this a little more carefully now."
        if intent == "clarify":
            return "I think we can open this a little more clearly now."
        return "I think we can open this a little more carefully now."

    def _inner_os_surface_policy_level(self, controls_used: Optional[Mapping[str, Any]]) -> str:
        policy = str((controls_used or {}).get("inner_os_surface_policy") or "").strip()
        if not policy:
            return "none"
        return "layered" if ":" in policy else "prefix_only"

    def _inner_os_surface_policy_intent(self, controls_used: Optional[Mapping[str, Any]]) -> Optional[str]:
        policy = str((controls_used or {}).get("inner_os_surface_policy") or "").strip()
        if ":" not in policy:
            return None
        return policy.split(":", 1)[1].strip() or None

    def _compose_inner_os_surface_text(self, prefix: str, body: str, closing: str) -> str:
        lines = []
        for part in (prefix, body, closing):
            value = str(part or "").strip()
            if value and value not in lines:
                lines.append(value)
        return "\n".join(lines)

    def _inner_os_surface_closing(
        self,
        *,
        intent: str = "",
        question_bias: float = 0.0,
        interaction_policy: Optional[Mapping[str, Any]] = None,
    ) -> str:
        packet = dict(interaction_policy or {})
        contract = self._inner_os_surface_contract_state(packet)
        closing_move = contract["closing_move"]
        operation_kind = contract["primary_operation_kind"]
        operation_kinds = contract["operation_kinds"]
        effect_kinds = contract["effect_kinds"]
        deferred_object_labels = contract["deferred_object_labels"]
        question_budget = contract["question_budget"]
        question_pressure = contract["question_pressure"]
        defer_dominance = contract["defer_dominance"]

        if (
            "avoid_forced_reopening" in effect_kinds
            or "protect_unfinished_part" in operation_kinds
            or (
                bool(deferred_object_labels)
                and (
                    operation_kind == "hold_without_probe"
                    or question_budget <= 0
                    or question_pressure >= 0.52
                    or defer_dominance >= 0.52
                )
            )
        ):
            return "We do not need to open the rest right now."
        if (
            "keep_next_step_connected" in effect_kinds
            or "enable_small_next_step" in effect_kinds
            or "anchor_next_step_in_theme" in operation_kinds
            or operation_kind == "offer_small_next_step"
        ):
            return "We can keep the next step connected from here."
        if "preserve_self_pacing" in effect_kinds or (
            "reduce_pressure" in effect_kinds and question_pressure >= 0.48
        ):
            return "You can keep your own pace from here."
        if closing_move == "hold_space":
            return "I can keep the space open without forcing it."
        if closing_move == "leave_space":
            return "I can leave this with room around it."
        if closing_move == "keep_pace_mutual":
            return "We can keep the pace shared from here."
        if closing_move == "do_not_overextend":
            return "I do not want to push this past what feels steady."
        if closing_move == "avoid_false_closure":
            return "I would rather leave this slightly open than close it too fast."
        if question_bias < INNER_OS_SURFACE_THRESHOLDS["closing_question_bias"]:
            return ""
        if intent == "check_in":
            return "I want to stay with this gently first."
        if intent == "clarify":
            return "Then I can answer a little more cleanly."
        return "Then I can continue a bit more carefully."

    def _inner_os_surface_contract_state(
        self,
        packet: Mapping[str, Any],
    ) -> Dict[str, Any]:
        conversation_contract = dict(packet.get("conversation_contract") or {})
        response_action = dict(conversation_contract.get("response_action_now") or {})
        primary_operation = dict(packet.get("primary_object_operation") or {})
        ordered_operation_kinds = [
            str(item).strip()
            for item in packet.get("ordered_operation_kinds") or []
            if str(item).strip()
        ]
        if not ordered_operation_kinds:
            ordered_operation_kinds = [
                str(item).strip()
                for item in response_action.get("ordered_operations") or []
                if str(item).strip()
            ]
        legacy_operation_kinds = [
            str(item).strip()
            for item in packet.get("object_operation_kinds") or []
            if str(item).strip()
        ]
        ordered_effect_kinds = [
            str(item).strip()
            for item in packet.get("ordered_effect_kinds") or []
            if str(item).strip()
        ]
        if not ordered_effect_kinds:
            ordered_effect_kinds = [
                str(item).strip()
                for item in conversation_contract.get("ordered_effects") or []
                if str(item).strip()
            ]
        legacy_effect_kinds = [
            str(item).strip()
            for item in packet.get("interaction_effect_kinds") or []
            if str(item).strip()
        ]
        if not legacy_effect_kinds:
            legacy_effect_kinds = [
                str(item.get("effect") or "").strip()
                for item in conversation_contract.get("wanted_effect_on_other") or conversation_contract.get("intended_effects") or []
                if isinstance(item, Mapping) and str(item.get("effect") or "").strip()
            ]
        deferred_object_labels = [
            str(item).strip()
            for item in packet.get("deferred_object_labels") or []
            if str(item).strip()
        ]
        if not deferred_object_labels:
            deferred_object_labels = [
                str(item).strip()
                for item in (
                    conversation_contract.get("leave_closed_for_now")
                    or conversation_contract.get("do_not_open_yet")
                    or conversation_contract.get("deferred_objects")
                    or []
                )
                if str(item).strip()
            ]
        operation_kinds = set(ordered_operation_kinds or legacy_operation_kinds)
        effect_kinds = set(ordered_effect_kinds or legacy_effect_kinds)
        return {
            "response_strategy": str(packet.get("response_strategy") or "").strip(),
            "opening_move": str(packet.get("opening_move") or "").strip(),
            "closing_move": str(packet.get("closing_move") or "").strip(),
            "primary_operation_kind": str(
                primary_operation.get("operation_kind")
                or response_action.get("primary_operation")
                or ""
            ).strip(),
            "operation_kinds": operation_kinds,
            "effect_kinds": effect_kinds,
            "deferred_object_labels": deferred_object_labels,
            "question_budget": max(0, int(packet.get("question_budget") or response_action.get("question_budget") or 0)),
            "question_pressure": max(0.0, min(1.0, float(packet.get("question_pressure", response_action.get("question_pressure", 0.0)) or 0.0))),
            "defer_dominance": max(0.0, min(1.0, float(packet.get("defer_dominance", response_action.get("defer_dominance", 0.0)) or 0.0))),
        }

    def _shape_inner_os_surface_text(self, text: str, *, intent: str = "") -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        if intent != "check_in":
            return cleaned
        first_line = cleaned.splitlines()[0].strip()
        if not first_line:
            return cleaned
        for marker in (". ", "? ", "! "):
            if marker in first_line:
                return first_line.split(marker, 1)[0].strip() + marker[0]
        return first_line

    def _shape_inner_os_surface_profile_text(
        self,
        text: str,
        *,
        surface_profile: Optional[Mapping[str, Any]] = None,
    ) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        profile = dict(surface_profile or {})
        response_length = str(profile.get("response_length") or "").strip()
        certainty_style = str(profile.get("certainty_style") or "").strip()
        sentence_temperature = str(profile.get("sentence_temperature") or "").strip()
        pause_insertion = str(profile.get("pause_insertion") or "").strip()
        opening_pace_windowed = str(profile.get("opening_pace_windowed") or "").strip()
        return_gaze_expectation = str(profile.get("return_gaze_expectation") or "").strip()
        locale = _runtime_response_locale(self)
        is_ja = locale.lower().startswith("ja")
        skip_soft_pause_prefixes: tuple[str, ...] = ()
        if is_ja:
            configured_prefixes = lookup_value(
                locale,
                "inner_os.surface_profile_cues.skip_soft_pause_prefixes",
            )
            if isinstance(configured_prefixes, list):
                skip_soft_pause_prefixes = tuple(
                    str(item).strip()
                    for item in configured_prefixes
                    if str(item).strip()
                )

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        body = " ".join(lines).strip()
        if not body:
            return cleaned

        ja_tentative_prefixes = ("たぶん、", "おそらく、", "最初は、", "切り出すなら、", "…", "...")
        ja_careful_prefixes = (
            "いまは、",
            "慎重に言うと、",
            "最初は、",
            "切り出すなら、",
            "助けてほしかったのに、",
            "まだ言えていないことが、",
            "言ったあとにどう見られるか、",
            "そのことが、いまも",
            "…",
            "...",
        )
        ja_soft_return_prefixes = (
            "ここにいます。",
            "いまは、",
            "最初は、",
            "切り出すなら、",
            "助けてほしかったのに、",
            "まだ言えていないことが、",
            "言ったあとにどう見られるか、",
            "そのことが、いまも",
            "…",
            "...",
        )
        ja_reflective_tail = "もう少しゆっくり見ていけます。"
        ja_forward_tail = "ここから先も、小さく一歩ずつで大丈夫です。"

        if response_length == "short":
            for marker in (". ", "? ", "! "):
                if marker in body:
                    body = body.split(marker, 1)[0].strip() + marker[0]
                    break
        elif response_length == "reflective":
            if is_ja and not body.endswith(ja_reflective_tail):
                body = f"{body} {ja_reflective_tail}"
            elif not is_ja and not body.endswith("I may need to stay with it a little longer."):
                body = f"{body} I may need to stay with it a little longer."
        elif response_length == "forward_leaning":
            if is_ja and not body.endswith(ja_forward_tail):
                body = f"{body} {ja_forward_tail}"
            elif not is_ja and not body.endswith("I think we can take one next step from here."):
                body = f"{body} I think we can take one next step from here."

        if certainty_style == "tentative":
            if is_ja and not body.startswith(ja_tentative_prefixes):
                body = f"たぶん、{body}"
            elif not is_ja and not body.lower().startswith("i think"):
                body = f"I think {body[0].lower() + body[1:]}" if len(body) > 1 else f"I think {body}"
        elif certainty_style == "careful":
            if is_ja and not body.startswith(ja_careful_prefixes):
                body = f"いまは、{body}"
            elif not is_ja and not body.lower().startswith(("carefully,", "let me ")):
                body = f"Carefully, {body[0].lower() + body[1:]}" if len(body) > 1 else f"Carefully, {body}"

        if opening_pace_windowed == "held" and not body.startswith(("…", "...")):
            body = f"… {body}" if is_ja else f"... {body}"
        elif opening_pace_windowed == "measured" and pause_insertion == "none" and not body.startswith(("…", "...", "..")):
            body = f"… {body}" if is_ja else f".. {body}"

        if return_gaze_expectation == "soft_return":
            if is_ja and not body.startswith(ja_soft_return_prefixes):
                if sentence_temperature == "warm":
                    body = f"ここにいます。{body}"
            elif not is_ja and not body.lower().startswith(("gently,", "carefully,", "...", "..", "let me ", "i'm here")):
                if sentence_temperature == "warm":
                    body = f"I'm here. {body}"
                elif sentence_temperature == "gentle":
                    body = body
                else:
                    body = f"Gently, {body[0].lower() + body[1:]}" if len(body) > 1 else f"Gently, {body}"
        elif return_gaze_expectation == "careful_return":
            if is_ja and not body.startswith(ja_careful_prefixes):
                body = f"いまは、{body}"
            elif not is_ja and not body.lower().startswith(("carefully,", "let me ")):
                body = f"Carefully, {body[0].lower() + body[1:]}" if len(body) > 1 else f"Carefully, {body}"
        elif return_gaze_expectation == "defer_return" and not body.startswith(("…", "...")) and not body.lower().startswith("let me "):
            body = f"… {body}" if is_ja else f"... {body}"

        skip_soft_pause = bool(
            is_ja
            and (
                (
                    skip_soft_pause_prefixes
                    and body.startswith(skip_soft_pause_prefixes)
                )
                or "のところなら、" in body[:24]
            )
        )
        if pause_insertion == "visible_pause" and not body.startswith(("…", "...")):
            body = f"… {body}" if is_ja else f"... {body}"
        elif (
            pause_insertion == "soft_pause"
            and not body.startswith(("…", "...", ".."))
            and not skip_soft_pause
        ):
            body = f"… {body}" if is_ja else f".. {body}"
        return shape_surface_language_text(body, surface_profile=profile, locale=locale)

    def _shape_inner_os_content_sequence(
        self,
        content_sequence: Optional[List[Mapping[str, Any]]],
        *,
        surface_profile: Optional[Mapping[str, Any]] = None,
        fallback_text: str = "",
        narrative_bridge_text: str = "",
    ) -> str:
        sequence = [dict(item) for item in (content_sequence or []) if str(item.get("text") or "").strip()]
        if not sequence:
            return self._shape_inner_os_surface_profile_text(fallback_text, surface_profile=surface_profile)
        if len(sequence) == 1 and str(sequence[0].get("act") or "") == "carry_text":
            return self._shape_inner_os_surface_profile_text(
                str(sequence[0].get("text") or "").strip(),
                surface_profile=surface_profile,
            )

        profile = dict(surface_profile or {})
        response_length = str(profile.get("response_length") or "").strip()
        if response_length == "short" and len(sequence) > 2:
            sequence = self._select_short_inner_os_sequence(sequence)
        locale = _runtime_response_locale(self)
        sequence = [
            {
                **item,
                "text": _choose_surface_act_text(
                    self,
                    act=str(item.get("act") or "").strip(),
                    text=str(item.get("text") or "").strip(),
                    locale=locale,
                    surface_profile=profile,
                ),
            }
            for item in sequence
        ]
        compact_text = self._compact_inner_os_sequence_text(
            sequence,
            surface_profile=profile,
        )
        if compact_text:
            return self._shape_inner_os_surface_profile_text(
                compact_text,
                surface_profile=surface_profile,
            )

        opening_profile = dict(profile)
        opening_profile["response_length"] = "balanced"
        opening_act = str(sequence[0].get("act") or "").strip()
        if opening_act == "acknowledge_overreach":
            opening_profile["return_gaze_expectation"] = "careful_return"
        elif opening_act == "shared_anchor":
            opening_profile["return_gaze_expectation"] = "steady_return"
        opening_text = self._shape_inner_os_surface_profile_text(
            str(sequence[0].get("text") or "").strip(),
            surface_profile=opening_profile,
        )
        rendered: List[str] = [opening_text]
        bridge_text = str(narrative_bridge_text or "").strip()
        if bridge_text:
            bridge_profile = dict(profile)
            bridge_profile["certainty_style"] = ""
            bridge_profile["pause_insertion"] = "none"
            bridge_profile["opening_pace_windowed"] = ""
            bridge_profile["return_gaze_expectation"] = ""
            shaped_bridge = self._shape_inner_os_surface_profile_text(
                bridge_text,
                surface_profile=bridge_profile,
            )
            if shaped_bridge and shaped_bridge not in rendered:
                rendered.append(shaped_bridge)

        trailing_segments = [
            str(item.get("text") or "").strip()
            for item in sequence[1:]
            if str(item.get("text") or "").strip()
        ]
        if trailing_segments:
            rendered.extend(trailing_segments)

        joined = " ".join(part for part in rendered if part).strip()
        if not joined:
            return self._shape_inner_os_surface_profile_text(fallback_text, surface_profile=surface_profile)

        response_length = str(profile.get("response_length") or "").strip()
        if response_length == "reflective" and not joined.endswith("I may need to stay with it a little longer."):
            joined = f"{joined} I may need to stay with it a little longer."
        elif response_length == "forward_leaning" and not joined.endswith("I think we can take one next step from here."):
            joined = f"{joined} I think we can take one next step from here."
        return joined

    def _compact_inner_os_sequence_text(
        self,
        sequence: List[Mapping[str, Any]],
        *,
        surface_profile: Optional[Mapping[str, Any]] = None,
    ) -> str:
        locale = _runtime_response_locale(self)
        if not locale.lower().startswith("ja"):
            return ""
        response_length = str((surface_profile or {}).get("response_length") or "").strip()
        if response_length != "short":
            return ""
        act_to_text = {
            str(item.get("act") or "").strip(): str(item.get("text") or "").strip()
            for item in sequence
            if str(item.get("act") or "").strip() and str(item.get("text") or "").strip()
        }
        disclosure_reflection_text = (
            act_to_text.get("reflect_hidden_need")
            or act_to_text.get("reflect_self_blame")
            or act_to_text.get("reflect_fear_of_being_seen")
            or act_to_text.get("reflect_unspoken_weight")
        )
        disclosure_question_text = (
            act_to_text.get("gentle_question_hidden_need")
            or act_to_text.get("gentle_question_hidden_need_continuing")
            or act_to_text.get("gentle_question_self_blame")
            or act_to_text.get("gentle_question_self_blame_continuing")
            or act_to_text.get("gentle_question_fear")
            or act_to_text.get("gentle_question_fear_continuing")
            or act_to_text.get("gentle_question_weight")
            or act_to_text.get("gentle_question_weight_continuing")
        )
        opening_text = act_to_text.get("offer_small_opening_line") or act_to_text.get(
            "offer_small_opening_frame"
        )
        thread_opening_text = (
            act_to_text.get("reopen_from_anchor_soft")
            or act_to_text.get("reopen_from_anchor")
            or act_to_text.get("leave_return_point_from_anchor_soft")
            or act_to_text.get("leave_return_point_from_anchor")
        )
        if disclosure_reflection_text:
            parts: List[str] = [disclosure_reflection_text]
            if disclosure_question_text:
                parts.append(disclosure_question_text)
            elif "stay_with_present_need" in act_to_text:
                compact_reflection_stay = _choose_compact_phrase(
                    self,
                    locale=locale,
                    primary_key="inner_os.content_policy_compact.deep_reflection_stay",
                    alternatives_key="inner_os.content_policy_compact.deep_reflection_stay_alternatives",
                    surface_profile=surface_profile,
                    candidate_profile="deep_reflection_stay",
                )
                parts.append(
                    compact_reflection_stay
                    or act_to_text.get("stay_with_present_need", "")
                )
            elif "quiet_presence" in act_to_text or any(
                act in act_to_text
                for act in (
                    "keep_shared_thread_visible",
                    "leave_return_point",
                    "leave_unfinished_closed",
                )
            ):
                compact_reflection_presence = _choose_compact_phrase(
                    self,
                    locale=locale,
                    primary_key="inner_os.content_policy_compact.deep_reflection_presence",
                    alternatives_key="inner_os.content_policy_compact.deep_reflection_presence_alternatives",
                    surface_profile=surface_profile,
                    candidate_profile="deep_reflection_presence",
                )
                parts.append(
                    compact_reflection_presence
                    or act_to_text.get("quiet_presence", "")
                )
            return " ".join(part.strip() for part in parts if str(part).strip()).strip()
        if (
            "respect_boundary" in act_to_text
            and "keep_shared_thread_visible" in act_to_text
            and "quiet_presence" in act_to_text
            and not opening_text
            and not thread_opening_text
        ):
            compact_continuity_opening = _choose_compact_phrase(
                self,
                locale=locale,
                primary_key="inner_os.content_policy_compact.continuity_opening",
                alternatives_key="inner_os.content_policy_compact.continuity_opening_alternatives",
                surface_profile=surface_profile,
                candidate_profile="continuity_opening",
            )
            if compact_continuity_opening:
                return compact_continuity_opening
        if not opening_text and not thread_opening_text:
            return ""
        parts: List[str] = [opening_text or thread_opening_text]
        if thread_opening_text and not opening_text:
            thread_anchor = normalize_anchor_hint(thread_opening_text, limit=32)
            if thread_anchor and any(
                act in act_to_text
                for act in (
                    "reopen_from_anchor",
                    "reopen_from_anchor_soft",
                    "leave_return_point_from_anchor",
                    "leave_return_point_from_anchor_soft",
                )
            ):
                compact_thread_opening = _format_locale_template(
                    locale,
                    "inner_os.thread_reopen_compact.from_anchor",
                    anchor=thread_anchor,
                )
                if compact_thread_opening:
                    parts = [compact_thread_opening]
                    if any(
                        act in act_to_text
                        for act in ("leave_return_point", "leave_unfinished_closed")
                    ):
                        compact_thread_return = _choose_compact_phrase(
                            self,
                            locale=locale,
                            primary_key="inner_os.thread_reopen_compact.return",
                            alternatives_key="inner_os.thread_reopen_compact.return_alternatives",
                            surface_profile=surface_profile,
                            candidate_profile="thread_reopen_return",
                        )
                        if compact_thread_return:
                            parts.append(compact_thread_return)
                    elif "quiet_presence" in act_to_text:
                        compact_thread_presence = _choose_compact_phrase(
                            self,
                            locale=locale,
                            primary_key="inner_os.content_policy_compact.thread_presence",
                            alternatives_key="inner_os.content_policy_compact.thread_presence_alternatives",
                            surface_profile=surface_profile,
                            candidate_profile="thread_presence",
                        )
                        if compact_thread_presence:
                            parts.append(compact_thread_presence)
                    return " ".join(
                        part.strip() for part in parts if str(part).strip()
                    ).strip()
            compact_thread_presence = _choose_compact_phrase(
                self,
                locale=locale,
                primary_key="inner_os.content_policy_compact.thread_presence",
                alternatives_key="inner_os.content_policy_compact.thread_presence_alternatives",
                surface_profile=surface_profile,
                candidate_profile="thread_presence",
            )
            if compact_thread_presence and (
                "quiet_presence" in act_to_text
                or any(
                    act in act_to_text
                    for act in ("leave_return_point", "leave_unfinished_closed")
                )
            ):
                parts.append(compact_thread_presence)
            return " ".join(part.strip() for part in parts if str(part).strip()).strip()
        if "quiet_presence" not in act_to_text:
            return ""
        compact_presence = lookup_text(
            locale,
            "inner_os.content_policy_compact.opening_presence",
        )
        if compact_presence:
            parts.append(compact_presence)
        if any(
            act in act_to_text
            for act in ("leave_return_point", "leave_unfinished_closed")
        ):
            compact_return = lookup_text(
                locale,
                "inner_os.content_policy_compact.opening_return",
            )
            if compact_return:
                parts.append(compact_return)
        return " ".join(part.strip() for part in parts if str(part).strip()).strip()

    def _select_short_inner_os_sequence(
        self,
        sequence: List[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized = [
            dict(item)
            for item in sequence
            if str(item.get("text") or "").strip()
        ]
        if len(normalized) <= 2:
            return normalized

        disclosure_candidate: Dict[str, Any] | None = None
        disclosure_question_candidate: Dict[str, Any] | None = None
        opening_candidate: Dict[str, Any] | None = None
        thread_opening_candidate: Dict[str, Any] | None = None
        for item in normalized:
            act = str(item.get("act") or "").strip()
            if disclosure_candidate is None and act in {
                "reflect_hidden_need",
                "reflect_self_blame",
                "reflect_fear_of_being_seen",
                "reflect_unspoken_weight",
            }:
                disclosure_candidate = dict(item)
            if disclosure_question_candidate is None and act in {
                "gentle_question_hidden_need",
                "gentle_question_hidden_need_continuing",
                "gentle_question_self_blame",
                "gentle_question_self_blame_continuing",
                "gentle_question_fear",
                "gentle_question_fear_continuing",
                "gentle_question_weight",
                "gentle_question_weight_continuing",
            }:
                disclosure_question_candidate = dict(item)
            if act in {"offer_small_opening_line", "offer_small_opening_frame"}:
                opening_candidate = dict(item)
                break
            if thread_opening_candidate is None and act in {
                "reopen_from_anchor",
                "reopen_from_anchor_soft",
                "leave_return_point_from_anchor",
                "leave_return_point_from_anchor_soft",
            }:
                thread_opening_candidate = dict(item)

        if disclosure_candidate is not None:
            selected = [disclosure_candidate]
            selected_acts = {
                str(disclosure_candidate.get("act") or "").strip()
            }
        elif opening_candidate is not None:
            selected = [opening_candidate]
            selected_acts = {
                str(opening_candidate.get("act") or "").strip()
            }
        elif thread_opening_candidate is not None:
            selected = [thread_opening_candidate]
            selected_acts = {
                str(thread_opening_candidate.get("act") or "").strip()
            }
        else:
            selected = [dict(normalized[0])]
            selected_acts = {str(normalized[0].get("act") or "").strip()}

        def append_first(*acts: str) -> None:
            for item in normalized:
                act = str(item.get("act") or "").strip()
                if act in acts and act not in selected_acts:
                    selected.append(dict(item))
                    selected_acts.add(act)
                    return

        if disclosure_candidate is not None:
            if disclosure_question_candidate is not None:
                append_first(
                    "gentle_question_hidden_need",
                    "gentle_question_hidden_need_continuing",
                    "gentle_question_self_blame",
                    "gentle_question_self_blame_continuing",
                    "gentle_question_fear",
                    "gentle_question_fear_continuing",
                    "gentle_question_weight",
                    "gentle_question_weight_continuing",
                )
            else:
                append_first("quiet_presence")
        elif opening_candidate is None:
            append_first(
                "reopen_from_anchor",
                "reopen_from_anchor_soft",
                "leave_return_point_from_anchor",
                "leave_return_point_from_anchor_soft",
                "keep_shared_thread_visible",
            )
        if disclosure_candidate is None:
            append_first("quiet_presence")
        if (
            disclosure_candidate is None
            and opening_candidate is None
            and thread_opening_candidate is None
            and len(selected) < 2
        ):
            append_first("respect_boundary", "respect_boundary_soft")
        if len(selected) < 3:
            append_first(
                "reopen_from_anchor",
                "reopen_from_anchor_soft",
                "leave_return_point_from_anchor",
                "leave_return_point_from_anchor_soft",
                "keep_shared_thread_visible",
                "leave_return_point",
                "leave_unfinished_closed",
            )
        if disclosure_candidate is not None and len(selected) < 3:
            append_first("quiet_presence")
        if len(selected) < 2:
            for item in normalized:
                act = str(item.get("act") or "").strip()
                if act and act not in selected_acts:
                    selected.append(dict(item))
                    selected_acts.add(act)
                    if len(selected) >= 2:
                        break
        if len(selected) < 3:
            for item in normalized:
                act = str(item.get("act") or "").strip()
                if act and act not in selected_acts:
                    selected.append(dict(item))
                    selected_acts.add(act)
                    if len(selected) >= 3:
                        break
        return selected

    def _extract_guarded_narrative_bridge_text(
        self,
        raw_text: str,
        *,
        locale: str = "ja-JP",
    ) -> str:
        body = str(raw_text or "").replace("\r\n", "\n").strip()
        if not body:
            return ""

        visible_lines: List[str] = []
        for raw_line in body.splitlines():
            line = re.sub(r"\s+", " ", str(raw_line or "").strip())
            if not line:
                continue
            if line.startswith("※"):
                continue
            if re.match(r"^[-*•]\s*", line):
                continue
            if re.match(r"^\d+[.)]\s*", line):
                continue
            if line.startswith(("例：", "例:", "ポイントは", "For example", "Examples:")):
                continue
            visible_lines.append(line)
            if len(visible_lines) >= 2:
                break

        if not visible_lines:
            return ""

        candidate = " ".join(visible_lines).strip()
        parts = [
            segment.strip()
            for segment in re.split(r"(?<=[。！？!?])\s*|\n+", candidate)
            if str(segment).strip()
        ]
        normalized_locale = str(locale or "").lower()
        if normalized_locale.startswith("ja"):
            japanese_parts = [
                segment
                for segment in parts
                if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", segment)
            ]
            if japanese_parts:
                candidate = japanese_parts[0]
            elif parts:
                return ""
        elif parts:
            candidate = parts[0]
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not candidate:
            return ""
        generic_phrases = (
            "お手伝いできること",
            "詳しく教えて",
            "内容が不明",
            "ご質問",
            "申し訳ありません",
            "ごめんなさい",
        )
        if any(phrase in candidate for phrase in generic_phrases):
            return ""
        return truncate_text(candidate, 110)

    def _attach_visual_reflection(
        self,
        response: Optional[HubResponse],
        perception_summary: Optional[Dict[str, Any]],
    ) -> Optional[HubResponse]:
        if response is None:
            return None
        response.perception_summary = perception_summary
        line = self._visual_reflection_line(perception_summary)
        if not line:
            return response
        existing = (response.text or "").strip()
        if not existing:
            response.text = line
            return response
        if line in existing:
            return response
        response.text = f"{existing}\n{line}"
        return response

    def _memory_cue_text(
        self,
        user_text: Optional[str],
        perception_summary: Optional[Dict[str, Any]],
    ) -> str:
        base = (user_text or "").strip()
        if not isinstance(perception_summary, dict):
            return base
        visual_text = perception_summary.get("text")
        if not isinstance(visual_text, str) or not visual_text.strip():
            return base
        cue = visual_text.strip().splitlines()[0].strip()
        if len(cue) > 160:
            cue = cue[:157].rstrip() + "..."
        if not base:
            return cue
        return f"{base}\n[vision-cue] {cue}"

    def _retrieve_from_memory_cue(
        self,
        memory_cue_text: str,
    ) -> Optional[Dict[str, Any]]:
        query = (memory_cue_text or "").strip()
        if not query:
            return None
        adapter = getattr(self, "_sse_adapter", None)
        if adapter is None:
            adapter = SSESearchAdapter()
            self._sse_adapter = adapter
        if not adapter.enabled:
            return None
        try:
            hits = adapter.search(query)
        except Exception as exc:
            return {
                "backend": "sse",
                "error": "sse_error",
                "detail": str(exc),
            }
        if not hits:
            return {
                "backend": "sse",
                "hit_count": 0,
                "hits": [],
            }
        return adapter.summarize_hits(hits)

    def _wrap_memory_response(
        self, text: str, *, controls_used: Optional[Dict[str, Any]] = None
    ) -> HubResponse:
        controls = {"mode": "memory_recall"}
        if controls_used:
            controls.update(controls_used)
        return HubResponse(
            text=text,
            model=None,
            trace_id=f"memory-{int(time.time() * 1000)}",
            latency_ms=0.0,
            controls_used=controls,
            safety={"rating": "G", "memory_recall": "true"},
        )

    def _build_emotion_vector(
        self, metrics: Mapping[str, float], affect: AffectSample
    ) -> EmotionVector:
        payload = dict(metrics)
        payload.setdefault("valence", float(getattr(affect, "valence", 0.0)))
        payload.setdefault("arousal", float(getattr(affect, "arousal", 0.0)))
        payload.setdefault("love", float(metrics.get("love", 0.0)))
        payload.setdefault("stress", float(metrics.get("stress", 0.0)))
        payload.setdefault("mask", float(metrics.get("mask", 0.0)))
        payload.setdefault(
            "heart_rate_norm", float(metrics.get("heart_rate_norm", self._heart_rate))
        )
        payload.setdefault(
            "breath_ratio_norm", float(metrics.get("breath_ratio_norm", 0.0))
        )
        payload["value_gradient"] = self._value_gradient_snapshot().to_dict()
        return EmotionVector.from_metrics(payload)

    def _value_gradient_snapshot(self) -> ValueGradient:
        base = self._default_value_gradient
        energy = getattr(self._self_model, "current_energy", None)
        if energy is None:
            return ValueGradient(**base.to_dict())
        energy = float(np.clip(energy, 0.0, 1.0))
        adjustment = 0.25
        return ValueGradient(
            survival_bias=base.survival_bias,
            physiological_bias=float(
                np.clip(base.physiological_bias + (0.5 - energy) * adjustment, 0.0, 1.0)
            ),
            social_bias=base.social_bias,
            exploration_bias=float(
                np.clip(base.exploration_bias + (energy - 0.5) * adjustment, 0.0, 1.0)
            ),
            attachment_bias=base.attachment_bias,
        )

    def _safety_lens(self, vg: ValueGradient) -> float:
        return float(
            np.clip(0.7 * vg.survival_bias + 0.3 * vg.physiological_bias, 0.0, 1.0)
        )

    def _empathy_lens(self, vg: ValueGradient) -> float:
        return float(np.clip(0.6 * vg.social_bias + 0.4 * vg.attachment_bias, 0.0, 1.0))

    def _exploration_lens(self, vg: ValueGradient) -> float:
        return float(np.clip(vg.exploration_bias, 0.0, 1.0))

    def _is_low_risk_context(
        self, prediction_error: float, gate_ctx: GateContext
    ) -> bool:
        if gate_ctx.force_listen:
            return False
        if prediction_error >= 0.4:
            return False
        return True

    def _current_impl_context(self, latency_ms: float) -> ImplementationContext:
        hardware = (
            os.getenv("EQNET_HARDWARE_PROFILE")
            or platform.node()
            or platform.system()
            or "unknown"
        )
        try:
            memory_load = float(os.getenv("EQNET_MEMORY_LOAD", 0.0))
        except (TypeError, ValueError):
            memory_load = 0.0
        sensor_snapshot = getattr(self._runtime_sensors, "snapshot", None)
        sensor_fidelity = 1.0 if sensor_snapshot is not None else 0.5
        return ImplementationContext(
            hardware_profile=str(hardware),
            latency_ms=float(latency_ms),
            memory_load=memory_load,
            sensor_fidelity=float(sensor_fidelity),
        )

    def _layer_force_score(
        self, layer: SelfLayer, vg: ValueGradient, matrix: Optional[ForceMatrix] = None
    ) -> float:
        fm = matrix or self._current_force_matrix()
        row = fm.row_for(layer)
        return (
            row.survival * vg.survival_bias
            + row.physiological * vg.physiological_bias
            + row.social * vg.social_bias
            + row.exploration * vg.exploration_bias
            + row.attachment * vg.attachment_bias
        )

    def _apply_context_damping(
        self,
        scores: Dict[SelfLayer, float],
        impl_ctx: Optional[ImplementationContext],
        context_tags: List[str],
    ) -> Dict[SelfLayer, float]:
        damping = getattr(self.config.conscious, "damping", None)
        if not damping or not damping.enabled:
            return dict(scores)
        adjusted: Dict[SelfLayer, float] = dict(scores)
        tags = {str(tag).lower() for tag in (context_tags or [])}
        latency = float(getattr(impl_ctx, "latency_ms", 0.0) or 0.0)
        excess = max(latency - damping.latency_L0_ms, 0.0)
        if excess > 0.0:
            cap = max(damping.latency_penalty_cap, 0.0)
            narr_penalty = damping.latency_k_narr * excess
            reflex_bonus = damping.latency_k_reflex * excess
            if cap > 0.0:
                narr_penalty = min(narr_penalty, cap)
                reflex_bonus = min(reflex_bonus, cap)
            adjusted[SelfLayer.NARRATIVE] = (
                adjusted.get(SelfLayer.NARRATIVE, 0.0) - narr_penalty
            )
            adjusted[SelfLayer.REFLEX] = (
                adjusted.get(SelfLayer.REFLEX, 0.0) + reflex_bonus
            )

        bump = float(damping.context_bump)
        if bump:

            def _hit(targets: Tuple[str, ...]) -> bool:
                return any(tag in tags for tag in (t.lower() for t in targets))

            if _hit(damping.reflex_tags):
                adjusted[SelfLayer.REFLEX] = adjusted.get(SelfLayer.REFLEX, 0.0) + bump
            if _hit(damping.affective_tags):
                adjusted[SelfLayer.AFFECTIVE] = (
                    adjusted.get(SelfLayer.AFFECTIVE, 0.0) + bump
                )
            if _hit(damping.narrative_tags):
                adjusted[SelfLayer.NARRATIVE] = (
                    adjusted.get(SelfLayer.NARRATIVE, 0.0) + bump
                )

        tag_bumps = getattr(damping, "tag_bumps", {}) or {}
        if tag_bumps:
            for tag in tags:
                config = tag_bumps.get(tag)
                if not isinstance(config, Mapping):
                    continue
                for layer_name, value in config.items():
                    layer = self._layer_from_name(layer_name)
                    if not layer:
                        continue
                    try:
                        bump_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    adjusted[layer] = adjusted.get(layer, 0.0) + bump_value

        if damping.score_floor is not None:
            for layer, value in adjusted.items():
                if value < damping.score_floor:
                    adjusted[layer] = damping.score_floor
        return adjusted

    @staticmethod
    def _force_snapshot_from_scores(
        scores: Mapping[SelfLayer, float],
    ) -> SelfForceSnapshot:
        return SelfForceSnapshot(
            reflex=float(scores.get(SelfLayer.REFLEX, 0.0)),
            affective=float(scores.get(SelfLayer.AFFECTIVE, 0.0)),
            narrative=float(scores.get(SelfLayer.NARRATIVE, 0.0)),
        )

    @staticmethod
    def _force_vector(snapshot: Optional[SelfForceSnapshot]) -> Optional[np.ndarray]:
        if snapshot is None:
            return None
        return np.array(
            [
                float(snapshot.reflex),
                float(snapshot.affective),
                float(snapshot.narrative),
            ],
            dtype=float,
        )

    @staticmethod
    def _layer_from_name(name: str) -> Optional[SelfLayer]:
        key = (name or "").strip().lower()
        if key in {"reflex", "reflex_self", SelfLayer.REFLEX.value}:
            return SelfLayer.REFLEX
        if key in {"affective", "affect", SelfLayer.AFFECTIVE.value}:
            return SelfLayer.AFFECTIVE
        if key in {"narrative", "story", SelfLayer.NARRATIVE.value}:
            return SelfLayer.NARRATIVE
        return None

    def _compute_boundary_signal(
        self, prediction_error: float
    ) -> Optional[BoundarySignal]:
        cfg = getattr(self.config.conscious, "boundary", None)
        current_damped = self._last_self_force
        current_raw = self._last_raw_force
        current_winner = self._last_dominant_layer
        if current_damped is None:
            self._prev_boundary_pred_error = prediction_error
            self._prev_boundary_damped = None
            self._prev_boundary_winner = current_winner
            self._last_boundary_signal = None
            return None
        if not cfg or not cfg.enabled:
            self._prev_boundary_pred_error = prediction_error
            self._prev_boundary_damped = current_damped
            self._prev_boundary_winner = current_winner
            self._last_boundary_signal = None
            return None

        sources: Dict[str, float] = {}
        components: List[Tuple[str, float, float]] = []
        if self._prev_boundary_pred_error is not None:
            delta = abs(prediction_error - self._prev_boundary_pred_error)
            norm = max(cfg.prediction_error_norm, 1e-6)
            components.append(
                (
                    "prediction_error_delta",
                    float(np.clip(delta / norm, 0.0, 1.0)),
                    cfg.weight_prediction_error,
                )
            )
        prev_vec = self._force_vector(self._prev_boundary_damped)
        curr_vec = self._force_vector(current_damped)
        if prev_vec is not None and curr_vec is not None:
            delta = float(np.linalg.norm(curr_vec - prev_vec))
            norm = max(cfg.force_delta_norm, 1e-6)
            components.append(
                (
                    "force_field_delta",
                    float(np.clip(delta / norm, 0.0, 1.0)),
                    cfg.weight_force_delta,
                )
            )
        if self._prev_boundary_winner is not None and current_winner is not None:
            flip = 1.0 if current_winner != self._prev_boundary_winner else 0.0
            components.append(("winner_flip", flip, cfg.weight_winner_flip))
        raw_vec = self._force_vector(current_raw)
        if raw_vec is not None and curr_vec is not None:
            gap = float(np.linalg.norm(curr_vec - raw_vec))
            norm = max(cfg.raw_gap_norm, 1e-6)
            components.append(
                (
                    "raw_damped_gap",
                    float(np.clip(gap / norm, 0.0, 1.0)),
                    cfg.weight_raw_gap,
                )
            )

        numerator = 0.0
        denom = 0.0
        for name, value, weight in components:
            if weight <= 0.0:
                continue
            sources[name] = value
            numerator += value * weight
            denom += weight
        score = float(np.clip(numerator / denom, 0.0, 1.0)) if denom > 0.0 else 0.0
        signal = BoundarySignal(score=score, sources=sources)
        self._prev_boundary_pred_error = prediction_error
        self._prev_boundary_damped = current_damped
        self._prev_boundary_winner = current_winner
        self._last_boundary_signal = signal
        return signal

    def _choose_self_layer(
        self,
        emotion: EmotionVector,
        impl_ctx: Optional[ImplementationContext],
        context_tags: Optional[List[str]] = None,
    ) -> SelfLayer:
        vg = getattr(emotion, "value_gradient", None) or self._value_gradient_snapshot()
        fm = self._current_force_matrix()
        base_scores: Dict[SelfLayer, float] = {}
        for layer in (SelfLayer.REFLEX, SelfLayer.AFFECTIVE, SelfLayer.NARRATIVE):
            base_scores[layer] = self._layer_force_score(layer, vg, fm)
        raw_snapshot = self._force_snapshot_from_scores(base_scores).with_winner()
        self._last_raw_force = raw_snapshot
        adjusted = self._apply_context_damping(
            base_scores, impl_ctx, context_tags or []
        )
        self._last_self_force = self._force_snapshot_from_scores(adjusted).with_winner()
        return max(adjusted.items(), key=lambda kv: kv[1])[0]

    def _compute_prediction_error(
        self, delta_m: float, metrics: Mapping[str, float]
    ) -> float:
        novelty = float(metrics.get("novelty", 0.0) or 0.0)
        body_flag = float(metrics.get("body_surprise", 0.0) or 0.0)
        return float(np.clip(max(delta_m, novelty, body_flag), 0.0, 2.0))

    def _decide_response_route(
        self,
        qualia: EmotionVector,
        prediction_error: float,
        text_input: bool,
        gate_ctx: GateContext,
    ) -> ResponseRoute:
        def _finish(choice: ResponseRoute) -> ResponseRoute:
            impl_ctx = self._last_impl_context or self._current_impl_context(0.0)
            tags: List[str] = []
            if text_input:
                tags.append("text_input")
            if gate_ctx.force_listen:
                tags.append("force_listen")
            self._last_dominant_layer = self._choose_self_layer(qualia, impl_ctx, tags)
            return choice

        if not self.config.conscious.enabled:
            return _finish(ResponseRoute.CONSCIOUS)
        vg = getattr(qualia, "value_gradient", None) or self._value_gradient_snapshot()
        safety_bias = self._safety_lens(vg)
        exploration_bias = self._exploration_lens(vg)
        empathy_bias = self._empathy_lens(vg)
        cfg = self.config.conscious
        if (
            safety_bias >= 0.8
            and prediction_error >= cfg.reflex.prediction_error_threshold
        ):
            return _finish(ResponseRoute.REFLEX)
        if self._should_use_reflex(qualia, prediction_error):
            return _finish(ResponseRoute.REFLEX)
        if self._should_use_habit(text_input, gate_ctx):
            return _finish(ResponseRoute.HABIT)
        if exploration_bias >= 0.7 and self._is_low_risk_context(
            prediction_error, gate_ctx
        ):
            return _finish(ResponseRoute.CONSCIOUS)
        if empathy_bias >= 0.65 and text_input:
            return _finish(ResponseRoute.CONSCIOUS)
        return _finish(ResponseRoute.CONSCIOUS)

    def _should_use_reflex(
        self, qualia: EmotionVector, prediction_error: float
    ) -> bool:
        cfg = self.config.conscious.reflex
        if prediction_error >= cfg.prediction_error_threshold:
            return True
        if qualia.stress >= cfg.stress_threshold:
            return True
        return False

    def _should_use_habit(self, text_input: bool, gate_ctx: GateContext) -> bool:
        if not text_input:
            return False
        if self._has_prioritized_inner_os_surface():
            return False
        low_motion = gate_ctx.delta_m < 0.05 and gate_ctx.jerk < 0.05
        low_voice = gate_ctx.voice_energy < 0.12
        return low_motion and low_voice

    def _has_prioritized_inner_os_surface(self) -> bool:
        expression_hint = self._last_gate_context.get("inner_os_expression_hint")
        if not isinstance(expression_hint, Mapping):
            return False

        turn_delta = expression_hint.get("turn_delta")
        if isinstance(turn_delta, Mapping):
            kind = str(turn_delta.get("kind") or "").strip().lower()
            preferred_act = str(turn_delta.get("preferred_act") or "").strip().lower()
            if kind in {
                "green_reflection_hold",
                "reopening_thread",
                "continuity_thread",
                "lingering_thread",
            }:
                return True
            if preferred_act in {
                "reflect_hidden_need",
                "reopen_from_anchor",
                "reopen_from_anchor_soft",
                "keep_shared_thread_visible",
                "stay_with_present_need",
            }:
                return True

        contact_reflection_state = expression_hint.get("contact_reflection_state")
        if isinstance(contact_reflection_state, Mapping):
            reflection_style = str(
                contact_reflection_state.get("reflection_style") or ""
            ).strip().lower()
            if reflection_style in {"reflect_only", "boundary_only"}:
                return True

        planned_content_sequence = expression_hint.get("planned_content_sequence")
        if isinstance(planned_content_sequence, list):
            prioritized_acts = {
                "reflect_hidden_need",
                "reflect_self_blame",
                "reflect_fear_of_being_seen",
                "reflect_unspoken_weight",
                "reopen_from_anchor",
                "reopen_from_anchor_soft",
                "keep_shared_thread_visible",
                "leave_return_point_from_anchor",
            }
            for item in planned_content_sequence:
                if not isinstance(item, Mapping):
                    continue
                act = str(item.get("act") or "").strip().lower()
                if act in prioritized_acts:
                    return True
        return False

    def _reflex_prompt(self, prediction_error: float) -> str:
        locale = self._response_locale()
        if prediction_error > 1.0:
            return (
                lookup_text(locale, "inner_os.runtime_route_prompts.reflex_alert")
                or "少し想定外の反応があります。いま安全かだけ先に確認したいです。"
            )
        return (
            lookup_text(locale, "inner_os.runtime_route_prompts.reflex_shift")
            or "少し違和感が強いので、いったん安全寄りで見ます。"
        )

    def _habit_prompt(self, user_text: Optional[str]) -> str:
        locale = self._response_locale()
        if not user_text:
            return (
                lookup_text(locale, "inner_os.runtime_route_prompts.habit_without_snippet")
                or "ここはひとまず、手短に返します。"
            )
        return (
            lookup_text(locale, "inner_os.runtime_route_prompts.habit_with_snippet")
            or "ひとまずそこだけ、短く返します。"
        )

    def _build_context_payload(
        self,
        user_text: Optional[str],
        context_text: Optional[str],
        metrics: Mapping[str, float],
        gate_ctx: GateContext,
        route: ResponseRoute,
    ) -> Dict[str, Any]:
        summary = (context_text or user_text or "").strip()
        if not summary:
            summary = f"talk_mode={self._talk_mode.value}"
        tags = [f"talk:{self._talk_mode.value}", f"route:{route.value}"]
        seed = self._last_gate_context.get("inner_os_working_memory_seed")
        if isinstance(seed, Mapping):
            focus = str(seed.get("semantic_seed_focus") or "").strip().lower().replace(" ", "_")
            anchor = str(seed.get("semantic_seed_anchor") or "").strip().lower().replace(" ", "_")
            theme_focus = str(seed.get("long_term_theme_focus") or "").strip().lower().replace(" ", "_")
            theme_anchor = str(seed.get("long_term_theme_anchor") or "").strip().lower().replace(" ", "_")
            theme_kind = str(seed.get("long_term_theme_kind") or "").strip().lower().replace(" ", "_")
            theme_summary = str(seed.get("long_term_theme_summary") or "").strip().lower().replace(" ", "_")
            if focus:
                tags.append(f"wm_seed_focus:{focus[:48]}")
            if anchor:
                tags.append(f"wm_seed_anchor:{anchor[:48]}")
            if theme_focus:
                tags.append(f"ltm_theme_focus:{theme_focus[:48]}")
            if theme_anchor:
                tags.append(f"ltm_theme_anchor:{theme_anchor[:64]}")
            if theme_kind:
                tags.append(f"ltm_theme_kind:{theme_kind[:32]}")
            if theme_summary:
                tags.append(f"ltm_theme_summary:{theme_summary[:96]}")
        if gate_ctx.force_listen:
            tags.append("force_listen")
        if gate_ctx.text_input:
            tags.append("text_input")
        salient_entities = metrics.get("salient_entities")
        if not isinstance(salient_entities, list):
            salient_entities = []
        flags = ["mark_as_conscious"] if route == ResponseRoute.CONSCIOUS else []
        return {
            "world_summary": summary,
            "salient_entities": list(salient_entities),
            "context_tags": tags,
            "flags": flags,
        }

    def _update_self_model_state(self, qualia: EmotionVector) -> None:
        self._self_model.current_mode = self._talk_mode
        energy = float(np.clip(1.0 - 0.3 * max(qualia.stress, 0.0), 0.1, 1.0))
        self._self_model.current_energy = energy
        attachment = 0.9 * self._self_model.attachment_to_user + 0.1 * max(
            qualia.love, 0.0
        )
        self._self_model.attachment_to_user = float(np.clip(attachment, 0.0, 1.0))

    def _maybe_store_conscious_episode(
        self,
        qualia: EmotionVector,
        prediction_error: float,
        narrative: Optional[str],
        route: ResponseRoute,
        context: Mapping[str, Any],
        implementation: Optional[ImplementationContext],
        *,
        boundary_signal: Optional[BoundarySignal] = None,
    ) -> Optional[ConsciousEpisode]:
        if not self.config.conscious.enabled or self._conscious_memory is None:
            return None
        if route == ResponseRoute.REFLEX:
            return None
        if not self._meets_conscious_threshold(qualia, prediction_error, context):
            return None
        impl_ctx = implementation or self._current_impl_context(0.0)
        episode = ConsciousEpisode(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            self_state=self._self_model.snapshot(),
            world_state=WorldStateSnapshot.from_context(
                context, prediction_error=prediction_error
            ),
            qualia=qualia,
            narrative=narrative,
            route=route,
            value_gradient=qualia.value_gradient,
            dominant_self_layer=self._last_dominant_layer,
            implementation=impl_ctx,
            self_force=self._last_self_force,
            raw_self_force=self._last_raw_force,
            boundary_signal=boundary_signal,
        )
        self._conscious_memory.add_conscious_episode(episode)
        if self._diary_writer is not None:
            self._diary_writer.write_conscious_episode(episode)
        return episode

    def _meets_conscious_threshold(
        self, qualia: EmotionVector, prediction_error: float, context: Mapping[str, Any]
    ) -> bool:
        cfg = self.config.conscious.conscious_threshold
        if prediction_error >= cfg.prediction_error_min:
            return True
        if abs(qualia.valence) >= cfg.valence_min:
            return True
        if qualia.love >= cfg.love_min:
            return True
        if qualia.stress >= cfg.stress_min:
            return True
        flags = context.get("flags") if isinstance(context, Mapping) else []
        if isinstance(flags, list) and "mark_as_conscious" in flags:
            return True
        return False

    def _log_memory_reference(
        self, result: Optional[Dict[str, Any]], user_text: str
    ) -> None:
        if not result or not self._memory_ref_log_path:
            return
        meta = result.get("meta") or {}
        record: Dict[str, Any] = {
            "ts": time.time(),
            "mode": meta.get("mode", "recall"),
            "fidelity": result.get("fidelity"),
            "reply_len": len((result.get("reply") or "")),
            "anchor": meta.get("anchor"),
            "disclaimer_source": meta.get("disclaimer_source"),
            "recall_render_mode": meta.get("recall_render_mode"),
            "cue_label": meta.get("cue_label"),
            "memory_kind": meta.get("memory_kind"),
            "record_kind": meta.get("record_kind"),
            "record_provenance": meta.get("record_provenance"),
            "source_class": meta.get("source_class"),
            "audit_event": meta.get("audit_event"),
            "evidence_keys": meta.get("evidence_keys"),
        }
        rarity = (result.get("meta") or {}).get("rarity_budget") or {}
        if isinstance(rarity, dict):
            record["rarity_budget"] = {
                "day_key": rarity.get("day_key"),
                "week_key": rarity.get("week_key"),
                "daily_limit": rarity.get("daily_limit"),
                "weekly_limit": rarity.get("weekly_limit"),
                "day_used": rarity.get("day_used"),
                "week_used": rarity.get("week_used"),
                "suppressed": rarity.get("suppressed"),
                "reason": rarity.get("reason"),
            }
        candidate = result.get("candidate") or {}
        if candidate:
            record["node"] = candidate.get("node")
            record["score"] = candidate.get("score")
            record["affective"] = candidate.get("affective")
        topic_hash = hashlib.sha1(
            user_text.strip().lower().encode("utf-8")
        ).hexdigest()[:12]
        record["topic_hash"] = topic_hash
        record["turn_id"] = self._turn_id
        record["memory_ref_id"] = f"memory-ref-{int(record['ts'] * 1000)}-{topic_hash}"
        try:
            with self._memory_ref_log_path.open("a", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")
        except Exception:
            pass
        audit_event = meta.get("audit_event")
        if (
            self._think_log_path is not None
            and isinstance(audit_event, str)
            and audit_event in {"SOURCE_FUZZY", "DOUBLE_TAKE"}
        ):
            think_event = {
                "ts": record["ts"],
                "turn_id": self._turn_id,
                "memory_ref_id": record["memory_ref_id"],
                "memory_kind": meta.get("memory_kind", "unknown"),
                "source_class": meta.get("source_class", "uncertain"),
                "audit_event": audit_event,
                "evidence_keys": meta.get("evidence_keys") or [],
                "trace_id": record["memory_ref_id"],
            }
            try:
                with self._think_log_path.open("a", encoding="utf-8") as handle:
                    json.dump(think_event, handle, ensure_ascii=False)
                    handle.write("\n")
            except Exception:
                pass








































def _apply_inner_os_expression_controls(controls: Mapping[str, Any], expression_hint: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(controls)
    if not isinstance(expression_hint, Mapping):
        return out
    tentative_bias = max(0.0, min(1.0, float(expression_hint.get("tentative_bias", 0.0) or 0.0)))
    assertiveness_cap = max(0.2, min(1.0, float(expression_hint.get("assertiveness_cap", 1.0) or 1.0)))
    question_bias = max(0.0, min(1.0, float(expression_hint.get("question_bias", 0.0) or 0.0)))
    directness = float(out.get("directness", 0.0) or 0.0)
    temperature = float(out.get("temperature", 0.0) or 0.0)
    top_p = float(out.get("top_p", 0.0) or 0.0)
    out["directness"] = max(-0.6, min(directness * assertiveness_cap - tentative_bias * 0.08 - question_bias * 0.04, 0.6))
    out["temperature"] = max(0.2, min(temperature - tentative_bias * 0.08, 1.0))
    out["top_p"] = max(0.5, min(top_p - tentative_bias * 0.05, 1.0))
    out["inner_os_tentative_bias"] = round(tentative_bias, 4)
    out["inner_os_question_bias"] = round(question_bias, 4)
    out["inner_os_assertiveness_cap"] = round(assertiveness_cap, 4)
    return out

def _inner_os_expression_context_line(expression_hint: Mapping[str, Any]) -> str:
    if not isinstance(expression_hint, Mapping):
        return ""
    if bool(expression_hint.get("clarify_first", False)):
        return "[inner-os] Context is still settling. Prefer grounded observations, avoid definitive interpretations, and ask one brief clarifying question before deeper interpretation."
    return "[inner-os] Context is still settling. Prefer grounded observations, avoid definitive interpretations, and keep wording tentative."
