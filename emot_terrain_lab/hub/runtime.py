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
from emot_terrain_lab.i18n.locale import lookup_text, truncate_text
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
from emot_terrain_lab.memory.recall_policy import (
    RarityBudgetState,
    apply_rarity_budget,
    render_recall_cue,
)
from emot_terrain_lab.perception.text_affect import quick_text_affect_v2

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
    eqnet_state_dir: str = "data/state_hub"
    robot: RobotBridgeConfig = field(default_factory=RobotBridgeConfig)
    mask_layer: MaskLayerConfig = field(default_factory=MaskLayerConfig)
    moment_log_path: Optional[str] = "logs/moment_log.jsonl"
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

    def process_turn(
        self,
        user_text: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        fast_only: bool = False,
        image_path: Optional[str] = None,
        transferred_lessons: Optional[list[Mapping[str, Any]]] = None,
    ) -> RuntimeTurnResult:
        sensor_snapshot = getattr(self._runtime_sensors, "snapshot", None)
        sensor_metrics = (
            getattr(sensor_snapshot, "metrics", {}) if sensor_snapshot is not None else {}
        )
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
        current_state["route"] = str(result.response_route or pre_hook.state.route)
        current_state["talk_mode"] = str(result.talk_mode or pre_hook.state.talk_mode)
        response_hook = self._integration_hooks.response_gate(
            draft={"text": getattr(result.response, "text", None)},
            current_state=current_state,
            safety_signals={"safety_bias": safety_bias},
        )
        if result.response is not None:
            result.response = self._apply_inner_os_surface_policy(
                result.response,
                response_hook.expression_hints,
                response_hook.conscious_access,
            )
            merged_retrieval = dict(result.response.retrieval_summary or {})
            merged_retrieval.setdefault("inner_os", {})
            merged_retrieval["inner_os"].update(recall_hook.ignition_hints)
            result.response.retrieval_summary = merged_retrieval
            merged_controls = dict(result.response.controls or {})
            inner_os_controls = response_hook.to_dict()
            inner_os_controls["surface_policy_level"] = self._inner_os_surface_policy_level(getattr(result.response, "controls_used", {}) or {})
            inner_os_controls["surface_policy_intent"] = self._inner_os_surface_policy_intent(getattr(result.response, "controls_used", {}) or {})
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
        surface_policy_level = self._inner_os_surface_policy_level(getattr(result.response, "controls_used", {}) if result.response is not None else {})
        surface_policy_intent = self._inner_os_surface_policy_intent(getattr(result.response, "controls_used", {}) if result.response is not None else {}) or ""
        result.metrics.setdefault("inner_os/surface_policy_active", 0.0 if surface_policy_level == "none" else 1.0)
        result.metrics.setdefault("inner_os/surface_policy_layered", 1.0 if surface_policy_level == "layered" else 0.0)
        result.metrics.setdefault("inner_os/surface_policy_intent_clarify", 1.0 if surface_policy_intent == "clarify" else 0.0)
        result.metrics.setdefault("inner_os/surface_policy_intent_check_in", 1.0 if surface_policy_intent == "check_in" else 0.0)
        current_state["surface_policy_active"] = 0.0 if surface_policy_level == "none" else 1.0
        current_state["surface_policy_level"] = surface_policy_level
        current_state["surface_policy_intent"] = surface_policy_intent or None
        memory_write_candidates = collect_runtime_memory_candidates(
            recall_payload=recall_hook.recall_payload,
            memory_reference=result.memory_reference,
            vision_entry=getattr(self, "_last_observed_vision_entry", None),
            relational_context=current_state,
        )
        post_hook = self._integration_hooks.post_turn_update(
            user_input={"text": user_text or ""},
            output={"reply_text": getattr(result.response, "text", None)},
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
                "reuse_trajectory": round(float(post_hook.state.reuse_trajectory), 4),
                "interference_pressure": round(float(post_hook.state.interference_pressure), 4),
                "consolidation_priority": round(float(post_hook.state.consolidation_priority), 4),
                "prospective_memory_pull": round(float(post_hook.state.prospective_memory_pull), 4),
            }
        )
        return result

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

        route_response: Optional[str] = None
        if route == ResponseRoute.REFLEX:
            route_response = self._reflex_prompt(prediction_error)
        elif route == ResponseRoute.HABIT and text_input:
            route_response = self._habit_prompt(user_text)

        response: Optional[HubResponse] = None
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
        should_call_llm = (
            text_input
            and self._talk_mode == TalkMode.TALK
            and route == ResponseRoute.CONSCIOUS
            and not route_response
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
                response = self.llm.generate(
                    user_text=masked_user_text,
                    context=masked_context,
                    controls=llm_controls,
                    intent=intent or self.llm.config.default_intent,
                    slos={"p95_ms": 180.0},
                    image_path=None,
                )
                response = self._attach_visual_reflection(response, perception_summary)
        elif response is None and ack_text:
            response = self._wrap_ack_response(ack_text, self._talk_mode)
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
            minimal = ack_text or ack_for_fast
            if not minimal and self._talk_mode == TalkMode.PRESENCE:
                minimal = self._render_presence_ack(gate_ctx)
            if not minimal:
                minimal = self._render_ack_for_mode(TalkMode.WATCH, gate_ctx)
            if minimal:
                response = self._wrap_ack_response(minimal, self._talk_mode)
            narrative_text = None
            payload["suppress_narrative"] = True
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
            for key, val in response.controls_used.items()
            if isinstance(val, (int, float, np.floating))
        }
        return {
            "model": response.model,
            "trace_id": response.trace_id,
            "latency_ms": float(response.latency_ms),
            "controls_used": controls_used,
            "safety": dict(response.safety),
            "confidence": float(response.confidence),
            "uncertainty_reason": list(response.uncertainty_reason),
            "perception_summary": dict(response.perception_summary) if isinstance(response.perception_summary, dict) else None,
            "retrieval_summary": dict(response.retrieval_summary) if isinstance(response.retrieval_summary, dict) else None,
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

    def _wrap_ack_response(self, text: str, mode: TalkMode) -> HubResponse:
        if mode == TalkMode.PRESENCE:
            self._last_presence_ack_ts = time.time()
        return HubResponse(
            text=text,
            model=None,
            trace_id=f"ack-{int(time.time() * 1000)}",
            latency_ms=0.0,
            controls_used={"mode": mode.name.lower()},
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
        if intent == "check_in":
            prefix = "Let me check in gently before I go further."
        elif intent == "clarify":
            prefix = "Let me check one small thing before I go further."
        else:
            prefix = "Let me pause for one brief check before I go further."
        question_bias = max(0.0, min(1.0, float(expression_hint.get("question_bias", 0.0) or 0.0)))
        closing = self._inner_os_surface_closing(intent=intent, question_bias=question_bias)
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

    def _inner_os_surface_closing(self, *, intent: str = "", question_bias: float = 0.0) -> str:
        if question_bias < INNER_OS_SURFACE_THRESHOLDS["closing_question_bias"]:
            return ""
        if intent == "check_in":
            return "I want to stay with this gently first."
        if intent == "clarify":
            return "Then I can answer a little more cleanly."
        return "Then I can continue a bit more carefully."

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
        low_motion = gate_ctx.delta_m < 0.05 and gate_ctx.jerk < 0.05
        low_voice = gate_ctx.voice_energy < 0.12
        return low_motion and low_voice

    def _reflex_prompt(self, prediction_error: float) -> str:
        if prediction_error > 1.0:
            return (
                "Hold on - I'm flagging something unexpected. Are you safe right now?"
            )
        return "Give me a second; something feels off so I'm switching to safety mode."

    def _habit_prompt(self, user_text: Optional[str]) -> str:
        if not user_text:
            return "I'll handle that routine step real quick."
        snippet = user_text.strip().splitlines()[0]
        if len(snippet) > 48:
            snippet = snippet[:45].rstrip() + "..."
        return f"Let me answer that quickly: {snippet}"

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
