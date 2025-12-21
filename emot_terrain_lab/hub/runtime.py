# -*- coding: utf-8 -*-
"""
End-to-end runtime scaffold that ties perception, EQNet metrics, policy
controls, and the LLM hub together.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, List, Mapping
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
from eqnet.qualia_model import (
    FutureReplayConfig,
    blend_emotion_axes,
    compute_future_risk,
    compute_future_hopefulness,
    sensor_to_emotion_axes,
)
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
from eqnet_core.qualia import AccessGate, AccessGateConfig, MetaMonitor
from .perception import PerceptionBridge, PerceptionConfig, AffectSample
from .policy import PolicyHead, PolicyConfig, AffectControls
from .llm_hub import LLMHub, LLMHubConfig, HubResponse
from .robot_bridge import RobotBridgeConfig, ROS2Bridge
from datetime import datetime
from emot_terrain_lab.terrain.system import EmotionalMemorySystem
from emot_terrain_lab.memory.reference_helper import handle_memory_reference
from emot_terrain_lab.perception.text_affect import quick_text_affect_v2

try:
    from runtime.config import load_runtime_cfg
except ImportError:
    load_runtime_cfg = None

LOGGER = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
LOGGER.setLevel(logging.DEBUG)


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


_ACK_TEMPLATES: Dict[TalkMode, str] = {
    TalkMode.WATCH: "I'm right here listening quietly; just wave if you need me.",
    TalkMode.SOOTHE: "Let's take a slow breath together. I'm here and you're safe.",
    TalkMode.ASK: "Something feels different; want to tell me what's on your mind?",
    TalkMode.TALK: "Okay, let's pick up the thread. Where should we start?",
}


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
        self._qualia_gate = AccessGate(AccessGateConfig())
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
        self.perceived_affect: Dict[str, float] = {
            "valence": 0.0,
            "arousal": 0.0,
            "confidence": 0.0,
        }
        self._pending_text_affect: Optional[AffectSample] = None
        self._memory_ref_cooldown_until = 0.0
        if self._memory_ref_cfg and hasattr(self._memory_ref_cfg, "log_path"):
            try:
                self._memory_ref_log_path = Path(self._memory_ref_cfg.log_path)
                self._memory_ref_log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                self._memory_ref_log_path = None

        self._fast_ack_state = FastAckState()
        self._arousal_tracker = ArousalTracker()
        self._last_persona_meta: Dict[str, Any] = {}
        self._last_fast_ack_sample: Optional[Dict[str, Any]] = None
        self._heart_rate = 0.85
        self._heart_phase = 0.0
        self._heart_last_ts = time.time()
        self._heart_base_rate = 0.85
        self._heart_gain = 0.45
        self._moment_log_writer = MomentLogWriter(self.config.moment_log_path)
        self._turn_id = 0
        self._session_id = None
        runtime_session = (
            getattr(self._runtime_cfg, "session", None) if self._runtime_cfg else None
        )
        if runtime_session is not None:
            self._session_id = getattr(runtime_session, "session_id", None)
        self._culture_recurrence: Dict[Tuple[str, str, str, str], int] = defaultdict(
            int
        )
        self._default_persona_meta = self._build_default_persona_meta()

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

        if base_rate is not None:
            try:
                self._heart_base_rate = float(base_rate)
            except (TypeError, ValueError):
                pass
        if gain is not None:
            try:
                self._heart_gain = float(gain)
            except (TypeError, ValueError):
                pass

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
            Intent label for the hub router (qa/chitchat/code遯ｶ・ｽE・ｽ).
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
        self._last_gate_context = {
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
        }
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
        elif self._talk_mode == TalkMode.WATCH:
            controls.gesture_amplitude = float(min(controls.gesture_amplitude, 0.25))
        if shadow_estimate is not None:
            self._apply_shadow_controls(controls, shadow_estimate)

        self._last_controls = controls

        ack_for_fast: Optional[str] = None
        if fast_only and text_input:
            ack_for_fast = self._sample_fast_ack_text(affect, gate_ctx)

        ack_text: Optional[str] = None
        if self._talk_mode in (TalkMode.SOOTHE, TalkMode.ASK):
            ack_text = _ACK_TEMPLATES.get(self._talk_mode)
        elif text_input and self._talk_mode == TalkMode.WATCH:
            ack_text = _ACK_TEMPLATES.get(TalkMode.WATCH)

        route_response: Optional[str] = None
        if route == ResponseRoute.REFLEX:
            route_response = self._reflex_prompt(prediction_error)
        elif route == ResponseRoute.HABIT and text_input:
            route_response = self._habit_prompt(user_text)

        response: Optional[HubResponse] = None
        memory_reference: Optional[Dict[str, Any]] = None
        persona_meta: Dict[str, Any] = {}

        if fast_only:
            ack_payload = ack_for_fast or route_response or ack_text
            if ack_payload:
                response = self._wrap_ack_response(ack_payload, self._talk_mode)
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
            }

        if route_response and response is None:
            response = self._wrap_ack_response(route_response, self._talk_mode)

        behavior_mod: Optional[BehaviorMod] = None
        should_call_llm = (
            text_input
            and self._talk_mode == TalkMode.TALK
            and route == ResponseRoute.CONSCIOUS
            and not route_response
        )
        if should_call_llm:
            memory_reference = self._maybe_memory_reference(user_text or "")
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
                llm_controls = {
                    "temperature": controls.temperature,
                    "top_p": controls.top_p,
                    "pause_ms": controls.pause_ms,
                    "directness": controls.directness,
                    "warmth": controls.warmth,
                    "prosody_energy": controls.prosody_energy,
                    "spoiler_mode": "warn",
                }
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
                )
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

        now = time.time()
        dt = 0.0
        if self._heart_last_ts is not None:
            dt = max(now - self._heart_last_ts, 0.0)
        self._heart_last_ts = now
        arousal_unit = float(np.clip(arousal, 0.0, 1.0))
        rate = float(
            np.clip(self._heart_base_rate + self._heart_gain * arousal_unit, 0.2, 3.0)
        )
        self._heart_rate = rate
        phase = float((self._heart_phase + rate * min(dt, 2.0)) % 1.0)
        self._heart_phase = phase
        noise_val = (
            noise_scale
            if noise_scale is not None
            else max(0.02, 0.01 + 0.02 * abs(arousal))
        )
        phi_vec = self._emotion_core.step(
            self._heart_base_rate,
            self._heart_gain,
            noise_scale=noise_val,
            damp=float(max(damp, 0.0)),
        )
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
        )
        payload.update(gate_result)
        payload["m_kind"] = "cosine"
        allow = bool(gate_result.get("allow", True))
        metrics["qualia/u_t"] = u_t
        metrics["qualia/m_t"] = m_t
        metrics["qualia/load"] = load_t
        metrics["qualia/gate_allow"] = 1.0 if allow else 0.0
        if not allow:
            minimal = ack_text or ack_for_fast or _ACK_TEMPLATES.get(TalkMode.WATCH, "...")
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
            "polite": "声の端を少し丁寧にして、言い回しも和らげてください。",
            "casual": "肩の力を抜いた柔らかい語尾で、距離を縮めるように話してください。",
            "neutral": "標準的な語尾で、落ち着いたトーンを保ってください。",
        }
        tone_line = tone_hints.get(behavior.tone, tone_hints["neutral"])

        empathy_line = (
            "気持ちの背景を一度言葉にすると安心します。"
            if behavior.empathy_level >= 0.65
            else "共感は一言添える程度で十分です。"
        )

        joke_line = (
            "軽いユーモアを1行だけ差し込んでも大丈夫です。"
            if behavior.joke_ratio >= 0.4
            else "冗談は控えめにして、落ち着いた語り口にしてください。"
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
        emotion_tag = self._infer_emotion_tag(affect)
        qualia_vec = self._qualia_vec_snapshot()
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
            gate_context=dict(self._last_gate_context),
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
            self._turn_id += 1
        except Exception:
            pass

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
            return "..."
        return _ACK_TEMPLATES.get(self._talk_mode, _ACK_TEMPLATES[TalkMode.WATCH])

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
        if (
            cfg is None
            or not getattr(cfg, "enabled", True)
            or self.eqnet_system is None
            or not user_text.strip()
            or time.time() < self._memory_ref_cooldown_until
        ):
            return None
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

    def _wrap_ack_response(self, text: str, mode: TalkMode) -> HubResponse:
        return HubResponse(
            text=text,
            model=None,
            trace_id=f"ack-{int(time.time() * 1000)}",
            latency_ms=0.0,
            controls_used={"mode": mode.name.lower()},
            safety={"rating": "G", "ack": "true"},
        )

    def _wrap_memory_response(self, text: str) -> HubResponse:
        return HubResponse(
            text=text,
            model=None,
            trace_id=f"memory-{int(time.time() * 1000)}",
            latency_ms=0.0,
            controls_used={"mode": "memory_recall"},
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
        topic_hash = hashlib.sha1(
            user_text.strip().lower().encode("utf-8")
        ).hexdigest()[:12]
        record["topic_hash"] = topic_hash
        try:
            with self._memory_ref_log_path.open("a", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")
        except Exception:
            pass
