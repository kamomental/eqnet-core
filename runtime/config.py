from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class IgnitionCfg:
    theta_on: float = field(default=0.62)
    theta_off: float = field(default=0.48)
    dwell_steps: int = field(default=8)


@dataclass
class TelemetryCfg:
    log_path: str = field(default="telemetry/ignition-%Y%m%d.jsonl")


@dataclass
class LatencyCfg:
    tight_max_ms: float = field(default=500.0)
    loose_max_ms: float = field(default=2000.0)
    enable_loose: bool = field(default=True)


@dataclass
class LLMCfg:
    model: str = field(default="gpt-oss20b(A3B)")
    stream: bool = field(default=True)
    max_tokens: int = field(default=96)
    temperature: float = field(default=0.35)


@dataclass
class ReplayCfg:
    min_interval_ms: int = field(default=10)
    sample_every: int = field(default=1)
    horizon: int = field(default=3)
    alpha: float = field(default=0.6)
    beta: float = field(default=0.8)
    gamma: float = field(default=0.1)
    future_ttl_sec: int = field(default=90)
    future_energy_threshold: float = field(default=1.2)


@dataclass
class PolicyRuntimeCfg:
    entropy_coef: float = field(default=0.03)
    temperature_base: float = field(default=0.9)
    temperature_k_by_entropy: float = field(default=0.2)


@dataclass
class GuardsCfg:
    talk_block_on_guard: bool = field(default=True)


@dataclass
class QualiaAccessGateCfg:
    alpha: float = field(default=1.0)
    beta: float = field(default=1.0)
    gamma: float = field(default=1.0)
    theta: float = field(default=1.0)
    ema: float = field(default=0.9)
    open_tau: float = field(default=0.6)
    close_tau: float = field(default=0.4)
    theta_lr: float = field(default=0.1)
    target_access: float = field(default=0.2)
    boundary_enabled: bool = field(default=False)
    boundary_weight: float = field(default=0.0)
    boundary_threshold: float = field(default=0.7)
    boundary_curve: str = field(default="linear")


@dataclass
class EmotionCfg:
    valence_w_rho: float = field(default=1.0)
    valence_w_s: float = field(default=1.0)
    resonance_k: float = field(default=0.05)
    affective_log_path: str = field(default="memory/affective_log.jsonl")


@dataclass
class AssocKernelCfg:
    sigma_init: float = field(default=1.2)
    beta_local_init: float = field(default=-3.0)
    beta_global_init: float = field(default=-3.0)
    window_w: int = field(default=128)
    stride_s: int = field(default=8)
    use_kv_asymmetry: bool = field(default=False)
    use_bandwidth_gating: bool = field(default=False)


@dataclass
class ModelCfg:
    use_assoc_kernel: bool = field(default=False)
    assoc_kernel: AssocKernelCfg = field(default_factory=AssocKernelCfg)


@dataclass
class CultureCfg:
    tag: str = field(default="default")
    politeness: float = field(default=0.5)
    intimacy: float = field(default=0.5)
    place_id: Optional[str] = field(default=None)
    partner_id: Optional[str] = field(default=None)
    object_id: Optional[str] = field(default=None)
    object_role: Optional[str] = field(default=None)
    activity_tag: Optional[str] = field(default=None)
    feedback: "CultureFeedbackCfg" = field(default_factory=lambda: CultureFeedbackCfg())


@dataclass
class AlertsCfg:
    max_abs_valence_mean: float = field(default=0.6)
    min_corr_rho_I: float = field(default=0.2)
    min_corr_arousal_I: float = field(default=0.1)
    min_corr_rho_rho: float = field(default=0.2)
    max_allowed_lag: float = field(default=8.0)
    min_resonance_samples: int = field(default=100)
    max_abs_culture_valence_mean: float = field(default=0.6)
    min_culture_rho_mean: float = field(default=0.2)
    min_culture_samples: int = field(default=20)


@dataclass
class ResonanceCfg:
    logs: list[str] = field(default_factory=list)
    resample_ms: float | None = field(default=20.0)
    zscore: bool = field(default=True)
    detrend: bool = field(default=True)
    window: str = field(default="hann")
    alpha: float = field(default=0.0)
    beta: float = field(default=0.0)


@dataclass
class PainLoopCfg:
    adaptive: bool = field(default=True)
    forgive_threshold: float = field(default=0.35)
    rotate_daily: bool = field(default=False)
    max_events_per_nightly: int = field(default=2000)
    policy_threshold_base: float = field(default=0.5)
    a2a_empathy_gain_base: float = field(default=0.1)
    ema_alpha: float = field(default=0.4)
    l1_budget: float = field(default=0.8)
    l_inf_budget: float = field(default=0.3)
    min_threshold: float = field(default=0.2)
    max_threshold: float = field(default=0.6)
    quantile: float = field(default=0.35)
    min_samples: int = field(default=50)
    severity_window: int = field(default=200)
    max_empathy: float = field(default=0.5)
    replay_eval: bool = field(default=False)
    care_targets: list[str] = field(default_factory=list)
    comfort_gain_base: float = field(default=0.15)
    protection_bias: float = field(default=0.3)
    growth_reward: float = field(default=0.2)
    patience_budget: float = field(default=0.5)
    care_mode: "CareModeCfg" = field(default_factory=lambda: CareModeCfg())


@dataclass
class CareModeBudgetsCfg:
    l1: float = field(default=0.8)
    l_inf: float = field(default=0.3)


@dataclass
class CareModeCfg:
    enabled: bool = field(default=True)
    canary_ratio: float = field(default=0.0)
    canary_seed: int = field(default=227)
    budgets: CareModeBudgetsCfg = field(default_factory=CareModeBudgetsCfg)


@dataclass
class BackchannelCfg:
    enabled: bool = field(default=True)
    nod_threshold: float = field(default=0.55)
    aizuchi_threshold: float = field(default=0.45)
    min_gap_ms: int = field(default=180)
    cancel_on_voice_ms: int = field(default=50)
    rate_limit_cpm: int = field(default=8)
    silence_ms_for_nod: int = field(default=260)
    silence_ms_for_aizuchi: int = field(default=420)
    energy_bias: float = field(default=0.05)
    culture: str = field(default="ja-JP")


@dataclass
class PresenceCfg:
    enable: bool = field(default=True)
    shadow_threshold: float = field(default=0.70)
    max_ack_interval_s: float = field(default=180.0)
    min_silence_ms: int = field(default=1200)
    short_after_silence_ms: int = field(default=15000)
    ack_key_short: str = field(default="presence_ack.short")
    ack_key_minimal: str = field(default="presence_ack.minimal")
    ack_max_chars: int = field(default=24)


@dataclass
class AckCfg:
    watch_key: str = field(default="ack.watch")
    soothe_key: str = field(default="ack.soothe")
    ask_key: str = field(default="ack.ask")
    talk_key: str = field(default="ack.talk")
    fast_ack_key: str = field(default="ack.fast.ack")
    fast_breath_key: str = field(default="ack.fast.breath")
    max_chars: int = field(default=64)


@dataclass
class MemoryHintPolicyCfg:
    enable: bool = field(default=False)
    allow_verbatim: bool = field(default=False)
    hint_style: str = field(default="temperature")
    max_label_chars: int = field(default=32)
    max_reply_chars: int = field(default=80)
    hint_key_prefix: str = field(default="memory_hint.")
    ban_patterns: list[str] = field(default_factory=list)
    social_mode: str = field(default="solo")
    turn_taking_mode: str = field(default="none")
    style_when_suppressed: str = field(default="none")
    min_silence_ms_for_hint: int = field(default=2000)
    pressure_threshold: float = field(default=0.55)
    pressure_margin: float = field(default=0.10)
    pressure_decay_per_s: float = field(default=0.985)
    pressure_floor: float = field(default=0.0)
    pressure_ceiling: float = field(default=1.0)
    pressure_w_interrupt: float = field(default=0.70)
    pressure_w_social: float = field(default=0.40)
    pressure_w_lowconf: float = field(default=0.60)


@dataclass
class MemoryReferenceCfg:
    enabled: bool = field(default=True)
    k: int = field(default=3)
    name_threshold: float = field(default=0.5)
    fidelity_high: float = field(default=0.65)
    fidelity_low: float = field(default=0.45)
    anchor_high: float = field(default=0.5)
    max_reply_chars: int = field(default=140)
    cooldown_s: float = field(default=20.0)
    log_path: str = field(default="logs/memory_ref.jsonl")
    per_culture: Dict[str, Dict[str, float | int]] = field(default_factory=dict)


@dataclass
class ObserverDisclaimerCfg:
    mode: str = field(default="llm_first")  # llm_first, template_only, fixed_only, llm_only, template_first
    max_latency_ms: int = field(default=150)
    rotation: str = field(default="deterministic")  # deterministic or random
    templates_path: str = field(default="config/i18n/disclaimer.yaml")
    community_bias: float = field(default=0.05)
    i_message_bias: float = field(default=0.0)


@dataclass
class UiCfg:
    community_update_ms: int = field(default=400)


@dataclass
class HeartosTransitionCfg:
    post_risk_scale_default: float = field(default=0.85)


@dataclass
class RuntimeCfg:
    ignition: IgnitionCfg = field(default_factory=IgnitionCfg)
    telemetry: TelemetryCfg = field(default_factory=TelemetryCfg)
    latency: LatencyCfg = field(default_factory=LatencyCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    policy: PolicyRuntimeCfg = field(default_factory=PolicyRuntimeCfg)
    guards: GuardsCfg = field(default_factory=GuardsCfg)
    qualia_access_gate: QualiaAccessGateCfg = field(default_factory=QualiaAccessGateCfg)
    emotion: EmotionCfg = field(default_factory=EmotionCfg)
    culture: CultureCfg = field(default_factory=CultureCfg)
    alerts: AlertsCfg = field(default_factory=AlertsCfg)
    resonance: ResonanceCfg = field(default_factory=ResonanceCfg)
    pain_loop: PainLoopCfg = field(default_factory=PainLoopCfg)
    backchannel: BackchannelCfg = field(default_factory=BackchannelCfg)
    presence: PresenceCfg = field(default_factory=PresenceCfg)
    ack: AckCfg = field(default_factory=AckCfg)
    memory_hint_policy: MemoryHintPolicyCfg = field(default_factory=MemoryHintPolicyCfg)
    memory_reference: MemoryReferenceCfg = field(default_factory=MemoryReferenceCfg)
    observer: ObserverDisclaimerCfg = field(default_factory=ObserverDisclaimerCfg)
    ui: UiCfg = field(default_factory=UiCfg)
    heartos_transition: HeartosTransitionCfg = field(default_factory=HeartosTransitionCfg)


def load_runtime_cfg(path: str | Path = "config/runtime.yaml") -> RuntimeCfg:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return RuntimeCfg()
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return RuntimeCfg()
    ignition = _merge_dataclass(IgnitionCfg(), payload.get("ignition", {}))
    telemetry = _merge_dataclass(TelemetryCfg(), payload.get("telemetry", {}))
    latency = _merge_dataclass(LatencyCfg(), payload.get("latency", {}))
    llm_cfg = _merge_dataclass(LLMCfg(), payload.get("llm", {}))
    model = _merge_dataclass(
        ModelCfg(),
        payload.get("model", {}),
        extra_factories={"assoc_kernel": AssocKernelCfg},
    )
    replay = _merge_dataclass(ReplayCfg(), payload.get("replay", {}))
    policy_cfg = _merge_dataclass(PolicyRuntimeCfg(), payload.get("policy", {}))
    guards_cfg = _merge_dataclass(GuardsCfg(), payload.get("guards", {}))
    qualia_access_gate = _merge_dataclass(
        QualiaAccessGateCfg(), payload.get("qualia_access_gate", {})
    )
    emotion = _merge_dataclass(EmotionCfg(), payload.get("emotion", {}))
    culture = _merge_dataclass(
        CultureCfg(),
        payload.get("culture", {}),
        extra_factories={"feedback": CultureFeedbackCfg},
    )
    alerts = _merge_dataclass(AlertsCfg(), payload.get("alerts", {}))
    resonance = _merge_dataclass(ResonanceCfg(), payload.get("resonance", {}))
    pain_loop = _merge_dataclass(
        PainLoopCfg(),
        payload.get("pain_loop", {}),
        extra_factories={"care_mode": (CareModeCfg, {"budgets": CareModeBudgetsCfg})},
    )
    backchannel = _merge_dataclass(BackchannelCfg(), payload.get("backchannel", {}))
    presence = _merge_dataclass(PresenceCfg(), payload.get("presence", {}))
    ack_cfg = _merge_dataclass(AckCfg(), payload.get("ack", {}))
    memory_hint_policy = _merge_dataclass(
        MemoryHintPolicyCfg(), payload.get("memory_hint_policy", {})
    )
    memory_reference = _merge_dataclass(MemoryReferenceCfg(), payload.get("memory_reference", {}))
    observer_cfg = _merge_dataclass(ObserverDisclaimerCfg(), payload.get("observer", {}))
    ui_cfg = _merge_dataclass(UiCfg(), payload.get("ui", {}))
    heartos_transition = _merge_dataclass(
        HeartosTransitionCfg(), payload.get("heartos_transition", {})
    )
    return RuntimeCfg(
        ignition=ignition,
        telemetry=telemetry,
        latency=latency,
        llm=llm_cfg,
        model=model,
        replay=replay,
        policy=policy_cfg,
        guards=guards_cfg,
        qualia_access_gate=qualia_access_gate,
        emotion=emotion,
        culture=culture,
        alerts=alerts,
        resonance=resonance,
        pain_loop=pain_loop,
        backchannel=backchannel,
        presence=presence,
        ack=ack_cfg,
        memory_hint_policy=memory_hint_policy,
        memory_reference=memory_reference,
        observer=observer_cfg,
        ui=ui_cfg,
        heartos_transition=heartos_transition,
    )


def _merge_dataclass(instance, overrides: dict[str, Any], extra_factories: dict[str, Any] | None = None):
    data = instance.__dict__.copy()
    for key, value in (overrides or {}).items():
        if key not in data:
            continue
        if extra_factories and key in extra_factories and isinstance(value, dict):
            factory_spec = extra_factories[key]
            nested_factories = None
            if isinstance(factory_spec, tuple):
                factory_cls, nested_factories = factory_spec
            else:
                factory_cls = factory_spec
            data[key] = _merge_dataclass(factory_cls(), value, nested_factories)
        else:
            data[key] = value
    return instance.__class__(**data)


@dataclass
class CultureFeedbackCfg:
    enabled: bool = field(default=False)
    corr_high: float = field(default=0.65)
    corr_low: float = field(default=0.35)
    delta: float = field(default=0.02)
    clamp_min: float = field(default=0.0)
    clamp_max: float = field(default=1.0)
    vision_coefficients: dict[str, dict[str, float]] = field(default_factory=dict)


__all__ = [
    "load_runtime_cfg",
    "RuntimeCfg",
    "IgnitionCfg",
    "TelemetryCfg",
    "LatencyCfg",
    "LLMCfg",
    "ModelCfg",
    "ReplayCfg",
    "EmotionCfg",
    "AssocKernelCfg",
    "CultureCfg",
    "CultureFeedbackCfg",
    "AlertsCfg",
    "ResonanceCfg",
    "PainLoopCfg",
    "CareModeCfg",
    "CareModeBudgetsCfg",
    "BackchannelCfg",
    "PresenceCfg",
    "AckCfg",
    "MemoryHintPolicyCfg",
    "MemoryReferenceCfg",
    "ObserverDisclaimerCfg",
    "PolicyRuntimeCfg",
    "GuardsCfg",
    "UiCfg",
    "HeartosTransitionCfg",
]
