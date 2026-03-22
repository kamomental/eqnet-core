from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SubjectiveHypothesis = Literal[
    "state_only",
    "attention_weighted",
    "access_mode",
    "geometry_interaction",
]


@dataclass
class DynamicFieldConfig:
    """EQNet Emotional DFT の最小動的場パラメータ。"""

    alpha_surprise: float = 0.9
    beta_surprise: float = 0.25
    gamma_surprise: float = 0.12
    relation_gamma: float = 0.18
    habituation_decay: float = 0.08
    habituation_gain: float = 0.24
    memory_fixation_gain: float = 0.18
    memory_fixation_decay: float = 0.03


@dataclass
class DynamicFieldState:
    """最小の内部状態。"""

    arousal: float
    prediction_error_mass: float
    relation_value: float
    habituation: float = 0.0
    memory_fixation: float = 0.0


@dataclass
class DynamicFieldInput:
    """刺激入力とその新奇性。"""

    arousal_drive: float = 0.0
    prediction_drive: float = 0.0
    relation_drive: float = 0.0
    novelty: float = 0.0
    repeated_exposure: float = 0.0
    noise: float = 0.0


@dataclass
class AccessState:
    """主観アクセスの最小状態。"""

    attention: float = 0.0
    interface_curvature: float = 0.0
    access_uncertainty: float = 1.0
    reportability: float = 0.0


@dataclass
class TerrainSnapshot:
    """感情地形の局所情報。"""

    state_energy: float
    gradient_norm: float
    max_curvature: float


SUBJECTIVE_HYPOTHESIS_REGISTRY: dict[SubjectiveHypothesis, str] = {
    "state_only": "状態量だけで主観強度を決める基準モデル。",
    "attention_weighted": "状態量に attention を掛ける最小 access モデル。",
    "access_mode": "attention と reportability で前景化された access モデル。",
    "geometry_interaction": "地形勾配と界面幾何の相互作用を見る幾何モデル。",
}


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def response_kernel_gain(
    novelty: float,
    habituation: float,
    *,
    config: DynamicFieldConfig | None = None,
) -> float:
    """新奇性と慣れから、その時点の応答核ゲインを計算する。"""
    cfg = config or DynamicFieldConfig()
    novelty_term = max(0.0, novelty)
    damped = novelty_term * (1.0 - clamp_unit(habituation))
    return max(0.0, damped + 0.15 - cfg.habituation_decay * clamp_unit(habituation))


def update_habituation(
    state: DynamicFieldState,
    external_input: DynamicFieldInput,
    *,
    dt: float,
    config: DynamicFieldConfig | None = None,
) -> float:
    """反復刺激で上がり、入力が弱いときはゆっくり戻る慣れ量。"""
    cfg = config or DynamicFieldConfig()
    growth = cfg.habituation_gain * max(0.0, external_input.repeated_exposure)
    decay = cfg.habituation_decay * clamp_unit(state.habituation)
    value = state.habituation + dt * (growth - decay)
    return clamp_unit(value)


def surprise_score(
    prediction_error_norm: float,
    gradient_norm: float,
    max_curvature: float,
    *,
    config: DynamicFieldConfig | None = None,
) -> float:
    cfg = config or DynamicFieldConfig()
    return (
        cfg.alpha_surprise * prediction_error_norm
        + cfg.beta_surprise * gradient_norm
        + cfg.gamma_surprise * max_curvature
    )


def update_memory_fixation(
    state: DynamicFieldState,
    surprise: float,
    access: AccessState,
    *,
    dt: float,
    config: DynamicFieldConfig | None = None,
) -> float:
    """驚きと attention に応じて記憶固定量を更新する。"""
    cfg = config or DynamicFieldConfig()
    fixation_drive = cfg.memory_fixation_gain * max(0.0, surprise) * clamp_unit(access.attention)
    decay = cfg.memory_fixation_decay * max(0.0, state.memory_fixation)
    return max(0.0, state.memory_fixation + dt * (fixation_drive - decay))


def dynamic_step(
    state: DynamicFieldState,
    terrain: TerrainSnapshot,
    graph_coupling: tuple[float, float, float],
    external_input: DynamicFieldInput,
    access: AccessState,
    *,
    dt: float,
    config: DynamicFieldConfig | None = None,
) -> DynamicFieldState:
    """地形勾配、関係結合、応答核ゲインで最小状態を 1 step 更新する。"""
    cfg = config or DynamicFieldConfig()
    cx, cy, cz = graph_coupling
    gain = response_kernel_gain(
        external_input.novelty,
        state.habituation,
        config=cfg,
    )
    habituation = update_habituation(
        state,
        external_input,
        dt=dt,
        config=cfg,
    )
    surprise = surprise_score(
        max(0.0, external_input.prediction_drive),
        terrain.gradient_norm,
        terrain.max_curvature,
        config=cfg,
    )
    memory_fixation = update_memory_fixation(
        state,
        surprise,
        access,
        dt=dt,
        config=cfg,
    )
    return DynamicFieldState(
        arousal=state.arousal + dt * (
            -terrain.gradient_norm
            + gain * external_input.arousal_drive
            - cfg.relation_gamma * cx
            + external_input.noise
        ),
        prediction_error_mass=state.prediction_error_mass + dt * (
            -0.6 * terrain.gradient_norm
            + gain * external_input.prediction_drive
            - cfg.relation_gamma * cy
            + external_input.noise
        ),
        relation_value=state.relation_value + dt * (
            -0.4 * terrain.gradient_norm
            + gain * external_input.relation_drive
            - cfg.relation_gamma * cz
            + external_input.noise
        ),
        habituation=habituation,
        memory_fixation=memory_fixation,
    )


def subjective_intensity(
    *,
    terrain: TerrainSnapshot,
    access: AccessState,
    hypothesis: SubjectiveHypothesis,
) -> float:
    """主観強度を比較可能な仮説群として返す。"""
    if hypothesis == "state_only":
        return max(0.0, terrain.state_energy)
    if hypothesis == "attention_weighted":
        return max(0.0, terrain.state_energy * clamp_unit(access.attention))
    if hypothesis == "access_mode":
        reportability = clamp_unit(access.reportability)
        uncertainty_gate = 1.0 - 0.5 * clamp_unit(access.access_uncertainty)
        return max(
            0.0,
            terrain.state_energy
            * (0.35 + 0.65 * clamp_unit(access.attention))
            * (0.25 + 0.75 * reportability)
            * uncertainty_gate,
        )
    return max(
        0.0,
        terrain.gradient_norm
        * clamp_unit(access.interface_curvature)
        * (0.2 + 0.8 * clamp_unit(access.attention)),
    )
