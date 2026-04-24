from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TerrainDynamicsFrame:
    """内的地形力学の最近フレーム。"""

    step: int = 0
    dominant_basin: str = "steady_basin"
    dominant_flow: str = "settle"
    terrain_energy: float = 0.0
    entropy: float = 0.0
    ignition_pressure: float = 0.0
    barrier_height: float = 0.0
    recovery_gradient: float = 0.0
    basin_pull: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "dominant_basin": self.dominant_basin,
            "dominant_flow": self.dominant_flow,
            "terrain_energy": round(self.terrain_energy, 4),
            "entropy": round(self.entropy, 4),
            "ignition_pressure": round(self.ignition_pressure, 4),
            "barrier_height": round(self.barrier_height, 4),
            "recovery_gradient": round(self.recovery_gradient, 4),
            "basin_pull": round(self.basin_pull, 4),
        }


@dataclass(frozen=True)
class TerrainDynamicsState:
    """地形・障壁・発火・回復の時間力学を束ねる state。"""

    dominant_basin: str = "steady_basin"
    dominant_flow: str = "settle"
    terrain_energy: float = 0.0
    entropy: float = 0.0
    ignition_pressure: float = 0.0
    barrier_height: float = 0.0
    recovery_gradient: float = 0.0
    basin_pull: float = 0.0
    trace: tuple[TerrainDynamicsFrame, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_basin": self.dominant_basin,
            "dominant_flow": self.dominant_flow,
            "terrain_energy": round(self.terrain_energy, 4),
            "entropy": round(self.entropy, 4),
            "ignition_pressure": round(self.ignition_pressure, 4),
            "barrier_height": round(self.barrier_height, 4),
            "recovery_gradient": round(self.recovery_gradient, 4),
            "basin_pull": round(self.basin_pull, 4),
            "trace": [frame.to_dict() for frame in self.trace],
        }

    def to_packet_axes(
        self,
        previous: Mapping[str, Any] | "TerrainDynamicsState" | None = None,
    ) -> dict[str, dict[str, float]]:
        previous_state = coerce_terrain_dynamics_state(previous)
        current_axes = _axis_values(self)
        previous_axes = _axis_values(previous_state)
        return {
            axis_name: {
                "value": round(axis_value, 4),
                "delta": round(axis_value - previous_axes.get(axis_name, 0.0), 4),
            }
            for axis_name, axis_value in current_axes.items()
        }


def derive_terrain_dynamics_state(
    *,
    previous_state: Mapping[str, Any] | TerrainDynamicsState | None = None,
    organism_state: Mapping[str, Any] | None = None,
    external_field_state: Mapping[str, Any] | None = None,
    memory_dynamics_state: Mapping[str, Any] | None = None,
    qualia_structure_state: Mapping[str, Any] | None = None,
    heartbeat_structure_state: Mapping[str, Any] | None = None,
    terrain_readout: Mapping[str, Any] | None = None,
    trace_limit: int = 8,
) -> TerrainDynamicsState:
    previous = coerce_terrain_dynamics_state(previous_state)
    organism = dict(organism_state or {})
    external_field = dict(external_field_state or {})
    memory_dynamics = dict(memory_dynamics_state or {})
    qualia = dict(qualia_structure_state or {})
    heartbeat = dict(heartbeat_structure_state or {})
    terrain = dict(terrain_readout or {})
    dominant_relation_type = _text(memory_dynamics.get("dominant_relation_type"))
    dominant_causal_type = _text(memory_dynamics.get("dominant_causal_type"))
    causal_generation_mode = _text(memory_dynamics.get("causal_generation_mode"))
    relation_generation_mode = _text(memory_dynamics.get("relation_generation_mode"))
    relation_meta_type = ""
    raw_meta_relations = memory_dynamics.get("meta_relations")
    if isinstance(raw_meta_relations, list) and raw_meta_relations and isinstance(raw_meta_relations[0], Mapping):
        relation_meta_type = _text(raw_meta_relations[0].get("meta_type"))

    causal_ignition_bias = 0.0
    if dominant_causal_type in {"triggered_by", "amplified_by"}:
        causal_ignition_bias += 0.08
    if causal_generation_mode == "reinforced":
        causal_ignition_bias += 0.06
    elif causal_generation_mode == "ignited":
        causal_ignition_bias += 0.04

    causal_barrier_bias = 0.0
    if dominant_causal_type in {"suppressed_by", "reopened_by"}:
        causal_barrier_bias += 0.08
    if causal_generation_mode == "contested":
        causal_barrier_bias += 0.08
    if relation_meta_type == "competes_with":
        causal_barrier_bias += 0.05

    causal_recovery_bias = 0.0
    if dominant_causal_type in {"reframed_by", "enabled_by"}:
        causal_recovery_bias += 0.08
    if dominant_causal_type == "amplified_by":
        causal_recovery_bias += 0.04
    if relation_meta_type == "reinforces":
        causal_recovery_bias += 0.04

    causal_entropy_bias = 0.0
    if dominant_causal_type in {"suppressed_by", "reopened_by"}:
        causal_entropy_bias += 0.08
    if causal_generation_mode == "contested":
        causal_entropy_bias += 0.08
    elif dominant_causal_type in {"reframed_by", "enabled_by"}:
        causal_entropy_bias -= 0.04

    causal_basin_bias = 0.0
    if dominant_relation_type in {"same_anchor", "cross_context_bridge"}:
        causal_basin_bias += 0.04
    if dominant_causal_type in {"reframed_by", "enabled_by"}:
        causal_basin_bias += 0.08
    if relation_generation_mode in {"anchored", "ignited"}:
        causal_basin_bias += 0.04

    terrain_protect_bias = _float01(terrain.get("protect_bias"))
    terrain_approach_bias = _float01(terrain.get("approach_bias"))
    terrain_value = _float01(abs(_float(terrain.get("value"), 0.0)))
    terrain_energy = _clamp01(
        _float01(qualia.get("emergence")) * 0.18
        + _float01(heartbeat.get("activation_drive")) * 0.22
        + _float01(organism.get("expressive_readiness")) * 0.16
        + _float01(memory_dynamics.get("ignition_readiness")) * 0.14
        + _float01(external_field.get("novelty")) * 0.1
        + _float01(external_field.get("continuity_pull")) * 0.08
        + terrain_value * 0.12
        + causal_ignition_bias * 0.18
    )
    entropy = _clamp01(
        _float01(qualia.get("drift")) * 0.24
        + max(0.0, 1.0 - _float01(qualia.get("stability"))) * 0.16
        + _float01(external_field.get("ambiguity_load")) * 0.2
        + _float01(memory_dynamics.get("memory_tension")) * 0.12
        + max(0.0, 1.0 - _float01(organism.get("coherence"))) * 0.12
        + _float01(external_field.get("novelty")) * 0.1
        + max(0.0, 1.0 - _float01(heartbeat.get("entrainment"))) * 0.06
        + causal_entropy_bias
    )
    barrier_height = _clamp01(
        _float01(organism.get("protective_tension")) * 0.24
        + _float01(heartbeat.get("containment_bias")) * 0.2
        + _float01(external_field.get("social_pressure")) * 0.16
        + _float01(external_field.get("environmental_load")) * 0.12
        + _float01(memory_dynamics.get("memory_tension")) * 0.12
        + terrain_protect_bias * 0.16
        + causal_barrier_bias
    )
    recovery_gradient = _clamp01(
        _float01(heartbeat.get("recovery_pull")) * 0.26
        + _float01(organism.get("grounding")) * 0.18
        + _float01(external_field.get("safety_envelope")) * 0.18
        + _float01(memory_dynamics.get("consolidation_pull")) * 0.12
        + _float01(qualia.get("temporal_coherence")) * 0.1
        + max(0.0, 1.0 - entropy) * 0.08
        + max(0.0, 1.0 - _float01(external_field.get("novelty"))) * 0.08
        + causal_recovery_bias
    )
    basin_pull = _clamp01(
        _float01(external_field.get("continuity_pull")) * 0.2
        + _float01(memory_dynamics.get("monument_salience")) * 0.18
        + _float01(organism.get("relation_pull")) * 0.14
        + _float01(qualia.get("memory_resonance")) * 0.18
        + terrain_approach_bias * 0.12
        + _float01(external_field.get("safety_envelope")) * 0.08
        + _float01(heartbeat.get("entrainment")) * 0.1
        + causal_basin_bias
    )
    ignition_pressure = _clamp01(
        terrain_energy * 0.28
        + _float01(memory_dynamics.get("ignition_readiness")) * 0.22
        + _float01(qualia.get("emergence")) * 0.12
        + _float01(heartbeat.get("activation_drive")) * 0.12
        + _float01(external_field.get("novelty")) * 0.1
        + basin_pull * 0.08
        + causal_ignition_bias
        - barrier_height * 0.16
        - entropy * 0.1
    )

    terrain_energy = _carry(previous.terrain_energy, terrain_energy, previous_state, 0.18)
    entropy = _carry(previous.entropy, entropy, previous_state, 0.16)
    ignition_pressure = _carry(previous.ignition_pressure, ignition_pressure, previous_state, 0.18)
    barrier_height = _carry(previous.barrier_height, barrier_height, previous_state, 0.22)
    recovery_gradient = _carry(previous.recovery_gradient, recovery_gradient, previous_state, 0.18)
    basin_pull = _carry(previous.basin_pull, basin_pull, previous_state, 0.2)

    dominant_basin = _dominant_basin(
        barrier_height=barrier_height,
        recovery_gradient=recovery_gradient,
        ignition_pressure=ignition_pressure,
        basin_pull=basin_pull,
        entropy=entropy,
        continuity_pull=_float01(external_field.get("continuity_pull")),
        monument_salience=_float01(memory_dynamics.get("monument_salience")),
        play_window=_float01(organism.get("play_window")),
    )
    dominant_flow = _dominant_flow(
        barrier_height=barrier_height,
        recovery_gradient=recovery_gradient,
        ignition_pressure=ignition_pressure,
        basin_pull=basin_pull,
        entropy=entropy,
    )
    step = previous.trace[-1].step + 1 if previous.trace else 1
    trace = list(previous.trace[-max(0, trace_limit - 1) :]) if trace_limit > 0 else []
    trace.append(
        TerrainDynamicsFrame(
            step=step,
            dominant_basin=dominant_basin,
            dominant_flow=dominant_flow,
            terrain_energy=terrain_energy,
            entropy=entropy,
            ignition_pressure=ignition_pressure,
            barrier_height=barrier_height,
            recovery_gradient=recovery_gradient,
            basin_pull=basin_pull,
        )
    )
    return TerrainDynamicsState(
        dominant_basin=dominant_basin,
        dominant_flow=dominant_flow,
        terrain_energy=terrain_energy,
        entropy=entropy,
        ignition_pressure=ignition_pressure,
        barrier_height=barrier_height,
        recovery_gradient=recovery_gradient,
        basin_pull=basin_pull,
        trace=tuple(trace[-trace_limit:] if trace_limit > 0 else ()),
    )


def coerce_terrain_dynamics_state(
    value: Mapping[str, Any] | TerrainDynamicsState | None,
) -> TerrainDynamicsState:
    if isinstance(value, TerrainDynamicsState):
        return value
    payload = dict(value or {})
    trace_items: list[TerrainDynamicsFrame] = []
    for item in payload.get("trace") or ():
        if isinstance(item, TerrainDynamicsFrame):
            trace_items.append(item)
        elif isinstance(item, Mapping):
            trace_items.append(
                TerrainDynamicsFrame(
                    step=int(_float(item.get("step"), 0.0)),
                    dominant_basin=_text(item.get("dominant_basin")) or "steady_basin",
                    dominant_flow=_text(item.get("dominant_flow")) or "settle",
                    terrain_energy=_float01(item.get("terrain_energy")),
                    entropy=_float01(item.get("entropy")),
                    ignition_pressure=_float01(item.get("ignition_pressure")),
                    barrier_height=_float01(item.get("barrier_height")),
                    recovery_gradient=_float01(item.get("recovery_gradient")),
                    basin_pull=_float01(item.get("basin_pull")),
                )
            )
    return TerrainDynamicsState(
        dominant_basin=_text(payload.get("dominant_basin")) or "steady_basin",
        dominant_flow=_text(payload.get("dominant_flow")) or "settle",
        terrain_energy=_float01(payload.get("terrain_energy")),
        entropy=_float01(payload.get("entropy")),
        ignition_pressure=_float01(payload.get("ignition_pressure")),
        barrier_height=_float01(payload.get("barrier_height")),
        recovery_gradient=_float01(payload.get("recovery_gradient")),
        basin_pull=_float01(payload.get("basin_pull")),
        trace=tuple(trace_items),
    )


def _axis_values(state: TerrainDynamicsState) -> dict[str, float]:
    return {
        "energy": _clamp01(state.terrain_energy),
        "entropy": _clamp01(state.entropy),
        "ignition": _clamp01(state.ignition_pressure),
        "barrier": _clamp01(state.barrier_height),
        "recovery": _clamp01(state.recovery_gradient),
        "basin": _clamp01(state.basin_pull),
    }


def _dominant_basin(
    *,
    barrier_height: float,
    recovery_gradient: float,
    ignition_pressure: float,
    basin_pull: float,
    entropy: float,
    continuity_pull: float,
    monument_salience: float,
    play_window: float,
) -> str:
    if barrier_height >= max(recovery_gradient, ignition_pressure, basin_pull) and barrier_height >= 0.5:
        return "protective_basin"
    if recovery_gradient >= max(barrier_height, ignition_pressure, basin_pull) and recovery_gradient >= 0.46:
        return "recovery_basin"
    if ignition_pressure >= max(barrier_height, recovery_gradient, basin_pull) and ignition_pressure >= 0.5:
        return "ignition_basin"
    if basin_pull >= max(barrier_height, recovery_gradient, ignition_pressure) and max(continuity_pull, monument_salience) >= 0.42:
        return "continuity_basin"
    if entropy >= 0.56:
        return "diffuse_basin"
    if play_window >= 0.48 and ignition_pressure >= 0.42:
        return "play_basin"
    return "steady_basin"


def _dominant_flow(
    *,
    barrier_height: float,
    recovery_gradient: float,
    ignition_pressure: float,
    basin_pull: float,
    entropy: float,
) -> str:
    if ignition_pressure >= 0.58 and entropy < 0.62:
        return "ignite"
    if recovery_gradient >= max(barrier_height, ignition_pressure, basin_pull) and recovery_gradient >= 0.44:
        return "recover"
    if barrier_height >= 0.54:
        return "contain"
    if entropy >= 0.56:
        return "diffuse"
    if basin_pull >= 0.48:
        return "reenter"
    return "settle"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _float01(value: Any, default: float = 0.0) -> float:
    return _clamp01(_float(value, default))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _carry(previous_value: float, current_value: float, previous_state: Any, alpha: float) -> float:
    if previous_state is None:
        return _clamp01(current_value)
    return _clamp01(previous_value * alpha + current_value * (1.0 - alpha))
