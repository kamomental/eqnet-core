from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from eqnet_core.models.conscious import (
    ConsciousEpisode,
    ResponseRoute,
    SelfLayer,
    SelfModel,
    WorldStateSnapshot,
)
from eqnet_core.models.emotion import EmotionVector, ValueGradient
from eqnet_core.models.talk_mode import TalkMode
from eqnet_core.memory.diary import DiaryWriter

from emot_terrain_lab.mind.inner_replay import (
    InnerReplayController,
    ReplayConfig,
    ReplayInputs,
    ReplayOutcome,
)


@dataclass
class MiniWorldStep:
    """Single state inside a scripted scenario."""

    name: str
    narrative: str
    salient_entities: List[str]
    context_tags: List[str]
    hazard_score: float = 0.0
    hazard_sources: List[str] = field(default_factory=list)
    chaos: float = 0.0
    risk: float = 0.0
    tom_cost: float = 0.0
    uncertainty: Optional[float] = None
    reward: float = 0.5
    valence: float = 0.0
    arousal: float = 0.0
    stress: float = 0.0
    love: float = 0.0
    mask: float = 0.0
    breath_ratio: float = 0.5
    heart_rate: float = 0.5
    anchor_label: Optional[str] = None
    log_episode: bool = False
    action: str = "OBSERVE"
    talk_mode: str = "watch"
    flags: List[str] = field(default_factory=list)
    value_gradient_override: Optional[Dict[str, float]] = None
    value_blend_ratio: float = 0.55
    timestamp: Optional[str] = None
    observations: Optional[Dict[str, Any]] = None
    membrane: Optional[Dict[str, float]] = None
    akorn: Optional[Dict[str, float]] = None
    emotion_label: Optional[str] = None

    def context_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "salient_entities": list(self.salient_entities),
            "context_tags": list(self.context_tags),
            "hazard_score": float(self.hazard_score),
            "hazard_sources": list(self.hazard_sources),
            "flags": list(self.flags),
        }
        if self.anchor_label:
            merged_flags = set(payload["flags"])
            merged_flags.add("anchor_cue")
            merged_flags.add(f"anchor:{self.anchor_label}")
            payload["flags"] = list(merged_flags)
        return payload


@dataclass
class MiniWorldScenario:
    name: str
    description: str
    steps: Sequence[MiniWorldStep]


@dataclass
class StepResult:
    scenario: str
    step: MiniWorldStep
    emotion: EmotionVector
    world_state: WorldStateSnapshot
    replay_inputs: ReplayInputs
    replay_outcome: ReplayOutcome
    route: ResponseRoute
    talk_mode: TalkMode
    step_index: int
    run_id: str
    seed: Optional[int]

    def trace_payload(self) -> Dict[str, object]:
        outcome = self.replay_outcome
        stats = outcome.stats
        return {
            "step_index": self.step_index,
            "scenario": self.scenario,
            "state": self.step.name,
            "action": self.step.action,
            "anchor": self.step.anchor_label,
            "context": self.world_state.to_dict(),
            "emotion": self.emotion.to_dict(),
            "replay_inputs": asdict(self.replay_inputs),
            "replay_outcome": {
                "decision": outcome.decision,
                "felt_intent_time": outcome.felt_intent_time,
                "u_hat": outcome.u_hat,
                "veto_score": outcome.veto_score,
                "prep_features": outcome.prep_features,
                "plan_features": outcome.plan_features,
                "trace": list(outcome.trace),
                "stats": None
                if stats is None
                else {
                    "top_signal": stats.top_signal,
                    "mean_signal": stats.mean_signal,
                    "entropy_proxy": stats.entropy_proxy,
                    "alignment": stats.alignment,
                },
            },
            "route": self.route.value,
            "talk_mode": self.talk_mode.value,
            "narrative": self.step.narrative,
            "timestamp": self.step.timestamp,
            "observations": self.step.observations,
            "internal": {
                "emotion_label": self.step.emotion_label,
                "membrane": self.step.membrane,
                "akorn": self.step.akorn,
            },
            "run": {"run_id": self.run_id, "seed": self.seed},
        }


@dataclass
class ScenarioStats:
    scenario: str
    steps: int
    mean_hazard: float
    anchors: int
    execute_count: int
    cancel_count: int
    conscious_count: int


class MiniWorldSimulator:
    """Scriptable box-world that feeds EQNet primitives."""

    def __init__(
        self,
        *,
        replay_config: Optional[ReplayConfig] = None,
        diary_path: str | Path = "logs/mini_world_diary.jsonl",
        telemetry_path: str | Path = "logs/mini_world_trace.jsonl",
        base_value_gradient: Optional[ValueGradient] = None,
    ) -> None:
        replay_cfg = replay_config or ReplayConfig()
        self.controller = InnerReplayController(replay_cfg)
        diary_path = Path(diary_path) if diary_path else None
        if diary_path:
            diary_path.parent.mkdir(parents=True, exist_ok=True)
            self.diary_writer = DiaryWriter(diary_path)
        else:
            self.diary_writer = None
        self.telemetry_path = Path(telemetry_path)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_gradient = base_value_gradient or ValueGradient()
        self.self_model = SelfModel(
            role_labels=["companion", "observer"],
            long_term_traits={"warmth": 0.6, "stability": 0.5},
            current_mode=TalkMode.WATCH,
            current_energy=0.85,
            attachment_to_user=0.5,
        )
        self._prev_emotion: Optional[EmotionVector] = None
        self._step_index = 0
        self.run_id = str(uuid.uuid4())
        self.seed = getattr(replay_cfg, "seed", None)

    def run_scenario(
        self, scenario: MiniWorldScenario
    ) -> Tuple[List[StepResult], ScenarioStats]:
        results: List[StepResult] = []
        hazard_sum = 0.0
        execute_count = 0
        conscious_count = 0
        anchors = 0
        for step in scenario.steps:
            result = self._run_step(scenario, step)
            results.append(result)
            self._maybe_log_episode(result)
            self._write_trace(result)
            self._prev_emotion = result.emotion
            hazard_sum += float(result.world_state.hazard_score)
            if result.replay_outcome.decision == "execute":
                execute_count += 1
            if result.route != ResponseRoute.REFLEX:
                conscious_count += 1
            if step.anchor_label:
                anchors += 1
        total_steps = len(results)
        mean_hazard = hazard_sum / total_steps if total_steps else 0.0
        stats = ScenarioStats(
            scenario=scenario.name,
            steps=total_steps,
            mean_hazard=mean_hazard,
            anchors=anchors,
            execute_count=execute_count,
            cancel_count=total_steps - execute_count,
            conscious_count=conscious_count,
        )
        return results, stats

    def _run_step(self, scenario: MiniWorldScenario, step: MiniWorldStep) -> StepResult:
        value_gradient = self._value_gradient_for_step(step)
        emotion = EmotionVector(
            valence=step.valence,
            arousal=step.arousal,
            love=step.love,
            stress=step.stress,
            mask=step.mask,
            heart_rate_norm=step.heart_rate,
            breath_ratio_norm=step.breath_ratio,
            value_gradient=value_gradient,
        )
        delta_aff = self._delta_aff(self._prev_emotion, emotion)
        inputs = self._build_replay_inputs(step, emotion, delta_aff)
        outcome = self.controller.run_cycle(inputs)
        context_payload = step.context_payload()
        prediction_error = self._prediction_error(step, delta_aff)
        world_state = WorldStateSnapshot.from_context(
            context_payload, prediction_error=prediction_error
        )
        talk_mode = TalkMode.from_any(step.talk_mode)
        self.self_model.current_mode = talk_mode
        self.self_model.current_energy = float(max(0.2, 0.95 - 0.4 * step.stress))
        self.self_model.attachment_to_user = float(
            min(1.0, 0.45 + max(emotion.love, 0.0))
        )
        route = self._route_for_step(step, outcome)
        result = StepResult(
            scenario=scenario.name,
            step=step,
            emotion=emotion,
            world_state=world_state,
            replay_inputs=inputs,
            replay_outcome=outcome,
            route=route,
            talk_mode=talk_mode,
            step_index=self._step_index,
            run_id=self.run_id,
            seed=self.seed,
        )
        self._step_index += 1
        return result

    def _maybe_log_episode(self, result: StepResult) -> None:
        if not (result.step.log_episode and self.diary_writer):
            return
        snapshot = self.self_model.snapshot()
        narrative = f"[{result.scenario}] {result.step.narrative}"
        episode = ConsciousEpisode(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            self_state=snapshot,
            world_state=result.world_state,
            qualia=result.emotion,
            narrative=narrative,
            route=result.route,
            value_gradient=result.emotion.value_gradient,
            dominant_self_layer=(
                SelfLayer.AFFECTIVE
                if abs(result.emotion.valence) >= 0.25
                else SelfLayer.NARRATIVE
            ),
        )
        self.diary_writer.write_conscious_episode(episode)

    def _write_trace(self, result: StepResult) -> None:
        record = result.trace_payload()
        with self.telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _value_gradient_for_step(self, step: MiniWorldStep) -> ValueGradient:
        base = self.base_gradient
        if not step.value_gradient_override:
            return base
        override = ValueGradient.from_mapping(step.value_gradient_override)
        ratio = float(max(0.0, min(1.0, step.value_blend_ratio)))
        return base.blend(override, ratio)

    @staticmethod
    def _delta_aff(prev: Optional[EmotionVector], current: EmotionVector) -> float:
        if prev is None:
            return 0.0
        delta = (
            abs(current.valence - prev.valence)
            + abs(current.arousal - prev.arousal)
            + abs(current.stress - prev.stress)
        ) / 3.0
        return float(delta)

    @staticmethod
    def _prediction_error(step: MiniWorldStep, delta_aff: float) -> float:
        return float(min(1.0, 0.35 * step.chaos + 0.35 * step.risk + 0.3 * delta_aff))

    def _build_replay_inputs(
        self, step: MiniWorldStep, emotion: EmotionVector, delta_aff: float
    ) -> ReplayInputs:
        chaos = float(step.chaos)
        risk = float(step.risk)
        tom_cost = float(step.tom_cost)
        uncertainty = step.uncertainty
        if step.observations:
            derived = self._derive_scalars_from_obs(step.observations)
            chaos = derived["chaos"]
            risk = derived["risk"]
            tom_cost = derived["tom_cost"]
            if uncertainty is None:
                uncertainty = derived["uncertainty"]
        if uncertainty is None:
            uncertainty = 0.5 * (chaos + risk)
        uncertainty = self._clamp01(float(uncertainty))
        reward = float(max(0.0, min(1.0, step.reward)))
        return ReplayInputs(
            chaos_sens=float(chaos),
            tom_cost=float(tom_cost),
            delta_aff_abs=float(delta_aff),
            risk=float(risk),
            uncertainty=uncertainty,
            reward_estimate=reward,
            mood_valence=emotion.valence,
            mood_arousal=emotion.arousal,
        )


    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @classmethod
    def _derive_scalars_from_obs(cls, obs: Mapping[str, Any]) -> Dict[str, float]:
        audio = obs.get("audio", {}) or {}
        video = obs.get("video", {}) or {}
        meta = obs.get("meta", {}) or {}

        def _coerce(container: Mapping[str, Any], key: str, default: float = 0.0) -> float:
            value = container.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        rms = _coerce(audio, "rms")
        peak = _coerce(audio, "peak")
        flux = _coerce(audio, "flux")
        overlap = _coerce(audio, "overlap_speech")

        flow = _coerce(video, "flow_mag")
        face_delta = _coerce(video, "face_delta")
        pose_delta = _coerce(video, "pose_delta")
        luma = _coerce(video, "luma_mean", 0.5)
        scene_change = _coerce(video, "scene_change")

        latency = _coerce(meta, "response_latency")

        darkness = cls._clamp01(1.0 - luma)
        chaos = cls._clamp01(
            0.40 * flow + 0.25 * flux + 0.20 * scene_change + 0.15 * pose_delta
        )
        risk = cls._clamp01(
            0.45 * peak + 0.25 * darkness + 0.20 * face_delta + 0.10 * rms
        )
        tom_cost = cls._clamp01(0.60 * overlap + 0.40 * cls._clamp01(latency / 2.0))
        uncertainty = cls._clamp01(0.5 * (chaos + risk))
        return {
            "chaos": chaos,
            "risk": risk,
            "tom_cost": tom_cost,
            "uncertainty": uncertainty,
            "darkness": darkness,
        }

    def _route_for_step(
        self, step: MiniWorldStep, outcome: ReplayOutcome
    ) -> ResponseRoute:
        if step.action.upper() in {"APPROACH", "HELP"}:
            return ResponseRoute.HABIT
        if step.log_episode or outcome.decision == "execute":
            return ResponseRoute.CONSCIOUS
        return ResponseRoute.REFLEX


def build_default_scenarios() -> Dict[str, MiniWorldScenario]:
    commute = MiniWorldScenario(
        name="commute",
        description="Landmarks trigger anchor cues during an old commute.",
        steps=[
            MiniWorldStep(
                name="station_arrival",
                narrative="Crowded station after decades; memory feels distant.",
                salient_entities=["station", "ticket_gate", "crowd"],
                context_tags=["commute", "morning", "public"],
                hazard_score=0.25,
                hazard_sources=["rush_hour", "crowd_push"],
                chaos=0.35,
                risk=0.25,
                tom_cost=0.4,
                valence=-0.05,
                arousal=0.32,
                stress=0.28,
                breath_ratio=0.58,
                heart_rate=0.6,
                action="OBSERVE",
            ),
            MiniWorldStep(
                name="unchanged_shop",
                narrative="Old bakery appears unchanged, anchoring the route memory.",
                salient_entities=["station", "bakery", "sweets"],
                context_tags=["commute", "anchor", "memory"],
                hazard_score=0.12,
                hazard_sources=["narrow_alley"],
                chaos=0.18,
                risk=0.12,
                tom_cost=0.2,
                valence=0.38,
                arousal=0.24,
                stress=0.15,
                love=0.22,
                breath_ratio=0.48,
                heart_rate=0.52,
                anchor_label="bakery",
                log_episode=True,
                action="APPROACH",
                value_gradient_override={"attachment_bias": 0.8, "exploration_bias": 0.4},
            ),
            MiniWorldStep(
                name="family_flashback",
                narrative="Grandmother insisting on souvenirs; father cautioning her.",
                salient_entities=["grandmother", "father", "mother", "souvenir_shop"],
                context_tags=["family", "memory", "social_rule"],
                hazard_score=0.2,
                hazard_sources=["time_pressure"],
                chaos=0.42,
                risk=0.31,
                tom_cost=0.5,
                valence=-0.04,
                arousal=0.46,
                stress=0.37,
                love=0.12,
                breath_ratio=0.62,
                heart_rate=0.64,
                log_episode=True,
                action="WAIT",
                value_gradient_override={"social_bias": 0.75, "attachment_bias": 0.7},
            ),
            MiniWorldStep(
                name="city_hall_corner",
                narrative="Empty lot opposite the city hall confirms the ryokan location.",
                salient_entities=["city_hall", "ryokan_site", "street_corner"],
                context_tags=["commute", "anchor", "verification"],
                hazard_score=0.18,
                hazard_sources=["traffic"],
                chaos=0.22,
                risk=0.18,
                tom_cost=0.25,
                valence=0.24,
                arousal=0.28,
                stress=0.18,
                love=0.18,
                breath_ratio=0.5,
                heart_rate=0.55,
                anchor_label="ryokan_site",
                log_episode=True,
                action="OBSERVE",
            ),
        ],
    )

    family = MiniWorldScenario(
        name="family_roles",
        description="Roles within the family activate attachment and caution cues.",
        steps=[
            MiniWorldStep(
                name="living_room_checkin",
                narrative="Morning check-in with mother; gentle but tired atmosphere.",
                salient_entities=["mother", "tea", "living_room"],
                context_tags=["home", "morning", "care"],
                hazard_score=0.1,
                hazard_sources=["fatigue"],
                chaos=0.18,
                risk=0.12,
                tom_cost=0.23,
                valence=0.22,
                arousal=0.21,
                stress=0.16,
                love=0.4,
                breath_ratio=0.46,
                heart_rate=0.48,
                action="SOOTHE",
                talk_mode="soothe",
                value_gradient_override={"attachment_bias": 0.9},
            ),
            MiniWorldStep(
                name="share_father_story",
                narrative="Talking about late father brings back stricter energy.",
                salient_entities=["father", "photo", "ritual"],
                context_tags=["home", "memory", "boundary"],
                hazard_score=0.22,
                hazard_sources=["grief_wave"],
                chaos=0.34,
                risk=0.28,
                tom_cost=0.35,
                valence=-0.08,
                arousal=0.44,
                stress=0.42,
                love=0.18,
                breath_ratio=0.61,
                heart_rate=0.63,
                log_episode=True,
                action="WAIT",
                talk_mode="ask",
                value_gradient_override={"social_bias": 0.62, "attachment_bias": 0.78},
            ),
            MiniWorldStep(
                name="rest_after_memory",
                narrative="Family collectively exhales; calm anchors return.",
                salient_entities=["mother", "altar", "window"],
                context_tags=["home", "evening", "rest"],
                hazard_score=0.09,
                hazard_sources=["emotional_fatigue"],
                chaos=0.16,
                risk=0.08,
                tom_cost=0.2,
                valence=0.35,
                arousal=0.18,
                stress=0.12,
                love=0.45,
                breath_ratio=0.42,
                heart_rate=0.46,
                log_episode=True,
                action="SOOTHE",
                talk_mode="soothe",
            ),
        ],
    )

    workplace = MiniWorldScenario(
        name="workplace_safety",
        description="Factory floor scenario focused on hazard anticipation.",
        steps=[
            MiniWorldStep(
                name="shift_briefing",
                narrative="Supervisor reviews hazard map; everyone slightly tense.",
                salient_entities=["manager", "hazard_board", "crew"],
                context_tags=["work", "briefing", "safety"],
                hazard_score=0.3,
                hazard_sources=["forklift_route"],
                chaos=0.25,
                risk=0.35,
                tom_cost=0.28,
                valence=0.05,
                arousal=0.36,
                stress=0.33,
                breath_ratio=0.55,
                heart_rate=0.57,
                action="OBSERVE",
                talk_mode="ask",
            ),
            MiniWorldStep(
                name="forklift_pass",
                narrative="Unexpected forklift swing causes spike in caution.",
                salient_entities=["forklift", "operator", "cargo"],
                context_tags=["work", "hazard", "avoid"],
                hazard_score=0.65,
                hazard_sources=["blind_corner", "noise"],
                chaos=0.58,
                risk=0.62,
                tom_cost=0.45,
                valence=-0.12,
                arousal=0.6,
                stress=0.58,
                breath_ratio=0.7,
                heart_rate=0.74,
                log_episode=True,
                action="AVOID",
                talk_mode="watch",
            ),
            MiniWorldStep(
                name="assist_peer",
                narrative="Colleague freezes; agent steps closer to help calmly.",
                salient_entities=["coworker", "safety_line", "toolkit"],
                context_tags=["work", "help", "social"],
                hazard_score=0.4,
                hazard_sources=["stress_echo"],
                chaos=0.32,
                risk=0.38,
                tom_cost=0.4,
                valence=0.12,
                arousal=0.38,
                stress=0.34,
                love=0.2,
                breath_ratio=0.5,
                heart_rate=0.52,
                log_episode=True,
                action="HELP",
                talk_mode="ask",
                value_gradient_override={"social_bias": 0.7, "survival_bias": 0.65},
            ),
        ],
    )

    scenarios = [commute, family, workplace]
    return {scenario.name: scenario for scenario in scenarios}


DEFAULT_SCENARIOS = build_default_scenarios()
