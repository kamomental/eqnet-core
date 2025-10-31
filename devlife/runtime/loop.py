"""Developmental runtime loop orchestrating wake/sleep cycles."""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Mapping, Optional
import uuid

import numpy as np

from devlife.logging.canary_logger import CanaryLogger
from devlife.value.model import compute_value_summary
from emot_terrain_lab.models.ssmax_encoder import register_assoc_health_hook, register_assoc_logger
from runtime.config import RuntimeCfg, load_runtime_cfg

logger = logging.getLogger(__name__)

try:  # Telemetry is optional for headless test runs
    from telemetry import event as telemetry_event
except Exception:  # pragma: no cover
    telemetry_event = None


@dataclass
class StageConfig:
    name: str
    duration_steps: int
    objectives: List[Dict[str, Any]] = field(default_factory=list)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: bool = True


@dataclass
class SleepConfig:
    interval_steps: int
    operations: List[Dict[str, Any]] = field(default_factory=list)


class DevelopmentLoop:
    """High-level orchestrator for genome ↁEdevelopment ↁEculture cycles."""

    def __init__(
        self,
        body,
        grn,
        policy,
        composer,
        archive,
        *,
        stages: Iterable[StageConfig],
        sleep: SleepConfig,
        log_hook: Optional[Any] = None,
        alert_logger: Optional[Any] = None,
        router: Optional[Any] = None,
        ignite_delta_R_thresh: float = 0.12,
        ignite_entropy_z_thresh: float = -1.0,
        ignite_ms_default: int = 250,
        assoc_link=None,
        selfother=None,
        mood_integrator=None,
        sim_plan=None,
        culture_logger=None,
        meta_confidence=None,
        aesthetic_guard=None,
        storygraph=None,
        theory_of_mind=None,
        value_committee=None,
        telemetry_hook: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
        field_metrics_source: Optional[Callable[[], Optional[Dict[str, float]]]] = None,
        runtime_cfg: RuntimeCfg | None = None,
    ) -> None:
        self.body = body
        self.grn = grn
        self.policy = policy
        self.composer = composer
        self.archive = archive
        self.stages = list(stages)
        self.sleep = sleep
        self.log_hook = log_hook
        self.alert_logger = alert_logger
        self.router = router
        self._runtime_cfg = runtime_cfg or load_runtime_cfg()
        self._emotion_cfg = getattr(self._runtime_cfg, "emotion", None)
        self._culture_cfg = getattr(self._runtime_cfg, "culture", None)
        self._valence_w_rho = float(getattr(self._emotion_cfg, "valence_w_rho", 1.0))
        self._valence_w_s = float(getattr(self._emotion_cfg, "valence_w_s", 1.0))
        self._resonance_k = float(getattr(self._emotion_cfg, "resonance_k", 0.05))
        self._affective_log_path = Path(getattr(self._emotion_cfg, "affective_log_path", "memory/affective_log.jsonl"))
        self._affective_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Ignition thresholds (tunable for dry-runs)
        self._ignite_delta_R_thresh = float(ignite_delta_R_thresh)
        self._ignite_entropy_z_thresh = float(ignite_entropy_z_thresh)
        self._ignite_ms_default = int(ignite_ms_default)
        self._step_counter = 0
        self._prev_sensor_stats: Dict[str, float] | None = None
        self.assoc_link = assoc_link
        self.selfother = selfother
        self.mood_integrator = mood_integrator
        self.sim_plan = sim_plan
        self.culture_logger = culture_logger
        self.meta_confidence = meta_confidence
        self.aesthetic_guard = aesthetic_guard
        self.storygraph = storygraph
        self.theory_of_mind = theory_of_mind
        self.value_committee = value_committee
        self._last_hormones: Dict[str, float] = {}
        # Rolling stats for ignition-index components
        self._entropy_mean: float | None = None
        self._entropy_var: float = 0.0
        self._entropy_n: int = 0
        self._prev_reward: float | None = None
        self._ignite_alerts: int = 0
        self._I_ema: float = 0.0
        self._episode_id: str = str(uuid.uuid4())
        self._delta_R_bounds: tuple[float, float] = (-0.25, 0.25)
        self._ignite_monotonic_violations: Dict[str, int] = {"S": 0, "R": 0}
        self._last_entropy_norm: float | None = None
        self._last_delta_R_norm: float | None = None
        self._last_s_term: float | None = None
        self._last_r_term: float | None = None
        self._last_ignition_terms: Dict[str, float] = {}
        self._gate_state: str = "forward"
        self._gate_dwell_counter: int = 0
        self._field_metrics_source = field_metrics_source
        self._field_metrics_queue: Deque[Dict[str, float]] = deque()
        self._field_metrics_log_iter: Optional[Iterator[Dict[str, float]]] = None
        self._telemetry_hook = telemetry_hook or telemetry_event
        self._field_metrics_backlog: Optional[Dict[str, float]] = None
        replay_cfg = self._runtime_cfg.replay
        self._prev_I_value: float | None = None
        self._rho_resonance_bias: float = 0.0
        self._prev_mood_vec: np.ndarray | None = None
        self._prev_prev_mood_vec: np.ndarray | None = None
        self._last_delta_m: float | None = None
        self._last_jerk: float | None = None
        self._assoc_canary_logger: Optional[CanaryLogger] = None
        self._assoc_score_spike_run: int = 0
        self._assoc_dm_spike_run: int = 0
        self._assoc_jerk_spike_run: int = 0
        self._assoc_icl_drop_run: int = 0
        self._assoc_score_threshold: float = 12.0
        self._assoc_dm_threshold: float = 0.35
        self._assoc_jerk_threshold: float = 0.2
        self._assoc_icl_baseline: Optional[float] = None
        self._assoc_icl_at1: Optional[float] = None
        self._assoc_icl_at3: Optional[float] = None
        self._assoc_icl_base_at1: Optional[float] = None
        self._assoc_icl_base_at3: Optional[float] = None
        try:
            self._assoc_canary_logger = CanaryLogger("logs/canary/assoc_kernel.jsonl")
            register_assoc_logger(self._assoc_canary_logger.write, self._assoc_log_context)
        except Exception:
            self._assoc_canary_logger = None
            register_assoc_logger(None)
        register_assoc_health_hook(self._assoc_guard_hook)
        self._field_sample_every = max(1, int(getattr(replay_cfg, "sample_every", 1)))
        self._field_log_sample_index = 0
        self._min_field_interval = max(0.0, float(getattr(replay_cfg, "min_interval_ms", 0)) / 1000.0)
        self._last_log_emit_time: float | None = None
        # If both are provided, wire alerts ↁErouter.downshift callback
        if getattr(self, "alert_logger", None) is not None and getattr(self, "router", None) is not None:
            try:
                if getattr(self.alert_logger, "downshift_fn", None) is None:
                    self.alert_logger.downshift_fn = getattr(self.router, "downshift", None)
            except Exception:
                pass

    def run(self) -> None:
        """Entry point: iterate over developmental stages with sleep cycles."""
        for stage in self.stages:
            self._run_stage(stage)
        print("Development cycle completed.")

    # ------------------------------------------------------------------ internals
    def _run_stage(self, stage: StageConfig) -> None:
        print(f"[{stage.name}] start for {stage.duration_steps} steps")
        for local_step in range(stage.duration_steps):
            self._step_counter += 1
            sensors_prior = self.body.observe()
            state_stats_prior = self.body.state_stats()
            sensory_stats_prior = self._sensor_stats(sensors_prior, state_stats_prior)
            hormones_prior = self.grn.forward(state_stats_prior, sensory_stats_prior)
            actuator_map = self.policy.act(sensors_prior, hormones_prior)
            actuator_array = self._actuator_map_to_array(actuator_map)
            # Apply autonomy gating on actuators if router present
            if self.router is not None:
                level = int(getattr(self.router, "level", 0)) if hasattr(self.router, "level") else 0
                gain = {0: 0.0, 1: 0.6, 2: 0.8, 3: 1.0}.get(level, 1.0)
                actuator_array = actuator_array * float(gain)
            sensors_post = self.body.step(actuator_array)
            state_stats_post = self.body.state_stats()
            sensory_stats_post = self._sensor_stats(sensors_post, state_stats_post)
            hormones_post = self.grn.forward(state_stats_post, sensory_stats_post)
            mood_out = None
            if self.mood_integrator is not None:
                mood_out = self.mood_integrator.update(
                    hormones_post.get("H_valence", 0.0),
                    hormones_post.get("H_arousal", 0.0),
                    reward=0.0,
                )
            # Mood metrics (for downstream gating)
            mood_metrics = {
                "mood_v": float(mood_out.get("mood_v", 0.0)) if mood_out else 0.0,
                "mood_a": float(mood_out.get("mood_a", 0.0)) if mood_out else 0.0,
                "mood_effort": 0.0,
                "mood_uncertainty": 0.0,
            }
            mood_vec = np.array(
                [
                    mood_metrics["mood_v"],
                    mood_metrics["mood_a"],
                    mood_metrics["mood_effort"],
                    mood_metrics["mood_uncertainty"],
                ],
                dtype=np.float32,
            )
            delta_m: float | None = None
            jerk: float | None = None
            if self._prev_mood_vec is not None:
                try:
                    delta_m = float(np.linalg.norm(mood_vec - self._prev_mood_vec))
                except Exception:
                    delta_m = None
            if self._prev_prev_mood_vec is not None and self._prev_mood_vec is not None:
                try:
                    jerk_vec = mood_vec - 2.0 * self._prev_mood_vec + self._prev_prev_mood_vec
                    jerk = float(np.linalg.norm(jerk_vec))
                except Exception:
                    jerk = None
            self._prev_prev_mood_vec = self._prev_mood_vec
            self._prev_mood_vec = mood_vec
            self._last_delta_m = delta_m
            self._last_jerk = jerk
            tokens = self.composer.bodylex_map(self.body.snapshot())
            utterance, delta_aff = self.composer.dialog(
                body_tokens=tokens,
                affect=hormones_post,
                world_tokens=[],
            )
            assoc_tokens, link_strength = [], 0.0
            if self.assoc_link is not None:
                assoc_tokens, link_strength = self.assoc_link.encode(self.body.snapshot())
            plan_info = None
            if self.sim_plan is not None:
                plan_info = self.sim_plan.simulate(self.body.snapshot())
            self_other = None
            if self.selfother is not None:
                ext_event = int(bool(sensors_post.get("events", 0)))
                self_other = self.selfother.classify(
                    state_stats_post.get("mean", 0.0),
                    state_stats_post.get("std", 0.0),
                    ext_event,
                )
            meta_out = None
            if self.meta_confidence is not None:
                logits = np.asarray(list(actuator_map.values()), dtype=np.float32)
                if logits.size == 0:
                    logits = np.zeros(1, dtype=np.float32)
                prediction_error = abs(
                    hormones_post.get("H_valence", 0.0) - self._last_hormones.get("H_valence", 0.0)
                )
                meta_out = self.meta_confidence.compute(logits, prediction_error)
            taste_out = None
            if self.aesthetic_guard is not None and mood_out is not None:
                taste_out = self.aesthetic_guard.evaluate(
                    mood_out.get("mood_v", 0.0),
                    [stage.name],
                )
            chosen_tokens = assoc_tokens if assoc_tokens else tokens
            story_summary = None
            if self.storygraph is not None:
                story_summary = self.storygraph.update(
                    {
                        "stage": stage.name,
                        "tokens": chosen_tokens,
                        "mood": mood_out,
                    }
                )
            tom_out = None
            if self.theory_of_mind is not None:
                tom_out = self.theory_of_mind.update(
                    {
                        "self_event": self_other.get("self", 0.0) if self_other else 0.0,
                        "other_event": self_other.get("other", 0.0) if self_other else 0.0,
                    }
                )
                # Router upshift with ToM hysteresis if available
                if self.router is not None:
                    try:
                        trust_s = tom_out.get("intent_trust_smoothed", tom_out.get("intent_trust", 0.5))
                        if hasattr(self.router, "maybe_upshift"):
                            self.router.maybe_upshift(float(trust_s))
                    except Exception:
                        pass
            value_vote = None
            if self.value_committee is not None:
                value_vote = self.value_committee.vote(
                    {
                        "mood_v": mood_out.get("mood_v", 0.0) if mood_out else 0.0,
                        "self_vs_other": (self_other.get("self", 0.0) - self_other.get("other", 0.0))
                        if self_other
                        else 0.0,
                        "taste_score": taste_out.get("taste_score", 0.0) if taste_out else 0.0,
                    }
                )
            if self.culture_logger is not None and mood_out is not None and stage.name.lower().startswith("social"):
                self.culture_logger.log(
                    speaker="self",
                    peer="peer",
                    delta_aff=mood_out.get("mood_v", 0.0),
                    tags=[stage.name],
                )
            # Field metrics (∥∇Ψ∥, ∥∂Ψ/∂t∥, ΔΨ + bounded S/H/ρ proxies)
            field_metrics = self._compute_field_metrics(sensors_prior, sensors_post, state_stats_post)

            # Ignition-Index components
            entropy_z = self._entropy_z(sensors_post)
            delta_R, I_value, ignite_ms, ignite_trigger = self._ignition_index(
                sensory_stats_post,
                entropy_z,
                field_metrics=field_metrics,
            )
            valence = self._compute_valence(field_metrics.get("S", 0.5), field_metrics.get("rho", 0.5))
            field_metrics["valence"] = valence
            dt_ms = getattr(self._runtime_cfg.replay, "min_interval_ms", 1)
            dt_sec = max(float(dt_ms) / 1000.0, 1e-3)
            arousal = self._compute_arousal(I_value, dt_sec=dt_sec)
            field_metrics["arousal"] = arousal
            self._emit_telemetry(
                "field.metrics",
                {
                    "stage": stage.name,
                    "step": self._step_counter,
                    "S": field_metrics.get("S"),
                    "H": field_metrics.get("H"),
                    "rho": field_metrics.get("rho"),
                    "Ignition": I_value,
                    "delta_R": delta_R,
                    "valence": valence,
                    "arousal": arousal,
                    "field_source": field_metrics.get("field_source", "proxy"),
                    "delta_aff": float(delta_aff) if isinstance(delta_aff, (int, float)) else None,
                    "gate_level": getattr(self.router, "level", None) if self.router is not None else None,
                    "ignite_trigger": ignite_trigger,
                    "gate_state": self._gate_state,
                    "delta_m": delta_m,
                    "jerk": jerk,
                },
            )
            context_summary = None
            if isinstance(utterance, str):
                context_summary = utterance[:160]
                self._log_affective_episode(
                    {
                        "ts": time.time(),
                        "stage": stage.name,
                        "step": self._step_counter,
                        "S": field_metrics.get("S"),
                        "H": field_metrics.get("H"),
                        "rho": field_metrics.get("rho"),
                        "I": I_value,
                        "valence": valence,
                        "arousal": arousal,
                        "delta_R": delta_R,
                        "field_source": field_metrics.get("field_source", "proxy"),
                        "delta_aff": float(delta_aff) if isinstance(delta_aff, (int, float)) else None,
                        "culture_tag": getattr(self._culture_cfg, "tag", "default"),
                        "politeness": float(getattr(self._culture_cfg, "politeness", 0.0)),
                        "intimacy": float(getattr(self._culture_cfg, "intimacy", 0.0)),
                        "delta_m": delta_m,
                        "jerk": jerk,
                        "context": {"utterance": context_summary},
                    }
                )

            social_alignment = 0.5
            if tom_out:
                social_alignment = float(
                    tom_out.get("intent_trust_smoothed", tom_out.get("intent_trust", 0.5))
                )
            if self_other:
                diff = abs(
                    float(self_other.get("self", 0.0)) - float(self_other.get("other", 0.0))
                )
                social_alignment = 0.5 * social_alignment + 0.5 * (1.0 - diff)
            coherence_score = 0.5
            if isinstance(plan_info, dict) and plan_info.get("match_score") is not None:
                coherence_score = float(plan_info.get("match_score", 0.5))
            elif isinstance(story_summary, dict) and story_summary.get("coherence") is not None:
                coherence_score = float(story_summary.get("coherence", 0.5))
            qualia_consistency = 0.5
            if taste_out and "taste_score" in taste_out:
                qualia_consistency = 0.5 + 0.5 * float(taste_out.get("taste_score", 0.0))
            norm_penalty = 0.0
            if value_vote:
                norm_penalty = max(0.0, -float(value_vote.get("score", 0.0)))
            value_summary = compute_value_summary(
                extrinsic_signal=float(delta_R),
                novelty_signal=float(sensory_stats_post.get("novelty_signal", 0.0)),
                social_alignment=social_alignment,
                coherence_score=coherence_score,
                homeostasis_error=float(sensory_stats_post.get("homeo_error", 0.0)),
                qualia_consistency=qualia_consistency,
                norm_penalty=norm_penalty,
                metadata={"stage": stage.name},
            )

            # Global foregrounding score G(t): combine I + proxies
            wm_occupancy = float(min(1.0, max(0.0, (len(chosen_tokens) if chosen_tokens else 0) / 16.0)))
            selfref_rate = 0.0
            if isinstance(utterance, str) and utterance:
                txt = utterance.lower()
                self_tokens = [" i ", " me ", " my ", " myself ", "わたし", "ぼく", "私", "自分"]
                hits = sum(txt.count(k) for k in self_tokens)
                selfref_rate = float(min(1.0, hits / 4.0))
            cf_lead = 0.0
            if isinstance(plan_info, dict) and plan_info.get("match_score") is not None:
                try:
                    cf_lead = float(max(0.0, min(1.0, plan_info.get("match_score", 0.0))))
                except Exception:
                    cf_lead = 0.0
            w1, w2, w3, w4 = 0.5, 0.2, 0.15, 0.15
            gscore = float(max(0.0, min(1.0, w1 * (0.5 + 0.5 * np.tanh(I_value)) + w2 * wm_occupancy + w3 * selfref_rate + w4 * cf_lead)))

            # Router observe and autonomy update (uses R/rho proxies)
            if self.router is not None:
                try:
                    R_proxy = self._router_sync_proxy(sensors_post)
                    rho_proxy = float(field_metrics.get("laplacian_abs_mean", sensors_post.get("kappa", 0.0)))
                    from runtime.router import RouterMetrics  # local import to avoid hard dep at module import

                    metrics_obj = RouterMetrics(rho=rho_proxy, synchrony=R_proxy, misfires=0, incidents=0)
                    self.router.observe(None, None, metrics_obj)  # core_state/stance unused in observe
                except Exception:
                    pass

            ignite_meta = {
                "entropy_z": float(entropy_z),
                "delta_R": float(delta_R),
                "I": float(I_value),
                "ignite_ms": int(ignite_ms) if ignite_ms is not None else None,
                "trigger": bool(ignite_trigger),
                "alerts": int(self._ignite_alerts),
                "violations": dict(self._ignite_monotonic_violations),
                "gate_state": self._gate_state,
            }
            if self._last_ignition_terms:
                for key, value in self._last_ignition_terms.items():
                    if isinstance(value, (int, float)):
                        ignite_meta[key] = float(value)
                    else:
                        ignite_meta[key] = value

            record = self._build_episode_record(
                stage.name,
                local_step,
                sensors_post,
                sensory_stats_post,
                hormones_post,
                actuator_map,
                utterance,
                delta_aff,
                chosen_tokens,
                link_strength,
                self_other,
                mood_out,
                plan_info,
                meta_out,
                taste_out,
                story_summary,
                tom_out,
                value_vote,
                value_summary,
                field_metrics,
                ignite_meta,
            )
            record["episode_id"] = self._episode_id
            # Attach mood metrics at top-level for downstream hubs
            record["mood_metrics"] = mood_metrics
            # Attach foregrounding signals
            try:
                record["foreground"] = {"G": gscore, "wm": wm_occupancy, "self_ref": selfref_rate, "cf_lead": cf_lead}
            except Exception:
                pass
            # Attach autonomy level if available
            if self.router is not None and hasattr(self.router, "level"):
                try:
                    record["autonomy"] = int(self.router.level)
                except Exception:
                    record["autonomy"] = None
            self.archive.log(record)
            self._last_hormones = hormones_post
            if self.log_hook:
                self.log_hook(record)
            if self.alert_logger is not None:
                try:
                    self.alert_logger.evaluate_and_log(record)
                except Exception:
                    pass
            if stage.checkpoints and self._should_sleep():
                self._sleep_cycle(stage.name)
        print(f"[{stage.name}] complete")

    def _should_sleep(self) -> bool:
        return self._step_counter % self.sleep.interval_steps == 0

    def _sleep_cycle(self, stage_name: str) -> None:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        print(f"[sleep] entering reversible distillation at {timestamp} (stage={stage_name})")
        for op in self.sleep.operations:
            name = op.get("name", "<unknown>")
            print(f"  - executing sleep op: {name}")
            # Placeholder: integrate reversible distillation / archive flush here.
            if name == "archive_flush" and self.culture_logger is not None:
                self.culture_logger.flush()
        print("[sleep] exiting")

    def _build_episode_record(
        self,
        stage: str,
        local_step: int,
        sensors: Any,
        sensory_stats: Dict[str, float],
        hormones: Any,
        actuators: Dict[str, float],
        utterance: Any,
        delta_aff: Any,
        tokens: List[int],
        link_strength: float,
        self_other: Optional[Dict[str, int]],
        mood_out: Optional[Dict[str, float]],
        plan_info: Optional[Dict[str, Any]],
        meta_out: Optional[Dict[str, float]],
        taste_out: Optional[Dict[str, float]],
        story_summary: Optional[Dict[str, Any]],
        tom_out: Optional[Dict[str, float]],
        value_vote: Optional[Dict[str, float]],
        value_summary: Optional[Dict[str, Any]],
        field_metrics: Dict[str, float],
        ignition: Dict[str, Any],
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "stage": stage,
            "step": local_step,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sensors": {
                "stats": sensors.get("stats", {}),
                "kappa": sensors.get("kappa", 0.0),
            },
            "sensory_stats": sensory_stats,
            "hormones": hormones,
            "actuators": actuators,
            "utterance": utterance,
            "delta_affect": delta_aff,
            "link_strength": link_strength,
            "field": field_metrics,
            "ignite": ignition,
        }
        if tokens:
            record["tokens"] = tokens
        if self_other:
            record.update(
                {
                    "self_event": self_other.get("self", 0),
                    "other_event": self_other.get("other", 0),
                    "conflict": self_other.get("conflict", 0),
                }
            )
        if mood_out:
            record["mood"] = mood_out
        if plan_info:
            record["plan"] = plan_info
        if meta_out:
            record["meta"] = meta_out
        if taste_out:
            record["taste"] = taste_out
        if story_summary:
            record["story"] = story_summary
        if tom_out:
            record["tom"] = tom_out
        if value_summary:
            record["value_summary"] = value_summary
        if value_vote:
            record["value_committee"] = value_vote
            record.setdefault("value", value_vote)
        elif value_summary:
            record["value"] = value_summary
        return record

    def _sensor_stats(self, sensors: Dict[str, Any], state_stats: Dict[str, float]) -> Dict[str, float]:
        stats_dict = sensors.get("stats", {})
        stats = {
            "stats_mean": float(stats_dict.get("mean", 0.0)),
            "stats_var": float(stats_dict.get("var", 0.0)),
            "stats_edge": float(stats_dict.get("edge", 0.0)),
            "kappa": float(sensors.get("kappa", 0.0)),
            "energy": float(state_stats.get("energy", 0.0)),
        }
        homeo_target = 0.0
        stats["homeo_error"] = stats["stats_mean"] - homeo_target
        if self._prev_sensor_stats is None:
            novelty = 0.0
        else:
            novelty = abs(stats["stats_mean"] - self._prev_sensor_stats.get("stats_mean", 0.0))
            novelty += abs(stats["stats_edge"] - self._prev_sensor_stats.get("stats_edge", 0.0))
        stats["novelty_signal"] = novelty
        self._prev_sensor_stats = stats.copy()
        return stats

    def _actuator_map_to_array(self, actuators: Dict[str, float]) -> np.ndarray:
        ordered_keys = sorted(actuators.keys())
        values = [float(actuators[key]) for key in ordered_keys]
        channels = getattr(self.body, "state", np.zeros((len(values),))).shape[0]
        vector = np.array(values, dtype=np.float32)
        if vector.size < channels:
            vector = np.pad(vector, (0, channels - vector.size))
        else:
            vector = vector[:channels]
        return vector.reshape(channels, 1, 1)

    # ------------------------------ added metrics helpers
    def _compute_field_metrics(
        self,
        sensors_prior: Dict[str, Any],
        sensors_post: Dict[str, Any],
        state_stats_post: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        ch_prior = np.asarray(sensors_prior.get("channels"))
        ch_post = np.asarray(sensors_post.get("channels"))
        metrics: Dict[str, float] = {}
        if ch_prior is None or ch_post is None or ch_post.size == 0:
            return {"grad_norm": 0.0, "dstate_norm": 0.0, "laplacian_abs_mean": float(sensors_post.get("kappa", 0.0))}
        # gradient on first channel
        c0 = ch_post[0]
        dy = np.roll(c0, -1, axis=0) - c0
        dx = np.roll(c0, -1, axis=1) - c0
        grad_norm = float(np.sqrt((dx * dx + dy * dy)).mean())
        # time derivative norm
        dstate = ch_post - ch_prior
        dstate_norm = float(np.sqrt(np.square(dstate)).mean())
        # laplacian proxy: reuse kappa from sensors_post
        lap_abs_mean = float(sensors_post.get("kappa", 0.0))
        metrics["grad_norm"] = grad_norm
        metrics["dstate_norm"] = dstate_norm
        metrics["laplacian_abs_mean"] = lap_abs_mean

        channel_entropy = self._entropy_norm_from_channel(ch_post[0])
        energy_scalar = 0.0
        if state_stats_post:
            energy_scalar = float(state_stats_post.get("energy", 0.0))
        enthalpy_norm = float(np.clip(0.5 + 0.5 * np.tanh(energy_scalar), 0.0, 1.0))
        rho_norm = float(np.clip(1.0 - math.tanh(grad_norm), 0.0, 1.0))
        if self._rho_resonance_bias:
            rho_norm = float(np.clip(rho_norm + self._rho_resonance_bias, 0.0, 1.0))
            self._rho_resonance_bias *= 0.9
        else:
            self._rho_resonance_bias *= 0.9
        metrics["entropy_norm"] = channel_entropy
        metrics["enthalpy_norm"] = enthalpy_norm
        metrics["rho_norm"] = rho_norm
        metrics["S"] = channel_entropy
        metrics["H"] = enthalpy_norm
        metrics["rho"] = rho_norm

        merged = self._merge_external_field_metrics(metrics)
        if "valence" not in merged:
            merged["valence"] = self._compute_valence(merged.get("S", 0.5), merged.get("rho", 0.5))
        return merged

    def _entropy_norm_from_channel(self, channel: Optional[np.ndarray]) -> float:
        if channel is None or channel.size == 0:
            return 0.5
        hist, _ = np.histogram(channel, bins=32, range=(-1.0, 1.0), density=True)
        hist = hist + 1e-9
        p = hist / hist.sum()
        entropy = float(-(p * np.log(p)).sum())
        max_entropy = math.log(len(hist))
        if max_entropy <= 0.0:
            return 0.5
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    def ingest_field_metrics(self, metrics: Mapping[str, float]) -> None:
        """Accept a real field metrics dict coming from the terrain layer."""
        self._field_metrics_queue.append(dict(metrics))

    def load_field_metrics_log(self, path: str | Path) -> None:
        """Replay recorded field metrics from a JSON/JSONL log."""
        entries = self._read_field_metrics_log(path)
        self._field_metrics_log_iter = iter(entries)
        self._field_log_sample_index = 0
        self._field_metrics_backlog = None

    def _merge_external_field_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        external = self._pull_field_metrics()
        if external:
            for key, value in external.items():
                if key == "field_source":
                    continue
                metrics[key] = value
            metrics["field_source"] = external.get("field_source", "external")
        else:
            metrics.setdefault("field_source", "proxy")
        metrics.setdefault("S", metrics.get("entropy_norm", 0.5))
        metrics.setdefault("H", metrics.get("enthalpy_norm", 0.5))
        metrics.setdefault("rho", metrics.get("rho_norm", 0.5))
        return metrics

    def _compute_valence(self, entropy_norm: float, rho_norm: float) -> float:
        try:
            val = self._valence_w_rho * float(rho_norm) - self._valence_w_s * float(entropy_norm)
        except Exception:
            val = 0.0
        return float(np.clip(val, -1.0, 1.0))

    def _compute_arousal(self, I_value: float, dt_sec: float) -> float:
        prev = self._prev_I_value
        self._prev_I_value = float(I_value)
        if prev is None:
            return 0.0
        try:
            delta = float(I_value) - float(prev)
            excitation = max(0.0, delta) / max(dt_sec, 1e-6)
            level = max(0.0, float(I_value))
            arousal = 0.2 * min(excitation, 1.0) + 0.8 * level
        except Exception:
            arousal = 0.0
        return float(min(max(0.0, arousal), 1.0))

    def ingest_resonance_from_peer(self, peer_I: float, k_res: Optional[float] = None) -> None:
        coeff = float(k_res) if k_res is not None else self._resonance_k
        if coeff <= 0.0:
            return
        base = self._prev_I_value
        if base is None:
            return
        try:
            delta = float(peer_I) - float(base)
        except Exception:
            return
        bias = self._rho_resonance_bias + coeff * delta
        self._rho_resonance_bias = float(np.clip(bias, -0.25, 0.25))

    def _log_affective_episode(self, payload: Dict[str, Any]) -> None:
        try:
            self._affective_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._affective_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _assoc_log_context(self) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {"step": self._step_counter}
        if self._last_delta_m is not None:
            ctx["delta_m"] = self._last_delta_m
        if self._last_jerk is not None:
            ctx["jerk"] = self._last_jerk
        if self._assoc_icl_at1 is not None:
            ctx["icl_at1"] = self._assoc_icl_at1
        if self._assoc_icl_at3 is not None:
            ctx["icl_at3"] = self._assoc_icl_at3
        if self._assoc_icl_base_at1 is not None:
            ctx["icl_base_at1"] = self._assoc_icl_base_at1
        if self._assoc_icl_base_at3 is not None:
            ctx["icl_base_at3"] = self._assoc_icl_base_at3
        if self._assoc_icl_baseline is not None:
            ctx["icl_baseline"] = self._assoc_icl_baseline
        ctx["score_threshold"] = self._assoc_score_threshold
        ctx["dm_threshold"] = self._assoc_dm_threshold
        ctx["jerk_threshold"] = self._assoc_jerk_threshold
        return ctx

    def _assoc_guard_hook(self, metrics: Dict[str, Any]) -> Optional[str]:
        action: Optional[str] = None
        self.record_assoc_icl(metrics.get("icl_at1"), metrics.get("icl_at3"))
        max_abs = metrics.get("max_score_abs")
        if isinstance(max_abs, (int, float)) and math.isfinite(float(max_abs)):
            value = float(max_abs)
            if value > self._assoc_score_threshold:
                self._assoc_score_spike_run += 1
            else:
                self._assoc_score_spike_run = 0
            if self._assoc_score_spike_run >= 3:
                self._assoc_score_spike_run = 0
                return "fallback_attention"
        else:
            self._assoc_score_spike_run = 0

        delta_m = metrics.get("delta_m", self._last_delta_m)
        if isinstance(delta_m, (int, float)) and math.isfinite(float(delta_m)):
            delta_val = float(delta_m)
            if delta_val > self._assoc_dm_threshold:
                self._assoc_dm_spike_run += 1
            else:
                self._assoc_dm_spike_run = 0
            if self._assoc_dm_spike_run >= 3:
                self._assoc_dm_spike_run = 0
                return "broaden_bandwidth"
        else:
            self._assoc_dm_spike_run = 0

        jerk_val = metrics.get("jerk", self._last_jerk)
        if isinstance(jerk_val, (int, float)) and math.isfinite(float(jerk_val)):
            jerk_val = float(jerk_val)
            if jerk_val > self._assoc_jerk_threshold:
                self._assoc_jerk_spike_run += 1
            else:
                self._assoc_jerk_spike_run = 0
            if self._assoc_jerk_spike_run >= 3:
                self._assoc_jerk_spike_run = 0
                return "broaden_bandwidth"
        else:
            self._assoc_jerk_spike_run = 0

        icl_at3 = metrics.get("icl_at3", self._assoc_icl_at3)
        baseline_at3 = self._assoc_icl_base_at3
        if baseline_at3 is None:
            baseline_at3 = self._assoc_icl_baseline
        if (
            isinstance(icl_at3, (int, float))
            and math.isfinite(float(icl_at3))
            and baseline_at3 is not None
            and math.isfinite(float(baseline_at3))
        ):
            icl_val = float(icl_at3)
            baseline_at3 = float(baseline_at3)
            if icl_val < baseline_at3 - 3.0:
                self._assoc_icl_drop_run += 1
            else:
                self._assoc_icl_drop_run = 0
            if self._assoc_icl_drop_run >= 2:
                self._assoc_icl_drop_run = 0
                return "narrow_bandwidth"
        else:
            self._assoc_icl_drop_run = 0
        return action

    def record_assoc_icl(
        self,
        icl_at1: Optional[float] = None,
        icl_at3: Optional[float] = None,
        *,
        update_baseline: bool = False,
    ) -> None:
        """Update ICL metrics used by assoc-kernel guards."""

        updated = False
        if isinstance(icl_at1, (int, float)) and math.isfinite(float(icl_at1)):
            value = float(icl_at1)
            self._assoc_icl_at1 = value
            updated = True
            if self._assoc_icl_base_at1 is None or update_baseline:
                self._assoc_icl_base_at1 = value
        if isinstance(icl_at3, (int, float)) and math.isfinite(float(icl_at3)):
            value = float(icl_at3)
            self._assoc_icl_at3 = value
            updated = True
            if self._assoc_icl_base_at3 is None or update_baseline:
                self._assoc_icl_base_at3 = value
        if updated and (self._assoc_icl_baseline is None or update_baseline):
            if self._assoc_icl_base_at3 is not None:
                self._assoc_icl_baseline = float(self._assoc_icl_base_at3)
            elif self._assoc_icl_at3 is not None:
                self._assoc_icl_baseline = float(self._assoc_icl_at3)
        if update_baseline and self._assoc_icl_baseline is not None:
            self._assoc_icl_drop_run = 0

    def set_assoc_thresholds(
        self,
        *,
        max_score: Optional[float] = None,
        delta_m: Optional[float] = None,
        jerk: Optional[float] = None,
    ) -> None:
        """Allow runtime adjustments of assoc-kernel guard thresholds."""

        if isinstance(max_score, (int, float)) and max_score > 0:
            self._assoc_score_threshold = float(max_score)
        if isinstance(delta_m, (int, float)) and delta_m > 0:
            self._assoc_dm_threshold = float(delta_m)
        if isinstance(jerk, (int, float)) and jerk > 0:
            self._assoc_jerk_threshold = float(jerk)
    def _pull_field_metrics(self) -> Optional[Dict[str, float]]:
        if self._field_metrics_backlog is not None:
            if self._should_emit_log_entry():
                entry = self._field_metrics_backlog
                self._field_metrics_backlog = None
                return entry
            return None
        if self._field_metrics_source is not None:
            try:
                candidate = self._field_metrics_source()
            except Exception:
                candidate = None
            if candidate:
                return self._sanitize_external_field_metrics(candidate, source="callback")
        if self._field_metrics_queue:
            return self._sanitize_external_field_metrics(self._field_metrics_queue.popleft(), source="push")
        if self._field_metrics_log_iter is not None:
            while True:
                try:
                    entry = next(self._field_metrics_log_iter)
                except StopIteration:
                    self._field_metrics_log_iter = None
                    break
                self._field_log_sample_index += 1
                if (self._field_log_sample_index - 1) % self._field_sample_every != 0:
                    continue
                sanitized = self._sanitize_external_field_metrics(entry, source="log")
                if sanitized is None:
                    continue
                if not self._should_emit_log_entry():
                    self._field_metrics_backlog = sanitized
                    return None
                return sanitized
        return None

    def _sanitize_external_field_metrics(self, payload: Mapping[str, float], *, source: str) -> Dict[str, float]:
        data = dict(payload)
        entropy_norm = self._coerce_unit_value(data, ("entropy_norm", "S"), default=0.5)
        enthalpy_norm = self._coerce_unit_value(data, ("enthalpy_norm", "H"), default=0.5)
        rho_norm = self._coerce_unit_value(data, ("rho_norm", "rho"), default=0.5)
        data["entropy_norm"] = entropy_norm
        data["enthalpy_norm"] = enthalpy_norm
        data["rho_norm"] = rho_norm
        data["S"] = entropy_norm
        data["H"] = enthalpy_norm
        data["rho"] = rho_norm
        data["field_source"] = source
        return data

    def _coerce_unit_value(self, payload: Mapping[str, float], keys: Iterable[str], default: float) -> float:
        for key in keys:
            if key in payload:
                try:
                    value = float(payload[key])
                    return float(np.clip(value, 0.0, 1.0))
                except Exception:
                    continue
        return float(default)

    def _read_field_metrics_log(self, path: str | Path) -> List[Dict[str, float]]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Field metrics log not found at {source}")
        text = source.read_text(encoding="utf-8").strip()
        if not text:
            return []
        entries: List[Dict[str, float]] = []
        if text.startswith("["):
            payload = json.loads(text)
            if isinstance(payload, list):
                entries = [dict(row) for row in payload if isinstance(row, dict)]
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    entries.append(dict(row))
        return entries

    def _should_emit_log_entry(self) -> bool:
        if self._min_field_interval <= 0.0:
            return True
        now = time.monotonic()
        if self._last_log_emit_time is None or (now - self._last_log_emit_time) >= self._min_field_interval:
            self._last_log_emit_time = now
            return True
        return False

    def _emit_telemetry(self, event_name: str, payload: Mapping[str, Any]) -> None:
        if self._telemetry_hook is None:
            return
        try:
            self._telemetry_hook(event_name, payload)
        except Exception as exc:  # pragma: no cover - telemetry failures should not stop runtime
            logger.warning("Telemetry hook failed for %s: %s", event_name, exc)

    def _entropy_z(self, sensors_post: Dict[str, Any]) -> float:
        channels = np.asarray(sensors_post.get("channels"))
        if channels is None or channels.size == 0:
            return 0.0
        c0 = channels[0]
        # histogram over [-1, 1]
        hist, _ = np.histogram(c0, bins=32, range=(-1.0, 1.0), density=True)
        hist = hist + 1e-9
        p = hist / hist.sum()
        entropy = float(-(p * np.log(p)).sum())
        # update rolling mean/var (Welford)
        n0 = self._entropy_n
        if n0 == 0:
            self._entropy_mean = entropy
            self._entropy_var = 0.0
            self._entropy_n = 1
            return 0.0
        mean = self._entropy_mean if self._entropy_mean is not None else 0.0
        n1 = n0 + 1
        delta = entropy - mean
        mean_new = mean + delta / n1
        var_new = self._entropy_var + delta * (entropy - mean_new)
        self._entropy_mean = mean_new
        self._entropy_var = var_new
        self._entropy_n = n1
        std = float(np.sqrt(max(var_new / max(1, n1 - 1), 1e-9)))
        z = (entropy - mean_new) / std if std > 1e-6 else 0.0
        return float(z)

    def _ignition_index(
        self,
        sensory_stats_post: Dict[str, float],
        entropy_z: float,
        field_metrics: Optional[Dict[str, float]] = None,
        *,
        lambda_gain: float = 0.5,
        ignite_ms_default: Optional[int] = None,
        delta_R_thresh: Optional[float] = None,
        entropy_z_thresh: Optional[float] = None,
    ) -> tuple[float, float, int | None, bool]:
        # success proxy R_t from homeostasis error (smaller error -> higher R)
        homeo_err = float(sensory_stats_post.get("homeo_error", 0.0))
        R_t = -abs(homeo_err)
        if self._prev_reward is None:
            delta_R = 0.0
        else:
            delta_R = R_t - self._prev_reward
        self._prev_reward = R_t

        metrics = field_metrics or {}
        entropy_norm = metrics.get("entropy_norm")
        if entropy_norm is None:
            entropy_norm = self._last_entropy_norm if self._last_entropy_norm is not None else 0.5
        entropy_norm = float(entropy_norm)
        if not math.isfinite(entropy_norm):
            entropy_norm = 0.5
        entropy_norm = float(np.clip(entropy_norm, 0.0, 1.0))

        delta_R_norm = self._normalize_scalar(delta_R, self._delta_R_bounds)
        w_delta = float(np.clip(lambda_gain, 0.0, 1.0))
        w_entropy = 1.0 - w_delta
        s_term = w_entropy * (1.0 - entropy_norm)
        r_term = w_delta * delta_R_norm
        I_raw = r_term + s_term

        # Keep trace for logging/monotonic guards
        self._last_ignition_terms = {
            "S_norm": entropy_norm,
            "delta_R_norm": delta_R_norm,
            "w_delta_R": w_delta,
            "w_entropy": w_entropy,
            "s_term": s_term,
            "r_term": r_term,
        }
        self._verify_ignition_monotonic(entropy_norm, s_term, delta_R_norm, r_term)
        self._last_entropy_norm = entropy_norm
        self._last_delta_R_norm = delta_R_norm
        self._last_s_term = s_term
        self._last_r_term = r_term

        # One-pole smoothing for I
        alpha = 0.2
        self._I_ema = (1 - alpha) * self._I_ema + alpha * I_raw
        I_value = float(self._I_ema)
        self._update_gate_state(I_value)
        dr_th = self._ignite_delta_R_thresh if delta_R_thresh is None else float(delta_R_thresh)
        ez_th = self._ignite_entropy_z_thresh if entropy_z_thresh is None else float(entropy_z_thresh)
        ignite_trigger = (delta_R > dr_th) and (entropy_z < ez_th) and (entropy_norm <= 0.85)
        ms_default = self._ignite_ms_default if ignite_ms_default is None else int(ignite_ms_default)
        ignite_ms = ms_default if ignite_trigger else None
        if ignite_trigger:
            self._ignite_alerts += 1
        return float(delta_R), float(I_value), ignite_ms, bool(ignite_trigger)

    def _normalize_scalar(self, value: float, bounds: tuple[float, float]) -> float:
        lo, hi = bounds
        if hi <= lo:
            return 0.5
        scaled = (float(value) - lo) / (hi - lo)
        return float(np.clip(scaled, 0.0, 1.0))

    def _verify_ignition_monotonic(
        self,
        entropy_norm: float,
        s_term: float,
        delta_R_norm: float,
        r_term: float,
    ) -> None:
        eps = 1e-6
        if self._last_entropy_norm is not None and entropy_norm > self._last_entropy_norm + eps:
            if self._last_s_term is not None and s_term >= self._last_s_term - eps:
                self._ignite_monotonic_violations["S"] += 1
        if self._last_delta_R_norm is not None and delta_R_norm > self._last_delta_R_norm + eps:
            if self._last_r_term is not None and r_term <= self._last_r_term + eps:
                self._ignite_monotonic_violations["R"] += 1

    def _update_gate_state(self, I_value: float) -> str:
        cfg = self._runtime_cfg.ignition
        theta_on = float(getattr(cfg, "theta_on", 0.6))
        theta_off = float(getattr(cfg, "theta_off", 0.5))
        dwell = max(1, int(getattr(cfg, "dwell_steps", 4)))
        state = self._gate_state
        if state == "forward":
            if I_value >= theta_on:
                self._gate_dwell_counter += 1
                if self._gate_dwell_counter >= dwell:
                    state = "reverse"
                    self._gate_dwell_counter = 0
            else:
                self._gate_dwell_counter = 0
        else:
            if I_value <= theta_off:
                self._gate_dwell_counter += 1
                if self._gate_dwell_counter >= dwell:
                    state = "forward"
                    self._gate_dwell_counter = 0
            else:
                self._gate_dwell_counter = 0
        self._gate_state = state
        return state

    def _router_sync_proxy(self, sensors_post: Dict[str, Any]) -> float:
        """Map curvature proxy to [0,1] synchrony for router.
        Uses 1 - exp(-kappa) for a bounded normalisation.
        """
        kappa = float(sensors_post.get("kappa", 0.0))
        import math

        return float(max(0.0, min(1.0, 1.0 - math.exp(-max(0.0, kappa)))))

