"""Developmental runtime loop orchestrating wake/sleep cycles."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import uuid

import numpy as np

from devlife.value.model import compute_value_summary


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
        # If both are provided, wire alerts → router.downshift callback
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
            # Field metrics (∥∇Ψ∥, ∥∂Ψ/∂t∥, ΔΨ)
            field_metrics = self._compute_field_metrics(sensors_prior, sensors_post)

            # Ignition-Index components
            entropy_z = self._entropy_z(sensors_post)
            delta_R, I_value, ignite_ms, ignite_trigger = self._ignition_index(
                sensory_stats_post,
                entropy_z,
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
                self_tokens = [" i ", " me ", " my ", " myself ", "わたし", "僕", "私", "自分"]
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
                {
                    "entropy_z": float(entropy_z),
                    "delta_R": float(delta_R),
                    "I": float(I_value),
                    "ignite_ms": int(ignite_ms) if ignite_ms is not None else None,
                    "trigger": bool(ignite_trigger),
                    "alerts": int(self._ignite_alerts),
                },
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
    def _compute_field_metrics(self, sensors_prior: Dict[str, Any], sensors_post: Dict[str, Any]) -> Dict[str, float]:
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
        return metrics

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
        # Robust components: clip entropy z and ΔR to reduce outliers
        s_comp = float(np.clip(-float(entropy_z), -2.0, 2.0))
        dr_comp = float(np.clip(float(delta_R), 0.0, 0.2))
        I_raw = s_comp + lambda_gain * dr_comp
        # One-pole smoothing for I
        alpha = 0.2
        self._I_ema = (1 - alpha) * self._I_ema + alpha * I_raw
        I_value = float(self._I_ema)
        dr_th = self._ignite_delta_R_thresh if delta_R_thresh is None else float(delta_R_thresh)
        ez_th = self._ignite_entropy_z_thresh if entropy_z_thresh is None else float(entropy_z_thresh)
        ignite_trigger = (delta_R > dr_th) and (entropy_z < ez_th)
        ms_default = self._ignite_ms_default if ignite_ms_default is None else int(ignite_ms_default)
        ignite_ms = ms_default if ignite_trigger else None
        if ignite_trigger:
            self._ignite_alerts += 1
        return float(delta_R), float(I_value), ignite_ms, bool(ignite_trigger)

    def _router_sync_proxy(self, sensors_post: Dict[str, Any]) -> float:
        """Map curvature proxy to [0,1] synchrony for router.
        Uses 1 - exp(-kappa) for a bounded normalisation.
        """
        kappa = float(sensors_post.get("kappa", 0.0))
        import math

        return float(max(0.0, min(1.0, 1.0 - math.exp(-max(0.0, kappa)))))
