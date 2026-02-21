from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from eqnet_core.models.activation_trace import (
    ActivationNode,
    ActivationTrace,
    ActivationTraceLogger,
    ConfidenceSample,
    ReplayEvent,
)
from eqnet_core.models.scene_frame import (
    AffectSnapshot,
    SceneAgent,
    SceneConstraint,
    SceneFrame,
    SceneObject,
)


@dataclass
class RecallEngineConfig:
    """Heuristics governing recall ignition logging."""

    anchor_gain: float = 0.45
    spread_gain: float = 0.25
    confirm_gain: float = 0.2
    external_gain: float = 0.25
    chain_decay: float = 0.82
    max_chain: int = 5
    min_activation: float = 0.05
    open_loop_gain: float = 0.12
    change_point_gain: float = 0.15
    max_related_per_seed: int = 2


class RecallEngine:
    """Record how anchor cues ignite replay without touching inner replay."""

    def __init__(
        self,
        config: Optional[RecallEngineConfig] = None,
        *,
        log_path: Optional[str] = "logs/activation_traces.jsonl",
        clock=None,
    ) -> None:
        self.config = config or RecallEngineConfig()
        self._clock = clock or time.time
        self._logger = ActivationTraceLogger(log_path)

    def ignite(
        self,
        *,
        ctx_time: Mapping[str, Any],
        plan: Mapping[str, Any],
        seeds: Sequence[Mapping[str, Any]],
        best_choice: Optional[Mapping[str, Any]],
        replay_details: Optional[Mapping[str, Any]] = None,
        field_signals: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[ActivationTrace], List[SceneFrame]]:
        """Return an activation trace + derived scenes for the given context."""

        anchor_label, anchor_strength = self._detect_anchor(ctx_time)
        if not anchor_label and not seeds and not best_choice:
            return None, []

        activation_chain = self._build_activation_chain(seeds, anchor_strength)
        if not activation_chain and not best_choice:
            # No ignition chain to report.
            return None, []

        scene_frames = self._build_scene_frames(
            ctx_time=ctx_time,
            plan=plan,
            seeds=seeds,
            best_choice=best_choice,
            anchor_label=anchor_label,
        )
        confidence_curve = self._build_confidence_curve(
            ctx_time,
            activation_chain,
            anchor_strength,
            field_signals=field_signals,
        )
        replay_events = self._build_replay_events(scene_frames, best_choice)
        trigger_context = self._build_trigger_context(ctx_time, plan)
        metadata = self._metadata_block(replay_details, field_signals)
        notes = self._compose_notes(anchor_label, confidence_curve)
        trace = ActivationTrace(
            trace_id=str(uuid.uuid4()),
            timestamp=float(self._clock()),
            trigger_context=trigger_context,
            anchor_hit=anchor_label,
            activation_chain=activation_chain,
            confidence_curve=confidence_curve,
            replay_events=replay_events,
            notes=notes,
            metadata=metadata,
            scene_frames=scene_frames,
        )
        self._logger.write(trace)
        return trace, scene_frames

    # ------------------------------------------------------------------
    # chain helpers
    # ------------------------------------------------------------------
    def _build_activation_chain(
        self,
        seeds: Sequence[Mapping[str, Any]],
        anchor_strength: float,
    ) -> List[ActivationNode]:
        cfg = self.config
        expanded = self._expand_chain_seeds(seeds)
        scored: List[Tuple[float, Mapping[str, Any]]] = []
        for seed in expanded:
            score = self._score_seed(seed, anchor_strength)
            if score <= 0.0:
                continue
            scored.append((score, seed))
        scored.sort(key=lambda item: item[0], reverse=True)
        chain: List[ActivationNode] = []
        for idx, (score, seed) in enumerate(scored[: cfg.max_chain]):
            decay = cfg.chain_decay ** idx
            activation = max(cfg.min_activation, score * decay)
            node_id = str(
                seed.get("trace_id")
                or seed.get("id")
                or seed.get("meta", {}).get("receipt", {}).get("id")
                or f"seed-{idx}"
            )
            meta = seed.get("meta", {}) or {}
            cue = meta.get("anchor") or meta.get("label")
            metadata = {
                "novelty": self._coerce_meta(meta, ("novelty", "novelty_score")),
                "constraint_weight": self._coerce_meta(
                    meta, ("constraint_weight", "mobility_block")
                ),
                "social": self._coerce_meta(meta, ("social_weight", "attachment")),
                "utility": self._coerce_meta(seed.get("value", {}), ("total",)),
            }
            chain.append(
                ActivationNode(
                    node_id=node_id,
                    activation=float(activation),
                    cue=cue,
                    metadata=metadata,
                )
            )
        return chain

    def _score_seed(self, seed: Mapping[str, Any], anchor_strength: float) -> float:
        cfg = self.config
        meta = seed.get("meta", {}) or {}
        novelty = self._coerce_meta(meta, ("novelty", "novelty_score"))
        constraint = self._coerce_meta(meta, ("constraint_weight", "mobility_block"))
        social = self._coerce_meta(meta, ("social_weight", "attachment"))
        open_loop = self._coerce_meta(meta, ("open_loops", "unresolved", "pending_ratio"))
        change_point = self._coerce_meta(
            meta,
            ("coherence_shift", "change_point", "coherence_drop"),
        )
        utility = self._coerce_meta(seed.get("value", {}), ("total",))
        score = (
            cfg.anchor_gain * max(anchor_strength, 0.1)
            + cfg.spread_gain * max(utility, 0.0)
            + 0.2 * novelty
            + 0.15 * constraint
            + 0.15 * social
            + cfg.open_loop_gain * max(0.0, open_loop)
            + cfg.change_point_gain * max(0.0, change_point)
        )
        return float(max(0.0, score))

    def _expand_chain_seeds(self, seeds: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """Expand top-level seeds with lightweight causal/temporal neighbors when present."""
        expanded: List[Mapping[str, Any]] = []
        seen: Dict[str, None] = {}
        for idx, seed in enumerate(seeds):
            sid = self._seed_id(seed, idx)
            if sid not in seen:
                seen[sid] = None
                expanded.append(seed)
            added = 0
            for related in self._iter_related(seed):
                if added >= self.config.max_related_per_seed:
                    break
                rid = self._seed_id(related, idx + added + 1)
                if rid in seen:
                    continue
                seen[rid] = None
                expanded.append(related)
                added += 1
        return expanded

    def _iter_related(self, seed: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        meta = seed.get("meta", {}) if isinstance(seed.get("meta"), Mapping) else {}
        related = meta.get("related")
        if isinstance(related, Mapping):
            related = [related]
        if isinstance(related, list):
            for item in related:
                if isinstance(item, Mapping):
                    yield item
        for key in ("prev", "next", "cause", "outcome", "counter"):
            item = meta.get(key)
            if isinstance(item, Mapping):
                yield item

    def _seed_id(self, seed: Mapping[str, Any], fallback_idx: int) -> str:
        meta = seed.get("meta")
        meta_map = meta if isinstance(meta, Mapping) else {}
        receipt = meta_map.get("receipt")
        receipt_map = receipt if isinstance(receipt, Mapping) else {}
        return str(
            seed.get("trace_id")
            or seed.get("id")
            or receipt_map.get("id")
            or meta_map.get("source_id")
            or f"seed-{fallback_idx}"
        )

    def _coerce_meta(
        self,
        payload: Mapping[str, Any],
        keys: Iterable[str],
        default: float = 0.0,
    ) -> float:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    # ------------------------------------------------------------------
    # context extraction helpers
    # ------------------------------------------------------------------
    def _detect_anchor(self, ctx_time: Mapping[str, Any]) -> Tuple[Optional[str], float]:
        flags = ctx_time.get("flags")
        label = None
        if isinstance(flags, list):
            for flag in flags:
                if isinstance(flag, str) and flag.startswith("anchor:"):
                    label = flag.split(":", 1)[1]
                    break
        if not label:
            for key in ("anchor", "anchor_label", "memory_anchor"):
                value = ctx_time.get(key)
                if isinstance(value, str) and value:
                    label = value
                    break
        if not label:
            label = ctx_time.get("landmark")
        strength = 1.0 if label else 0.3
        return label, float(strength)

    def _build_scene_frames(
        self,
        *,
        ctx_time: Mapping[str, Any],
        plan: Mapping[str, Any],
        seeds: Sequence[Mapping[str, Any]],
        best_choice: Optional[Mapping[str, Any]],
        anchor_label: Optional[str],
    ) -> List[SceneFrame]:
        agents = self._extract_agents(ctx_time)
        objects = self._extract_objects(ctx_time)
        constraints = self._extract_constraints(ctx_time)
        snapshots = self._build_affect_snapshots(plan, best_choice, seeds)
        norm_event = self._extract_norm_event(ctx_time)
        if not agents:
            agents = [SceneAgent(name="self", role="observer", perspective="first_person", certainty=1.0)]
        scene_id = anchor_label or f"scene-{uuid.uuid4().hex[:6]}"
        frame = SceneFrame(
            scene_id=scene_id,
            anchor=anchor_label,
            agents=agents,
            objects=objects,
            constraints=constraints,
            norm_event=norm_event,
            affect_snapshots=snapshots,
            replay_source="replay",
        )
        return [frame]

    def _extract_agents(self, ctx_time: Mapping[str, Any]) -> List[SceneAgent]:
        agents: Dict[str, SceneAgent] = {}
        agents["self"] = SceneAgent(
            name="self",
            role="self",
            perspective="first_person",
            certainty=1.0,
        )
        for key in ("agents", "salient_entities", "participants", "peers"):
            values = ctx_time.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                label = None
                if isinstance(value, str):
                    label = value.strip()
                elif isinstance(value, Mapping):
                    label = str(value.get("name") or value.get("label") or value.get("id"))
                if not label:
                    continue
                norm = label.strip()
                if not norm or norm.lower() == "self":
                    continue
                agents.setdefault(
                    norm,
                    SceneAgent(name=norm, role="other", perspective="observer", certainty=0.6),
                )
        return list(agents.values())

    def _extract_objects(self, ctx_time: Mapping[str, Any]) -> List[SceneObject]:
        objects: List[SceneObject] = []
        for key in ("landmarks", "objects", "context_tags"):
            values = ctx_time.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str):
                    label = value.strip()
                elif isinstance(value, Mapping):
                    label = str(value.get("name") or value.get("label") or "")
                else:
                    continue
                if not label:
                    continue
                objects.append(SceneObject(name=label, salience=0.6))
        return objects[:5]

    def _extract_constraints(self, ctx_time: Mapping[str, Any]) -> List[SceneConstraint]:
        constraints: List[SceneConstraint] = []
        payload = ctx_time.get("constraints")
        if isinstance(payload, Mapping):
            payload = [payload]
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    label = item.get("label") or item.get("name")
                    intensity = self._coerce_meta(item, ("intensity", "weight", "value"), 0.4)
                    description = item.get("description") or item.get("reason")
                else:
                    label = str(item)
                    intensity = 0.4
                    description = None
                if not label:
                    continue
                constraints.append(
                    SceneConstraint(
                        label=str(label),
                        intensity=float(max(0.0, min(1.0, intensity))),
                        description=str(description) if description else "",
                    )
                )
        flags = ctx_time.get("flags")
        if isinstance(flags, list):
            for flag in flags:
                if isinstance(flag, str) and flag.startswith("constraint:"):
                    label = flag.split(":", 1)[1]
                    constraints.append(SceneConstraint(label=label, intensity=0.5))
        return constraints[:4]

    def _extract_norm_event(self, ctx_time: Mapping[str, Any]) -> Optional[str]:
        for key in ("norm_event", "boundary_event", "rule_event"):
            value = ctx_time.get(key)
            if isinstance(value, str) and value:
                return value
        flags = ctx_time.get("flags")
        if isinstance(flags, list):
            for flag in flags:
                if isinstance(flag, str) and flag.startswith("norm:"):
                    return flag.split(":", 1)[1]
        return None

    def _build_affect_snapshots(
        self,
        plan: Mapping[str, Any],
        best_choice: Optional[Mapping[str, Any]],
        seeds: Sequence[Mapping[str, Any]],
    ) -> List[AffectSnapshot]:
        snapshots: List[AffectSnapshot] = []
        mood = plan.get("mood", {}) if isinstance(plan.get("mood"), Mapping) else {}
        if mood:
            intensity = min(
                1.0,
                abs(float(mood.get("valence", 0.0))) + abs(float(mood.get("arousal", 0.0))),
            )
            snapshots.append(
                AffectSnapshot(
                    label="current_mood",
                    intensity=float(intensity),
                    emotion_tag=mood.get("label"),
                    replay_source="live",
                )
            )
        if best_choice:
            snapshots.append(
                AffectSnapshot(
                    label=str(best_choice.get("a", "replay")),
                    intensity=float(min(1.0, abs(float(best_choice.get("U", 0.0))))),
                    emotion_tag="replay",
                    replay_source="replay",
                )
            )
        elif seeds:
            top = max(seeds, key=lambda s: self._coerce_meta(s.get("value", {}), ("total",)))
            snapshots.append(
                AffectSnapshot(
                    label=str(top.get("imagined", {}).get("best_action", "seed")),
                    intensity=float(min(1.0, abs(self._coerce_meta(top.get("value", {}), ("total",))))),
                    emotion_tag="replay",
                    replay_source="replay",
                )
            )
        return snapshots

    def _build_trigger_context(self, ctx_time: Mapping[str, Any], plan: Mapping[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "flags": list(ctx_time.get("flags", []) or []),
            "salient_entities": list(ctx_time.get("salient_entities", []) or []),
            "context_tags": list(ctx_time.get("context_tags", []) or []),
            "mode": ctx_time.get("mode") or plan.get("mode"),
        }
        if "talk_mode" in ctx_time:
            payload["talk_mode"] = ctx_time["talk_mode"]
        if "observations" in ctx_time:
            payload["observations"] = ctx_time["observations"]
        return payload

    def _metadata_block(
        self,
        replay_details: Optional[Mapping[str, Any]],
        field_signals: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if replay_details:
            meta["reverse_ratio"] = replay_details.get("reverse_ratio")
            meta["steps"] = replay_details.get("steps")
        if field_signals:
            meta["field_load"] = float(field_signals.get("inflammation_global", 0.0))
        return meta

    def _compose_notes(
        self,
        anchor_label: Optional[str],
        curve: Sequence[ConfidenceSample],
    ) -> str:
        if not curve:
            return ""
        start = curve[0].conf_internal
        end = curve[-1].conf_internal
        anchor_text = anchor_label or "an unnamed anchor"
        return f"Anchor {anchor_text} lifted internal confidence from {start:.2f} to {end:.2f}."

    def _build_confidence_curve(
        self,
        ctx_time: Mapping[str, Any],
        chain: Sequence[ActivationNode],
        anchor_strength: float,
        *,
        field_signals: Optional[Mapping[str, Any]],
    ) -> List[ConfidenceSample]:
        cfg = self.config
        curve: List[ConfidenceSample] = []
        conf_internal = max(0.05, min(1.0, 0.35 + 0.45 * anchor_strength))
        conf_external = float(ctx_time.get("external_confidence", 0.0))
        curve.append(ConfidenceSample(step=0, conf_internal=conf_internal, conf_external=conf_external))
        for idx, node in enumerate(chain, start=1):
            conf_internal = min(1.0, conf_internal + cfg.confirm_gain * node.activation)
            curve.append(
                ConfidenceSample(
                    step=idx,
                    conf_internal=conf_internal,
                    conf_external=conf_external,
                )
            )
        matches = self._observation_matches(ctx_time, field_signals)
        if matches > 0.0:
            conf_external = min(1.0, conf_external + cfg.external_gain * matches)
            curve.append(
                ConfidenceSample(
                    step=len(chain) + 1,
                    conf_internal=conf_internal,
                    conf_external=conf_external,
                )
            )
        return curve

    def _observation_matches(
        self,
        ctx_time: Mapping[str, Any],
        field_signals: Optional[Mapping[str, Any]],
    ) -> float:
        if isinstance(ctx_time.get("observation_matches"), Mapping):
            matches = ctx_time["observation_matches"].get("count", 0.0)
            return float(max(0.0, matches))
        if isinstance(ctx_time.get("observation_matches"), (int, float)):
            return float(max(0.0, ctx_time.get("observation_matches", 0.0)))
        if isinstance(ctx_time.get("landmarks"), list):
            return min(1.0, 0.2 * len(ctx_time["landmarks"]))
        if field_signals and field_signals.get("coherence_gain"):
            return float(max(0.0, field_signals["coherence_gain"]))
        return 0.0

    def _build_replay_events(
        self,
        frames: Sequence[SceneFrame],
        best_choice: Optional[Mapping[str, Any]],
    ) -> List[ReplayEvent]:
        events: List[ReplayEvent] = []
        for frame in frames:
            events.append(
                ReplayEvent(
                    scene_id=frame.scene_id,
                    replay_source=frame.replay_source,
                    payload={"anchor": frame.anchor, "agents": [agent.name for agent in frame.agents]},
                )
            )
        if not frames and best_choice:
            events.append(
                ReplayEvent(
                    scene_id=str(best_choice.get("a", "best")),
                    replay_source="replay",
                    payload={"utility": float(best_choice.get("U", 0.0))},
                )
            )
        return events


__all__ = ["RecallEngine", "RecallEngineConfig"]
