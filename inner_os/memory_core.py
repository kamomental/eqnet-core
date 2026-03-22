from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .memory_records import normalize_memory_record


MEMORY_RERANK_WEIGHTS = {
    "culture_match_base": 0.12,
    "culture_resonance": 0.05,
    "community_match_base": 0.14,
    "community_resonance": 0.06,
    "role_match": 0.08,
    "anchor_match": 0.18,
    "verified_caution": 0.06,
    "verified_hazard": 0.05,
    "verified_terrain": 0.06,
    "verified_afterglow": 0.03,
    "observed_afterglow": 0.05,
    "verified_anticipation": 0.02,
    "verified_stabilization": 0.03,
    "verified_reopening": 0.02,
    "trace_affiliation": 0.06,
    "trace_ritual": 0.04,
    "trace_institution": 0.04,
    "relationship_institution": 0.02,
    "relationship_afterglow": 0.05,
    "relationship_check_in_afterglow": 0.08,
    "relationship_replay": 0.04,
    "relationship_clarity": 0.05,
    "relationship_reopening": 0.04,
    "relationship_person_match": 0.08,
    "identity_person_match": 0.05,
    "community_culture": 0.08,
    "community_resonance_trace": 0.1,
    "community_terrain": 0.05,
    "identity_terrain": 0.04,
    "identity_clarify_afterglow": 0.03,
    "identity_clarity": 0.03,
    "identity_inertia": 0.03,
    "identity_reopening": 0.04,
    "reconstructed_ritual": 0.03,
    "reconstructed_terrain_penalty": 0.04,
    "reconstructed_tentative_penalty": 0.06,
    "reconstructed_terrain_tentative_penalty": 0.05,
    "reconstructed_afterglow_penalty": 0.03,
    "reconstructed_anticipation_penalty": 0.02,
    "reconstructed_inertia_penalty": 0.03,
    "reconstructed_reopening": 0.08,
    "reconstructed_reopen_clarity": 0.03,
    "continuity": 0.04,
    "reconstructed_resource_penalty": 0.02,
    "monument_verified": 0.05,
    "monument_relationship": 0.06,
    "monument_identity": 0.04,
    "mosaic_observed": 0.03,
    "mosaic_relationship": 0.04,
    "object_observed": 0.05,
    "object_relationship": 0.03,
    "object_identity": 0.02,
    "object_reconstructed_penalty": 0.05,
    "peripersonal_observed": 0.04,
    "peripersonal_relationship": 0.02,
    "peripersonal_reconstructed_penalty": 0.05,
    "consolidation_verified": 0.04,
    "prospective_verified": 0.03,
    "interference_reconstructed_penalty": 0.05,
    "working_memory_verified": 0.04,
    "working_memory_observed": 0.03,
    "working_memory_reconstructed_penalty": 0.05,
    "working_memory_social_trace": 0.04,
    "working_memory_identity_trace": 0.04,
    "working_memory_replay_focus_match": 0.06,
    "working_memory_replay_anchor_match": 0.08,
    "working_memory_replay_strength_scale": 0.08,
    "working_memory_seed_focus_match": 0.04,
    "working_memory_seed_anchor_match": 0.05,
    "working_memory_seed_strength_scale": 0.05,
    "long_term_theme_focus_match": 0.05,
    "long_term_theme_anchor_match": 0.06,
    "long_term_theme_summary_match": 0.04,
    "long_term_theme_strength_scale": 0.06,
    "partner_seed_summary_match": 0.07,
    "partner_social_interpretation_match": 0.05,
}

RECALL_ALLOCATION_WEIGHTS = {
    "base": 0.5,
    "caution_to_anchor": 0.18,
    "roughness_to_anchor": 0.22,
    "anticipation_to_anchor": 0.12,
    "stabilization_to_anchor": 0.1,
    "affiliation_to_reconstructive": 0.08,
    "replay_to_reconstructive": 0.06,
    "reopening_to_reconstructive": 0.12,
    "clarity_to_reconstructive": 0.05,
    "anchored_bonus": 0.08,
    "reconstructive_bonus": 0.06,
    "anchored_penalty_for_reconstructive": 0.03,
}

MEMORY_ACCESS_WEIGHTS = {
    "access_cap": 3.0,
    "access_scale": 0.04,
    "primed_scale": 0.06,
    "primed_replay_gain": 0.24,
    "primed_clarity_gain": 0.16,
    "primed_observed_bonus": 0.05,
    "primed_relationship_bonus": 0.06,
    "primed_identity_bonus": 0.04,
    "primed_reconstructed_bonus": 0.03,
    "touch_increment": 1.0,
    "touch_prime_floor": 0.28,
    "touch_prime_gain": 0.22,
    "touch_decay": 0.92,
}

MEMORY_FORGETTING_WEIGHTS = {
    "priming_pressure_penalty": 0.45,
    "reconstructed_pressure_penalty": 0.06,
    "reconstructed_short_horizon_penalty": 0.03,
}


@dataclass
class MemorySearchHit:
    record_id: str
    score: float
    record: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.record_id,
            "score": float(self.score),
            "record": dict(self.record),
        }


@dataclass
class MemoryCore:
    path: Path = Path("logs/inner_os_memory.jsonl")

    def append_records(self, records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        ts = time.time()
        for index, record in enumerate(records):
            payload = normalize_memory_record(record)
            payload.setdefault("id", f"inner-os-{int(ts * 1000)}-{index}")
            payload.setdefault("timestamp", ts)
            payload.setdefault("access_count", 0.0)
            payload.setdefault("primed_weight", 0.0)
            prepared.append(payload)
        if not prepared:
            return []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for payload in prepared:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return prepared

    def search(self, cue_text: str, *, limit: int = 3) -> List[MemorySearchHit]:
        text = str(cue_text or "").strip()
        if not text or not self.path.exists():
            return []
        query_terms = _terms(text)
        if not query_terms:
            return []
        hits: List[MemorySearchHit] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    haystack = " ".join(
                        str(record.get(key) or "")
                        for key in (
                            "text",
                            "summary",
                            "cue_text",
                            "user_text",
                            "assistant_text",
                            "memory_anchor",
                            "kind",
                            "policy_hint",
                            "provenance",
                            "source_episode_id",
                            "reinterpretation_mode",
                            "reinterpretation_summary",
                            "environment_summary",
                        )
                    ).strip()
                    score = _score_terms(query_terms, haystack)
                    if score <= 0.0:
                        continue
                    hits.append(
                        MemorySearchHit(
                            record_id=str(record.get("id") or "memory-hit"),
                            score=score,
                            record=record,
                        )
                    )
        except OSError:
            return []
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(int(limit), 1)]

    def touch_record_usage(self, record_id: str) -> Dict[str, Any]:
        target_id = str(record_id or "").strip()
        if not target_id or not self.path.exists():
            return {}
        records = self._read_all_records()
        if not records:
            return {}
        now_ts = time.time()
        touched: Dict[str, Any] = {}
        changed = False
        for record in records:
            existing_prime = _safe_float(record.get("primed_weight"), 0.0)
            if str(record.get("id") or "") == target_id:
                record["access_count"] = round(_safe_float(record.get("access_count"), 0.0) + MEMORY_ACCESS_WEIGHTS["touch_increment"], 4)
                record["last_accessed_at"] = now_ts
                record["primed_weight"] = round(max(MEMORY_ACCESS_WEIGHTS["touch_prime_floor"], existing_prime * MEMORY_ACCESS_WEIGHTS["touch_decay"] + MEMORY_ACCESS_WEIGHTS["touch_prime_gain"]), 4)
                touched = dict(record)
                changed = True
            elif existing_prime > 0.0:
                record["primed_weight"] = round(existing_prime * MEMORY_ACCESS_WEIGHTS["touch_decay"], 4)
                changed = True
        if changed:
            self._write_all_records(records)
        return touched

    def load_latest_profile_record(
        self,
        *,
        kind: str,
        culture_id: str | None = None,
        community_id: str | None = None,
        social_role: str | None = None,
        memory_anchor: str | None = None,
        related_person_id: str | None = None,
    ) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        latest: Dict[str, Any] = {}
        wanted_kind = str(kind or "").strip()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(record.get("kind") or "") != wanted_kind:
                        continue
                    if culture_id and str(record.get("culture_id") or "") != culture_id:
                        continue
                    if community_id and str(record.get("community_id") or "") != community_id:
                        continue
                    if social_role and str(record.get("social_role") or "") != social_role:
                        continue
                    if memory_anchor and str(record.get("memory_anchor") or "") != memory_anchor:
                        continue
                    if related_person_id and str(record.get("related_person_id") or "") != related_person_id:
                        continue
                    latest = record
        except OSError:
            return {}
        return latest

    def load_latest_identity_trace(
        self,
        *,
        culture_id: str | None = None,
        community_id: str | None = None,
        social_role: str | None = None,
        related_person_id: str | None = None,
    ) -> Dict[str, Any]:
        return self.load_latest_profile_record(
            kind="identity_trace",
            culture_id=culture_id,
            community_id=community_id,
            social_role=social_role,
            related_person_id=related_person_id,
        )

    def build_recall_payload(
        self,
        cue_text: str,
        *,
        limit: int = 3,
        bias_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        hits = self.search(cue_text, limit=max(limit, 6 if bias_context else limit))
        if not hits:
            return {}
        allocation = _recall_allocation(bias_context or {}) if bias_context else None
        if bias_context:
            hits = self._rerank_hits(hits, bias_context=bias_context)
        hits = hits[: max(int(limit), 1)]
        top = hits[0].record
        anchor = str(top.get("memory_anchor") or top.get("summary") or top.get("text") or cue_text).strip()
        summary = str(top.get("summary") or top.get("text") or anchor).strip()
        text = str(top.get("text") or top.get("summary") or anchor).strip()
        kind_breakdown: Dict[str, int] = {}
        for hit in hits:
            kind = str(hit.record.get("kind") or "unknown")
            kind_breakdown[kind] = kind_breakdown.get(kind, 0) + 1
        return {
            "memory_anchor": anchor[:160],
            "summary": summary,
            "text": text,
            "cue_text": cue_text,
            "record_id": top.get("id"),
            "record_kind": str(top.get("kind") or "observed_real"),
            "record_provenance": str(top.get("provenance") or "inner_os_memory"),
            "source_episode_id": top.get("source_episode_id") or top.get("id"),
            "policy_hint": top.get("policy_hint"),
            "reinterpretation_mode": top.get("reinterpretation_mode"),
            "reinterpretation_summary": top.get("reinterpretation_summary"),
            "tentative_bias": top.get("tentative_bias"),
            "environment_summary": top.get("environment_summary"),
            "reflective_tension": top.get("reflective_tension"),
            "social_self_pressure": top.get("social_self_pressure"),
            "meaning_shift": top.get("meaning_shift"),
            "recovery_reopening": top.get("recovery_reopening"),
            "access_count": top.get("access_count"),
            "primed_weight": top.get("primed_weight"),
            "last_accessed_at": top.get("last_accessed_at"),
            "forgetting_pressure": (bias_context or {}).get("forgetting_pressure"),
            "replay_horizon": (bias_context or {}).get("replay_horizon"),
            "monument_salience": (bias_context or {}).get("monument_salience"),
            "monument_kind": (bias_context or {}).get("monument_kind"),
            "conscious_mosaic_density": (bias_context or {}).get("conscious_mosaic_density"),
            "anchored_allocation": (allocation or {}).get("anchored_allocation"),
            "reconstructive_allocation": (allocation or {}).get("reconstructive_allocation"),
            "hits": [hit.to_dict() for hit in hits],
            "kind_breakdown": kind_breakdown,
            "culture_id": top.get("culture_id"),
            "community_id": top.get("community_id"),
            "social_role": top.get("social_role"),
            "related_person_id": top.get("related_person_id"),
        }

    def _rerank_hits(self, hits: List[MemorySearchHit], *, bias_context: Mapping[str, Any]) -> List[MemorySearchHit]:
        culture_id = str(bias_context.get("culture_id") or "").strip()
        community_id = str(bias_context.get("community_id") or "").strip()
        social_role = str(bias_context.get("social_role") or "").strip()
        related_person_id = str(bias_context.get("related_person_id") or "").strip()
        memory_anchor = str(bias_context.get("memory_anchor") or bias_context.get("place_memory_anchor") or "").strip()
        culture_resonance = _safe_float(bias_context.get("culture_resonance"), 0.0)
        community_resonance = _safe_float(bias_context.get("community_resonance"), 0.0)
        affiliation_bias = _safe_float(bias_context.get("affiliation_bias"), 0.45)
        caution_bias = _safe_float(bias_context.get("caution_bias"), 0.4)
        continuity_score = _safe_float(bias_context.get("continuity_score"), 0.48)
        hazard_pressure = _safe_float(bias_context.get("hazard_pressure"), 0.0)
        institutional_pressure = _safe_float(bias_context.get("institutional_pressure"), 0.0)
        ritual_pressure = _safe_float(bias_context.get("ritual_pressure"), 0.0)
        resource_pressure = _safe_float(bias_context.get("resource_pressure"), 0.0)
        terrain_transition_roughness = _safe_float(bias_context.get("terrain_transition_roughness"), 0.0)
        interaction_afterglow = _safe_float(bias_context.get("interaction_afterglow"), 0.0)
        interaction_afterglow_intent = str(bias_context.get("interaction_afterglow_intent") or "").strip()
        replay_intensity = _safe_float(bias_context.get("replay_intensity"), 0.0)
        anticipation_tension = _safe_float(bias_context.get("anticipation_tension"), 0.0)
        stabilization_drive = _safe_float(bias_context.get("stabilization_drive"), 0.0)
        relational_clarity = _safe_float(bias_context.get("relational_clarity"), 0.0)
        meaning_inertia = _safe_float(bias_context.get("meaning_inertia"), 0.0)
        recovery_reopening = _safe_float(bias_context.get("recovery_reopening"), 0.0)
        forgetting_pressure = _safe_float(bias_context.get("forgetting_pressure"), 0.0)
        working_memory_pressure = _safe_float(bias_context.get("working_memory_pressure"), 0.0)
        pending_meaning = _safe_float(bias_context.get("pending_meaning"), 0.0)
        current_focus = str(bias_context.get("current_focus") or "").strip()
        replay_signature_focus = str(bias_context.get("replay_signature_focus") or "").strip()
        replay_signature_anchor = str(bias_context.get("replay_signature_anchor") or "").strip()
        replay_signature_strength = _safe_float(bias_context.get("replay_signature_strength"), 0.0)
        semantic_seed_focus = str(bias_context.get("semantic_seed_focus") or "").strip()
        semantic_seed_anchor = str(bias_context.get("semantic_seed_anchor") or "").strip()
        semantic_seed_strength = _safe_float(bias_context.get("semantic_seed_strength"), 0.0)
        long_term_theme_focus = str(bias_context.get("long_term_theme_focus") or "").strip()
        long_term_theme_anchor = str(bias_context.get("long_term_theme_anchor") or "").strip()
        long_term_theme_summary = str(bias_context.get("long_term_theme_summary") or "").strip()
        long_term_theme_strength = _safe_float(bias_context.get("long_term_theme_strength"), 0.0)
        relation_seed_summary = str(bias_context.get("relation_seed_summary") or "").strip()
        partner_social_interpretation = str(bias_context.get("partner_social_interpretation") or "").strip()
        replay_horizon = max(1, int(_safe_float(bias_context.get("replay_horizon"), 3.0)))
        monument_salience = _safe_float(bias_context.get("monument_salience"), 0.0)
        conscious_mosaic_density = _safe_float(bias_context.get("conscious_mosaic_density"), 0.0)
        object_affordance_bias = _safe_float(bias_context.get("object_affordance_bias"), 0.0)
        fragility_guard = _safe_float(bias_context.get("fragility_guard"), 0.0)
        object_attachment = _safe_float(bias_context.get("object_attachment"), 0.0)
        object_avoidance = _safe_float(bias_context.get("object_avoidance"), 0.0)
        reachability = _safe_float(bias_context.get("reachability"), 0.0)
        near_body_risk = _safe_float(bias_context.get("near_body_risk"), 0.0)
        defensive_salience = _safe_float(bias_context.get("defensive_salience"), 0.0)
        interference_pressure = _safe_float(bias_context.get("interference_pressure"), 0.0)
        consolidation_priority = _safe_float(bias_context.get("consolidation_priority"), 0.0)
        prospective_memory_pull = _safe_float(bias_context.get("prospective_memory_pull"), 0.0)
        kind_biases = bias_context.get("kind_biases") if isinstance(bias_context.get("kind_biases"), Mapping) else {}
        allocation = _recall_allocation(bias_context)
        anchored_allocation = _safe_float(allocation.get("anchored_allocation"), 0.5)
        reconstructive_allocation = _safe_float(allocation.get("reconstructive_allocation"), 0.5)

        rescored: List[MemorySearchHit] = []
        for hit in hits:
            bonus = 0.0
            record = hit.record
            record_tentative_bias = _safe_float(record.get("tentative_bias"), 0.0)
            access_count = min(_safe_float(record.get("access_count"), 0.0), MEMORY_ACCESS_WEIGHTS["access_cap"])
            primed_weight = _effective_primed_weight(record)
            priming_gain = primed_weight * (
                MEMORY_ACCESS_WEIGHTS["primed_scale"]
                + replay_intensity * MEMORY_ACCESS_WEIGHTS["primed_replay_gain"]
                + relational_clarity * MEMORY_ACCESS_WEIGHTS["primed_clarity_gain"]
            )
            priming_gain *= max(0.25, 1.0 - forgetting_pressure * MEMORY_FORGETTING_WEIGHTS["priming_pressure_penalty"])
            if culture_id and str(record.get("culture_id") or "") == culture_id:
                bonus += MEMORY_RERANK_WEIGHTS["culture_match_base"] + culture_resonance * MEMORY_RERANK_WEIGHTS["culture_resonance"]
            if community_id and str(record.get("community_id") or "") == community_id:
                bonus += MEMORY_RERANK_WEIGHTS["community_match_base"] + community_resonance * MEMORY_RERANK_WEIGHTS["community_resonance"]
            if social_role and str(record.get("social_role") or "") == social_role:
                bonus += MEMORY_RERANK_WEIGHTS["role_match"]
            if memory_anchor and str(record.get("memory_anchor") or "") == memory_anchor:
                bonus += MEMORY_RERANK_WEIGHTS["anchor_match"]
            if replay_signature_focus and _record_matches_replay_signature_focus(record, replay_signature_focus):
                bonus += MEMORY_RERANK_WEIGHTS["working_memory_replay_focus_match"]
                bonus += replay_signature_strength * MEMORY_RERANK_WEIGHTS["working_memory_replay_strength_scale"]
            if replay_signature_anchor and str(record.get("memory_anchor") or "").strip() == replay_signature_anchor:
                bonus += MEMORY_RERANK_WEIGHTS["working_memory_replay_anchor_match"]
                bonus += replay_signature_strength * MEMORY_RERANK_WEIGHTS["working_memory_replay_strength_scale"]
            if semantic_seed_focus and _record_matches_replay_signature_focus(record, semantic_seed_focus):
                bonus += MEMORY_RERANK_WEIGHTS["working_memory_seed_focus_match"]
                bonus += semantic_seed_strength * MEMORY_RERANK_WEIGHTS["working_memory_seed_strength_scale"]
            if semantic_seed_anchor and str(record.get("memory_anchor") or "").strip() == semantic_seed_anchor:
                bonus += MEMORY_RERANK_WEIGHTS["working_memory_seed_anchor_match"]
                bonus += semantic_seed_strength * MEMORY_RERANK_WEIGHTS["working_memory_seed_strength_scale"]
            if long_term_theme_focus and _record_matches_replay_signature_focus(record, long_term_theme_focus):
                bonus += MEMORY_RERANK_WEIGHTS["long_term_theme_focus_match"]
                bonus += long_term_theme_strength * MEMORY_RERANK_WEIGHTS["long_term_theme_strength_scale"]
            if long_term_theme_anchor and str(record.get("memory_anchor") or "").strip() == long_term_theme_anchor:
                bonus += MEMORY_RERANK_WEIGHTS["long_term_theme_anchor_match"]
                bonus += long_term_theme_strength * MEMORY_RERANK_WEIGHTS["long_term_theme_strength_scale"]
            if long_term_theme_summary and _record_matches_theme_summary(record, long_term_theme_summary):
                bonus += MEMORY_RERANK_WEIGHTS["long_term_theme_summary_match"]
                bonus += long_term_theme_strength * MEMORY_RERANK_WEIGHTS["long_term_theme_strength_scale"]
            if relation_seed_summary and _record_matches_theme_summary(record, relation_seed_summary):
                bonus += MEMORY_RERANK_WEIGHTS["partner_seed_summary_match"]
            if partner_social_interpretation and _record_matches_partner_social_interpretation(record, partner_social_interpretation):
                bonus += MEMORY_RERANK_WEIGHTS["partner_social_interpretation_match"]

            kind = str(record.get("kind") or "")
            if kind in {"verified", "observed_real"}:
                bonus += caution_bias * MEMORY_RERANK_WEIGHTS["verified_caution"]
                bonus += hazard_pressure * MEMORY_RERANK_WEIGHTS["verified_hazard"]
                bonus += terrain_transition_roughness * MEMORY_RERANK_WEIGHTS["verified_terrain"]
                bonus += interaction_afterglow * MEMORY_RERANK_WEIGHTS["verified_afterglow"] if kind == "verified" else interaction_afterglow * MEMORY_RERANK_WEIGHTS["observed_afterglow"]
                bonus += anticipation_tension * MEMORY_RERANK_WEIGHTS["verified_anticipation"] + stabilization_drive * MEMORY_RERANK_WEIGHTS["verified_stabilization"]
                bonus += recovery_reopening * MEMORY_RERANK_WEIGHTS["verified_reopening"]
                bonus += anchored_allocation * RECALL_ALLOCATION_WEIGHTS["anchored_bonus"]
                bonus += access_count * MEMORY_ACCESS_WEIGHTS["access_scale"]
                bonus += priming_gain * MEMORY_ACCESS_WEIGHTS["primed_observed_bonus"]
                bonus += monument_salience * MEMORY_RERANK_WEIGHTS["monument_verified"]
                bonus += conscious_mosaic_density * MEMORY_RERANK_WEIGHTS["mosaic_observed"]
                bonus += object_affordance_bias * MEMORY_RERANK_WEIGHTS["object_observed"]
                bonus += object_attachment * MEMORY_RERANK_WEIGHTS["object_observed"] * 0.5
                bonus += (reachability + defensive_salience) * MEMORY_RERANK_WEIGHTS["peripersonal_observed"]
                bonus += consolidation_priority * MEMORY_RERANK_WEIGHTS["consolidation_verified"]
                bonus += prospective_memory_pull * MEMORY_RERANK_WEIGHTS["prospective_verified"]
                bonus += working_memory_pressure * (MEMORY_RERANK_WEIGHTS["working_memory_verified"] if kind == "verified" else MEMORY_RERANK_WEIGHTS["working_memory_observed"])
            if kind in {"reconstructed", "relationship_trace", "identity_trace", "community_profile_trace"}:
                bonus += affiliation_bias * MEMORY_RERANK_WEIGHTS["trace_affiliation"]
                bonus += ritual_pressure * MEMORY_RERANK_WEIGHTS["trace_ritual"]
                bonus += institutional_pressure * MEMORY_RERANK_WEIGHTS["trace_institution"]
                bonus += access_count * MEMORY_ACCESS_WEIGHTS["access_scale"]
            if kind == "relationship_trace":
                if related_person_id and str(record.get("related_person_id") or "") == related_person_id:
                    bonus += MEMORY_RERANK_WEIGHTS["relationship_person_match"]
                bonus += institutional_pressure * MEMORY_RERANK_WEIGHTS["relationship_institution"]
                bonus += interaction_afterglow * (MEMORY_RERANK_WEIGHTS["relationship_check_in_afterglow"] if interaction_afterglow_intent == "check_in" else MEMORY_RERANK_WEIGHTS["relationship_afterglow"])
                bonus += replay_intensity * MEMORY_RERANK_WEIGHTS["relationship_replay"] + relational_clarity * MEMORY_RERANK_WEIGHTS["relationship_clarity"]
                bonus += recovery_reopening * MEMORY_RERANK_WEIGHTS["relationship_reopening"]
                bonus += priming_gain * MEMORY_ACCESS_WEIGHTS["primed_relationship_bonus"]
                bonus += monument_salience * MEMORY_RERANK_WEIGHTS["monument_relationship"]
                bonus += conscious_mosaic_density * MEMORY_RERANK_WEIGHTS["mosaic_relationship"]
                bonus += object_attachment * MEMORY_RERANK_WEIGHTS["object_relationship"]
                bonus += defensive_salience * MEMORY_RERANK_WEIGHTS["peripersonal_relationship"]
                if current_focus == "social":
                    bonus += MEMORY_RERANK_WEIGHTS["working_memory_social_trace"]
            if kind == "community_profile_trace":
                bonus += culture_resonance * MEMORY_RERANK_WEIGHTS["community_culture"] + community_resonance * MEMORY_RERANK_WEIGHTS["community_resonance_trace"]
                bonus += terrain_transition_roughness * MEMORY_RERANK_WEIGHTS["community_terrain"]
            if kind == "identity_trace":
                if related_person_id and str(record.get("related_person_id") or "") == related_person_id:
                    bonus += MEMORY_RERANK_WEIGHTS["identity_person_match"]
                bonus += terrain_transition_roughness * MEMORY_RERANK_WEIGHTS["identity_terrain"]
                bonus += interaction_afterglow * MEMORY_RERANK_WEIGHTS["identity_clarify_afterglow"] if interaction_afterglow_intent == "clarify" else 0.0
                bonus += relational_clarity * MEMORY_RERANK_WEIGHTS["identity_clarity"] + meaning_inertia * MEMORY_RERANK_WEIGHTS["identity_inertia"]
                bonus += recovery_reopening * MEMORY_RERANK_WEIGHTS["identity_reopening"]
                bonus += priming_gain * MEMORY_ACCESS_WEIGHTS["primed_identity_bonus"]
                bonus += monument_salience * MEMORY_RERANK_WEIGHTS["monument_identity"]
                bonus += object_attachment * MEMORY_RERANK_WEIGHTS["object_identity"]
                if current_focus == "meaning":
                    bonus += MEMORY_RERANK_WEIGHTS["working_memory_identity_trace"]
            if kind == "reconstructed":
                bonus += ritual_pressure * MEMORY_RERANK_WEIGHTS["reconstructed_ritual"]
                bonus -= terrain_transition_roughness * MEMORY_RERANK_WEIGHTS["reconstructed_terrain_penalty"]
                bonus -= record_tentative_bias * MEMORY_RERANK_WEIGHTS["reconstructed_tentative_penalty"]
                bonus -= terrain_transition_roughness * record_tentative_bias * MEMORY_RERANK_WEIGHTS["reconstructed_terrain_tentative_penalty"]
                bonus -= interaction_afterglow * MEMORY_RERANK_WEIGHTS["reconstructed_afterglow_penalty"]
                bonus -= anticipation_tension * MEMORY_RERANK_WEIGHTS["reconstructed_anticipation_penalty"]
                bonus -= meaning_inertia * MEMORY_RERANK_WEIGHTS["reconstructed_inertia_penalty"]
                bonus += recovery_reopening * MEMORY_RERANK_WEIGHTS["reconstructed_reopening"] * max(0.0, 1.0 - record_tentative_bias)
                bonus += relational_clarity * recovery_reopening * MEMORY_RERANK_WEIGHTS["reconstructed_reopen_clarity"]
                bonus += reconstructive_allocation * RECALL_ALLOCATION_WEIGHTS["reconstructive_bonus"] * max(0.0, 1.0 - record_tentative_bias)
                bonus -= anchored_allocation * RECALL_ALLOCATION_WEIGHTS["anchored_penalty_for_reconstructive"]
                bonus += priming_gain * MEMORY_ACCESS_WEIGHTS["primed_reconstructed_bonus"] * max(0.0, 1.0 - record_tentative_bias)
                bonus -= forgetting_pressure * MEMORY_FORGETTING_WEIGHTS["reconstructed_pressure_penalty"]
                if replay_horizon <= 2:
                    bonus -= MEMORY_FORGETTING_WEIGHTS["reconstructed_short_horizon_penalty"]
                bonus -= (fragility_guard + object_avoidance) * MEMORY_RERANK_WEIGHTS["object_reconstructed_penalty"]
                bonus -= (near_body_risk + defensive_salience) * MEMORY_RERANK_WEIGHTS["peripersonal_reconstructed_penalty"]
                bonus -= interference_pressure * MEMORY_RERANK_WEIGHTS["interference_reconstructed_penalty"]
                bonus -= pending_meaning * MEMORY_RERANK_WEIGHTS["working_memory_reconstructed_penalty"]
            bonus += continuity_score * MEMORY_RERANK_WEIGHTS["continuity"]
            bonus -= resource_pressure * MEMORY_RERANK_WEIGHTS["reconstructed_resource_penalty"] if kind == "reconstructed" else 0.0
            bonus += _safe_float(kind_biases.get(kind), 0.0)
            rescored.append(MemorySearchHit(record_id=hit.record_id, score=hit.score + bonus, record=record))

        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored

    def _read_all_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return records

    def _write_all_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _recall_allocation(bias_context: Mapping[str, Any]) -> Dict[str, float]:
    caution_bias = _safe_float(bias_context.get("caution_bias"), 0.4)
    affiliation_bias = _safe_float(bias_context.get("affiliation_bias"), 0.45)
    terrain_transition_roughness = _safe_float(bias_context.get("terrain_transition_roughness"), 0.0)
    replay_intensity = _safe_float(bias_context.get("replay_intensity"), 0.0)
    anticipation_tension = _safe_float(bias_context.get("anticipation_tension"), 0.0)
    stabilization_drive = _safe_float(bias_context.get("stabilization_drive"), 0.0)
    relational_clarity = _safe_float(bias_context.get("relational_clarity"), 0.0)
    recovery_reopening = _safe_float(bias_context.get("recovery_reopening"), 0.0)
    anchored_allocation = _clamp01(
        RECALL_ALLOCATION_WEIGHTS["base"]
        + caution_bias * RECALL_ALLOCATION_WEIGHTS["caution_to_anchor"]
        + terrain_transition_roughness * RECALL_ALLOCATION_WEIGHTS["roughness_to_anchor"]
        + anticipation_tension * RECALL_ALLOCATION_WEIGHTS["anticipation_to_anchor"]
        + stabilization_drive * RECALL_ALLOCATION_WEIGHTS["stabilization_to_anchor"]
        - affiliation_bias * RECALL_ALLOCATION_WEIGHTS["affiliation_to_reconstructive"]
        - replay_intensity * RECALL_ALLOCATION_WEIGHTS["replay_to_reconstructive"]
        - recovery_reopening * RECALL_ALLOCATION_WEIGHTS["reopening_to_reconstructive"]
        - relational_clarity * RECALL_ALLOCATION_WEIGHTS["clarity_to_reconstructive"]
    )
    reconstructive_allocation = 1.0 - anchored_allocation
    return {
        "anchored_allocation": round(anchored_allocation, 4),
        "reconstructive_allocation": round(reconstructive_allocation, 4),
    }


def _effective_primed_weight(record: Mapping[str, Any]) -> float:
    primed_weight = _safe_float(record.get("primed_weight"), 0.0)
    last_accessed_at = _safe_float(record.get("last_accessed_at"), 0.0)
    if primed_weight <= 0.0 or last_accessed_at <= 0.0:
        return primed_weight
    elapsed_hours = max(0.0, (time.time() - last_accessed_at) / 3600.0)
    decay = max(0.25, 1.0 - min(elapsed_hours / 24.0, 0.75))
    return _clamp01(primed_weight * decay)


def _terms(text: str) -> List[str]:
    parts = [part.strip().lower() for part in text.replace("\n", " ").split(" ")]
    return [part for part in parts if len(part) >= 2]


def _score_terms(query_terms: List[str], haystack: str) -> float:
    content = str(haystack or "").lower()
    if not content:
        return 0.0
    score = 0.0
    for term in query_terms:
        if term in content:
            score += 1.0 / max(len(query_terms), 1)
    return float(score)


def _record_matches_replay_signature_focus(record: Mapping[str, Any], focus: str) -> bool:
    target = str(focus or "").strip().lower()
    if not target:
        return False
    haystack = " ".join(
        str(record.get(key) or "")
        for key in (
            "summary",
            "text",
            "cue_text",
            "memory_anchor",
            "environment_summary",
            "reinterpretation_summary",
        )
    ).lower()
    return target in haystack


def _record_matches_theme_summary(record: Mapping[str, Any], summary: str) -> bool:
    tokens = _terms(summary)
    if not tokens:
        return False
    haystack = " ".join(
        str(record.get(key) or "")
        for key in (
            "summary",
            "text",
            "cue_text",
            "memory_anchor",
            "environment_summary",
            "reinterpretation_summary",
        )
    ).lower()
    if not haystack:
        return False
    hits = sum(1 for token in tokens[:4] if token and token in haystack)
    return hits >= 2 or (hits >= 1 and len(tokens) == 1)


def _record_matches_partner_social_interpretation(record: Mapping[str, Any], social_interpretation: str) -> bool:
    target = str(social_interpretation or "").strip().lower()
    if not target:
        return False
    haystack = " ".join(
        str(record.get(key) or "")
        for key in (
            "social_interpretation",
            "address_hint",
            "timing_hint",
            "stance_hint",
            "summary",
            "text",
        )
    ).lower()
    if not haystack:
        return False
    return target in haystack


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
