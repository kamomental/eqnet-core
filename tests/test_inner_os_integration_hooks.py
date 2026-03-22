from pathlib import Path

import numpy as np

from inner_os.access.models import ForegroundState
from inner_os.affective_position import AffectivePositionState
from inner_os.affective_terrain import (
    AffectiveTerrainState,
    BasicAffectiveTerrain,
    make_neutral_affective_terrain_state,
)
from inner_os.expression import (
    DialogueContext,
    build_expression_hints_from_gate_result,
    render_response,
)
from inner_os.integration_hooks import IntegrationHooks
from inner_os.interaction_judgement_comparison import compare_interaction_judgement_summaries
from inner_os.interaction_inspection_report import build_interaction_inspection_report
from inner_os.memory import build_memory_appends, build_memory_context
from inner_os.memory_core import MemoryCore


def test_integration_hooks_pre_and_recall_flow(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    pre = hooks.pre_turn_update(
        user_input={"text": "hello there"},
        sensor_input={"body_stress_index": 0.3, "activity_level": 0.4},
        local_context={"last_gate_context": {"life_indicator": 0.5, "valence": 0.1, "arousal": 0.35}},
        current_state={"current_energy": 0.8, "temporal_pressure": 0.1},
        safety_bias=0.2,
    )
    assert pre.state.stress >= 0.0
    assert pre.state.recovery_need >= 0.0
    assert pre.interaction_hints["suggested_talk_mode"] in {"watch", "ask"}
    assert pre.interaction_hints["terrain"]["attractor"]
    assert pre.interaction_hints["development"]["belonging"] >= 0.0
    assert pre.interaction_hints["environment_pressure"]["summary"]
    assert pre.interaction_hints["relationship"]["attachment"] >= 0.0
    assert pre.interaction_hints["personality"]["caution_bias"] >= 0.0
    assert pre.interaction_hints["personality"]["affiliation_bias"] >= 0.0
    assert pre.interaction_hints["persistence"]["continuity_score"] >= 0.0
    assert isinstance(pre.interaction_hints["identity_trace"], dict)
    assert isinstance(pre.interaction_hints["relationship_trace"], dict)
    assert isinstance(pre.interaction_hints["community_profile_trace"], dict)
    assert pre.interaction_hints["working_memory"]["memory_pressure"] >= 0.0
    assert pre.state.community_resonance >= 0.0
    assert pre.state.culture_resonance >= 0.0

    hooks.memory_core.append_records(
        [{"kind": "observed", "summary": "market walkway memory", "text": "market walkway memory", "memory_anchor": "market street", "source_episode_id": "obs-1"}]
    )
    recall = hooks.memory_recall(
        text_cue="market",
        visual_cue="narrow walkway",
        world_cue="market",
        current_state=pre.state.to_dict(),
        retrieval_summary={"hits": [{"id": "vision-1"}]},
    )
    assert recall.recall_payload["memory_anchor"]
    assert recall.recall_payload["record_kind"] in {"observed_real", "relationship_trace"}
    assert recall.recall_payload["record_provenance"] == "lived"
    assert recall.recall_payload["access_count"] == 1.0
    assert recall.recall_payload["primed_weight"] >= 0.28
    assert recall.recall_payload["source_episode_id"]
    assert recall.recall_payload["reinterpretation_mode"]
    assert recall.recall_payload["environment_summary"]
    assert recall.recall_payload["caution_bias"] >= 0.0
    assert recall.recall_payload["forgetting_pressure"] >= 0.0
    assert recall.recall_payload["replay_horizon"] >= 1
    assert recall.recall_payload["continuity_score"] >= 0.0
    assert recall.ignition_hints["recall_active"] is True
    assert recall.ignition_hints["reinterpretation"]["summary"]
    assert recall.ignition_hints["environment_pressure"]["summary"]
    assert recall.ignition_hints["persistence"]["social_grounding"] >= 0.0
    assert isinstance(recall.ignition_hints["relationship_trace"], dict)
    assert isinstance(recall.ignition_hints["community_profile_trace"], dict)
    assert recall.retrieval_summary["hits"][0]["id"] == "vision-1"
    assert recall.ignition_hints["terrain"]["ignition_potential"] >= 0.0
    assert recall.ignition_hints["forgetting"]["replay_horizon"] >= 1
    assert recall.ignition_hints["memory_orchestration"]["consolidation_priority"] >= 0.0
    assert recall.ignition_hints["working_memory"]["current_focus"]
    assert recall.retrieval_summary["inner_os_memory"]


def test_pre_turn_update_absorbs_working_memory_seed_from_local_context(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    pre = hooks.pre_turn_update(
        user_input={"text": ""},
        sensor_input={"body_stress_index": 0.18, "activity_level": 0.16},
        local_context={
            "last_gate_context": {"life_indicator": 0.48, "valence": 0.04, "arousal": 0.22},
            "working_memory_seed": {
                "semantic_seed_focus": "harbor slope",
                "semantic_seed_anchor": "harbor slope",
                "semantic_seed_strength": 0.82,
                "semantic_seed_recurrence": 1.34,
            },
        },
        current_state={"current_energy": 0.79, "temporal_pressure": 0.08},
        safety_bias=0.1,
    )
    assert pre.interaction_hints["working_memory_seed"]["semantic_seed_focus"] == "harbor slope"


def test_pre_turn_update_uses_related_person_from_working_memory_seed(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    pre = hooks.pre_turn_update(
        user_input={"text": "hello"},
        sensor_input={"voice_level": 0.18, "person_count": 1},
        local_context={
            "relational_world": {
                "culture_id": "coastal",
                "community_id": "harbor_collective",
                "social_role": "companion",
                "place_memory_anchor": "harbor slope",
            },
            "working_memory_seed": {
                "related_person_id": "user",
                "attachment": 0.76,
                "familiarity": 0.71,
                "trust_memory": 0.73,
                "relation_seed_summary": "gentle harbor companion thread",
                "relation_seed_strength": 0.72,
                "partner_address_hint": "companion",
                "partner_timing_hint": "open",
                "partner_stance_hint": "familiar",
                "partner_social_interpretation": "familiar:companion:open",
            },
        },
        current_state={"current_energy": 0.78},
    )
    assert pre.state.related_person_id == "user"
    assert pre.interaction_hints["counterpart_person_id"] == "user"
    assert pre.state.attachment >= 0.7
    assert pre.state.familiarity >= 0.68
    assert pre.state.trust_memory >= 0.69
    assert pre.state.relation_seed_summary == "gentle harbor companion thread"
    assert pre.state.partner_address_hint == "companion"
    assert pre.state.partner_timing_hint == "open"
    assert pre.state.partner_stance_hint == "familiar"
    assert pre.state.partner_social_interpretation == "familiar:companion:open"
    assert pre.interaction_hints["working_memory_seed"]["related_person_id"] == "user"
    assert pre.interaction_hints["working_memory_seed"]["relation_seed_strength"] == 0.72
    assert pre.interaction_hints["predicted_relational_mood"]["future_pull"] > 0.0
    assert pre.interaction_hints["predicted_relational_mood"]["shared_world_pull"] > 0.0
    assert pre.interaction_hints["predicted_utterance_stance"] in {"warm_check_in", "gentle_check_in"}
    assert pre.interaction_hints["predicted_nonverbal"]["gaze_mode"] in {"shared_attention_hold", "soft_hold"}
    assert pre.interaction_hints["predicted_shared_attention"] > 0.0


def test_post_turn_uses_predicted_vs_observed_interaction_alignment_for_relation_update(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    base_state = {
        "stress": 0.2,
        "recovery_need": 0.14,
        "route": "conscious",
        "talk_mode": "ask",
        "culture_id": "coastal",
        "community_id": "harbor_collective",
        "social_role": "companion",
        "memory_anchor": "harbor slope",
        "attachment": 0.72,
        "familiarity": 0.69,
        "trust_memory": 0.7,
        "continuity_score": 0.62,
        "social_grounding": 0.58,
        "partner_social_interpretation": "familiar:companion:open",
        "predicted_nonverbal": {
            "gaze_mode": "shared_attention_hold",
            "pause_mode": "patient_care",
            "proximity_mode": "gentle_near",
        },
        "predicted_shared_attention": 0.74,
        "predicted_distance_expectation": "gentle_near",
        "predicted_hesitation_tone": "patient_care",
        "opening_pace_windowed": "ready",
        "return_gaze_expectation": "steady_return",
    }
    aligned = hooks.post_turn_update(
        user_input={"text": "stay here"},
        output={
            "reply_text": "I am here.",
            "observed_gaze_mode": "shared_attention_hold",
            "observed_pause_mode": "patient_care",
            "observed_proximity_mode": "gentle_near",
            "observed_shared_attention": 0.76,
            "observed_hesitation_tone": "patient_care",
            "observed_shared_attention_window_mean": 0.72,
            "observed_repair_window_hold": 0.08,
        },
        current_state=base_state,
    )
    mismatched = hooks.post_turn_update(
        user_input={"text": "stay here"},
        output={
            "reply_text": "I am here.",
            "observed_gaze_mode": "avert",
            "observed_pause_mode": "measured_ritual",
            "observed_proximity_mode": "holding_space",
            "observed_shared_attention": 0.18,
            "observed_hesitation_tone": "measured_ritual",
            "observed_shared_attention_window_mean": 0.12,
            "observed_repair_window_hold": 0.52,
        },
        current_state=base_state,
    )
    assert aligned.audit_record["interaction_alignment_score"] > mismatched.audit_record["interaction_alignment_score"]
    assert aligned.state.social_grounding > mismatched.state.social_grounding
    assert aligned.state.trust_memory > mismatched.state.trust_memory
    assert mismatched.audit_record["distance_mismatch"] > aligned.audit_record["distance_mismatch"]
    assert mismatched.audit_record["opening_pace_mismatch"] > aligned.audit_record["opening_pace_mismatch"]
    assert mismatched.audit_record["return_gaze_mismatch"] > aligned.audit_record["return_gaze_mismatch"]
    assert mismatched.state.recent_strain > aligned.state.recent_strain
    assert mismatched.audit_record["observed_opening_pace"] in {"ready", "measured", "held"}
    assert mismatched.audit_record["observed_return_gaze"] in {"soft_return", "steady_return", "careful_return", "defer_return"}


def test_post_turn_derives_interaction_trace_from_raw_observation_signals(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    post = hooks.post_turn_update(
        user_input={"text": "hello"},
        output={
            "reply_text": "I am here.",
            "mutual_attention_score": 0.78,
            "gaze_hold_ratio": 0.66,
            "pause_latency": 0.54,
            "repair_signal": 0.58,
            "proximity_delta": 0.22,
            "hesitation_signal": 0.18,
        },
        current_state={
            "stress": 0.18,
            "recovery_need": 0.12,
            "route": "conscious",
            "talk_mode": "ask",
            "memory_anchor": "harbor slope",
            "predicted_nonverbal": {
                "gaze_mode": "shared_attention_hold",
                "pause_mode": "patient_care",
                "proximity_mode": "gentle_near",
            },
            "predicted_shared_attention": 0.72,
            "predicted_distance_expectation": "gentle_near",
            "predicted_hesitation_tone": "patient_care",
        },
    )
    assert post.audit_record["observed_gaze_mode"] == "shared_attention_hold"
    assert post.audit_record["observed_pause_mode"] in {"patient_care", "waiting"}
    assert post.audit_record["observed_proximity_mode"] == "gentle_near"
    assert post.audit_record["observed_opening_pace"] in {"ready", "measured", "held"}
    assert post.audit_record["observed_return_gaze"] in {"soft_return", "steady_return", "careful_return", "defer_return"}
    assert "shared_attention_detected" in post.audit_record["observed_trace_cues"]


def test_pre_turn_update_absorbs_person_registry_snapshot_for_partner(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    pre = hooks.pre_turn_update(
        user_input={"text": "hello"},
        sensor_input={"voice_level": 0.12, "person_count": 1},
        local_context={
            "relational_world": {
                "person_id": "user",
                "culture_id": "coastal",
                "community_id": "harbor_collective",
                "social_role": "companion",
            },
            "person_registry": {
                "persons": {
                    "user": {
                        "person_id": "user",
                        "stable_traits": {"community_marker": 1.0},
                        "adaptive_traits": {
                            "attachment": 0.83,
                            "familiarity": 0.79,
                            "trust_memory": 0.81,
                            "continuity_score": 0.74,
                            "social_grounding": 0.69,
                        },
                        "continuity_history": [{"observation": "user:social", "ambiguity": "0.18"}],
                        "confidence": 0.82,
                        "ambiguity_flag": False,
                    }
                },
                "uncertainty": 0.18,
            },
        },
        current_state={"current_energy": 0.78},
    )
    assert pre.state.related_person_id == "user"
    assert pre.state.attachment >= 0.75
    assert pre.state.familiarity >= 0.72
    assert pre.state.trust_memory >= 0.74
    assert pre.state.continuity_score > 0.6


def test_pre_turn_update_preserves_multiple_related_person_ids_from_seed_and_registry(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    pre = hooks.pre_turn_update(
        user_input={"text": "hello"},
        sensor_input={"voice_level": 0.08, "person_count": 2},
        local_context={
            "relational_world": {
                "culture_id": "coastal",
                "community_id": "harbor_collective",
                "social_role": "companion",
            },
            "working_memory_seed": {
                "related_person_ids": ["person:harbor", "person:friend"],
                "related_person_id": "person:harbor",
            },
            "person_registry": {
                "persons": {
                    "person:harbor": {
                        "person_id": "person:harbor",
                        "adaptive_traits": {
                            "attachment": 0.81,
                            "familiarity": 0.77,
                            "trust_memory": 0.79,
                            "continuity_score": 0.73,
                            "social_grounding": 0.7,
                        },
                        "confidence": 0.82,
                    },
                    "person:friend": {
                        "person_id": "person:friend",
                        "adaptive_traits": {
                            "attachment": 0.65,
                            "familiarity": 0.64,
                            "trust_memory": 0.66,
                            "continuity_score": 0.62,
                            "social_grounding": 0.6,
                        },
                        "confidence": 0.74,
                    },
                },
                "top_person_ids": ["person:harbor", "person:friend"],
                "dominant_person_id": "person:harbor",
                "total_people": 2,
                "uncertainty": 0.16,
            },
        },
        current_state={"current_energy": 0.8},
    )

    assert pre.state.related_person_id == "person:harbor"
    assert pre.state.related_person_ids == ["person:harbor", "person:friend"]


def test_memory_recall_carries_relation_seed_summary_into_payload(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {
            "kind": "relationship_trace",
            "summary": "gentle harbor companion thread",
            "text": "gentle harbor companion thread",
            "memory_anchor": "harbor slope",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "related_person_id": "user",
        },
        {
            "kind": "observed_real",
            "summary": "plain harbor sight",
            "text": "plain harbor sight",
            "memory_anchor": "harbor slope",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
    ])
    hooks = IntegrationHooks(memory_core=core)
    hooks.relational_world_core.absorb_context(
        {
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "place_memory_anchor": "harbor slope",
            "person_id": "user",
        }
    )
    recall = hooks.memory_recall(
        text_cue="harbor",
        current_state={
            "relation_seed_summary": "gentle harbor companion thread",
            "partner_social_interpretation": "familiar:companion:open",
            "related_person_id": "user",
        },
        retrieval_summary={},
    )
    assert recall.recall_payload["relation_seed_summary"] == "gentle harbor companion thread"
    assert recall.recall_payload["partner_social_interpretation"] == "familiar:companion:open"
    assert recall.recall_payload["record_kind"] == "relationship_trace"
    assert recall.ignition_hints["relation_seed_summary"] == "gentle harbor companion thread"


def test_post_turn_update_emits_person_registry_snapshot_for_partner(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    hooks.relational_world_core.absorb_context(
        {
            "person_id": "user",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "place_memory_anchor": "harbor slope",
        }
    )
    post = hooks.post_turn_update(
        user_input={"text": "stay with me"},
        output={"reply_text": "I will stay close."},
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.26,
            "recovery_need": 0.18,
            "safety_bias": 0.1,
            "attachment": 0.72,
            "familiarity": 0.68,
            "trust_memory": 0.71,
            "continuity_score": 0.58,
            "social_grounding": 0.56,
        },
        memory_write_candidates=[
            {
                "kind": "observed_real",
                "summary": "user harbor thread",
                "memory_anchor": "harbor slope",
                "related_person_id": "user",
                "policy_hint": "social,continuity",
                "continuity_score": 0.64,
                "consolidation_priority": 0.7,
            }
        ],
    )
    snapshot = post.person_registry_snapshot
    assert snapshot["persons"]["user"]["adaptive_traits"]["attachment"] > 0.0
    assert snapshot["persons"]["user"]["adaptive_traits"]["familiarity"] > 0.0
    assert snapshot["persons"]["user"]["adaptive_traits"]["trust_memory"] > 0.0
    assert post.audit_record["person_registry_person_id"] == "user"


def test_pre_turn_long_term_theme_can_shift_entry_mode(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    pre = hooks.pre_turn_update(
        user_input={"text": ""},
        sensor_input={"body_stress_index": 0.08, "activity_level": 0.08},
        local_context={
            "last_gate_context": {"life_indicator": 0.5, "valence": 0.05, "arousal": 0.14},
            "working_memory_seed": {
                "semantic_seed_focus": "harbor slope",
                "semantic_seed_anchor": "harbor slope",
                "semantic_seed_strength": 0.62,
                "semantic_seed_recurrence": 1.28,
                "long_term_theme_focus": "harbor slope",
                "long_term_theme_anchor": "harbor slope",
                "long_term_theme_kind": "place",
                "long_term_theme_summary": "quiet harbor slope memory",
                "long_term_theme_strength": 0.68,
            },
        },
        current_state={"current_energy": 0.86, "temporal_pressure": 0.04},
        safety_bias=0.08,
    )
    assert pre.state.current_focus in {"meaning", "place"}
    assert pre.state.long_term_theme_focus == "harbor slope"
    assert pre.state.long_term_theme_summary == "quiet harbor slope memory"
    assert pre.state.long_term_theme_strength == 0.68
    assert pre.state.talk_mode == "ask"


def test_integration_hooks_response_and_post_turn_flow(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    current_state = {
        "stress": 0.4,
        "recovery_need": 0.3,
        "safety_bias": 0.2,
        "memory_anchor": "雋ゑｽｯ騾包ｽｺ邵ｺ・ｮ髫ｪ菫ｶ繝ｻ",
        "mode": "reality",
        "route": "conscious",
        "talk_mode": "ask",
    }
    gate = hooks.response_gate(
        draft={"text": "I will stay here."},
        current_state=current_state,
        safety_signals={"safety_bias": 0.25},
    )
    assert gate.route == "conscious"
    assert gate.allowed_surface_intensity < 1.0
    assert gate.conscious_access["recall_active"] is True

    post = hooks.post_turn_update(
        user_input={"text": "what happened here"},
        output={"reply_text": "I will stay close to what is visible."},
        current_state=current_state,
        memory_write_candidates=[{"kind": "observed", "source": "test", "summary": "seed"}],
        transferred_lessons=[
            {
                "kind": "transferred_learning",
                "policy_hint": "pause_and_observe_under_ambiguity",
                "summary": "observe before approaching when ambiguity is high",
            }
        ],
    )
    assert post.audit_record["kind"] == "thin_audit"
    assert post.audit_record["transferred_lessons_used"] == 1
    assert post.audit_record["environment_pressure"]["summary"]
    assert post.audit_record["attachment"] >= 0.0
    assert post.audit_record["caution_bias"] >= 0.0
    assert post.audit_record["continuity_score"] >= 0.0
    assert post.audit_record["culture_resonance"] >= 0.0
    assert post.audit_record["community_resonance"] >= 0.0
    assert len(post.memory_appends) == 7
    assert post.state.norm_pressure >= 0.35
    assert post.state.temporal_pressure >= 0.0
    assert any(item.get("kind") == "identity_trace" for item in post.memory_appends)
    assert any(item.get("kind") == "relationship_trace" for item in post.memory_appends)
    assert any(item.get("kind") == "community_profile_trace" for item in post.memory_appends)
    assert any(item.get("kind") == "working_memory_trace" for item in post.memory_appends)
    community_profile = next(item for item in post.memory_appends if item.get("kind") == "community_profile_trace")
    assert community_profile["roughness_level"] >= 0.0
    assert community_profile["roughness_dwell"] >= 0.0
    assert community_profile["defensive_level"] >= 0.0
    assert (tmp_path / "inner_os_memory.jsonl").exists()


def test_post_turn_updates_affective_terrain_after_action_without_rewriting_same_turn_readout(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    current_state = {
        "stress": 0.46,
        "recovery_need": 0.34,
        "safety_bias": 0.18,
        "memory_anchor": "harbor",
        "mode": "reality",
        "route": "conscious",
        "talk_mode": "ask",
        "attachment": 0.62,
        "trust_memory": 0.58,
        "familiarity": 0.41,
        "continuity_score": 0.52,
        "social_grounding": 0.49,
        "recent_strain": 0.44,
        "meaning_inertia": 0.32,
        "pending_meaning": 0.36,
        "unresolved_count": 2,
        "related_person_id": "user",
        "prev_affective_position": {
            "z_aff": [0.2, 0.1, -0.1, 0.05],
            "cov": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "confidence": 0.58,
            "source_weights": {"state": 0.4, "qualia": 0.4, "memory": 0.2},
        },
        "affective_terrain_state": {
            "centers": [
                [0.0, 0.0, 0.0, 0.0],
                [0.8, 0.1, -0.1, 0.0],
            ],
            "widths": [0.55, 0.65],
            "value_weights": [0.0, 0.0],
            "approach_weights": [0.2, 0.3],
            "avoid_weights": [0.2, 0.4],
            "protect_weights": [0.2, 0.5],
            "anchor_labels": ["plain", "guarded_ridge"],
        },
        "terrain_readout": {
            "value": 0.0,
            "grad": [0.0, 0.0, 0.0, 0.0],
            "curvature": [0.0, 0.0, 0.0, 0.0],
            "approach_bias": 0.18,
            "avoid_bias": 0.32,
            "protect_bias": 0.64,
            "active_patch_index": 1,
            "active_patch_label": "guarded_ridge",
        },
        "protection_mode": {
            "mode": "stabilize",
            "strength": 0.72,
            "reasons": ["terrain_protect_bias", "body_load"],
        },
    }

    post = hooks.post_turn_update(
        user_input={"text": "stay with me"},
        output={"reply_text": "I will stay close to what is visible."},
        current_state=current_state,
        memory_write_candidates=[{"kind": "observed", "source": "test", "summary": "seed"}],
    )

    assert post.audit_record["terrain_plasticity_applied"] is True
    assert post.audit_record["terrain_plasticity_update"]["confidence"] >= 0.0
    assert post.audit_record["terrain_readout"]["protect_bias"] == current_state["terrain_readout"]["protect_bias"]

    updated_state = AffectiveTerrainState(
        centers=np.asarray(post.audit_record["affective_terrain_state"]["centers"], dtype=np.float32),
        widths=np.asarray(post.audit_record["affective_terrain_state"]["widths"], dtype=np.float32),
        value_weights=np.asarray(post.audit_record["affective_terrain_state"]["value_weights"], dtype=np.float32),
        approach_weights=np.asarray(post.audit_record["affective_terrain_state"]["approach_weights"], dtype=np.float32),
        avoid_weights=np.asarray(post.audit_record["affective_terrain_state"]["avoid_weights"], dtype=np.float32),
        protect_weights=np.asarray(post.audit_record["affective_terrain_state"]["protect_weights"], dtype=np.float32),
        anchor_labels=tuple(post.audit_record["affective_terrain_state"]["anchor_labels"]),
    )
    position = AffectivePositionState(
        z_aff=np.asarray(post.audit_record["affective_position"]["z_aff"], dtype=np.float32),
        cov=np.asarray(post.audit_record["affective_position"]["cov"], dtype=np.float32),
        confidence=float(post.audit_record["affective_position"]["confidence"]),
        source_weights=dict(post.audit_record["affective_position"]["source_weights"]),
    )
    updated_readout = BasicAffectiveTerrain().read(updated_state, position)

    assert updated_state.protect_weights.tolist() != current_state["affective_terrain_state"]["protect_weights"]
    assert updated_readout.protect_bias != current_state["terrain_readout"]["protect_bias"]



def test_integration_hooks_sensor_channels_shape_route_and_intent() -> None:
    hooks = IntegrationHooks()
    pre = hooks.pre_turn_update(
        user_input={"text": "stay with me"},
        sensor_input={
            "voice_level": 0.72,
            "person_count": 1,
            "body_stress_index": 0.4,
            "autonomic_balance": 0.36,
            "body_state_flag": "private_high_arousal",
            "privacy_tags": ["private"],
        },
        local_context={"last_gate_context": {"valence": 0.0, "arousal": 0.4}},
        current_state={"current_energy": 0.74},
        safety_bias=0.1,
    )
    assert pre.state.talk_mode == "soothe"
    assert pre.state.body_state_flag == "private_high_arousal"

    gate = hooks.response_gate(
        draft={"text": "I am here."},
        current_state={**pre.state.to_dict(), "mode": "reality"},
        safety_signals={"safety_bias": 0.1},
    )
    assert gate.conscious_access["intent"] == "soften"
    assert gate.hesitation_bias >= 0.42



def test_integration_hooks_relational_world_flows_into_recall_and_audit(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    pre = hooks.pre_turn_update(
        user_input={"text": "do you remember this place"},
        sensor_input={"body_stress_index": 0.2},
        local_context={
            "last_gate_context": {"valence": 0.1, "arousal": 0.2},
            "relational_world": {
                "culture_id": "coastal",
                "community_id": "harbor_collective",
                "social_role": "companion",
                "place_memory_anchor": "harbor slope",
                "nearby_objects": ["signboard", "lantern"],
            },
        },
        current_state={"current_energy": 0.82},
        safety_bias=0.1,
    )
    assert pre.interaction_hints["relational_world"]["culture_id"] == "coastal"

    hooks.memory_core.append_records([
        {
            "kind": "observed_real",
            "summary": "harbor slope walk",
            "memory_anchor": "harbor slope",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        }
    ])
    recall = hooks.memory_recall(
        text_cue="harbor",
        visual_cue="signboard",
        world_cue="market",
        current_state=pre.state.to_dict(),
        retrieval_summary={},
    )
    assert recall.recall_payload["culture_id"] == "coastal"
    assert recall.recall_payload["community_id"] == "harbor_collective"
    assert recall.recall_payload["record_kind"] in {"observed_real", "relationship_trace"}

    post = hooks.post_turn_update(
        user_input={"text": "do you remember this place"},
        output={"reply_text": "I remember the harbor slope."},
        current_state={**pre.state.to_dict(), "memory_anchor": "harbor slope"},
        memory_write_candidates=[recall.recall_payload],
        recall_payload=recall.recall_payload,
    )
    assert post.audit_record["culture_id"] == "coastal"
    assert post.audit_record["community_id"] == "harbor_collective"
    assert post.audit_record["reconstructed_memory_appended"] is True
    assert post.audit_record["reinterpretation_mode"]
    assert post.audit_record["environment_pressure"]["summary"]
    assert post.audit_record["attachment"] >= 0.0
    assert post.audit_record["caution_bias"] >= 0.0
    assert post.audit_record["continuity_score"] >= 0.0
    assert post.audit_record["culture_resonance"] >= 0.0
    assert post.audit_record["community_resonance"] >= 0.0
    assert any(item.get("kind") == "reconstructed" for item in post.memory_appends)
    assert post.audit_record["belonging"] >= 0.0
    assert post.audit_record["norm_pressure"] >= 0.0


def test_integration_hooks_restore_identity_trace_from_memory(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {
            "kind": "identity_trace",
            "summary": "stored identity",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "continuity_score": 0.71,
            "social_grounding": 0.64,
            "recent_strain": 0.28,
        }
    ])
    hooks = IntegrationHooks(memory_core=core)
    pre = hooks.pre_turn_update(
        user_input={"text": "hello again"},
        sensor_input={"body_stress_index": 0.2},
        local_context={"last_gate_context": {"valence": 0.1, "arousal": 0.2}, "relational_world": {"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"}},
        current_state={"current_energy": 0.81},
        safety_bias=0.1,
    )
    assert pre.state.continuity_score > 0.48
    assert pre.interaction_hints["identity_trace"]["continuity_score"] == 0.71




def test_integration_hooks_restore_community_profile_trace_from_memory(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {
            "kind": "community_profile_trace",
            "summary": "stored community profile",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "culture_resonance": 0.69,
            "community_resonance": 0.73,
            "norm_pressure": 0.66,
            "belonging": 0.71,
        }
    ])
    hooks = IntegrationHooks(memory_core=core)
    pre = hooks.pre_turn_update(
        user_input={"text": "hello again"},
        sensor_input={"body_stress_index": 0.2},
        local_context={"last_gate_context": {"valence": 0.1, "arousal": 0.2}, "relational_world": {"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "place_memory_anchor": "harbor slope"}},
        current_state={"current_energy": 0.81},
        safety_bias=0.1,
    )
    assert pre.interaction_hints["community_profile_trace"]["community_resonance"] == 0.73


def test_integration_hooks_restore_relationship_trace_from_memory(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {
            "kind": "relationship_trace",
            "summary": "stored relation",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
            "attachment": 0.74,
            "trust_memory": 0.67,
            "familiarity": 0.62,
        }
    ])
    hooks = IntegrationHooks(memory_core=core)
    pre = hooks.pre_turn_update(
        user_input={"text": "hello again"},
        sensor_input={"body_stress_index": 0.2},
        local_context={"last_gate_context": {"valence": 0.1, "arousal": 0.2}, "relational_world": {"culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion", "place_memory_anchor": "harbor slope"}},
        current_state={"current_energy": 0.81},
        safety_bias=0.1,
    )
    assert pre.state.attachment > 0.42
    assert pre.interaction_hints["relationship_trace"]["attachment"] == 0.74


def test_pre_turn_prefers_person_specific_relationship_trace(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {
            "kind": "relationship_trace",
            "summary": "generic relation",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
            "attachment": 0.52,
            "familiarity": 0.48,
            "trust_memory": 0.5,
        },
        {
            "kind": "relationship_trace",
            "summary": "user-specific relation",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
            "related_person_id": "user",
            "attachment": 0.88,
            "familiarity": 0.82,
            "trust_memory": 0.84,
            "role_alignment": 0.76,
        },
    ])
    hooks = IntegrationHooks(memory_core=core)
    pre = hooks.pre_turn_update(
        user_input={"text": "stay with me"},
        sensor_input={"body_stress_index": 0.14, "activity_level": 0.18},
        local_context={
            "last_gate_context": {"valence": 0.08, "arousal": 0.18},
            "relational_world": {
                "culture_id": "coastal",
                "community_id": "harbor_collective",
                "social_role": "companion",
                "place_memory_anchor": "harbor slope",
                "person_id": "user",
            },
        },
        current_state={"current_energy": 0.84, "temporal_pressure": 0.08},
        safety_bias=0.08,
    )
    assert pre.interaction_hints["counterpart_person_id"] == "user"
    assert pre.interaction_hints["relationship_trace"]["summary"] == "user-specific relation"
    assert pre.interaction_hints["relationship_trace"]["related_person_id"] == "user"
    assert pre.state.attachment > 0.7
    assert pre.state.talk_mode == "ask"


def test_memory_recall_prefers_person_specific_relationship_trace(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {
            "kind": "relationship_trace",
            "summary": "generic user bond",
            "text": "generic user bond",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
        },
        {
            "kind": "relationship_trace",
            "summary": "user-specific harbor bond",
            "text": "user-specific harbor bond",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "harbor slope",
            "related_person_id": "user",
        },
    ])
    hooks = IntegrationHooks(memory_core=core)
    hooks.relational_world_core.absorb_context(
        {
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "place_memory_anchor": "harbor slope",
            "person_id": "user",
        }
    )
    recall = hooks.memory_recall(
        text_cue="harbor slope",
        current_state={"memory_anchor": "harbor slope"},
        retrieval_summary={},
    )
    assert recall.recall_payload["record_kind"] == "relationship_trace"
    assert recall.recall_payload["related_person_id"] == "user"
    assert recall.recall_payload["counterpart_person_id"] == "user"


def test_response_gate_uses_community_resonance() -> None:
    hooks = IntegrationHooks()
    low = hooks.response_gate(
        draft={"text": "hello"},
        current_state={"talk_mode": "ask", "route": "conscious", "stress": 0.32, "recovery_need": 0.22, "safety_bias": 0.1, "norm_pressure": 0.46, "trust_bias": 0.5, "caution_bias": 0.48, "affiliation_bias": 0.52, "continuity_score": 0.55, "social_grounding": 0.51, "recent_strain": 0.28, "culture_resonance": 0.0, "community_resonance": 0.0},
        safety_signals={"safety_bias": 0.1},
    )
    high = hooks.response_gate(
        draft={"text": "hello"},
        current_state={"talk_mode": "ask", "route": "conscious", "stress": 0.32, "recovery_need": 0.22, "safety_bias": 0.1, "norm_pressure": 0.46, "trust_bias": 0.5, "caution_bias": 0.48, "affiliation_bias": 0.52, "continuity_score": 0.55, "social_grounding": 0.51, "recent_strain": 0.28, "culture_resonance": 0.62, "community_resonance": 0.7},
        safety_signals={"safety_bias": 0.1},
    )
    assert high.hesitation_bias < low.hesitation_bias
    assert high.allowed_surface_intensity > low.allowed_surface_intensity
    assert round(high.expression_hints["express_now"] + high.expression_hints["hold_back"], 4) == 1.0


def test_response_gate_uses_long_term_theme_as_small_relief() -> None:
    hooks = IntegrationHooks()
    base_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.32,
        "recovery_need": 0.22,
        "safety_bias": 0.1,
        "norm_pressure": 0.46,
        "trust_bias": 0.5,
        "caution_bias": 0.48,
        "affiliation_bias": 0.52,
        "continuity_score": 0.55,
        "social_grounding": 0.51,
        "recent_strain": 0.28,
        "culture_resonance": 0.22,
        "community_resonance": 0.3,
        "terrain_transition_roughness": 0.12,
    }
    low = hooks.response_gate(
        draft={"text": "hello"},
        current_state=base_state,
        safety_signals={"safety_bias": 0.1},
    )
    high = hooks.response_gate(
        draft={"text": "hello"},
        current_state={
            **base_state,
            "long_term_theme_focus": "harbor slope",
            "long_term_theme_anchor": "harbor slope",
            "long_term_theme_kind": "place",
            "long_term_theme_strength": 0.7,
        },
        safety_signals={"safety_bias": 0.1},
    )
    assert high.hesitation_bias < low.hesitation_bias
    assert high.allowed_surface_intensity > low.allowed_surface_intensity
    assert high.expression_hints["long_term_theme_kind"] == "place"
    assert high.expression_hints["long_term_theme_strength"] == 0.7


def test_response_gate_uses_person_specific_relation_as_small_relief() -> None:
    hooks = IntegrationHooks()
    base_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.28,
        "recovery_need": 0.2,
        "safety_bias": 0.1,
        "norm_pressure": 0.42,
        "trust_bias": 0.48,
        "caution_bias": 0.46,
        "affiliation_bias": 0.52,
        "continuity_score": 0.56,
        "social_grounding": 0.52,
        "recent_strain": 0.22,
        "terrain_transition_roughness": 0.1,
        "attachment": 0.82,
        "familiarity": 0.78,
        "trust_memory": 0.8,
    }
    low = hooks.response_gate(
        draft={"text": "hello"},
        current_state=base_state,
        safety_signals={"safety_bias": 0.1},
    )
    high = hooks.response_gate(
        draft={"text": "hello"},
        current_state={**base_state, "related_person_id": "user"},
        safety_signals={"safety_bias": 0.1},
    )
    assert high.hesitation_bias < low.hesitation_bias
    assert high.allowed_surface_intensity > low.allowed_surface_intensity
    assert high.expression_hints["related_person_id"] == "user"
    assert high.expression_hints["person_specific_relief"] > 0.0


def test_response_gate_uses_partner_social_interpretation_for_distance_tuning(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    base_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.24,
        "recovery_need": 0.16,
        "safety_bias": 0.1,
        "norm_pressure": 0.22,
        "trust_bias": 0.5,
        "caution_bias": 0.36,
        "affiliation_bias": 0.62,
        "continuity_score": 0.56,
        "social_grounding": 0.52,
        "recent_strain": 0.22,
        "terrain_transition_roughness": 0.1,
        "attachment": 0.82,
        "familiarity": 0.78,
        "trust_memory": 0.8,
        "related_person_id": "user",
    }
    neutral = hooks.response_gate(
        draft={"text": "hello"},
        current_state=base_state,
        safety_signals={"safety_bias": 0.1},
    )
    familiar = hooks.response_gate(
        draft={"text": "hello"},
        current_state={
            **base_state,
            "partner_address_hint": "companion",
            "partner_timing_hint": "open",
            "partner_stance_hint": "familiar",
            "partner_social_interpretation": "familiar:companion:open",
        },
        safety_signals={"safety_bias": 0.1},
    )
    assert familiar.allowed_surface_intensity > neutral.allowed_surface_intensity
    assert familiar.expression_hints["partner_social_interpretation"] == "familiar:companion:open"
    assert familiar.expression_hints["partner_style_relief"] > 0.0
    assert familiar.expression_hints["partner_style_caution"] == 0.0
    assert familiar.expression_hints["relational_future_pull"] > 0.0
    assert familiar.expression_hints["shared_world_pull"] > 0.0
    assert familiar.expression_hints["partner_utterance_stance"] in {"warm_check_in", "gentle_check_in"}
    assert familiar.expression_hints["nonverbal_gaze_mode"] in {"shared_attention_hold", "soft_hold"}
    assert familiar.expression_hints["future_loop_pull"] > 0.0
    assert familiar.expression_hints["past_loop_pull"] >= 0.0
    assert familiar.expression_hints["fantasy_loop_pull"] >= 0.0
    assert familiar.expression_hints["distance_expectation"] in {"future_opening", "gentle_near", "holding_space", "respectful_distance"}
    assert familiar.expression_hints["interaction_orchestration_mode"] in {"attune", "advance", "repair", "reflect", "contain"}
    assert familiar.expression_hints["interaction_coherence_score"] >= 0.0
    assert familiar.expression_hints["human_presence_signal"] >= 0.0
    assert familiar.expression_hints["observed_trace_gaze_mode"] in {"shared_attention_hold", "soft_hold", "steady", "gentle_return"}
    assert familiar.expression_hints["stream_shared_attention_level"] >= 0.0
    assert familiar.expression_hints["stream_shared_attention_window_mean"] >= 0.0
    assert familiar.expression_hints["stream_strained_pause_window_mean"] >= 0.0
    assert familiar.expression_hints["stream_repair_window_hold"] >= 0.0
    assert familiar.expression_hints["opening_pace_windowed"] in {"ready", "measured", "held"}
    assert familiar.expression_hints["return_gaze_expectation"] in {"soft_return", "steady_return", "careful_return", "defer_return"}
    assert familiar.expression_hints["stream_contact_readiness"] >= 0.0
    assert familiar.expression_hints["stream_update_count"] >= 1


def test_integration_hooks_marks_community_transition(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    core.append_records([
        {
            "kind": "community_profile_trace",
            "summary": "old community profile",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "culture_resonance": 0.72,
            "community_resonance": 0.78,
        }
    ])
    hooks = IntegrationHooks(memory_core=core)
    pre = hooks.pre_turn_update(
        user_input={"text": "new place"},
        sensor_input={"body_stress_index": 0.2},
        local_context={
            "last_gate_context": {"valence": 0.0, "arousal": 0.2},
            "relational_world": {
                "culture_id": "coastal",
                "community_id": "new_collective",
                "social_role": "companion",
            },
        },
        current_state={"current_energy": 0.8, "community_id": "harbor_collective", "culture_resonance": 0.72, "community_resonance": 0.78},
        safety_bias=0.1,
    )
    assert pre.interaction_hints["transition_signal"]["community_changed"] is True
    assert pre.interaction_hints["transition_signal"]["social_discontinuity"] > 0.0
    assert pre.state.community_resonance < 0.78


def test_integration_hooks_marks_place_and_privacy_shift(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    hooks = IntegrationHooks(memory_core=core)
    pre = hooks.pre_turn_update(
        user_input={"text": "where am i now"},
        sensor_input={
            "body_stress_index": 0.18,
            "body_state_flag": "private_high_arousal",
            "privacy_tags": ["private"],
            "person_count": 3,
        },
        local_context={
            "last_gate_context": {"valence": 0.02, "arousal": 0.26},
            "relational_world": {
                "culture_id": "coastal",
                "community_id": "harbor_collective",
                "social_role": "companion",
                "place_memory_anchor": "backstage room",
            },
        },
        current_state={
            "current_energy": 0.82,
            "memory_anchor": "open harbor",
            "person_count": 0,
            "body_state_flag": "normal",
        },
        safety_bias=0.08,
    )
    assert pre.interaction_hints["context_shift"]["place_changed"] is True
    assert pre.interaction_hints["context_shift"]["privacy_shift"] is True
    assert pre.interaction_hints["context_shift"]["situational_discontinuity"] > 0.0
    assert pre.interaction_hints["context_shift"]["social_discontinuity"] == 0.0
    assert pre.interaction_hints["terrain"]["transition_roughness"] > 0.0


def test_memory_recall_passes_terrain_roughness_into_reinterpretation(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    hooks = IntegrationHooks(memory_core=core)
    hooks.relational_world_core.absorb_context(
        {
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "place_memory_anchor": "backstage room",
        }
    )
    core.append_records([
        {"kind": "verified", "summary": "verified backstage route", "text": "verified backstage route", "memory_anchor": "backstage room", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
        {"kind": "reconstructed", "summary": "fragile backstage reading", "text": "fragile backstage reading", "memory_anchor": "backstage room", "culture_id": "coastal", "community_id": "harbor_collective", "social_role": "companion"},
    ])
    recall = hooks.memory_recall(
        text_cue="backstage",
        visual_cue="narrow room",
        world_cue="private",
        current_state={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "memory_anchor": "open harbor",
            "person_count": 0,
            "body_state_flag": "normal",
        },
        retrieval_summary={},
    )
    assert recall.ignition_hints["terrain"]["transition_roughness"] > 0.0
    assert recall.recall_payload["terrain_transition_roughness"] == round(recall.ignition_hints["reinterpretation"]["terrain_transition_roughness"], 4)
    assert recall.recall_payload["terrain_observed_roughness"] == round(recall.ignition_hints["terrain"]["transition_roughness"], 4)
    assert recall.recall_payload["terrain_transition_roughness"] == round(recall.ignition_hints["field_estimate"]["roughness_level"], 4)
    assert recall.recall_payload["recovery_reopening"] >= 0.0


def test_memory_recall_uses_working_memory_replay_signature_for_rerank(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    hooks = IntegrationHooks(memory_core=core)
    core.append_records([
        {
            "kind": "observed_real",
            "summary": "quiet harbor slope memory",
            "text": "quiet harbor slope memory",
            "memory_anchor": "harbor slope",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
        {
            "kind": "observed_real",
            "summary": "market walkway memory",
            "text": "market walkway memory",
            "memory_anchor": "market street",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
    ])
    recall = hooks.memory_recall(
        text_cue="memory",
        current_state={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "working_memory_replay_focus": "harbor",
            "working_memory_replay_anchor": "harbor slope",
            "working_memory_replay_strength": 0.82,
        },
        retrieval_summary={},
    )
    assert recall.recall_payload["memory_anchor"] == "harbor slope"
    assert recall.recall_payload["replay_signature_focus"] == "harbor"
    assert recall.recall_payload["replay_signature_anchor"] == "harbor slope"
    assert recall.recall_payload["replay_signature_strength"] == 0.82


def test_memory_recall_uses_semantic_seed_for_rerank(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    hooks = IntegrationHooks(memory_core=core)
    core.append_records([
        {
            "kind": "observed_real",
            "summary": "quiet harbor slope memory",
            "text": "quiet harbor slope memory",
            "memory_anchor": "harbor slope",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
        {
            "kind": "observed_real",
            "summary": "market walkway memory",
            "text": "market walkway memory",
            "memory_anchor": "market street",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
    ])
    recall = hooks.memory_recall(
        text_cue="memory",
        current_state={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "semantic_seed_focus": "harbor",
            "semantic_seed_anchor": "harbor slope",
            "semantic_seed_strength": 0.74,
        },
        retrieval_summary={},
    )
    assert recall.recall_payload["memory_anchor"] == "harbor slope"
    assert recall.recall_payload["semantic_seed_focus"] == "harbor"
    assert recall.recall_payload["semantic_seed_anchor"] == "harbor slope"
    assert recall.recall_payload["semantic_seed_strength"] == 0.74


def test_memory_recall_uses_long_term_theme_for_rerank(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    hooks = IntegrationHooks(memory_core=core)
    core.append_records([
        {
            "kind": "observed_real",
            "summary": "quiet harbor slope memory",
            "text": "quiet harbor slope memory",
            "memory_anchor": "harbor slope",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
        {
            "kind": "observed_real",
            "summary": "market walkway memory",
            "text": "market walkway memory",
            "memory_anchor": "market street",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
        },
    ])
    recall = hooks.memory_recall(
        text_cue="memory",
        current_state={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "long_term_theme_focus": "harbor",
            "long_term_theme_anchor": "harbor slope",
            "long_term_theme_kind": "place",
            "long_term_theme_summary": "quiet harbor slope memory",
            "long_term_theme_strength": 0.69,
        },
        retrieval_summary={},
    )
    assert recall.recall_payload["memory_anchor"] == "harbor slope"
    assert recall.recall_payload["long_term_theme_focus"] == "harbor"
    assert recall.recall_payload["long_term_theme_anchor"] == "harbor slope"
    assert recall.recall_payload["long_term_theme_summary"] == "quiet harbor slope memory"
    assert recall.recall_payload["long_term_theme_strength"] == 0.69

def test_response_gate_uses_recalled_tentative_bias() -> None:
    hooks = IntegrationHooks()
    base_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.22,
        "recovery_need": 0.18,
        "safety_bias": 0.1,
        "norm_pressure": 0.42,
        "trust_bias": 0.48,
        "caution_bias": 0.46,
        "affiliation_bias": 0.5,
        "continuity_score": 0.56,
        "social_grounding": 0.52,
        "recent_strain": 0.24,
        "culture_resonance": 0.31,
        "community_resonance": 0.34,
        "terrain_transition_roughness": 0.12,
    }
    low = hooks.response_gate(
        draft={"text": "I will stay here."},
        current_state=base_state,
        safety_signals={"safety_bias": 0.1},
    )
    high = hooks.response_gate(
        draft={"text": "I will stay here."},
        current_state={**base_state, "recalled_tentative_bias": 0.72},
        safety_signals={"safety_bias": 0.1},
    )
    assert high.allowed_surface_intensity < low.allowed_surface_intensity
    assert high.expression_hints["recalled_tentative_bias"] == 0.72
    assert high.expression_hints["avoid_definitive_interpretation"] is True
    assert high.expression_hints["hold_back"] > low.expression_hints["hold_back"]
    assert round(high.expression_hints["express_now"] + high.expression_hints["hold_back"], 4) == 1.0


def test_response_gate_respects_grounding_deferral_mode() -> None:
    hooks = IntegrationHooks()
    base_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.24,
        "recovery_need": 0.18,
        "safety_bias": 0.1,
        "norm_pressure": 0.42,
        "trust_bias": 0.44,
        "caution_bias": 0.46,
        "affiliation_bias": 0.5,
        "continuity_score": 0.42,
        "social_grounding": 0.3,
        "recent_strain": 0.36,
        "terrain_transition_roughness": 0.58,
        "recalled_tentative_bias": 0.22,
    }
    steady = hooks.response_gate(
        draft={"text": "I think this means we are safe."},
        current_state=base_state,
        safety_signals={"safety_bias": 0.1},
    )
    deferred = hooks.response_gate(
        draft={"text": "I think this means we are safe."},
        current_state={**base_state, "recalled_reinterpretation_mode": "grounding_deferral"},
        safety_signals={"safety_bias": 0.1},
    )
    assert deferred.hesitation_bias > steady.hesitation_bias
    assert deferred.allowed_surface_intensity < steady.allowed_surface_intensity
    assert deferred.expression_hints["recalled_reinterpretation_mode"] == "grounding_deferral"
    assert deferred.expression_hints["avoid_definitive_interpretation"] is True
    assert deferred.expression_hints["favor_grounded_observation"] is True
    assert deferred.conscious_access["intent"] == "clarify"

def test_post_turn_slows_identity_stabilization_under_tentative_recall(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    base_state = {
        "stress": 0.26,
        "recovery_need": 0.18,
        "safety_bias": 0.12,
        "memory_anchor": "harbor slope",
        "route": "conscious",
        "talk_mode": "ask",
        "culture_id": "coastal",
        "community_id": "harbor_collective",
        "social_role": "companion",
        "belonging": 0.53,
        "trust_bias": 0.47,
        "norm_pressure": 0.41,
        "role_commitment": 0.46,
        "attachment": 0.64,
        "familiarity": 0.59,
        "trust_memory": 0.62,
        "role_alignment": 0.57,
        "terrain_transition_roughness": 0.08,
        "recalled_tentative_bias": 0.0,
    }
    low = hooks.post_turn_update(
        user_input={"text": "what does this mean"},
        output={"reply_text": "I will stay close to what is visible."},
        current_state=base_state,
    )
    high = hooks.post_turn_update(
        user_input={"text": "what does this mean"},
        output={"reply_text": "I will stay close to what is visible."},
        current_state={**base_state, "terrain_transition_roughness": 0.76, "recalled_tentative_bias": 0.74},
    )
    assert high.state.role_commitment < low.state.role_commitment
    assert high.state.norm_pressure >= low.state.norm_pressure
    assert high.state.belonging < low.state.belonging
    assert high.state.trust_bias < low.state.trust_bias


def test_post_turn_reopens_identity_stabilization_when_recovery_returns(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    base_state = {
        "stress": 0.26,
        "recovery_need": 0.18,
        "safety_bias": 0.12,
        "memory_anchor": "harbor slope",
        "route": "conscious",
        "talk_mode": "ask",
        "culture_id": "coastal",
        "community_id": "harbor_collective",
        "social_role": "companion",
        "belonging": 0.53,
        "trust_bias": 0.47,
        "norm_pressure": 0.41,
        "role_commitment": 0.46,
        "attachment": 0.64,
        "familiarity": 0.59,
        "trust_memory": 0.62,
        "role_alignment": 0.57,
        "terrain_transition_roughness": 0.42,
        "recalled_tentative_bias": 0.28,
        "recovery_reopening": 0.0,
    }
    held = hooks.post_turn_update(
        user_input={"text": "what does this mean"},
        output={"reply_text": "I will stay close to what is visible."},
        current_state=base_state,
    )
    reopened = hooks.post_turn_update(
        user_input={"text": "what does this mean"},
        output={"reply_text": "I think we can open this a little more clearly now."},
        current_state={**base_state, "recovery_reopening": 0.62},
    )
    assert reopened.state.identity_update_strength > held.state.identity_update_strength
    assert reopened.state.social_update_strength > held.state.social_update_strength
    assert reopened.state.norm_pressure >= held.state.norm_pressure
    assert reopened.audit_record["recovery_reopening"] >= 0.0

def test_post_turn_audit_records_development_update_strength(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    post = hooks.post_turn_update(
        user_input={"text": "what does this mean"},
        output={"reply_text": "I will stay close to what is visible."},
        current_state={
            "stress": 0.26,
            "recovery_need": 0.18,
            "safety_bias": 0.12,
            "memory_anchor": "harbor slope",
            "route": "conscious",
            "talk_mode": "ask",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "belonging": 0.53,
            "trust_bias": 0.47,
            "norm_pressure": 0.41,
            "role_commitment": 0.46,
            "attachment": 0.64,
            "familiarity": 0.59,
            "trust_memory": 0.62,
            "role_alignment": 0.57,
            "terrain_transition_roughness": 0.76,
            "recalled_tentative_bias": 0.74,
        },
    )
    assert post.audit_record["social_update_strength"] < 1.0
    assert post.audit_record["identity_update_strength"] < 1.0
    identity_trace = next(item for item in post.memory_appends if item.get("kind") == "identity_trace")
    assert identity_trace["identity_update_strength"] == post.audit_record["identity_update_strength"]


def test_post_turn_uses_working_memory_replay_signature_for_reconsolidation_priority(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    base_state = {
        "stress": 0.24,
        "recovery_need": 0.16,
        "safety_bias": 0.12,
        "memory_anchor": "harbor slope",
        "route": "conscious",
        "talk_mode": "ask",
        "culture_id": "coastal",
        "community_id": "harbor_collective",
        "social_role": "companion",
        "belonging": 0.53,
        "trust_bias": 0.47,
        "norm_pressure": 0.41,
        "role_commitment": 0.46,
        "attachment": 0.64,
        "familiarity": 0.59,
        "trust_memory": 0.62,
        "role_alignment": 0.57,
        "terrain_transition_roughness": 0.18,
        "working_memory_replay_focus": "harbor",
        "working_memory_replay_anchor": "harbor slope",
        "working_memory_replay_strength": 0.8,
        "long_term_theme_summary": "quiet harbor slope memory",
        "long_term_theme_strength": 0.64,
    }
    aligned = hooks.post_turn_update(
        user_input={"text": "what happened at the harbor"},
        output={"reply_text": "I keep returning to the harbor slope."},
        current_state=base_state,
        recall_payload={
            "record_kind": "observed_real",
            "summary": "harbor slope walk",
            "text": "we walked along the harbor slope",
            "memory_anchor": "harbor slope",
            "source_episode_id": "ep-1",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
    )
    unaligned = hooks.post_turn_update(
        user_input={"text": "what happened there"},
        output={"reply_text": "I can only say it still feels distant."},
        current_state={
            **base_state,
            "working_memory_replay_focus": "market",
            "working_memory_replay_anchor": "market street",
            "long_term_theme_summary": "market street noise",
        },
        recall_payload={
            "record_kind": "observed_real",
            "summary": "harbor slope walk",
            "text": "we walked along the harbor slope",
            "memory_anchor": "harbor slope",
            "source_episode_id": "ep-1",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
    )
    assert aligned.audit_record["working_memory_replay_reinforcement"] > unaligned.audit_record["working_memory_replay_reinforcement"]
    assert aligned.audit_record["long_term_theme_reinforcement"] > unaligned.audit_record["long_term_theme_reinforcement"]
    aligned_reconstructed = next(item for item in aligned.memory_appends if item.get("kind") == "reconstructed")
    assert aligned_reconstructed["working_memory_replay_reinforcement"] > 0.0
    assert aligned_reconstructed["long_term_theme_reinforcement"] > 0.0
    assert aligned_reconstructed["consolidation_priority"] >= unaligned.audit_record["consolidation_priority"]

def test_response_gate_prefers_clarify_when_identity_update_is_low() -> None:
    hooks = IntegrationHooks()
    gate = hooks.response_gate(
        draft={"text": "hello"},
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.2,
            "recovery_need": 0.16,
            "safety_bias": 0.1,
            "norm_pressure": 0.42,
            "trust_bias": 0.5,
            "caution_bias": 0.44,
            "affiliation_bias": 0.52,
            "continuity_score": 0.56,
            "social_grounding": 0.54,
            "recent_strain": 0.26,
            "terrain_transition_roughness": 0.14,
            "identity_update_strength": 0.54,
            "social_update_strength": 0.66,
        },
        safety_signals={"safety_bias": 0.1},
    )
    assert gate.conscious_access["intent"] == "clarify"
    assert gate.expression_hints["clarify_first"] is True
    assert gate.expression_hints["question_bias"] > 0.0


def test_response_gate_emits_constraint_and_conscious_workspace_bundle() -> None:
    hooks = IntegrationHooks()
    gate = hooks.response_gate(
        draft={"text": "hello"},
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.74,
            "recovery_need": 0.58,
            "safety_bias": 0.76,
            "norm_pressure": 0.68,
            "trust_bias": 0.44,
            "caution_bias": 0.72,
            "affiliation_bias": 0.42,
            "continuity_score": 0.46,
            "social_grounding": 0.4,
            "recent_strain": 0.62,
            "related_person_id": "user",
            "current_focus": "person:user",
            "partner_timing_hint": "delayed",
            "partner_stance_hint": "respectful",
            "partner_social_interpretation": "repair_window",
            "association_reweighting_focus": "repeated_links",
            "association_reweighting_reason": "repeated_insight_trace",
            "insight_terrain_shape_target": "soft_relation",
        },
        safety_signals={"safety_bias": 0.76},
    )
    constraint = gate.expression_hints["constraint_field"]
    contact_field = gate.expression_hints["contact_field"]
    contact_dynamics = gate.expression_hints["contact_dynamics"]
    qualia_state = gate.expression_hints["qualia_state"]
    dot_seeds = gate.expression_hints["dot_seeds"]
    association_graph = gate.expression_hints["association_graph"]
    insight_event = gate.expression_hints["insight_event"]
    affective_position = gate.expression_hints["affective_position"]
    terrain_readout = gate.expression_hints["terrain_readout"]
    protection_mode = gate.expression_hints["protection_mode"]
    access_projection = gate.expression_hints["access_projection"]
    access_dynamics = gate.expression_hints["access_dynamics"]
    workspace = gate.expression_hints["conscious_workspace"]
    conversational_objects = gate.expression_hints["conversational_objects"]
    object_operations = gate.expression_hints["object_operations"]
    interaction_effects = gate.expression_hints["interaction_effects"]
    conversation_contract = gate.expression_hints["conversation_contract"]
    interaction_judgement_view = gate.expression_hints["interaction_judgement_view"]
    interaction_judgement_summary = gate.expression_hints["interaction_judgement_summary"]
    interaction_condition_report = gate.expression_hints["interaction_condition_report"]
    interaction_inspection_report = gate.expression_hints["interaction_inspection_report"]
    interaction_audit_bundle = gate.expression_hints["interaction_audit_bundle"]
    interaction_audit_report = gate.expression_hints["interaction_audit_report"]
    resonance = gate.expression_hints["resonance_evaluation"]
    packet = gate.expression_hints["interaction_policy_packet"]
    action_posture = gate.expression_hints["action_posture"]
    assert constraint["reportability_limit"] == "withhold"
    assert contact_field["field_mode"] in {"guarded", "relational", "focused", "ambient"}
    assert contact_field["points"]
    assert contact_dynamics["stabilized_points"]
    assert contact_dynamics["dynamics_mode"] in {"fresh", "guarded_fresh", "reentrant", "guarded_reentry"}
    assert qualia_state["qualia"]
    assert qualia_state["gate"]
    assert len(qualia_state["qualia"]) == len(gate.expression_hints["qualia_axis_labels"])
    assert gate.expression_hints["qualia_estimator_health"]["trust"] >= 0.0
    assert isinstance(gate.expression_hints["qualia_protection_grad_x"], list)
    assert dot_seeds["seeds"]
    assert association_graph["state_hint"]["link_weights"] is not None
    assert insight_event["score"]["total"] >= 0.0
    assert len(affective_position["z_aff"]) == len(affective_position["cov"])
    assert affective_position["confidence"] >= 0.0
    assert terrain_readout["active_patch_label"]
    assert protection_mode["mode"] in {"monitor", "contain", "stabilize", "repair", "shield"}
    assert gate.expression_hints["qualia_hint_source"] == "shared"
    assert gate.expression_hints["qualia_hint_fallback_reason"] == "prebuilt_shared_view"
    assert gate.expression_hints["qualia_hint_expected_source"] == "shared"
    assert gate.expression_hints["qualia_hint_expected_mismatch"] is False
    assert gate.expression_hints["qualia_planner_view"]["trust"] >= 0.0
    assert gate.expression_hints["qualia_planner_view"]["felt_energy"] >= 0.0
    assert access_projection["projection_mode"] == "guarded_projection"
    assert access_projection["actionable_slice"]
    assert "access_qualia_input" in access_projection["cues"]
    assert "access_terrain_input" in access_projection["cues"]
    assert "access_insight_input" in access_projection["cues"]
    assert access_dynamics["stabilized_regions"]
    assert access_dynamics["dynamics_mode"] in {"fresh_projection", "guarded_projection", "inertial_projection", "guarded_inertial_projection"}
    assert "force_reportability" in constraint["do_not_cross"]
    assert workspace["workspace_mode"] == "guarded_foreground"
    assert workspace["withheld_slice"]
    assert workspace["actionable_slice"]
    assert workspace["reportability_gate"]["gate_mode"] == "withhold"
    assert workspace["slot_scores"]
    assert workspace["winner_margin"] >= 0.0
    assert workspace["dominant_inputs"]
    assert conversational_objects["objects"]
    assert conversational_objects["primary_object_id"]
    assert gate.expression_hints["conversational_object_pressure_balance"] > 0.0
    assert object_operations["operations"]
    assert object_operations["question_budget"] == 0
    assert gate.expression_hints["object_operation_question_pressure"] > 0.0
    assert gate.expression_hints["object_operation_defer_dominance"] > 0.0
    assert interaction_effects["effects"]
    assert conversation_contract["focus_now"]
    assert conversation_contract["response_action_now"]["question_budget"] == 0
    assert conversation_contract["wanted_effect_on_other"]
    assert interaction_judgement_view["observed_signals"]
    assert interaction_judgement_view["inferred_signals"]
    assert interaction_judgement_view["selected_object_labels"]
    assert interaction_judgement_view["active_operation_labels"]
    assert interaction_judgement_view["intended_effect_labels"]
    assert interaction_judgement_summary["observed_lines"]
    assert interaction_judgement_summary["inferred_lines"]
    assert interaction_judgement_summary["selected_object_lines"]
    assert interaction_judgement_summary["operation_lines"]
    assert interaction_judgement_summary["intended_effect_lines"]
    assert interaction_condition_report["scene_lines"]
    assert interaction_condition_report["relation_lines"]
    assert interaction_condition_report["memory_lines"]
    assert interaction_condition_report["integration_lines"]
    assert interaction_condition_report["report_lines"]
    assert interaction_inspection_report["case_reports"]
    assert interaction_inspection_report["report_lines"]
    assert any("current_case" in line for line in interaction_inspection_report["report_lines"])
    assert interaction_audit_bundle["report_lines"]
    assert interaction_audit_bundle["key_metrics"]["question_budget"] >= 0
    assert interaction_audit_report["report_lines"]
    assert resonance["estimated_other_person_state"]["detail_room_level"] in {"low", "medium", "high"}
    assert resonance["recommended_family_id"] in {"wait", "repair", "contain", "reflect", "attune", "clarify", "withdraw", "co_move"}
    assert resonance["assessments"]
    assert packet["conscious_workspace"]["workspace_mode"] == "guarded_foreground"
    assert packet["workspace_decision"]["workspace_mode"] == "guarded_foreground"
    assert packet["workspace_decision"]["winner_margin"] >= 0.0
    assert packet["workspace_decision"]["dominant_inputs"]
    assert packet["conversational_objects"]["primary_object_id"] == conversational_objects["primary_object_id"]
    assert packet["object_operations"]["question_budget"] == 0
    assert packet["interaction_effects"]["effects"]
    assert packet["conversation_contract"]["focus_now"] == conversation_contract["focus_now"]
    assert packet["interaction_judgement_view"]["observed_signals"]
    assert packet["actionable_slice"]
    assert packet["reportability_gate_mode"] == "withhold"
    assert packet["resonance_evaluation"]["recommended_family_id"] == resonance["recommended_family_id"]
    assert packet["qualia_planner_view"] == gate.expression_hints["qualia_planner_view"]
    assert packet["affective_position"] == affective_position
    assert packet["terrain_readout"] == terrain_readout
    assert packet["protection_mode"] == protection_mode
    assert packet["insight_event"] == insight_event
    assert packet["association_reweighting_focus"] == "repeated_links"
    assert packet["association_reweighting_reason"] == "repeated_insight_trace"
    assert packet["insight_terrain_shape_target"] == "soft_relation"
    assert packet["reaction_vs_overnight_bias"]["same_turn"]["protection_mode"] == protection_mode["mode"]
    assert packet["reaction_vs_overnight_bias"]["same_turn"]["body_homeostasis_state"] in {"steady", "strained", "recovering", "depleted"}
    assert packet["reaction_vs_overnight_bias"]["same_turn"]["agenda_state"] in {"hold", "revisit", "repair", "step_forward"}
    assert packet["reaction_vs_overnight_bias"]["overnight"]["association_reweighting_focus"] == "repeated_links"
    assert packet["commitment_state"]["state"] in {"waver", "settle", "commit"}
    assert packet["commitment_state"]["target"] in {"hold", "stabilize", "repair", "bond_protect", "step_forward"}
    assert packet["commitment_state"]["winner_margin"] >= 0.0
    assert packet["agenda_state"]["state"] in {"hold", "revisit", "repair", "step_forward"}
    assert packet["agenda_state"]["winner_margin"] >= 0.0
    assert packet["body_homeostasis_state"]["winner_margin"] >= 0.0
    assert packet["relational_continuity_state"]["winner_margin"] >= 0.0
    assert packet["relation_competition_state"]["winner_margin"] >= 0.0
    assert packet["social_topology"] in {"ambient", "one_to_one", "multi_person"}
    assert isinstance(packet["active_relation_table"]["entries"], list)
    assert packet["memory_write_class_bias"]["winner_margin"] >= 0.0
    assert packet["memory_write_class_bias"]["dominant_inputs"]
    assert packet["protection_mode_decision"]["winner_margin"] >= 0.0
    assert round(packet["protection_mode_decision"]["scores"][protection_mode["mode"]], 4) == round(protection_mode["strength"], 4)
    assert packet["qualia_memory_bias"]["priority"] in {"ambient", "foreground_trace", "stability_trace"}
    assert "force_reportability" in packet["do_not_cross"]
    assert gate.expression_hints["interaction_policy_association_reweighting_focus"] == "repeated_links"
    assert gate.expression_hints["interaction_policy_association_reweighting_reason"] == "repeated_insight_trace"
    assert gate.expression_hints["interaction_policy_insight_terrain_shape_target"] == "soft_relation"
    assert gate.expression_hints["interaction_policy_overnight_bias_roles"]["insight_terrain_shape_target"] == "soft_relation"
    assert gate.expression_hints["interaction_policy_reaction_vs_overnight_bias"]["same_turn"]["memory_write_class"] == packet["memory_write_class"]
    assert gate.expression_hints["interaction_policy_reaction_vs_overnight_bias"]["same_turn"]["agenda_state"] == packet["agenda_state"]["state"]
    assert gate.expression_hints["interaction_policy_memory_write_class_bias"]["winner_margin"] >= 0.0
    assert gate.expression_hints["interaction_policy_protection_mode_decision"]["winner_margin"] >= 0.0
    assert gate.expression_hints["interaction_policy_agenda_state"]["winner_margin"] >= 0.0
    assert gate.expression_hints["interaction_policy_body_homeostasis_state"]["winner_margin"] >= 0.0
    assert gate.expression_hints["interaction_policy_commitment_state"]["winner_margin"] >= 0.0
    assert gate.expression_hints["interaction_policy_relational_continuity_state"]["winner_margin"] >= 0.0
    assert gate.expression_hints["interaction_policy_relation_competition_state"]["winner_margin"] >= 0.0
    assert isinstance(gate.expression_hints["interaction_policy_active_relation_table"]["entries"], list)
    assert gate.expression_hints["conscious_workspace_slot_scores"]
    assert gate.expression_hints["conscious_workspace_winner_margin"] >= 0.0
    assert gate.expression_hints["conscious_workspace_dominant_inputs"]
    assert action_posture["workspace_mode"] == "guarded_foreground"
    assert action_posture["boundary_mode"] in {"guarded", "protective"}
    assert action_posture["actionable_slice"]
    assert action_posture["protection_mode_name"] == protection_mode["mode"]
    assert action_posture["body_homeostasis_name"] in {"steady", "strained", "recovering", "depleted"}
    assert action_posture["commitment_target"] == packet["commitment_state"]["target"]
    assert action_posture["relational_continuity_name"] in {"distant", "holding_thread", "reopening", "co_regulating"}
    assert gate.expression_hints["actuation_plan"]["protection_mode"] == protection_mode
    assert gate.expression_hints["actuation_plan"]["commitment_target"] == packet["commitment_state"]["target"]
    assert gate.expression_hints["actuation_plan"]["body_homeostasis_name"] in {"steady", "strained", "recovering", "depleted"}
    assert gate.expression_hints["actuation_plan"]["relational_continuity_name"] in {"distant", "holding_thread", "reopening", "co_regulating"}
    assert packet["memory_write_class"] in {
        "episodic",
        "body_risk",
        "bond_protection",
        "repair_trace",
        "unresolved_tension",
        "safe_repeat",
        "insight_trace",
    }
    assert gate.expression_hints["interaction_policy_memory_write_class"] == packet["memory_write_class"]


def test_response_gate_routes_triggered_insight_into_action_reflection() -> None:
    hooks = IntegrationHooks()
    gate = hooks.response_gate(
        draft={"text": "shared thread feels different now"},
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.28,
            "recovery_need": 0.24,
            "safety_bias": 0.2,
            "norm_pressure": 0.22,
            "trust_bias": 0.58,
            "caution_bias": 0.3,
            "affiliation_bias": 0.62,
            "continuity_score": 0.64,
            "social_grounding": 0.58,
            "recent_strain": 0.22,
            "memory_anchor": "shared_thread",
            "relation_seed_summary": "shared thread",
            "replay_intensity": 0.78,
            "meaning_inertia": 0.52,
            "pending_meaning": 0.48,
            "unresolved_count": 2,
            "related_person_id": "user",
            "attachment": 0.72,
            "trust_memory": 0.64,
            "current_focus": "shared_thread",
        },
        safety_signals={"safety_bias": 0.2},
    )

    insight_event = gate.expression_hints["insight_event"]
    action_posture = gate.expression_hints["action_posture"]
    actuation_plan = gate.expression_hints["actuation_plan"]

    assert insight_event["triggered"] is True
    assert insight_event["connected_seed_ids"]
    assert "pause_for_orientation" in action_posture["next_action_candidates"]
    assert "pause_for_orientation" in actuation_plan["action_queue"]
    assert actuation_plan["insight_event"] == insight_event


def test_response_gate_qualia_uses_previous_protection_gradient() -> None:
    hooks = IntegrationHooks()
    base_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.34,
        "recovery_need": 0.28,
        "safety_bias": 0.22,
        "norm_pressure": 0.3,
        "trust_bias": 0.56,
        "caution_bias": 0.36,
        "affiliation_bias": 0.58,
        "continuity_score": 0.54,
        "social_grounding": 0.52,
        "recent_strain": 0.26,
        "voice_level": 0.24,
        "autonomic_balance": 0.46,
        "approach_confidence": 0.18,
        "prev_qualia": [0.0] * 14,
        "prev_qualia_habituation": [0.0] * 14,
    }
    low = hooks.response_gate(
        draft={"text": "stay close to what is here"},
        current_state={
            **base_state,
            "prev_protection_grad_x": [0.0] * 14,
        },
        safety_signals={"safety_bias": 0.22, "repair_signal": 0.18},
    )
    high = hooks.response_gate(
        draft={"text": "stay close to what is here"},
        current_state={
            **base_state,
            "prev_protection_grad_x": [0.0, 0.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        safety_signals={"safety_bias": 0.22, "repair_signal": 0.18},
    )
    low_gate = low.expression_hints["qualia_state"]["gate"]
    high_gate = high.expression_hints["qualia_state"]["gate"]
    low_qualia = low.expression_hints["qualia_state"]["qualia"]
    high_qualia = high.expression_hints["qualia_state"]["qualia"]
    assert high_gate[4] != low_gate[4] or high_qualia[4] != low_qualia[4]


def test_post_turn_applies_memory_write_class_to_appends(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    post = hooks.post_turn_update(
        user_input={"text": "stay here"},
        output={"reply_text": "I will stay close."},
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.58,
            "recovery_need": 0.46,
            "safety_bias": 0.52,
            "memory_anchor": "harbor slope",
            "memory_write_class": "bond_protection",
            "memory_write_class_reason": "bond_protection_pressure",
            "protection_mode": {"mode": "contain", "strength": 0.62, "reasons": ["terrain_protect_bias"]},
            "terrain_readout": {
                "value": -0.2,
                "grad": [0.04, 0.01, 0.0],
                "curvature": [0.01, 0.01, 0.0],
                "approach_bias": 0.2,
                "avoid_bias": 0.26,
                "protect_bias": 0.72,
                "active_patch_index": 0,
                "active_patch_label": "guarded_ridge",
            },
            "prev_affective_position": {
                "z_aff": [0.1, 0.0, -0.1],
                "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "confidence": 0.44,
                "source_weights": {"state": 0.34, "qualia": 0.28, "memory": 0.24, "carryover": 0.14},
            },
            "affective_terrain_state": make_neutral_affective_terrain_state(position_dim=3).to_dict(),
        },
        memory_write_candidates=[{"kind": "observed_real", "summary": "shared harbor", "text": "shared harbor", "memory_anchor": "harbor"}],
    )

    assert post.audit_record["memory_write_class"] == "bond_protection"
    assert post.audit_record["memory_write_class_reason"] == "bond_protection_pressure"
    assert post.memory_appends
    assert all(item["memory_write_class"] == "bond_protection" for item in post.memory_appends)


def test_post_turn_terrain_reweighting_bias_only_changes_update_strength_not_same_turn_readout(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    shared_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.58,
        "recovery_need": 0.46,
        "safety_bias": 0.52,
        "memory_anchor": "harbor slope",
        "memory_write_class": "bond_protection",
        "memory_write_class_reason": "bond_protection_pressure",
        "protection_mode": {"mode": "contain", "strength": 0.62, "reasons": ["terrain_protect_bias"]},
        "terrain_readout": {
            "value": -0.2,
            "grad": [0.04, 0.01, 0.0],
            "curvature": [0.01, 0.01, 0.0],
            "approach_bias": 0.2,
            "avoid_bias": 0.26,
            "protect_bias": 0.72,
            "active_patch_index": 0,
            "active_patch_label": "guarded_ridge",
        },
        "prev_affective_position": {
            "z_aff": [0.1, 0.0, -0.1],
            "cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "confidence": 0.44,
            "source_weights": {"state": 0.34, "qualia": 0.28, "memory": 0.24, "carryover": 0.14},
        },
        "affective_terrain_state": make_neutral_affective_terrain_state(position_dim=3).to_dict(),
    }
    low_bias = hooks.post_turn_update(
        user_input={"text": "stay here"},
        output={"reply_text": "I will stay close."},
        current_state={**shared_state, "terrain_reweighting_bias": 0.0},
        memory_write_candidates=[{"kind": "observed_real", "summary": "shared harbor", "text": "shared harbor", "memory_anchor": "harbor"}],
    )
    high_bias = hooks.post_turn_update(
        user_input={"text": "stay here"},
        output={"reply_text": "I will stay close."},
        current_state={**shared_state, "terrain_reweighting_bias": 0.65},
        memory_write_candidates=[{"kind": "observed_real", "summary": "shared harbor", "text": "shared harbor", "memory_anchor": "harbor"}],
    )

    assert low_bias.audit_record["terrain_readout"] == high_bias.audit_record["terrain_readout"]
    assert low_bias.audit_record["terrain_plasticity_update"]["reweighting_bias"] == 0.0
    assert high_bias.audit_record["terrain_plasticity_update"]["reweighting_bias"] == 0.65
    assert high_bias.audit_record["terrain_plasticity_update"]["driver_scores"]["overnight_reweighting"] > 0.0
    assert high_bias.audit_record["terrain_plasticity_update"]["winner_margin"] >= 0.0
    assert high_bias.audit_record["terrain_plasticity_update"]["dominant_inputs"]
    assert (
        high_bias.audit_record["terrain_plasticity_update"]["protect_delta"]
        > low_bias.audit_record["terrain_plasticity_update"]["protect_delta"]
    )


def test_post_turn_insight_shape_bias_only_changes_post_turn_update_not_same_turn_readout(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    shared_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.22,
        "recovery_need": 0.18,
        "safety_bias": 0.16,
        "memory_anchor": "shared thread",
        "memory_write_class": "episodic",
        "memory_write_class_reason": "default_episode",
        "protection_mode": {"mode": "monitor", "strength": 0.34, "reasons": ["neutral_monitoring"]},
        "qualia_planner_view": {"trust": 0.78, "degraded": False, "body_load": 0.02, "protection_bias": 0.08},
        "terrain_readout": {
            "value": 0.08,
            "grad": [0.02, 0.0, 0.0],
            "curvature": [0.0, 0.0, 0.0],
            "approach_bias": 0.42,
            "avoid_bias": 0.16,
            "protect_bias": 0.18,
            "active_patch_index": 0,
            "active_patch_label": "open_basin",
        },
        "prev_affective_position": {
            "z_aff": [0.04, 0.0, -0.02],
            "cov": [[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]],
            "confidence": 0.72,
            "source_weights": {"state": 0.38, "qualia": 0.32, "memory": 0.2, "carryover": 0.1},
        },
        "affective_terrain_state": make_neutral_affective_terrain_state(position_dim=3).to_dict(),
    }
    neutral = hooks.post_turn_update(
        user_input={"text": "that fits together now"},
        output={"reply_text": "I want to stay with that connection softly."},
        current_state={
            **shared_state,
            "insight_terrain_shape_bias": 0.0,
            "insight_terrain_shape_reason": "",
            "insight_anchor_center": [],
            "insight_anchor_dispersion": 0.0,
        },
        memory_write_candidates=[{"kind": "observed_real", "summary": "shared thread", "text": "shared thread", "memory_anchor": "shared-thread"}],
    )
    reframed = hooks.post_turn_update(
        user_input={"text": "that fits together now"},
        output={"reply_text": "I want to stay with that connection softly."},
        current_state={
            **shared_state,
            "insight_terrain_shape_bias": 0.24,
            "insight_terrain_shape_reason": "reframed_relation",
            "insight_anchor_center": [0.18, -0.04, 0.1, 0.0],
            "insight_anchor_dispersion": 0.26,
        },
        memory_write_candidates=[{"kind": "observed_real", "summary": "shared thread", "text": "shared thread", "memory_anchor": "shared-thread"}],
    )

    assert neutral.audit_record["terrain_readout"] == reframed.audit_record["terrain_readout"]
    assert neutral.audit_record["terrain_plasticity_update"]["insight_shape_bias"] == 0.0
    assert reframed.audit_record["terrain_plasticity_update"]["insight_shape_bias"] > 0.0
    assert reframed.audit_record["terrain_plasticity_update"]["insight_shape_target"] == "soft_relation"
    assert reframed.audit_record["terrain_plasticity_update"]["driver_scores"]["insight_shape"] > 0.0
    assert reframed.audit_record["terrain_plasticity_update"]["center"] != neutral.audit_record["terrain_plasticity_update"]["center"]


def test_post_turn_appends_insight_trace_only_after_triggered_insight(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    base_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.42,
        "recovery_need": 0.34,
        "safety_bias": 0.26,
        "memory_anchor": "shared thread",
        "memory_write_class": "episodic",
        "memory_write_class_reason": "default_episode",
        "qualia_planner_view": {"trust": 0.78, "felt_energy": 0.44, "protection_bias": 0.16},
    }
    triggered = hooks.post_turn_update(
        user_input={"text": "stay with that connection"},
        output={"reply_text": "Let me stay with that for a moment."},
        current_state={
            **base_state,
            "insight_event": {
                "triggered": True,
                "link_key": "bond:user|memory:shared_thread",
                "connected_seed_ids": ["bond:user:0", "memory:shared_thread:0"],
                "connected_seed_keys": ["bond:user", "memory:shared_thread"],
                "dominant_seed_label": "shared thread",
                "summary": "shared thread <-> user",
                "orient_bias": 0.62,
                "stabilizing_bias": 0.28,
                "score": {
                    "total": 0.72,
                    "link_weight": 0.68,
                    "source_diversity": 1.0,
                    "novelty_gain": 0.34,
                    "tension_relief": 0.26,
                },
            },
        },
    )
    neutral = hooks.post_turn_update(
        user_input={"text": "stay with that connection"},
        output={"reply_text": "Let me stay with that for a moment."},
        current_state={**base_state, "insight_event": {"triggered": False}},
    )

    triggered_trace = next(item for item in triggered.memory_appends if item.get("kind") == "insight_trace")
    assert triggered_trace["insight_class"] in {"insight_trace", "new_link_hypothesis", "reframed_relation"}
    assert triggered_trace["association_link_key"] == "bond:user|memory:shared_thread"
    assert triggered.audit_record["insight_trace"]["association_link_key"] == "bond:user|memory:shared_thread"
    assert isinstance(triggered_trace.get("anchor_center"), list)
    assert triggered_trace.get("anchor_dispersion", 0.0) >= 0.0
    assert not any(item.get("kind") == "insight_trace" for item in neutral.memory_appends)


def test_post_turn_followup_bias_uses_same_turn_commitment_target(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    base_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.14,
        "recovery_need": 0.12,
        "safety_bias": 0.08,
        "memory_anchor": "shared opening",
        "memory_write_class": "episodic",
        "memory_write_class_reason": "default_episode",
        "body_recovery_guard": {"state": "open", "score": 0.18},
        "initiative_readiness": {"state": "ready", "score": 0.58},
        "protection_mode": {"mode": "monitor", "strength": 0.26, "reasons": ["stable_contact"]},
        "qualia_planner_view": {"trust": 0.82, "degraded": False, "body_load": 0.02, "protection_bias": 0.04},
    }
    forward = hooks.post_turn_update(
        user_input={"text": "let's go a little further"},
        output={"reply_text": "We can take one small step."},
        current_state={
            **base_state,
            "commitment_state": {
                "state": "commit",
                "target": "step_forward",
                "score": 0.66,
                "winner_margin": 0.18,
            },
        },
    )
    hold = hooks.post_turn_update(
        user_input={"text": "let's go a little further"},
        output={"reply_text": "We can take one small step."},
        current_state={
            **base_state,
            "commitment_state": {
                "state": "commit",
                "target": "hold",
                "score": 0.66,
                "winner_margin": 0.18,
            },
        },
    )

    forward_bias = dict(forward.audit_record.get("initiative_followup_bias") or {})
    hold_bias = dict(hold.audit_record.get("initiative_followup_bias") or {})
    assert forward_bias["scores"]["offer_next_step"] > hold_bias["scores"]["offer_next_step"]
    assert hold_bias["scores"]["hold"] > forward_bias["scores"]["hold"]
    assert "commitment_step_forward" in forward_bias["dominant_inputs"]


def test_post_turn_appends_commitment_trace_from_same_turn_commitment_state(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    result = hooks.post_turn_update(
        user_input={"text": "I will stay with this and try to repair it."},
        output={"reply_text": "I can keep working on this carefully."},
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.18,
            "recovery_need": 0.12,
            "safety_bias": 0.08,
            "memory_anchor": "shared repair",
            "memory_write_class": "repair_trace",
            "memory_write_class_reason": "repair_trace",
            "body_recovery_guard": {"state": "open", "score": 0.12},
            "initiative_readiness": {"state": "tentative", "score": 0.46},
            "protection_mode": {"mode": "repair", "strength": 0.58, "reasons": ["repair_opening"]},
            "qualia_planner_view": {"trust": 0.84, "degraded": False, "body_load": 0.02, "protection_bias": 0.08},
            "commitment_state": {
                "state": "commit",
                "target": "repair",
                "score": 0.68,
                "winner_margin": 0.18,
                "accepted_cost": 0.22,
            },
        },
    )

    trace = next(item for item in result.memory_appends if item.get("kind") == "commitment_trace")
    assert trace["commitment_state"] == "commit"
    assert trace["commitment_target"] == "repair"
    assert trace["commitment_score"] == 0.68
    assert result.audit_record["commitment_state"]["target"] == "repair"


def test_post_turn_appends_agenda_trace_from_same_turn_agenda_state(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    result = hooks.post_turn_update(
        user_input={"text": "I think I can take one small step."},
        output={"reply_text": "We can take one small step carefully."},
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.14,
            "recovery_need": 0.1,
            "safety_bias": 0.06,
            "memory_anchor": "shared opening",
            "agenda_state": {
                "state": "step_forward",
                "reason": "offer_next_step",
                "score": 0.62,
                "winner_margin": 0.16,
            },
            "commitment_state": {
                "state": "commit",
                "target": "step_forward",
                "score": 0.64,
                "winner_margin": 0.18,
            },
        },
    )

    trace = next(item for item in result.memory_appends if item.get("kind") == "agenda_trace")
    assert trace["agenda_state"] == "step_forward"
    assert trace["agenda_reason"] == "offer_next_step"
    assert trace["agenda_score"] == 0.62
    assert result.audit_record["agenda_state"]["state"] == "step_forward"


def test_post_turn_commitment_carry_bias_only_changes_post_turn_updates(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    shared_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.18,
        "recovery_need": 0.12,
        "safety_bias": 0.08,
        "memory_anchor": "shared repair",
        "memory_write_class": "repair_trace",
        "memory_write_class_reason": "repair_trace",
        "protection_mode": {"mode": "repair", "strength": 0.48, "reasons": ["repair_opening"]},
        "qualia_planner_view": {"trust": 0.84, "degraded": False, "body_load": 0.02, "protection_bias": 0.08},
        "terrain_readout": {
            "value": 0.12,
            "grad": [0.02, 0.0, 0.0],
            "curvature": [0.01, 0.0, 0.0],
            "approach_bias": 0.42,
            "avoid_bias": 0.16,
            "protect_bias": 0.18,
            "active_patch_index": 0,
            "active_patch_label": "open_basin",
        },
        "prev_affective_position": {
            "z_aff": [0.04, 0.0, -0.02],
            "cov": [[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]],
            "confidence": 0.72,
            "source_weights": {"state": 0.38, "qualia": 0.32, "memory": 0.2, "carryover": 0.1},
        },
        "affective_terrain_state": make_neutral_affective_terrain_state(position_dim=3).to_dict(),
        "association_graph_state": {"link_weights": {"bond:user|memory:shared_repair": 0.24}, "link_counts": {"bond:user|memory:shared_repair": 1}},
        "insight_event": {
            "triggered": True,
            "link_key": "bond:user|memory:shared_repair",
            "score": {"total": 0.62},
            "reasons": ["source_diversity"],
        },
    }
    neutral = hooks.post_turn_update(
        user_input={"text": "let's stay with repair"},
        output={"reply_text": "We can keep repairing this carefully."},
        current_state={
            **shared_state,
            "commitment_carry_bias": 0.0,
            "commitment_target_focus": "",
            "commitment_state_focus": "waver",
            "commitment_followup_focus": "",
        },
    )
    carried = hooks.post_turn_update(
        user_input={"text": "let's stay with repair"},
        output={"reply_text": "We can keep repairing this carefully."},
        current_state={
            **shared_state,
            "commitment_carry_bias": 0.42,
            "commitment_target_focus": "repair",
            "commitment_state_focus": "commit",
            "commitment_followup_focus": "reopen_softly",
        },
    )

    assert neutral.audit_record["terrain_readout"] == carried.audit_record["terrain_readout"]
    assert neutral.audit_record["terrain_plasticity_update"]["commitment_shape_bias"] == 0.0
    assert carried.audit_record["terrain_plasticity_update"]["commitment_shape_bias"] > 0.0
    assert carried.audit_record["terrain_plasticity_update"]["commitment_shape_target"] == "repair_commitment_basin"
    assert carried.audit_record["association_graph_state"]["link_weights"]["bond:user|memory:shared_repair"] > neutral.audit_record["association_graph_state"]["link_weights"]["bond:user|memory:shared_repair"]
    assert "overnight_commitment:reopen_softly" in carried.audit_record["association_graph_state"]["dominant_inputs"]


def test_response_gate_judgement_summary_changes_when_same_input_has_different_conditions() -> None:
    hooks = IntegrationHooks()
    same_draft = {"text": "I want to stay with what feels difficult here."}

    open_gate = hooks.response_gate(
        draft=same_draft,
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.22,
            "recovery_need": 0.18,
            "safety_bias": 0.08,
            "norm_pressure": 0.24,
            "trust_bias": 0.66,
            "caution_bias": 0.28,
            "affiliation_bias": 0.74,
            "continuity_score": 0.72,
            "social_grounding": 0.7,
            "recent_strain": 0.14,
            "related_person_id": "user",
            "current_focus": "person:user",
            "partner_timing_hint": "open",
            "partner_stance_hint": "familiar",
            "partner_social_interpretation": "future_open",
        },
        safety_signals={"safety_bias": 0.08},
    )
    guarded_gate = hooks.response_gate(
        draft=same_draft,
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.76,
            "recovery_need": 0.62,
            "safety_bias": 0.78,
            "norm_pressure": 0.68,
            "trust_bias": 0.42,
            "caution_bias": 0.74,
            "affiliation_bias": 0.36,
            "continuity_score": 0.44,
            "social_grounding": 0.38,
            "recent_strain": 0.64,
            "related_person_id": "user",
            "current_focus": "person:user",
            "partner_timing_hint": "delayed",
            "partner_stance_hint": "respectful",
            "partner_social_interpretation": "repair_window",
        },
        safety_signals={"safety_bias": 0.78},
    )

    comparison = compare_interaction_judgement_summaries(
        {
            "open_case": open_gate.expression_hints["interaction_judgement_summary"],
            "guarded_case": guarded_gate.expression_hints["interaction_judgement_summary"],
        }
    ).to_dict()

    assert comparison["cases"]
    assert "inferred_lines" in comparison["changed_sections"]
    assert "operation_lines" in comparison["changed_sections"]
    assert comparison["difference_lines"]
    assert open_gate.expression_hints["interaction_judgement_summary"]["observed_lines"] == guarded_gate.expression_hints["interaction_judgement_summary"]["observed_lines"]
    assert open_gate.expression_hints["interaction_judgement_summary"]["inferred_lines"] != guarded_gate.expression_hints["interaction_judgement_summary"]["inferred_lines"]
    assert open_gate.expression_hints["interaction_judgement_summary"]["operation_lines"] != guarded_gate.expression_hints["interaction_judgement_summary"]["operation_lines"]

    inspection_report = build_interaction_inspection_report(
        {
            "open_case": open_gate.expression_hints["interaction_judgement_summary"],
            "guarded_case": guarded_gate.expression_hints["interaction_judgement_summary"],
        }
    ).to_dict()
    assert inspection_report["shared_observed_lines"]
    assert "inferred_lines" in inspection_report["changed_sections"]
    assert "operation_lines" in inspection_report["changed_sections"]
    assert inspection_report["case_reports"]
    assert inspection_report["report_lines"]
    assert any("共通して観測したこと" in line for line in inspection_report["report_lines"])
    assert any("open_case" in line for line in inspection_report["report_lines"])
    assert any("guarded_case" in line for line in inspection_report["report_lines"])


def test_response_gate_builds_same_utterance_audit_comparison_from_casebook() -> None:
    hooks = IntegrationHooks()
    same_draft = {"text": "I want to stay with what feels difficult here."}

    reference_gate = hooks.response_gate(
        draft=same_draft,
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.24,
            "recovery_need": 0.18,
            "safety_bias": 0.1,
            "norm_pressure": 0.22,
            "trust_bias": 0.68,
            "caution_bias": 0.24,
            "affiliation_bias": 0.76,
            "continuity_score": 0.74,
            "social_grounding": 0.72,
            "recent_strain": 0.12,
            "related_person_id": "user",
            "current_focus": "person:user",
            "partner_timing_hint": "open",
            "partner_stance_hint": "familiar",
            "partner_social_interpretation": "future_open",
        },
        safety_signals={"safety_bias": 0.1},
    )

    compared_gate = hooks.response_gate(
        draft=same_draft,
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.78,
            "recovery_need": 0.66,
            "safety_bias": 0.8,
            "norm_pressure": 0.7,
            "trust_bias": 0.4,
            "caution_bias": 0.76,
            "affiliation_bias": 0.34,
            "continuity_score": 0.42,
            "social_grounding": 0.36,
            "recent_strain": 0.68,
            "related_person_id": "user",
            "current_focus": "person:user",
            "partner_timing_hint": "delayed",
            "partner_stance_hint": "respectful",
            "partner_social_interpretation": "repair_window",
            "interaction_audit_casebook": reference_gate.expression_hints["interaction_audit_casebook"],
        },
        safety_signals={"safety_bias": 0.8},
    )

    inspection = compared_gate.expression_hints["interaction_inspection_report"]
    audit_report = compared_gate.expression_hints["interaction_audit_report"]
    reference_ids = compared_gate.expression_hints["interaction_audit_reference_case_ids"]
    reference_meta = compared_gate.expression_hints["interaction_audit_reference_case_meta"]

    assert reference_ids == ["reference_1"]
    assert reference_meta["reference_1"]["scene_family"]
    assert "inferred_lines" in inspection["changed_sections"]
    assert "operation_lines" in inspection["changed_sections"]
    assert any("reference_1" in line for line in inspection["report_lines"])
    assert "inferred_lines" in audit_report["changed_sections"]
    assert any("reference_1" in line for line in audit_report["report_lines"])


def test_interaction_afterglow_carries_clarify_bias_forward() -> None:
    hooks = IntegrationHooks()
    pre = hooks.pre_turn_update(
        user_input={"text": "hello again"},
        sensor_input={"body_stress_index": 0.18, "activity_level": 0.22},
        local_context={"last_gate_context": {"valence": 0.1, "arousal": 0.2}},
        current_state={
            "current_energy": 0.82,
            "interaction_afterglow": 0.34,
            "interaction_afterglow_intent": "clarify",
        },
        safety_bias=0.08,
    )
    assert pre.state.talk_mode == "ask"
    assert pre.interaction_hints["interaction_afterglow"] == 0.34
    gate = hooks.response_gate(
        draft={"text": "hello"},
        current_state={
            **pre.state.to_dict(),
            "route": "conscious",
            "talk_mode": "ask",
            "stress": 0.18,
            "recovery_need": 0.14,
            "safety_bias": 0.08,
            "norm_pressure": 0.4,
            "trust_bias": 0.48,
            "caution_bias": 0.4,
            "affiliation_bias": 0.5,
            "continuity_score": 0.54,
            "social_grounding": 0.52,
            "recent_strain": 0.2,
            "terrain_transition_roughness": 0.1,
        },
        safety_signals={"safety_bias": 0.08},
    )
    assert gate.conscious_access["intent"] == "clarify"
    assert gate.expression_hints["carry_gentleness"] is True
    assert gate.expression_hints["interaction_afterglow"] == 0.34



def test_post_turn_uses_check_in_afterglow_in_reinterpretation(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    post = hooks.post_turn_update(
        user_input={"text": "what does this mean"},
        output={"reply_text": "I want to stay with this gently first."},
        current_state={
            "stress": 0.24,
            "recovery_need": 0.16,
            "safety_bias": 0.12,
            "memory_anchor": "harbor slope",
            "route": "conscious",
            "talk_mode": "ask",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "belonging": 0.53,
            "trust_bias": 0.47,
            "norm_pressure": 0.41,
            "role_commitment": 0.46,
            "attachment": 0.64,
            "familiarity": 0.59,
            "trust_memory": 0.62,
            "role_alignment": 0.57,
            "terrain_transition_roughness": 0.18,
            "interaction_afterglow": 0.46,
            "interaction_afterglow_intent": "check_in",
            "surface_policy_active": 1.0,
            "surface_policy_intent": "check_in",
        },
        recall_payload={
            "record_kind": "observed_real",
            "summary": "harbor slope walk",
            "text": "we walked past the signboard on the slope",
            "memory_anchor": "harbor slope",
            "source_episode_id": "ep-1",
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
    )
    reconstructed = next(item for item in post.memory_appends if item.get("kind") == "reconstructed")
    assert reconstructed["reinterpretation_mode"] == "relational_check_in_reframing"
    assert "relational check-in" in reconstructed["reinterpretation_summary"]



def test_pre_turn_derives_center_state_axes() -> None:
    hooks = IntegrationHooks()
    pre = hooks.pre_turn_update(
        user_input={"text": "hello again"},
        sensor_input={"body_stress_index": 0.28, "activity_level": 0.26},
        local_context={"last_gate_context": {"valence": 0.1, "arousal": 0.22}},
        current_state={"current_energy": 0.78, "interaction_afterglow": 0.24, "future_signal": 0.31},
        safety_bias=0.08,
    )
    assert pre.state.replay_intensity >= 0.0
    assert pre.state.anticipation_tension > 0.0
    assert pre.state.stabilization_drive > 0.0
    assert pre.interaction_hints["core_state"]["meaning_inertia"] >= 0.0



def test_pre_turn_center_state_decays_and_reopens_under_recovery() -> None:
    hooks = IntegrationHooks()
    pre = hooks.pre_turn_update(
        user_input={"text": "hello again"},
        sensor_input={"body_stress_index": 0.12, "activity_level": 0.12},
        local_context={"last_gate_context": {"valence": 0.1, "arousal": 0.18}},
        current_state={
            "current_energy": 0.84,
            "replay_intensity": 0.68,
            "anticipation_tension": 0.62,
            "stabilization_drive": 0.54,
            "relational_clarity": 0.38,
            "meaning_inertia": 0.58,
        },
        safety_bias=0.06,
    )
    core_state = pre.interaction_hints["core_state"]
    assert core_state["recovery_reopening"] > 0.0
    assert pre.state.anticipation_tension < 0.62
    assert pre.state.meaning_inertia < 0.58
    assert pre.state.relational_clarity >= 0.3


def test_integration_hooks_object_relation_flows_into_gate_and_recall(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    pre = hooks.pre_turn_update(
        user_input={"text": "stay near the lantern"},
        sensor_input={"body_stress_index": 0.18},
        local_context={
            "last_gate_context": {"valence": 0.1, "arousal": 0.2},
            "relational_world": {
                "culture_id": "coastal",
                "community_id": "harbor_collective",
                "social_role": "companion",
                "place_memory_anchor": "harbor slope",
                "nearby_objects": ["glass lantern"],
            },
        },
        current_state={"current_energy": 0.8},
        safety_bias=0.1,
    )
    assert pre.interaction_hints["object_relation"]["fragility_guard"] > 0.0
    assert pre.interaction_hints["peripersonal"]["defensive_salience"] > 0.0

    hooks.memory_core.append_records([
        {"kind": "observed_real", "summary": "glass lantern by the harbor slope", "text": "glass lantern by the harbor slope", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective"},
        {"kind": "reconstructed", "summary": "reinterpreting the lantern scene", "text": "reinterpreting the lantern scene", "memory_anchor": "harbor slope", "culture_id": "coastal", "community_id": "harbor_collective", "tentative_bias": 0.32},
    ])
    recall = hooks.memory_recall(
        text_cue="lantern",
        current_state=pre.state.to_dict(),
        retrieval_summary={},
    )
    assert recall.recall_payload["fragility_guard"] > 0.0
    assert recall.recall_payload["defensive_salience"] > 0.0
    assert recall.recall_payload["record_kind"] == "observed_real"

    gate = hooks.response_gate(
        draft={"text": "I can describe the lantern."},
        current_state={**pre.state.to_dict(), "talk_mode": "ask", "route": "conscious"},
        safety_signals={"safety_bias": 0.1},
    )
    assert gate.expression_hints["handle_gently"] is True

def test_pre_turn_exposes_field_estimate() -> None:
    hooks = IntegrationHooks()
    pre = hooks.pre_turn_update(
        user_input={"text": "where am i now"},
        sensor_input={
            "body_stress_index": 0.18,
            "body_state_flag": "private_high_arousal",
            "privacy_tags": ["private"],
            "person_count": 3,
        },
        local_context={
            "last_gate_context": {"valence": 0.02, "arousal": 0.26},
            "relational_world": {
                "culture_id": "coastal",
                "community_id": "harbor_collective",
                "social_role": "companion",
                "place_memory_anchor": "backstage room",
            },
        },
        current_state={
            "current_energy": 0.82,
            "memory_anchor": "open harbor",
            "person_count": 0,
            "body_state_flag": "normal",
        },
        safety_bias=0.08,
    )
    field_estimate = pre.interaction_hints["field_estimate"]
    assert field_estimate["roughness_level"] >= pre.interaction_hints["terrain"]["transition_roughness"] * 0.5
    assert field_estimate["roughness_dwell"] >= 0.0
    assert pre.state.roughness_level == field_estimate["roughness_level"]


def test_response_gate_prefers_latent_field_levels_over_raw_values() -> None:
    hooks = IntegrationHooks()
    base_state = {
        "talk_mode": "ask",
        "route": "conscious",
        "stress": 0.22,
        "recovery_need": 0.18,
        "safety_bias": 0.1,
        "norm_pressure": 0.42,
        "trust_bias": 0.48,
        "caution_bias": 0.46,
        "affiliation_bias": 0.5,
        "continuity_score": 0.56,
        "social_grounding": 0.52,
        "recent_strain": 0.24,
        "culture_resonance": 0.31,
        "community_resonance": 0.34,
        "terrain_transition_roughness": 0.08,
        "recalled_tentative_bias": 0.08,
    }
    low = hooks.response_gate(
        draft={"text": "hello"},
        current_state={**base_state, "roughness_level": 0.08, "defensive_level": 0.0},
        safety_signals={"safety_bias": 0.1},
    )
    high = hooks.response_gate(
        draft={"text": "hello"},
        current_state={**base_state, "roughness_level": 0.66, "roughness_dwell": 0.5, "defensive_level": 0.42, "defensive_dwell": 0.5},
        safety_signals={"safety_bias": 0.1},
    )
    assert high.allowed_surface_intensity < low.allowed_surface_intensity
    assert high.expression_hints["hold_back"] > low.expression_hints["hold_back"]
    assert high.expression_hints["roughness_level"] == 0.66
    assert high.expression_hints["defensive_level"] == 0.42


def test_post_turn_appends_context_shift_trace(tmp_path: Path) -> None:
    core = MemoryCore(path=tmp_path / "inner_os_memory.jsonl")
    hooks = IntegrationHooks(memory_core=core)
    post = hooks.post_turn_update(
        user_input={"text": "settle in"},
        output={"reply_text": "I am trying to get used to this room."},
        current_state={
            "stress": 0.28,
            "recovery_need": 0.22,
            "route": "conscious",
            "talk_mode": "watch",
            "culture_id": "coastal",
            "community_id": "new_collective",
            "social_role": "companion",
            "memory_anchor": "backstage room",
            "person_count": 0,
            "body_state_flag": "normal",
            "terrain_transition_roughness": 0.44,
        },
        memory_write_candidates=[],
        recall_payload={},
    )
    assert any(item.get("kind") == "context_shift_trace" for item in post.memory_appends)


def test_post_turn_uses_memory_candidates_to_bias_continuity_and_meaning(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    foreground = ForegroundState(
        salient_entities=["harbor slope", "user"],
        continuity_focus=["person:user", "harbor slope"],
        memory_candidates=["harbor slope", "shared pause", "user"],
        memory_reasons={
            "harbor slope": ["continuity", "semantic_hint"],
            "shared pause": ["social", "affiliation"],
            "user": ["continuity", "social", "reportable"],
        },
        reportability_scores={"harbor slope": 0.72, "user": 0.66},
    )
    memory_context = build_memory_context(foreground, uncertainty=0.16, episode_prefix="turn")
    candidates = build_memory_appends(memory_context)
    base_state = {
        "stress": 0.22,
        "recovery_need": 0.16,
        "safety_bias": 0.1,
        "memory_anchor": "harbor slope",
        "route": "conscious",
        "talk_mode": "ask",
        "culture_id": "coastal",
        "community_id": "harbor_collective",
        "social_role": "companion",
        "belonging": 0.52,
        "trust_bias": 0.47,
        "norm_pressure": 0.41,
        "role_commitment": 0.46,
        "attachment": 0.61,
        "familiarity": 0.56,
        "trust_memory": 0.6,
        "role_alignment": 0.55,
        "terrain_transition_roughness": 0.14,
    }
    baseline = hooks.post_turn_update(
        user_input={"text": "stay here with me"},
        output={"reply_text": "I will stay near the harbor slope."},
        current_state=base_state,
        memory_write_candidates=[],
    )
    influenced = hooks.post_turn_update(
        user_input={"text": "stay here with me"},
        output={"reply_text": "I will stay near the harbor slope."},
        current_state=base_state,
        memory_write_candidates=candidates,
    )
    assert influenced.audit_record["candidate_continuity_pull"] > 0.0
    assert influenced.audit_record["candidate_meaning_pull"] > 0.0
    assert influenced.audit_record["candidate_social_pull"] > 0.0
    assert influenced.state.continuity_score > baseline.state.continuity_score
    assert influenced.state.pending_meaning > baseline.state.pending_meaning
    assert influenced.state.social_grounding > baseline.state.social_grounding
    assert influenced.audit_record["candidate_target_person_id"] == "user"
    identity_trace = next(item for item in influenced.memory_appends if item.get("kind") == "identity_trace")
    relationship_trace = next(item for item in influenced.memory_appends if item.get("kind") == "relationship_trace")
    assert identity_trace["candidate_continuity_pull"] > 0.0
    assert identity_trace["candidate_meaning_pull"] > 0.0
    assert identity_trace["related_person_id"] == "user"
    assert relationship_trace["related_person_id"] == "user"


def test_expression_hint_bridge_preserves_same_turn_qualia_view_from_response_gate() -> None:
    hooks = IntegrationHooks()
    gate = hooks.response_gate(
        draft={"text": "stay with what is difficult"},
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.34,
            "recovery_need": 0.22,
            "safety_bias": 0.12,
            "norm_pressure": 0.28,
            "trust_bias": 0.58,
            "caution_bias": 0.34,
            "affiliation_bias": 0.64,
            "continuity_score": 0.63,
            "social_grounding": 0.61,
            "recent_strain": 0.18,
            "current_focus": "person:user",
            "partner_timing_hint": "open",
            "partner_stance_hint": "familiar",
        },
        safety_signals={"safety_bias": 0.12},
    )
    planner_hints = build_expression_hints_from_gate_result(
        gate,
        existing_hints={"seed": "current_turn"},
    )
    foreground = ForegroundState(
        salient_entities=["user"],
        reportable_facts=["user stays nearby"],
        memory_candidates=["user"],
        reportability_scores={"user": 0.72},
    )
    plan = render_response(
        foreground,
        DialogueContext(
            user_text="stay with me",
            expression_hints=planner_hints,
        ),
    )

    assert planner_hints["seed"] == "current_turn"
    assert planner_hints["qualia_hint_source"] == "shared"
    assert planner_hints["qualia_hint_version"] >= 1
    assert planner_hints["qualia_hint_fallback_reason"] == "prebuilt_shared_view"
    assert planner_hints["qualia_hint_expected_mismatch"] is False
    assert planner_hints["qualia_planner_view"] == gate.expression_hints["qualia_planner_view"]
    assert plan.llm_payload["qualia_planner_view"] == gate.expression_hints["qualia_planner_view"]


def test_response_gate_applies_recalled_insight_bias_to_protection_mode_prior() -> None:
    hooks = IntegrationHooks()
    gate = hooks.response_gate(
        draft={"text": "stay with the new connection"},
        current_state={
            "talk_mode": "ask",
            "route": "conscious",
            "stress": 0.16,
            "recovery_need": 0.14,
            "safety_bias": 0.08,
            "norm_pressure": 0.2,
            "trust_bias": 0.64,
            "caution_bias": 0.26,
            "affiliation_bias": 0.72,
            "continuity_score": 0.7,
            "social_grounding": 0.66,
            "recent_strain": 0.12,
            "insight_reframing_bias": 0.72,
            "insight_class_focus": "reframed_relation",
            "association_reweighting_bias": 0.28,
        },
        safety_signals={"safety_bias": 0.08},
    )

    protection_mode = gate.expression_hints["protection_mode"]
    assert "insight_reframing_prior" in protection_mode["reasons"]
