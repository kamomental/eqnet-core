from pathlib import Path

from inner_os.integration_hooks import IntegrationHooks
from inner_os.memory_core import MemoryCore


def test_integration_hooks_pre_and_recall_flow(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    pre = hooks.pre_turn_update(
        user_input={"text": "こんにちは"},
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
    assert pre.state.community_resonance >= 0.0
    assert pre.state.culture_resonance >= 0.0

    hooks.memory_core.append_records(
        [{"kind": "observed", "summary": "懐かしい港町の坂道と看板", "memory_anchor": "港町の坂道"}]
    )
    recall = hooks.memory_recall(
        text_cue="懐かしい港町",
        visual_cue="夕方の坂道と看板",
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
    assert recall.retrieval_summary["inner_os_memory"]


def test_integration_hooks_response_and_post_turn_flow(tmp_path: Path) -> None:
    hooks = IntegrationHooks(memory_core=MemoryCore(path=tmp_path / "inner_os_memory.jsonl"))
    current_state = {
        "stress": 0.4,
        "recovery_need": 0.3,
        "safety_bias": 0.2,
        "memory_anchor": "港町の記憶",
        "mode": "reality",
        "route": "conscious",
        "talk_mode": "ask",
    }
    gate = hooks.response_gate(
        draft={"text": "少し気になっています"},
        current_state=current_state,
        safety_signals={"safety_bias": 0.25},
    )
    assert gate.route == "conscious"
    assert gate.allowed_surface_intensity < 1.0
    assert gate.conscious_access["recall_active"] is True

    post = hooks.post_turn_update(
        user_input={"text": "どうしたの"},
        output={"reply_text": "少し考えていました"},
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
    assert len(post.memory_appends) == 6
    assert post.state.norm_pressure >= 0.35
    assert post.state.temporal_pressure >= 0.0
    assert any(item.get("kind") == "identity_trace" for item in post.memory_appends)
    assert any(item.get("kind") == "relationship_trace" for item in post.memory_appends)
    assert any(item.get("kind") == "community_profile_trace" for item in post.memory_appends)
    assert (tmp_path / "inner_os_memory.jsonl").exists()



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
    assert recall.recall_payload["terrain_transition_roughness"] == round(recall.ignition_hints["terrain"]["transition_roughness"], 4)
    assert recall.recall_payload["recovery_reopening"] >= 0.0

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
    low = hooks.response_gate(draft={"text": "hello"}, current_state={**base_state, "recalled_tentative_bias": 0.0}, safety_signals={"safety_bias": 0.1})
    high = hooks.response_gate(draft={"text": "hello"}, current_state={**base_state, "recalled_tentative_bias": 0.72}, safety_signals={"safety_bias": 0.1})
    assert high.allowed_surface_intensity < low.allowed_surface_intensity
    assert high.expression_hints["recalled_tentative_bias"] == 0.72
    assert high.expression_hints["avoid_definitive_interpretation"] is True
    assert high.expression_hints["hold_back"] > low.expression_hints["hold_back"]
    assert round(high.expression_hints["express_now"] + high.expression_hints["hold_back"], 4) == 1.0

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
    assert high.state.norm_pressure < low.state.norm_pressure
    assert high.state.role_commitment < low.state.role_commitment


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
