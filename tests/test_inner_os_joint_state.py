from inner_os.joint_state import JointState, coerce_joint_state, derive_joint_state


def test_joint_state_integrates_shared_moment_listener_and_relation_fields() -> None:
    state = derive_joint_state(
        previous_state=None,
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.74,
            "jointness": 0.68,
            "afterglow": 0.61,
        },
        listener_action_state={
            "state": "warm_laugh_ack",
            "acknowledgement_room": 0.63,
            "laughter_room": 0.7,
            "filler_room": 0.52,
        },
        live_engagement_state={
            "state": "riff_with_comment",
            "comment_pickup_room": 0.58,
            "riff_room": 0.64,
            "topic_seed_room": 0.32,
        },
        meaning_update_state={
            "relation_update": "shared_smile_window",
            "world_update": "small_moment_on_known_thread",
            "preserve_guard": "keep_it_small",
        },
        organism_state={
            "attunement": 0.71,
            "protective_tension": 0.22,
            "expressive_readiness": 0.68,
            "play_window": 0.74,
            "relation_pull": 0.66,
        },
        external_field_state={
            "continuity_pull": 0.69,
            "social_pressure": 0.18,
            "ambiguity_load": 0.21,
            "safety_envelope": 0.73,
        },
        terrain_dynamics_state={
            "barrier_height": 0.24,
            "recovery_gradient": 0.58,
            "basin_pull": 0.63,
        },
        memory_dynamics_state={
            "monument_salience": 0.57,
            "activation_confidence": 0.54,
            "memory_tension": 0.22,
            "dominant_relation_type": "same_anchor",
            "relation_generation_mode": "ignited",
            "meta_relations": [{"meta_type": "reinforces"}],
        },
    )

    assert state.shared_delight > 0.0
    assert state.common_ground > 0.0
    assert state.coupling_strength > 0.0
    assert state.dominant_mode in {
        "ambient",
        "shared_attention",
        "repair_attunement",
        "strained_jointness",
        "delighted_jointness",
    }
    axes = state.to_packet_axes()
    assert axes["delight"]["value"] >= 0.0
    assert axes["coupling"]["value"] >= 0.0


def test_joint_state_preserves_trace_and_can_shift_to_strained_mode() -> None:
    previous = coerce_joint_state(
        {
            "shared_tension": 0.24,
            "shared_delight": 0.52,
            "repair_readiness": 0.48,
            "common_ground": 0.57,
            "joint_attention": 0.53,
            "mutual_room": 0.46,
            "coupling_strength": 0.55,
            "dominant_mode": "delighted_jointness",
            "trace": [{"step": 1, "dominant_mode": "delighted_jointness"}],
        }
    )

    state = derive_joint_state(
        previous_state=previous,
        shared_moment_state={"state": "none"},
        listener_action_state={"acknowledgement_room": 0.18, "filler_room": 0.14},
        live_engagement_state={"state": "hold", "comment_pickup_room": 0.12, "riff_room": 0.08},
        meaning_update_state={"preserve_guard": "hold_line"},
        organism_state={
            "attunement": 0.26,
            "protective_tension": 0.71,
            "expressive_readiness": 0.24,
            "play_window": 0.11,
            "relation_pull": 0.33,
        },
        external_field_state={
            "continuity_pull": 0.36,
            "social_pressure": 0.63,
            "ambiguity_load": 0.47,
            "safety_envelope": 0.38,
        },
        terrain_dynamics_state={
            "barrier_height": 0.61,
            "recovery_gradient": 0.24,
            "basin_pull": 0.31,
        },
        memory_dynamics_state={
            "memory_tension": 0.59,
            "activation_confidence": 0.22,
        },
    )

    assert len(state.trace) == 2
    assert state.shared_tension >= previous.shared_tension
    assert state.dominant_mode in {"strained_jointness", "repair_attunement", "ambient"}


def test_joint_state_coercion_accepts_plain_mapping() -> None:
    state = coerce_joint_state(
        {
            "shared_tension": 0.22,
            "shared_delight": 0.46,
            "repair_readiness": 0.41,
            "common_ground": 0.52,
            "joint_attention": 0.5,
            "mutual_room": 0.39,
            "coupling_strength": 0.49,
            "dominant_mode": "shared_attention",
        }
    )

    assert isinstance(state, JointState)
    assert state.dominant_mode == "shared_attention"
    assert state.common_ground == 0.52


def test_joint_state_reflects_causal_memory_shift() -> None:
    reopened = derive_joint_state(
        previous_state=None,
        shared_moment_state={"state": "none"},
        listener_action_state={"acknowledgement_room": 0.2, "filler_room": 0.12},
        live_engagement_state={"state": "hold", "comment_pickup_room": 0.14, "riff_room": 0.08},
        meaning_update_state={"preserve_guard": "keep_it_small"},
        organism_state={
            "attunement": 0.52,
            "protective_tension": 0.34,
            "expressive_readiness": 0.4,
            "play_window": 0.22,
            "relation_pull": 0.44,
        },
        external_field_state={
            "continuity_pull": 0.52,
            "social_pressure": 0.26,
            "ambiguity_load": 0.22,
            "safety_envelope": 0.64,
        },
        terrain_dynamics_state={
            "barrier_height": 0.32,
            "recovery_gradient": 0.41,
            "basin_pull": 0.5,
        },
        memory_dynamics_state={
            "dominant_relation_type": "unfinished_carry",
            "dominant_causal_type": "reopened_by",
            "activation_confidence": 0.41,
            "monument_salience": 0.44,
            "memory_tension": 0.38,
        },
    )

    reframed = derive_joint_state(
        previous_state=None,
        shared_moment_state={"state": "none"},
        listener_action_state={"acknowledgement_room": 0.2, "filler_room": 0.12},
        live_engagement_state={"state": "hold", "comment_pickup_room": 0.14, "riff_room": 0.08},
        meaning_update_state={"preserve_guard": "keep_it_small"},
        organism_state={
            "attunement": 0.52,
            "protective_tension": 0.34,
            "expressive_readiness": 0.4,
            "play_window": 0.22,
            "relation_pull": 0.44,
        },
        external_field_state={
            "continuity_pull": 0.52,
            "social_pressure": 0.26,
            "ambiguity_load": 0.22,
            "safety_envelope": 0.64,
        },
        terrain_dynamics_state={
            "barrier_height": 0.32,
            "recovery_gradient": 0.41,
            "basin_pull": 0.5,
        },
        memory_dynamics_state={
            "dominant_relation_type": "same_anchor",
            "dominant_causal_type": "reframed_by",
            "activation_confidence": 0.41,
            "monument_salience": 0.44,
            "memory_tension": 0.38,
        },
    )

    assert reopened.shared_tension > reframed.shared_tension
    assert reopened.repair_readiness > reframed.repair_readiness
    assert reframed.common_ground > reopened.common_ground
