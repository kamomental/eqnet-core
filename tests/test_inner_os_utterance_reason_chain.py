from inner_os.appraisal_state import derive_appraisal_state
from inner_os.meaning_update_state import derive_meaning_update_state
from inner_os.utterance_reason_packet import derive_utterance_reason_packet


def test_appraisal_state_marks_shared_laugh_as_concrete_shift() -> None:
    state = derive_appraisal_state(
        current_focus="comment:after",
        current_risks=[],
        self_state={"recent_strain": 0.56},
        recent_dialogue_state={"state": "continuing_thread"},
        issue_state={"state": "light_tension"},
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.78,
            "jointness": 0.64,
            "afterglow": 0.58,
            "fragility": 0.22,
        },
        lightness_budget_state={"banter_room": 0.52},
        memory_dynamics_state={
            "dominant_mode": "ignite",
            "recall_anchor": "harbor",
            "dominant_relation_type": "same_anchor",
            "dominant_causal_type": "enabled_by",
            "causal_edges": [
                {
                    "causal_key": "harbor->promise:enabled_by",
                }
            ],
            "relation_edges": [
                {
                    "relation_key": "harbor->promise",
                }
            ],
            "meta_relations": [{"meta_type": "reinforces"}],
            "palace_topology": 0.48,
            "monument_salience": 0.54,
            "ignition_readiness": 0.62,
            "activation_confidence": 0.58,
            "memory_tension": 0.18,
        },
    ).to_dict()

    assert state["state"] == "active"
    assert state["moment_event"] == "laugh_break"
    assert state["shared_shift"] == "shared_smile_window"
    assert state["background_state"] in {"awkwardness_present", "strain_present"}
    assert state["dominant_relation_type"] == "same_anchor"
    assert state["dominant_relation_key"] == "harbor->promise"
    assert state["relation_meta_type"] == "reinforces"
    assert state["dominant_causal_type"] == "enabled_by"
    assert state["dominant_causal_key"] == "harbor->promise:enabled_by"
    assert state["memory_mode"] == "ignite"
    assert state["recall_anchor"] == "harbor"
    assert state["memory_resonance"] > 0.3
    assert state["easing_shift"] > 0.3


def test_meaning_update_state_keeps_small_shared_change_concrete() -> None:
    state = derive_meaning_update_state(
        appraisal_state={
            "state": "active",
            "moment_event": "laugh_break",
            "shared_shift": "shared_smile_window",
            "dominant_relation_type": "same_anchor",
            "dominant_relation_key": "harbor->promise",
            "relation_meta_type": "reinforces",
            "dominant_causal_type": "enabled_by",
            "dominant_causal_key": "harbor->promise:enabled_by",
            "memory_mode": "ignite",
            "recall_anchor": "harbor",
            "memory_resonance": 0.56,
            "easing_shift": 0.62,
            "fragility": 0.28,
        },
        recent_dialogue_state={"state": "bright_continuity"},
        discussion_thread_state={"state": "open_thread"},
        live_engagement_state={"state": "riff_with_comment"},
        memory_dynamics_state={
            "dominant_mode": "ignite",
            "recall_anchor": "harbor",
            "activation_confidence": 0.56,
        },
    ).to_dict()

    assert state["state"] == "active"
    assert state["self_update"] == "guard_relaxes_for_moment"
    assert state["relation_update"] == "shared_smile_window"
    assert state["relation_frame"] == "same_anchor_link"
    assert state["relation_key"] == "harbor->promise"
    assert state["relation_meta_type"] == "reinforces"
    assert state["causal_frame"] == "same_anchor_cause"
    assert state["causal_key"] == "harbor->promise:enabled_by"
    assert state["world_update"] == "small_moment_on_known_thread"
    assert state["memory_update"] == "known_thread_returns"
    assert state["recall_anchor"] == "harbor"
    assert state["preserve_guard"] in {"keep_it_small", "do_not_overclaim", "keep_it_small_and_linked"}


def test_utterance_reason_packet_prefers_brief_shared_smile_offer() -> None:
    packet = derive_utterance_reason_packet(
        appraisal_state={
            "state": "active",
            "moment_event": "laugh_break",
            "shared_shift": "shared_smile_window",
            "memory_mode": "ignite",
            "recall_anchor": "harbor",
            "memory_resonance": 0.58,
            "dominant_causal_key": "harbor->promise:enabled_by",
        },
        meaning_update_state={
            "state": "active",
            "relation_update": "shared_smile_window",
            "relation_frame": "same_anchor_link",
            "relation_key": "harbor->promise",
            "causal_frame": "same_anchor_cause",
            "causal_key": "harbor->promise:enabled_by",
            "memory_update": "known_thread_returns",
            "recall_anchor": "harbor",
            "memory_resonance": 0.58,
            "preserve_guard": "keep_it_small",
        },
        listener_action_state={"state": "warm_laugh_ack"},
        live_engagement_state={"state": "riff_with_comment"},
        memory_dynamics_state={
            "dominant_mode": "ignite",
            "recall_anchor": "harbor",
            "activation_confidence": 0.58,
        },
    ).to_dict()

    assert packet["state"] == "active"
    assert packet["reaction_target"] == "small_laugh_moment"
    assert packet["reason_frame"] == "name_shared_shift"
    assert packet["relation_frame"] == "same_anchor_link"
    assert packet["relation_key"] == "harbor->promise"
    assert packet["causal_frame"] == "same_anchor_cause"
    assert packet["causal_key"] == "harbor->promise:enabled_by"
    assert packet["offer"] == "brief_shared_smile"
    assert packet["memory_frame"] == "echo_known_thread"
    assert packet["memory_anchor"] == "harbor"
    assert packet["question_policy"] == "none"
    assert packet["tone_hint"] == "chatty_ack"
