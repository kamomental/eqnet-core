from inner_os.listener_action_state import derive_listener_action_state


def test_listener_action_state_prefers_warm_laugh_ack_for_shared_laugh() -> None:
    state = derive_listener_action_state(
        expressive_style_state={"state": "light_playful", "lightness_room": 0.64},
        cultural_conversation_state={
            "state": "casual_shared",
            "joke_ratio_ceiling": 0.58,
            "politeness_pressure": 0.14,
        },
        live_engagement_state={"state": "riff_with_comment", "score": 0.62},
        shared_moment_state={
            "state": "shared_moment",
            "moment_kind": "laugh",
            "score": 0.74,
            "afterglow": 0.68,
        },
        recent_dialogue_state={"state": "continuing_thread"},
    ).to_dict()

    assert state["state"] == "warm_laugh_ack"
    assert state["filler_mode"] == "playful"
    assert state["token_profile"] == "soft_laugh"
