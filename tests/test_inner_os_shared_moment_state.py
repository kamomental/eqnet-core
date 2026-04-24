from inner_os.shared_moment_state import derive_shared_moment_state


def test_shared_moment_state_detects_laugh_moment_from_recent_text() -> None:
    state = derive_shared_moment_state(
        current_focus="comment:latest",
        current_risks=[],
        self_state={
            "surface_user_text": "さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
            "recent_dialogue_history": [
                "今日はちょっといいことがあって、帰り道は少し気分が軽かったんです。"
            ],
        },
        recent_dialogue_state={"state": "continuing_thread"},
        discussion_thread_state={"state": "open_thread"},
        issue_state={"state": "light_tension"},
        lightness_budget_state={"state": "open_play", "banter_room": 0.44},
    ).to_dict()

    assert state["state"] == "shared_moment"
    assert state["moment_kind"] == "laugh"
    assert state["score"] >= 0.42
    assert state["jointness"] >= 0.5
    assert "cue:laugh" in state["dominant_inputs"]
