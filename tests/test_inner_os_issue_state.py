from inner_os.issue_state import derive_issue_state


def test_issue_state_detects_pausing_issue_from_pause_markers() -> None:
    state = derive_issue_state(
        "いったんここまでにして、またあとで戻れたら助かります。",
        [
            "さっきの引っかかりをどう言えばいいか迷っています。",
            "まだ整理できていない感じがあります。",
        ],
        discussion_thread_state={
            "state": "revisit_issue",
            "unresolved_pressure": 0.44,
            "revisit_readiness": 0.62,
            "topic_anchor": "さっきの引っかかり",
        },
        recent_dialogue_state={"state": "reopening_thread", "thread_carry": 0.68},
    ).to_dict()

    assert state["state"] == "pausing_issue"
    assert state["pause_readiness"] >= 0.38
    assert "pause_marker" in state["dominant_inputs"]


def test_issue_state_detects_resolving_issue_from_resolution_markers() -> None:
    state = derive_issue_state(
        "ひとまず大丈夫そうです。だいぶ見えてきました。",
        ["さっきの引っかかりをどう言えばいいか迷っています。"],
        discussion_thread_state={
            "state": "settling_issue",
            "unresolved_pressure": 0.12,
            "topic_anchor": "さっきの引っかかり",
        },
    ).to_dict()

    assert state["state"] == "resolving_issue"
    assert state["resolution_readiness"] >= 0.42
