from inner_os.discussion_thread_state import derive_discussion_thread_state


def test_discussion_thread_state_detects_revisit_issue() -> None:
    state = derive_discussion_thread_state(
        "前の話で、まだ引っかかっているところに戻りたいです。",
        [
            "まだ引っかかっているところが残っています。",
            "いったん止めて、また戻れるようにしておきましょう。",
        ],
        recent_dialogue_state={"state": "reopening_thread", "thread_carry": 0.72},
    ).to_dict()

    assert state["state"] == "revisit_issue"
    assert state["revisit_readiness"] >= 0.42
    assert state["unresolved_pressure"] >= 0.18
    assert "revisit_marker" in state["dominant_inputs"]


def test_discussion_thread_state_can_mark_settling_issue() -> None:
    state = derive_discussion_thread_state(
        "ひとまず納得できたので、今日はここまでで大丈夫です。",
        ["さっきの引っかかりをどう置くか話していました。"],
    ).to_dict()

    assert state["state"] == "settling_issue"
    assert state["unresolved_pressure"] < 0.26


def test_discussion_thread_state_extracts_quoted_anchor_from_history() -> None:
    state = derive_discussion_thread_state(
        "前に少し引っかかっていた話の続きを戻したいです。",
        [
            "前に出ていた「港での約束」のところから、いま話せるぶんだけ戻れば大丈夫です。",
        ],
        recent_dialogue_state={"state": "reopening_thread", "thread_carry": 0.72},
    ).to_dict()

    assert state["topic_anchor"] == "港での約束"


def test_discussion_thread_state_prefers_quoted_anchor_when_revisit_text_is_generic() -> None:
    state = derive_discussion_thread_state(
        "その続きを、いま話せるところから少しだけ。",
        [
            "うん、話せるところからで大丈夫です。急がなくていいです。",
            "前に出ていた「港での約束」のところを、いま話せるぶんだけ拾う感じで大丈夫です。",
        ],
        recent_dialogue_state={"state": "reopening_thread", "thread_carry": 0.72},
    ).to_dict()

    assert state["topic_anchor"] == "港での約束"
    assert "quoted_history_anchor" in state["dominant_inputs"]
    assert "revisit_marker" in state["dominant_inputs"]
