from inner_os.recent_dialogue_state import derive_recent_dialogue_state


def test_recent_dialogue_state_detects_continuing_thread_from_overlap() -> None:
    state = derive_recent_dialogue_state(
        "さっきの引っかかりを、いま少しだけ見直したいです。",
        [
            "さっきの引っかかりがまだ残っている感じがします。",
            "いまは急がなくて大丈夫です。",
        ],
    ).to_dict()

    assert state["state"] == "continuing_thread"
    assert state["overlap_score"] > 0.14
    assert state["thread_carry"] > 0.28
    assert "history_overlap" in state["dominant_inputs"]


def test_recent_dialogue_state_detects_reopening_thread_from_marker() -> None:
    state = derive_recent_dialogue_state(
        "前の話の続きを、いま話せるところから。",
        [
            "いったんここで止めて、また戻れる形にしておきましょう。",
            "残りはまた話せそうなときに。",
        ],
    ).to_dict()

    assert state["state"] == "reopening_thread"
    assert state["reopen_pressure"] >= 0.24
    assert "reopen_marker" in state["dominant_inputs"]


def test_recent_dialogue_state_extracts_quoted_anchor_from_history() -> None:
    state = derive_recent_dialogue_state(
        "前に触れた話の続きを少しだけ。",
        [
            "前に出ていた「港での約束」のところから、いま話せるぶんだけ戻れば大丈夫です。",
        ],
    ).to_dict()

    assert state["recent_anchor"] == "港での約束"


def test_recent_dialogue_state_prefers_quoted_anchor_when_reopening_text_is_generic() -> None:
    state = derive_recent_dialogue_state(
        "その続きを、いま話せるところから少しだけ。",
        [
            "うん、話せるところからで大丈夫です。急がなくていいです。",
            "前に出ていた「港での約束」のところを、いま話せるぶんだけ拾う感じで大丈夫です。",
        ],
    ).to_dict()

    assert state["state"] == "reopening_thread"
    assert state["recent_anchor"] == "港での約束"
    assert "quoted_history_anchor" in state["dominant_inputs"]


def test_recent_dialogue_state_detects_continuation_marker_without_history() -> None:
    state = derive_recent_dialogue_state(
        "さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        [],
        interaction_policy={
            "live_engagement_state": {"state": "pickup_comment"},
            "lightness_budget_state": {"state": "open_play"},
        },
    ).to_dict()

    assert state["state"] == "continuing_thread"
    assert state["thread_carry"] >= 0.3
    assert "continuation_marker" in state["dominant_inputs"]


def test_recent_dialogue_state_prefers_continuing_thread_for_shared_smile_followup() -> None:
    state = derive_recent_dialogue_state(
        "さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        ["前の流れはまだしんどさが残っていた。"],
        interaction_policy={
            "live_engagement_state": {"state": "riff_with_comment"},
            "lightness_budget_state": {"state": "open_play"},
            "shared_moment_state": {
                "state": "shared_moment",
                "moment_kind": "laugh",
                "score": 0.72,
                "jointness": 0.7,
                "afterglow": 0.62,
            },
            "utterance_reason_packet": {
                "state": "active",
                "offer": "brief_shared_smile",
                "question_policy": "none",
            },
            "organism_state": {
                "dominant_posture": "play",
                "play_window": 0.44,
                "expressive_readiness": 0.58,
            },
        },
    ).to_dict()

    assert state["state"] == "continuing_thread"
    assert "shared_moment_reentry" in state["dominant_inputs"]
