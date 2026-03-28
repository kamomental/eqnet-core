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
