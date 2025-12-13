from emot_terrain_lab.text.filler_inserter import insert_fillers


BASE_TEXT = "This is a sample sentence. Another line follows."


def test_insert_fillers_sentence_start():
    out = insert_fillers(BASE_TEXT, [("filler", "sentence_start", "um")])
    assert out.startswith("um")
    assert "Another line" in out


def test_insert_fillers_clause_injection():
    text = "I walked, paused, and kept going."
    out = insert_fillers(text, [("filler", "clause", "you know")])
    assert "you know" in out
