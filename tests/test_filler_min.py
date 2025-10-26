from emot_terrain_lab.text.filler_inserter import insert_fillers


def test_insert_fillers_basic():
    base = "–¾“ú‚Ì—\’è‚ğ‹³‚¦‚Ä‚­‚¾‚³‚¢B‰ï‹c‚ÍŒßŒã‚Å‚·B"
    out = insert_fillers(base, [("filler", "sentence_start", "‚¦‚Á‚Æ")])
    assert "‚¦‚Á‚Æ" in out
    assert "c" in out
