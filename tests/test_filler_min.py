from emot_terrain_lab.text.filler_inserter import insert_fillers


def test_insert_fillers_basic():
    base = "�����̗\��������Ă��������B��c�͌ߌ�ł��B"
    out = insert_fillers(base, [("filler", "sentence_start", "������")])
    assert "������" in out
    assert "�c" in out
