# -*- coding: utf-8 -*-

from emot_terrain_lab.utils.referent import sanitize_referent_label


def test_sanitize_referent_label_filters_pii() -> None:
    assert sanitize_referent_label("右上の棒グラフ") == "右上の棒グラフ"
    assert sanitize_referent_label("face_id_123") is None
    assert sanitize_referent_label("x=120,y=45") is None
