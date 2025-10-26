# -*- coding: utf-8 -*-

from emot_terrain_lab.causality.do_graph import DOgraph


def test_do_graph_topk_has_entries_and_normalizes() -> None:
    graph = DOgraph(normalize_topk=True, contrib_cap=0.05)
    graph.load_weights({"色→受": 0.3, "受→想": 0.2})
    graph.update({"色": 0.8, "受": 0.5, "想": 0.2, "行": 0.1, "識": 0.1}, d_tau=1.0)
    top = graph.topk(k=5)
    assert len(top) >= 1
    names = {name for name, _ in top}
    assert "色→受" in names
    assert "受→想" in names
    total = sum(max(0.0, val) for _, val in top)
    assert 0.99 <= total <= 1.01
