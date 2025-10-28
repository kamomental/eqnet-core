from emot_terrain_lab.ops.care_canary import select_canary_ids


def test_select_canary_ids_basic():
    candidates = [f"id{i}" for i in range(10)]
    selected = select_canary_ids(candidates, ratio=0.2, seed=42)
    assert len(selected) == 2
    # determinism
    assert selected == select_canary_ids(candidates, ratio=0.2, seed=42)


def test_select_canary_ids_ratio_bounds():
    candidates = ["a", "b", "c"]
    assert select_canary_ids(candidates, ratio=0.0, seed=1) == set()
    assert len(select_canary_ids(candidates, ratio=1.0, seed=1)) == 3


def test_select_canary_ids_empty():
    assert select_canary_ids([], ratio=0.5, seed=1) == set()
