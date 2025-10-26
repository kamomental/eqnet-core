# -*- coding: utf-8 -*-

from emot_terrain_lab.sense.residuals import compute_residual


def test_delta_stays_between_zero_and_one() -> None:
    f = {"firmness": 0.7, "aftertaste": 0.6}
    fhat = {"firmness": 0.4, "aftertaste": 0.1}
    shareability = {"firmness": 0.6, "aftertaste": 0.3}
    weights = {"firmness": 0.5, "aftertaste": 0.5}

    result = compute_residual(f, fhat, shareability, weights)

    assert 0.0 <= result["delta"] <= 1.0
    assert len(result["top"]) == 2
