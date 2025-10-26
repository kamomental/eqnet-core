# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.phase_ratio_filter import HysteresisCfg, PhaseRatioFilter


def test_small_oscillations_do_not_flip_direction() -> None:
    cfg = HysteresisCfg(ema_tau=10.0, hysteresis_up=0.65, hysteresis_down=0.35, min_window_tau=0.0, consecutive_up=2, consecutive_down=2)
    filt = PhaseRatioFilter(cfg)
    tau = 0.0
    for value in [0.45, 0.5, 0.55, 0.48, 0.52]:
        tau += 1.0
        filt.update(value, tau)
    assert filt.direction == "mixed"


def test_direction_requires_consecutive_observations() -> None:
    cfg = HysteresisCfg(ema_tau=5.0, hysteresis_up=0.6, hysteresis_down=0.4, min_window_tau=0.0, consecutive_up=3, consecutive_down=2)
    filt = PhaseRatioFilter(cfg)
    tau = 0.0
    for value in [0.62, 0.63]:
        tau += 1.0
        filt.update(value, tau)
    assert filt.direction != "reverse"
    tau += 1.0
    filt.update(0.64, tau)
    assert filt.direction == "reverse"
    tau += 1.0
    filt.update(0.38, tau)
    assert filt.direction == "reverse"
    tau += 1.0
    filt.update(0.35, tau)
    assert filt.direction == "forward"
