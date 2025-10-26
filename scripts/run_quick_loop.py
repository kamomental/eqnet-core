#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a minimal DevelopmentLoop with Router + Alerts wired for a quick demo.

This constructs BodyNCA, SimpleGRN, SimplePolicy, SimpleComposer, EpisodeArchive,
RuntimeRouter, AlertsLogger, and the DevelopmentLoop, then runs a short stage.

Usage:
  python scripts/run_quick_loop.py --steps 120 --ignite_dr 0.08 --ignite_ez -0.5
  In another shell, tail alerts: python scripts/alerts_tail.py --follow
"""

from __future__ import annotations

import argparse

from devlife.core.body_nca import BodyNCA, BodyConfig
from devlife.core.grn import SimpleGRN
from devlife.core.policy_neat import SimplePolicy
from devlife.core.composer_eq import SimpleComposer
from devlife.runtime.archive import EpisodeArchive
from devlife.runtime.loop import DevelopmentLoop, StageConfig, SleepConfig
from devlife.runtime.alerts import AlertsLogger
from runtime.router import RuntimeRouter, AutonomyLevel, RouterConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ignite_dr", type=float, default=0.02, help="ΔR threshold for ignition")
    ap.add_argument("--ignite_ez", type=float, default=0.20, help="entropy_z threshold for ignition")
    ap.add_argument("--ignite_ms_default", type=int, default=250, help="default ignite duration (ms) on trigger")
    ap.add_argument("--tom_trust_thresh", type=float, default=0.30, help="threshold to downshift when intent_trust falls below")
    ap.add_argument("--tom_trust_high", type=float, default=0.45, help="hysteresis high threshold to allow upshift")
    ap.add_argument("--tom_alpha", type=float, default=0.20, help="ToM EMA alpha")
    ap.add_argument("--tom_med_window", type=int, default=5, help="ToM median filter window")
    ap.add_argument("--tom_rate_limit", type=float, default=0.15, help="ToM rate limit per step (fraction)")
    ap.add_argument("--selfother_thresh", type=float, default=0.15, help="Self/Other conflict threshold")
    args = ap.parse_args()

    body = BodyNCA(BodyConfig())
    grn = SimpleGRN()
    policy = SimplePolicy()
    composer = SimpleComposer()
    archive = EpisodeArchive()
    router = RuntimeRouter(RouterConfig(upgrade_hold_seconds=0.0))
    router.force_level(AutonomyLevel.L2)
    from devlife.runtime.alerts import AlertsConfig
    alerts = AlertsLogger(AlertsConfig(tom_trust_threshold=args.tom_trust_thresh))

    # Optional mood integrator for suffering/tension KPIs
    try:
        from devlife.bricks.affect_mood_integrator import MoodIntegrator
        mood = MoodIntegrator()
    except Exception:
        mood = None

    # Theory of Mind + Self/Other
    try:
        from devlife.social.tom import TheoryOfMind
        from devlife.bricks.threads_selfother import SelfOtherClassifier, SelfOtherConfig
        tom = TheoryOfMind()
        # smoothing knobs
        tom._cfg.alpha = args.tom_alpha
        tom._cfg.med_window = int(args.tom_med_window)
        tom._cfg.rate_limit = args.tom_rate_limit
        selfother = SelfOtherClassifier(SelfOtherConfig(threshold=args.selfother_thresh))
    except Exception:
        tom = None
        selfother = None

    loop = DevelopmentLoop(
        body,
        grn,
        policy,
        composer,
        archive,
        stages=[StageConfig(name="test", duration_steps=args.steps)],
        sleep=SleepConfig(interval_steps=999999),
        alert_logger=alerts,
        router=router,
        ignite_delta_R_thresh=args.ignite_dr,
        ignite_entropy_z_thresh=args.ignite_ez,
        ignite_ms_default=args.ignite_ms_default,
        mood_integrator=mood,
        selfother=selfother,
        theory_of_mind=tom,
    )
    # Router ToM thresholds
    router.config.trust_low = args.tom_trust_thresh
    router.config.trust_high = args.tom_trust_high
    router.config.min_hold_s = 6.0

    # Wire alerts -> router downshift
    alerts.downshift_fn = router.downshift

    loop.run()
    print("Run complete. Check logs/episodes/* and logs/alerts.jsonl")


if __name__ == "__main__":
    main()
