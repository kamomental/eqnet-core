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
from datetime import datetime
from pathlib import Path
import random
import time

import numpy as np

from devlife.core.body_nca import BodyNCA, BodyConfig
from devlife.core.grn import SimpleGRN
from devlife.core.policy_neat import SimplePolicy
from devlife.core.composer_eq import SimpleComposer
from devlife.runtime.archive import EpisodeArchive
from devlife.runtime.loop import DevelopmentLoop, StageConfig, SleepConfig
from actuate.learner_hooks import LearnerHooks
from devlife.mind.self_model import SelfReporter
from control.mcp import MCPController, MCPConfig
from devlife.runtime.alerts import AlertsLogger
from runtime.router import RuntimeRouter, AutonomyLevel, RouterConfig
from runtime.config import load_runtime_cfg
from telemetry import event as telemetry_event


def _resolve_log_path(template: str) -> Path:
    if "%" in template:
        return Path(datetime.utcnow().strftime(template))
    return Path(template)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ignite_dr", type=float, default=0.02, help="ﾎ燃 threshold for ignition")
    ap.add_argument("--ignite_ez", type=float, default=0.20, help="entropy_z threshold for ignition")
    ap.add_argument("--ignite_ms_default", type=int, default=250, help="default ignite duration (ms) on trigger")
    ap.add_argument("--tom_trust_thresh", type=float, default=0.30, help="threshold to downshift when intent_trust falls below")
    ap.add_argument("--tom_trust_high", type=float, default=0.45, help="hysteresis high threshold to allow upshift")
    ap.add_argument("--tom_alpha", type=float, default=0.20, help="ToM EMA alpha")
    ap.add_argument("--tom_med_window", type=int, default=5, help="ToM median filter window")
    ap.add_argument("--tom_rate_limit", type=float, default=0.15, help="ToM rate limit per step (fraction)")
    ap.add_argument("--love_low", type=float, default=0.45, help="affect.love threshold to trigger tone softening")
    ap.add_argument("--love_high", type=float, default=0.82, help="affect.love threshold to trigger FAST-path style override")
    ap.add_argument("--selfother_thresh", type=float, default=0.15, help="Self/Other conflict threshold")
    ap.add_argument("--tag", type=str, default="", help="Optional experiment label recorded into telemetry/logs.")
    ap.add_argument(
        "--fastpath_style_profile",
        action="append",
        default=None,
        help="FAST-path profiles allowed to perform style overrides during the run (lab only).",
    )
    ap.add_argument(
        "--field_metrics_log",
        type=str,
        default="",
        help="Optional path to a terrain field_metrics log (JSON or JSONL) to replay into the loop.",
    )
    ap.add_argument(
        "--telemetry_log",
        type=str,
        default="",
        help="Override telemetry log path (default: telemetry/ignition-YYYYMMDD.jsonl).",
    )
    ap.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved runtime configuration and exit.",
    )
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = ap.parse_args()
    runtime_cfg = load_runtime_cfg()
    if args.print_config:
        print(runtime_cfg)
        return
    seed = args.seed if args.seed is not None else int(time.time() * 1000) % 1_000_000
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    print(f"[run_quick_loop] seed={seed}")

    body = BodyNCA(BodyConfig())
    grn = SimpleGRN()
    policy = SimplePolicy()
    composer = SimpleComposer()
    archive = EpisodeArchive()
    router = RuntimeRouter(RouterConfig(upgrade_hold_seconds=0.0))
    router.force_level(AutonomyLevel.L2)
    from devlife.runtime.alerts import AlertsConfig
    alerts = AlertsLogger(AlertsConfig(tom_trust_threshold=args.tom_trust_thresh))

    experiment_tag = args.tag.strip() or None
    learner_hooks = LearnerHooks(experiment_tag=experiment_tag)
    self_reporter = SelfReporter()
    mcp_cfg = MCPConfig(love_low=args.love_low, love_high=args.love_high)
    mcp = MCPController(
        config=mcp_cfg,
        learner_hooks=learner_hooks,
        self_reporter=self_reporter,
        router=router,
        eligible_fastpath_profiles=args.fastpath_style_profile,
        experiment_tag=experiment_tag,
    )

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

    telemetry_template = args.telemetry_log or runtime_cfg.telemetry.log_path
    telemetry_path = _resolve_log_path(telemetry_template)

    def _telemetry_hook(name, payload):
        if telemetry_event is None:
            return
        data = dict(payload) if isinstance(payload, dict) else payload
        if experiment_tag and isinstance(data, dict):
            data = dict(data)
            data.setdefault("tag", experiment_tag)
        telemetry_event(name, data, log_path=telemetry_path)

    loop = DevelopmentLoop(
        body,
        grn,
        policy,
        composer,
        archive,
        log_hook=mcp.handle_episode,
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
        telemetry_hook=_telemetry_hook,
        runtime_cfg=runtime_cfg,
    )
    print(f"[info] telemetry log: {telemetry_path}")
    if telemetry_event is not None:
        payload = {"seed": seed}
        if experiment_tag:
            payload["tag"] = experiment_tag
        telemetry_event("run.seed", payload, log_path=telemetry_path)
    if args.field_metrics_log:
        fm_path = Path(args.field_metrics_log)
        if fm_path.exists():
            loop.load_field_metrics_log(fm_path)
            print(f"[info] loaded field metrics log from {fm_path}")
        else:
            print(f"[warn] field metrics log not found: {fm_path}")
    # Router ToM thresholds
    router.config.trust_low = args.tom_trust_thresh
    router.config.trust_high = args.tom_trust_high
    router.config.min_hold_s = 6.0

    # Wire alerts -> router downshift
    alerts.downshift_fn = router.downshift

    loop.run()
    print("[info] KPI log -> logs/kpi_rollup.jsonl")
    print("[info] MCP actions -> logs/mcp_actions.jsonl")
    print("[info] Learner hooks -> logs/learner_hooks.jsonl")
    print("[info] Fast-path tracker -> logs/fastpath_state.jsonl")
    print("Run complete. Check logs/episodes/* and logs/alerts.jsonl")


if __name__ == "__main__":
    main()




