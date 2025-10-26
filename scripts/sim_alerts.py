#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick simulation to emit ignite and downshift alerts without full runtime.

Writes a few synthetic episodes into AlertsLogger so you can verify:
- ignite trigger
- inhibit (long ignite_ms)
- autonomy.downshift (low intent_trust)

Usage:
  python scripts/sim_alerts.py
Then in another shell:
  python scripts/alerts_tail.py --follow
"""

from __future__ import annotations

import time
from devlife.runtime.alerts import AlertsLogger


def main() -> None:
    logger = AlertsLogger()

    # 1) ignite event
    ep = {
        "stage": "test",
        "step": 1,
        "ignite": {"trigger": True, "I": 0.35, "delta_R": 0.2, "entropy_z": -1.4, "ignite_ms": 250, "alerts": 1},
    }
    logger.evaluate_and_log(ep)

    time.sleep(0.1)

    # 2) long ignition, should emit inhibit suggestion
    ep2 = {
        "stage": "test",
        "step": 2,
        "ignite": {"trigger": False, "I": 0.10, "delta_R": 0.05, "entropy_z": -0.5, "ignite_ms": 1300, "alerts": 1},
    }
    logger.evaluate_and_log(ep2)

    time.sleep(0.1)

    # 3) low ToM trust should downshift
    ep3 = {"stage": "test", "step": 3, "tom": {"intent_trust": 0.2}}
    logger.evaluate_and_log(ep3)

    print("Simulated 3 alerts → see logs/alerts.jsonl")


if __name__ == "__main__":
    main()

