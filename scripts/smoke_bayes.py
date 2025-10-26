# -*- coding: utf-8 -*-
"""Smoke test for Bayesian persona and safety helpers."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from emot_terrain_lab.bayes.posteriors import BetaPosterior
from emot_terrain_lab.persona.selector_bayes import BayesPersonaSelector
from emot_terrain_lab.safety.bayes_gate import BayesSafetyGate


def main() -> None:
    beta = BetaPosterior(1.0, 1.0, half_life_tau=4.0)
    beta.update(True, d_tau=1.0)
    beta.update(False, d_tau=0.5)
    print(
        "beta mean/ci:",
        round(beta.mean(), 3),
        round(beta.ci_lower(), 3),
        round(beta.ci_upper(), 3),
    )

    selector = BayesPersonaSelector(["professional", "playful"], half_life_tau=6.0)
    chosen = selector.choose()
    selector.update(chosen, success=True, d_tau=0.7)
    print("persona chosen:", chosen, round(selector.metrics()[chosen]["mean"], 3))

    gate = BayesSafetyGate(read_only_th=0.2, block_th=0.4)
    print("safety decision@cold:", gate.decide("dialogue"))
    gate.update("dialogue", misfire=True, d_tau=0.8)
    print("safety decision@after_fail:", gate.decide("dialogue"))
    print("OK")


if __name__ == "__main__":
    main()
