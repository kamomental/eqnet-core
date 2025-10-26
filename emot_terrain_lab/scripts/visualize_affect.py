# -*- coding: utf-8 -*-
"""
Simple real-time affect visualiser using matplotlib.

Usage:
    python scripts/visualize_affect.py --duration 30 --interval 0.5

This script periodically runs the hub runtime step (with synthetic input in the
current scaffold) and plots valence / arousal / confidence. Once actual
perception inputs are connected, the same visualiser can be reused.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from hub import EmotionalHubRuntime, RuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0, help="Total duration to run (s)")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval (s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))

    timestamps = deque(maxlen=200)
    valences = deque(maxlen=200)
    arousals = deque(maxlen=200)
    confidences = deque(maxlen=200)

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("EQNet Affect (valence / arousal / confidence)")

    start_time = time.time()
    last_time = start_time

    while True:
        now = time.time()
        if now - start_time > args.duration:
            break

        result = runtime.step(user_text=None)
        affect = result["affect"]

        timestamps.append(now - start_time)
        valences.append(affect.valence)
        arousals.append(affect.arousal)
        confidences.append(affect.confidence)

        ax[0].cla()
        ax[0].plot(list(timestamps), list(valences), color="tab:orange")
        ax[0].set_ylabel("Valence")
        ax[0].set_ylim([-1, 1])

        ax[1].cla()
        ax[1].plot(list(timestamps), list(arousals), color="tab:blue")
        ax[1].set_ylabel("Arousal")
        ax[1].set_ylim([-1, 1])

        ax[2].cla()
        ax[2].plot(list(timestamps), list(confidences), color="tab:green")
        ax[2].set_ylabel("Confidence")
        ax[2].set_ylim([0, 1])
        ax[2].set_xlabel("Time (s)")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        sleep_time = max(0.0, args.interval - (time.time() - last_time))
        time.sleep(sleep_time)
        last_time = time.time()

    plt.show(block=True)


if __name__ == "__main__":
    main()
