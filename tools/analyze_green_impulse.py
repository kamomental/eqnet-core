# -*- coding: utf-8 -*-
"""Analyze EQNet green impulse logs (FFT + decay)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np


def load_series(path: Path, key: str) -> Tuple[np.ndarray, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    series = np.asarray(data["series"].get(key), dtype=np.float32)
    meta = data.get("meta", {})
    return series, meta


def compute_fft(series: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    centered = series - float(np.mean(series))
    fft = np.fft.rfft(centered)
    freq = np.fft.rfftfreq(len(series), d=dt)
    power = np.abs(fft)
    return freq, power


def estimate_decay(series: np.ndarray) -> float:
    amplitude = np.abs(series - np.mean(series))
    amplitude = amplitude[amplitude > 1e-6]
    if amplitude.size < 3:
        return float("nan")
    x = np.arange(amplitude.size)
    y = np.log(amplitude)
    slope, _ = np.polyfit(x, y, 1)
    return float(-1.0 / slope) if slope < 0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="green impulse JSON log")
    parser.add_argument("--series", default="phi_local", help="series key to analyze")
    parser.add_argument("--dt", type=float, default=1.0, help="sampling interval (steps)")
    parser.add_argument("--top", type=int, default=3, help="number of dominant frequencies to display")
    parser.add_argument("--emit", type=Path, default=None, help="optional JSONL path to log tau and dominant frequencies")
    args = parser.parse_args()

    series, meta = load_series(args.input, args.series)
    freq, power = compute_fft(series, args.dt)
    decay_tau = estimate_decay(series)

    idx = np.argsort(power)[::-1][: args.top]
    print(f"[green_analyze] series='{args.series}' length={len(series)} dt={args.dt}")
    if np.isfinite(decay_tau):
        print(f"  estimated decay tau ~= {decay_tau:.2f} steps")
    else:
        print("  decay estimate unavailable")
    print("  dominant frequencies (1/steps):")
    for rank, i in enumerate(idx, 1):
        print(f"    {rank}. f={freq[i]:.4f}, power={power[i]:.4f}")
    if meta:
        print("  meta:")
        for key, value in meta.items():
            print(f"    {key}: {value}")

    summary = {
        "ts": time.time(),
        "input": str(args.input),
        "series": args.series,
        "decay_tau": float(decay_tau) if np.isfinite(decay_tau) else None,
        "top_freqs": [
            {"freq": float(freq[i]), "power": float(power[i])} for i in idx
        ],
        "meta": meta,
    }
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        with args.emit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
