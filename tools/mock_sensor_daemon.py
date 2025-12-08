#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate mock StreamingSensor raw_frame payloads for Phase 1 bring-up.

This daemon exposes a pluggable sensor source. `--source-class mock` keeps the
built-in generator, while `--source-class pkg.module:ClassName` allows swapping
in a real sensor bridge without touching downstream code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Type

try:
    from eqnet.hub.streaming_sensor import StreamingSensorState
except Exception:  # pragma: no cover - optional during bootstrap
    StreamingSensorState = None  # type: ignore


@dataclass
class WaveParams:
    baseline: float
    amplitude: float
    period_s: float


class SensorSource:
    """Interface for anything that can produce ``raw_frame`` dicts."""

    def next_raw_frame(self, t: float) -> Dict[str, object]:  # pragma: no cover - protocol shim
        raise NotImplementedError


class MockSensorSource(SensorSource):
    """Built-in mock generator with simple sinusoidal dynamics."""

    def __init__(
        self,
        *,
        rng: random.Random,
        hr_params: WaveParams,
        breath_params: WaveParams,
        activity_cfg: Dict[str, float],
        hr_noise: float = 1.2,
        breath_noise: float = 0.02,
    ) -> None:
        self._rng = rng
        self._hr = hr_params
        self._breath = breath_params
        self._activity = activity_cfg
        self._hr_noise = hr_noise
        self._breath_noise = breath_noise

    def next_raw_frame(self, t: float) -> Dict[str, object]:
        phase = self._rng.random() * math.pi * 2.0
        hr = self._hr.baseline + self._hr.amplitude * math.sin(2.0 * math.pi * t / self._hr.period_s + phase)
        if self._hr_noise > 0:
            hr += self._rng.gauss(0.0, self._hr_noise)
        activity = max(0.0, min(1.5, self._activity["pose"] + self._rng.random() * self._activity["burst"]))
        voice = max(0.0, min(1.0, self._activity["voice"] + self._rng.random() * 0.1))
        breath = self._breath.baseline + self._breath.amplitude * math.sin(
            2.0 * math.pi * t / self._breath.period_s + phase * 0.5
        )
        if self._breath_noise > 0:
            breath += self._rng.gauss(0.0, self._breath_noise)
        breath = max(0.0, min(1.0, breath))
        delta_hr = hr - self._hr.baseline
        pose_vec = [activity * math.sin(t * 0.7), activity * math.cos(t * 0.5)]
        inner_emotion = max(0.0, min(1.0, 0.4 + 0.3 * math.tanh(delta_hr / 10.0)))
        raw_frame: Dict[str, object] = {
            "timestamp": time.time(),
            "heart_rate_raw": round(hr, 3),
            "heart_rate_baseline": round(self._hr.baseline, 3),
            "breath_rate": round(breath, 4),
            "pose_vec": pose_vec,
            "voice_level": round(voice, 4),
            "inner_emotion_score": round(inner_emotion, 4),
            "pose_detected": True,
            "person_count": 1,
            "has_face": True,
        }
        if self._rng.random() < 0.15:
            raw_frame["privacy_tags"] = ["private"]
        return raw_frame


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-ms", type=int, default=500, help="Sampling interval in milliseconds (default: 500)")
    parser.add_argument("--duration", type=float, default=None, help="Total duration in seconds (default: run until Ctrl-C)")
    parser.add_argument("--steps", type=int, default=None, help="Number of frames to emit (overrides --duration when set)")
    parser.add_argument("--hr-baseline", type=float, default=78.0, help="Heart rate baseline (bpm)")
    parser.add_argument("--hr-amp", type=float, default=6.0, help="Sinusoidal heart rate amplitude")
    parser.add_argument("--breath-baseline", type=float, default=0.23, help="Baseline breath rate (0-1 proxy)")
    parser.add_argument("--breath-amp", type=float, default=0.05, help="Breath amplitude")
    parser.add_argument("--activity-mode", choices=["calm", "mixed", "active"], default="mixed", help="Preset for pose/voice energy")
    parser.add_argument("--log-path", type=Path, default=Path("logs/streaming_sensor_raw.jsonl"), help="Output JSONL path")
    parser.add_argument("--emit-metrics", action="store_true", help="Pass frames through StreamingSensorState to attach derived metrics")
    parser.add_argument("--quiet", action="store_true", help="Silence stdout summaries")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic RNG seed")
    parser.add_argument(
        "--source-class",
        default="mock",
        help="Either 'mock' or a dotted import path 'pkg.module:ClassName' implementing SensorSource",
    )
    parser.add_argument(
        "--source-kwargs",
        default="{}",
        help="JSON dict forwarded to the source-class constructor",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _activity_profile(mode: str) -> Dict[str, float]:
    if mode == "calm":
        return {"pose": 0.05, "voice": 0.08, "burst": 0.08}
    if mode == "active":
        return {"pose": 0.35, "voice": 0.3, "burst": 0.28}
    return {"pose": 0.15, "voice": 0.12, "burst": 0.18}


def _load_source_class(spec: str) -> Type[SensorSource]:
    if spec == "mock":
        return MockSensorSource
    if ":" not in spec:
        raise ValueError("Custom source must be in 'module:ClassName' format")
    module_name, _, class_name = spec.partition(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class '{class_name}' not found in module '{module_name}'")
    return cls  # type: ignore[return-value]


def _create_source(
    args: argparse.Namespace,
    rng: random.Random,
    hr_params: WaveParams,
    breath_params: WaveParams,
    activity_cfg: Dict[str, float],
) -> SensorSource:
    cls = _load_source_class(args.source_class)
    if args.source_class == "mock":
        return cls(rng=rng, hr_params=hr_params, breath_params=breath_params, activity_cfg=activity_cfg)
    extra = json.loads(args.source_kwargs)
    return cls(
        rng=rng,
        hr_params=hr_params,
        breath_params=breath_params,
        activity_cfg=activity_cfg,
        **extra,
    )


def _maybe_attach_metrics(raw_frame: Dict[str, object]) -> Optional[Dict[str, object]]:
    if StreamingSensorState is None:
        return None
    snapshot = StreamingSensorState.from_raw(raw_frame)
    return {
        "fused_vec": snapshot.fused_vec.tolist(),
        "metrics": snapshot.metrics,
    }


def run_daemon(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    hr_params = WaveParams(baseline=args.hr_baseline, amplitude=args.hr_amp, period_s=28.0)
    breath_params = WaveParams(baseline=args.breath_baseline, amplitude=args.breath_amp, period_s=7.0)
    activity_cfg = _activity_profile(args.activity_mode)
    source = _create_source(args, rng, hr_params, breath_params, activity_cfg)
    interval = max(0.05, args.interval_ms / 1000.0)

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    next_emit = start
    steps = 0
    with args.log_path.open("a", encoding="utf-8") as handle:
        while True:
            now = time.time()
            if args.steps is not None and steps >= args.steps:
                break
            if args.duration is not None and (now - start) >= args.duration:
                break
            if now < next_emit:
                time.sleep(min(0.05, next_emit - now))
                continue
            t = now - start
            raw_frame = source.next_raw_frame(t)
            record = {"raw_frame": raw_frame}
            if args.emit_metrics:
                metrics = _maybe_attach_metrics(raw_frame)
                if metrics is not None:
                    record.update(metrics)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            steps += 1
            if not args.quiet:
                print(
                    f"[sensor] t={t:6.2f}s hr={raw_frame['heart_rate_raw']:6.2f} "
                    f"breath={raw_frame['breath_rate']:5.3f} voice={raw_frame['voice_level']:5.3f}"
                )
            next_emit += interval


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    try:
        run_daemon(args)
    except KeyboardInterrupt:
        print("\n[sensor] stopped", file=sys.stderr)


if __name__ == "__main__":
    main()

