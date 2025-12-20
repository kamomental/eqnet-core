from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from emot_terrain_lab.sim.mini_world import MiniWorldScenario, MiniWorldSimulator, MiniWorldStep


def _parse_ts(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        print(f"[WARN] missing file: {path}")
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _extract_commit_meta(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    commit_ts = record.get("commit_ts")
    window_len = record.get("window_len")
    if commit_ts is None or window_len is None:
        obs = record.get("observations") or {}
        meta = (obs.get("meta") or {}) if isinstance(obs, dict) else {}
        commit_ts = commit_ts or meta.get("commit_ts")
        window_len = window_len or meta.get("window_len")
    if isinstance(window_len, str):
        try:
            window_len = int(window_len)
        except ValueError:
            window_len = None
    if isinstance(window_len, (float, int)):
        window_len = int(window_len)
    if not isinstance(commit_ts, str):
        commit_ts = None
    if not isinstance(window_len, int) or window_len <= 0:
        window_len = None
    return commit_ts, window_len


def _build_step(window: List[Dict[str, Any]], step_ms: int, commit_ts: str, window_len: int) -> MiniWorldStep:
    audio_rms = _mean([_safe_float(obs.get("audio", {}).get("rms")) for obs in window])
    audio_peak = _mean([_safe_float(obs.get("audio", {}).get("peak")) for obs in window])
    audio_flux = _mean([_safe_float(obs.get("audio", {}).get("flux")) for obs in window])
    vad_mean = _mean([_safe_float(obs.get("audio", {}).get("vad")) for obs in window])
    flow_mean = _mean([_safe_float(obs.get("video", {}).get("flow_mag")) for obs in window])
    luma_mean = _mean([_safe_float(obs.get("video", {}).get("luma_mean"), 0.5) for obs in window])
    scene_mean = _mean([_safe_float(obs.get("video", {}).get("scene_change")) for obs in window])
    darkness = _clamp01(1.0 - luma_mean)
    hazard_score = _clamp01(0.45 * audio_peak + 0.35 * darkness + 0.20 * scene_mean)
    hazard_sources: List[str] = []
    if audio_peak > 0.6:
        hazard_sources.append("audio_peak")
    if darkness > 0.6:
        hazard_sources.append("darkness")
    if scene_mean > 0.5:
        hazard_sources.append("scene_change")
    observations = {
        "timestamp": window[-1].get("timestamp"),
        "audio": {
            "rms": audio_rms,
            "peak": audio_peak,
            "flux": audio_flux,
            "vad": vad_mean,
            "overlap_speech": 0.0,
        },
        "video": {
            "flow_mag": flow_mean,
            "pose_delta": 0.0,
            "face_delta": 0.0,
            "luma_mean": luma_mean,
            "scene_change": scene_mean,
        },
        "meta": {
            "dt_ms": float(step_ms),
            "fps": 0.0,
            "response_latency": 0.0,
            "dropped_frames": 0,
            "commit_ts": commit_ts,
            "window_len": int(window_len),
        },
    }
    entities = ["user", "desk"]
    tags = ["home", "quiet" if vad_mean < 0.2 else "call"]
    return MiniWorldStep(
        name="replay_step",
        narrative="replay from live observations (commit-aligned)",
        salient_entities=entities,
        context_tags=tags,
        hazard_score=hazard_score,
        hazard_sources=hazard_sources,
        chaos=0.0,
        risk=0.0,
        tom_cost=0.0,
        uncertainty=None,
        valence=0.0,
        arousal=_clamp01(0.6 * audio_flux + 0.4 * flow_mean),
        stress=_clamp01(0.7 * hazard_score + 0.3 * audio_flux),
        love=0.0,
        mask=0.0,
        breath_ratio=_clamp01(0.5 + (vad_mean - 0.5) * 0.2),
        heart_rate=_clamp01(0.5 + (hazard_score - 0.5) * 0.3),
        action="OBSERVE",
        talk_mode="watch",
        flags=[],
        timestamp=observations["timestamp"],
        observations=observations,
    )


def _windows_from_commit(
    observations: List[Dict[str, Any]],
    commits: List[Tuple[datetime, str, int]],
) -> List[Tuple[str, int, List[Dict[str, Any]]]]:
    obs_sorted: List[Tuple[datetime, Dict[str, Any]]] = []
    for obs in observations:
        ts = _parse_ts(obs.get("timestamp"))
        if ts is None:
            continue
        obs_sorted.append((ts, obs))
    obs_sorted.sort(key=lambda pair: pair[0])
    windows: List[Tuple[str, int, List[Dict[str, Any]]]] = []
    if not obs_sorted:
        return windows
    idx = 0
    n = len(obs_sorted)
    for commit_dt, commit_ts_str, win_len in commits:
        while idx < n and obs_sorted[idx][0] <= commit_dt:
            idx += 1
        if idx <= 0:
            windows.append((commit_ts_str, win_len, []))
            continue
        start = max(0, idx - win_len)
        window_obs = [pair[1] for pair in obs_sorted[start:idx]]
        windows.append((commit_ts_str, win_len, window_obs))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", default="logs/live_observations.jsonl")
    parser.add_argument("--live-telemetry", default="logs/live_telemetry.jsonl")
    parser.add_argument("--step-ms", type=int, default=750)
    parser.add_argument("--out-telemetry", default="logs/replay_telemetry.jsonl")
    parser.add_argument("--out-diary", default="logs/replay_diary.jsonl")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    obs_path = Path(args.observations)
    live_tel_path = Path(args.live_telemetry)

    observations = _load_jsonl(obs_path)
    live_tel = _load_jsonl(live_tel_path)
    if not observations:
        print("[ERR] observations empty; generate logs/live_observations.jsonl first")
        return
    if not live_tel:
        print("[ERR] live telemetry empty; run live bridge first")
        return

    commits: List[Tuple[datetime, str, int]] = []
    for record in live_tel:
        commit_ts_str, window_len = _extract_commit_meta(record)
        if not commit_ts_str or not window_len:
            continue
        commit_dt = _parse_ts(commit_ts_str)
        if commit_dt is None:
            continue
        commits.append((commit_dt, commit_ts_str, window_len))

    if not commits:
        print("[ERR] no commit_ts/window_len in live telemetry")
        return

    commits.sort(key=lambda entry: entry[0])
    if args.max_steps > 0:
        commits = commits[: args.max_steps]

    win_specs = _windows_from_commit(observations, commits)
    steps: List[MiniWorldStep] = []
    empty_windows = 0
    for commit_ts_str, win_len, window in win_specs:
        if not window:
            empty_windows += 1
            continue
        steps.append(_build_step(window, args.step_ms, commit_ts_str, win_len))

    print(f"[INFO] commits={len(commits)} windows={len(win_specs)} steps={len(steps)} empty_windows={empty_windows}")
    if not steps:
        print("[ERR] no steps constructed; check timestamps/window sizes")
        return

    simulator = MiniWorldSimulator(
        diary_path=args.out_diary,
        telemetry_path=args.out_telemetry,
    )

    scenario = MiniWorldScenario(
        name="replay_commit_aligned",
        description="Replay from live observations using commit boundaries",
        steps=steps,
    )

    results, stats = simulator.run_scenario(scenario)
    print("[OK] replay completed:", stats)
    for r in results[:5]:
        outcome = r.replay_outcome
        meta = (r.step.observations or {}).get("meta", {}) if r.step.observations else {}
        print(
            f"idx={r.step_index} decision={outcome.decision} veto={outcome.veto_score:.3f} "
            f"u_hat={outcome.u_hat:.3f} commit={meta.get('commit_ts')} win={meta.get('window_len')}"
        )


if __name__ == "__main__":
    main()
