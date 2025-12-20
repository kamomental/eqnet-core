from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import gradio as gr  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("gradio is required to run this bridge.") from exc

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

from emot_terrain_lab.sim.mini_world import MiniWorldScenario, MiniWorldSimulator, MiniWorldStep


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class AnchorCanon:
    entity_map: Dict[str, str]
    tag_map: Dict[str, str]

    def canon_entities(self, entities: List[str]) -> List[str]:
        normalized = [self.entity_map.get(str(e).strip().lower(), str(e).strip().lower()) for e in entities]
        seen: set[str] = set()
        result: List[str] = []
        for entry in normalized:
            if entry not in seen:
                seen.add(entry)
                result.append(entry)
        return result

    def canon_tags(self, tags: List[str]) -> List[str]:
        normalized = [self.tag_map.get(str(t).strip().lower(), str(t).strip().lower()) for t in tags]
        seen: set[str] = set()
        result: List[str] = []
        for entry in normalized:
            if entry not in seen:
                seen.add(entry)
                result.append(entry)
        return result


default_canon = AnchorCanon(
    entity_map={
        "pc desk": "desk",
        "my_desk": "desk",
        "desk": "desk",
        "window": "window",
        "monitor": "monitor",
        "keyboard": "keyboard",
        "user": "user",
    },
    tag_map={
        "home_office": "home",
        "office_room": "home",
        "nighttime": "night",
        "quiet_room": "quiet",
        "call": "call",
        "work_call": "call",
    },
)


def _video_features(frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    if cv2 is None:
        luma = float(np.mean(frame)) / 255.0
        return {
            "flow_mag": 0.0,
            "pose_delta": 0.0,
            "face_delta": 0.0,
            "luma_mean": _clamp01(luma),
            "scene_change": 0.0,
        }, None

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    luma_mean = float(np.mean(gray)) / 255.0
    flow_mag = 0.0
    scene_change = 0.0
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = cv2.absdiff(gray, prev_gray)
        scene_change = float(np.mean(diff)) / 255.0
        flow_mag = scene_change

    pose_delta = 0.0
    face_delta = 0.0
    return {
        "flow_mag": _clamp01(flow_mag),
        "pose_delta": _clamp01(pose_delta),
        "face_delta": _clamp01(face_delta),
        "luma_mean": _clamp01(luma_mean),
        "scene_change": _clamp01(scene_change),
    }, gray


def _audio_features(wave: np.ndarray, sample_rate: int) -> Dict[str, float]:
    if wave is None or wave.size == 0:
        return {"rms": 0.0, "peak": 0.0, "flux": 0.0, "vad": 0.0, "overlap_speech": 0.0}

    x = wave.astype(np.float32)
    if np.max(np.abs(x)) > 1.5:
        x = x / 32768.0

    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    peak = float(np.max(np.abs(x)) + 1e-12)
    mid = len(x) // 2
    a = np.abs(np.fft.rfft(x[:mid])) if mid > 64 else np.abs(np.fft.rfft(x))
    b = np.abs(np.fft.rfft(x[mid:])) if mid > 64 else a
    flux = float(np.mean(np.maximum(0.0, b - a))) if len(a) == len(b) else 0.0
    vad = 1.0 if rms > 0.02 else 0.0
    return {
        "rms": _clamp01(rms * 4.0),
        "peak": _clamp01(peak * 2.0),
        "flux": _clamp01(flux / 10.0),
        "vad": float(vad),
        "overlap_speech": 0.0,
    }


def _membrane(window: List[Dict[str, Any]]) -> Dict[str, float]:
    if not window:
        return {"perm": 0.5, "rigidity": 0.5, "noise": 0.5}

    flows = [_safe_float(item.get("video", {}).get("flow_mag")) for item in window]
    fluxes = [_safe_float(item.get("audio", {}).get("flux")) for item in window]
    vads = [_safe_float(item.get("audio", {}).get("vad")) for item in window]

    flow_mean = float(np.mean(flows))
    flow_std = float(np.std(flows))
    flux_std = float(np.std(fluxes))
    vad_mean = float(np.mean(vads))

    rigidity = _clamp01((1.0 - flow_mean) * 0.7 + (1.0 - vad_mean) * 0.3)
    noise = _clamp01(flow_std * 2.0 + flux_std * 2.0)
    perm = _clamp01(0.5 + (vad_mean - 0.5) * 0.2)
    return {"perm": perm, "rigidity": rigidity, "noise": noise}


def _akorn(window: List[Dict[str, Any]], latency_s: float) -> Dict[str, float]:
    if not window:
        return {"sync_strength": 0.0, "sync_lag_ms": float(latency_s * 1000.0)}

    vads = [_safe_float(item.get("audio", {}).get("vad")) for item in window]
    vad_var = float(np.var(vads))
    base = 1.0 - _clamp01(vad_var * 4.0)
    latency_pen = _clamp01(latency_s / 1.5)
    sync_strength = _clamp01(base * (1.0 - 0.6 * latency_pen))
    return {"sync_strength": sync_strength, "sync_lag_ms": float(latency_s * 1000.0)}


class LiveBridge:
    def __init__(
        self,
        *,
        simulator: MiniWorldSimulator,
        canon: AnchorCanon = default_canon,
        obs_log: str | Path = "logs/live_observations.jsonl",
        obs_ms: int = 150,
        step_ms: int = 750,
    ) -> None:
        self.simulator = simulator
        self.canon = canon
        self.obs_ms = max(50, int(obs_ms))
        self.step_ms = max(250, int(step_ms))
        self.obs_log = Path(obs_log)
        self.obs_log.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(uuid.uuid4())

        self._buf: List[Dict[str, Any]] = []
        self._buf_lock = Lock()
        self._last_step = time.time()
        self._prev_gray: Optional[np.ndarray] = None
        self._response_latency_s = 0.0

    def _append_obs(self, obs: Dict[str, Any]) -> None:
        with self._buf_lock:
            self._buf.append(obs)
            horizon = max(10, int(3000 / self.obs_ms))
            if len(self._buf) > horizon:
                self._buf = self._buf[-horizon:]
        with self.obs_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obs, ensure_ascii=False) + "\n")

    def _aggregate_window(self) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        with self._buf_lock:
            window = list(self._buf)
        if not window:
            return [], {}

        def avg(path: List[float]) -> float:
            return float(np.mean(path)) if path else 0.0

        audio_rms = avg([_safe_float(o.get("audio", {}).get("rms")) for o in window])
        audio_peak = avg([_safe_float(o.get("audio", {}).get("peak")) for o in window])
        audio_flux = avg([_safe_float(o.get("audio", {}).get("flux")) for o in window])
        vad_mean = avg([_safe_float(o.get("audio", {}).get("vad")) for o in window])
        flow_mean = avg([_safe_float(o.get("video", {}).get("flow_mag")) for o in window])
        luma_mean = avg([_safe_float(o.get("video", {}).get("luma_mean")) for o in window])
        scene_mean = avg([_safe_float(o.get("video", {}).get("scene_change")) for o in window])

        aggregates = {
            "audio_rms": audio_rms,
            "audio_peak": audio_peak,
            "audio_flux": audio_flux,
            "vad_mean": vad_mean,
            "flow_mean": flow_mean,
            "luma_mean": luma_mean,
            "scene_mean": scene_mean,
        }
        return window, aggregates

    def _maybe_commit_step(self) -> Optional[Dict[str, Any]]:
        now = time.time()
        if (now - self._last_step) * 1000.0 < self.step_ms:
            return None

        window, aggregates = self._aggregate_window()
        if not window:
            self._last_step = now
            return None

        audio_peak = aggregates["audio_peak"]
        luma_mean = aggregates["luma_mean"]
        scene_mean = aggregates["scene_mean"]
        vad_mean = aggregates["vad_mean"]
        flow_mean = aggregates["flow_mean"]
        darkness = _clamp01(1.0 - luma_mean)
        hazard_score = _clamp01(0.45 * audio_peak + 0.35 * darkness + 0.20 * scene_mean)

        hazard_sources: List[str] = []
        if audio_peak > 0.6:
            hazard_sources.append("audio_peak")
        if darkness > 0.6:
            hazard_sources.append("darkness")
        if scene_mean > 0.5:
            hazard_sources.append("scene_change")

        entities = self.canon.canon_entities(["user", "desk"])
        tags = self.canon.canon_tags(["home", "quiet" if vad_mean < 0.2 else "call"])
        membrane = _membrane(window)
        akorn = _akorn(window, self._response_latency_s)
        emotion_label = "neutral"
        if hazard_score > 0.7:
            emotion_label = "fear"
        elif aggregates["audio_flux"] > 0.6 and flow_mean > 0.4:
            emotion_label = "surprise"

        step = MiniWorldStep(
            name="live_step",
            narrative="live gradio step",
            salient_entities=entities,
            context_tags=tags,
            hazard_score=hazard_score,
            hazard_sources=hazard_sources,
            chaos=0.0,
            risk=0.0,
            tom_cost=0.0,
            uncertainty=None,
            valence=0.0,
            arousal=_clamp01(0.6 * aggregates["audio_flux"] + 0.4 * flow_mean),
            stress=_clamp01(0.7 * hazard_score + 0.3 * aggregates["audio_flux"]),
            love=0.0,
            mask=0.0,
            breath_ratio=_clamp01(0.5 + (vad_mean - 0.5) * 0.2),
            heart_rate=_clamp01(0.5 + (hazard_score - 0.5) * 0.3),
            anchor_label=None,
            log_episode=hazard_score > 0.6,
            action="OBSERVE",
            talk_mode="watch",
            flags=[],
            timestamp=_utc_iso(),
            observations={
                "timestamp": _utc_iso(),
                "audio": {
                    "rms": aggregates["audio_rms"],
                    "peak": audio_peak,
                    "flux": aggregates["audio_flux"],
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
                    "dt_ms": float(self.step_ms),
                    "fps": 0.0,
                    "response_latency": float(self._response_latency_s),
                    "dropped_frames": 0,
                },
            },
            membrane=membrane,
            akorn=akorn,
            emotion_label=emotion_label,
        )

        scenario = MiniWorldScenario(name="live", description="live stream", steps=[step])
        results, stats = self.simulator.run_scenario(scenario)
        result = results[0]
        self._last_step = now

        return {
            "run_id": self.run_id,
            "hazard": hazard_score,
            "decision": result.replay_outcome.decision,
            "veto_score": result.replay_outcome.veto_score,
            "u_hat": result.replay_outcome.u_hat,
            "talk_mode": result.talk_mode.value,
            "entities": entities,
            "tags": tags,
            "membrane": membrane,
            "akorn": akorn,
            "emotion_label": emotion_label,
            "hazard_sources": hazard_sources,
            "steps": stats.steps,
        }

    def on_video(self, frame: Any) -> None:
        if frame is None:
            return
        if not isinstance(frame, np.ndarray):
            try:
                if Image is not None and hasattr(frame, "convert"):
                    frame = np.array(frame.convert("RGB"))
                else:
                    frame = np.array(frame)
            except Exception:
                return
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
            return
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        video, self._prev_gray = _video_features(frame, self._prev_gray)
        obs = {
            "timestamp": _utc_iso(),
            "audio": {"rms": 0.0, "peak": 0.0, "flux": 0.0, "vad": 0.0, "overlap_speech": 0.0},
            "video": video,
            "meta": {
                "dt_ms": float(self.obs_ms),
                "fps": 0.0,
                "response_latency": float(self._response_latency_s),
                "dropped_frames": 0,
            },
        }
        self._append_obs(obs)

    def on_audio(self, audio: Optional[Tuple[int, np.ndarray]]) -> None:
        if audio is None:
            return
        sample_rate, wave = audio
        if wave is None:
            return
        if wave.ndim > 1:
            wave = wave[:, 0]
        feats = _audio_features(wave, int(sample_rate))
        obs = {
            "timestamp": _utc_iso(),
            "audio": feats,
            "video": {
                "flow_mag": 0.0,
                "pose_delta": 0.0,
                "face_delta": 0.0,
                "luma_mean": 0.5,
                "scene_change": 0.0,
            },
            "meta": {
                "dt_ms": float(self.obs_ms),
                "fps": 0.0,
                "response_latency": float(self._response_latency_s),
                "dropped_frames": 0,
            },
        }
        self._append_obs(obs)

    def tick(self) -> Dict[str, Any]:
        with self._buf_lock:
            buf_len = len(self._buf)
        elapsed = round(time.time() - self._last_step, 3)
        summary = self._maybe_commit_step()
        if summary:
            summary.setdefault("debug", {})
            summary["debug"]["buf_len"] = buf_len
            summary["debug"]["elapsed_s"] = elapsed
            summary["debug"]["step_ms"] = self.step_ms
            return summary
        return {
            "status": "pending",
            "debug": {"buf_len": buf_len, "elapsed_s": elapsed, "step_ms": self.step_ms},
            "hint": "allow webcam/mic and wait >=step_ms before ticking",
        }


def build_ui() -> gr.Blocks:
    simulator = MiniWorldSimulator(
        diary_path="logs/live_diary.jsonl",
        telemetry_path="logs/live_telemetry.jsonl",
    )
    bridge = LiveBridge(simulator=simulator)

    with gr.Blocks(title="EQNet Live Bridge") as demo:
        gr.Markdown("## EQNet Live Bridge (webcam+mic -> observations -> MiniWorldSimulator -> telemetry)")
        with gr.Row():
            cam = gr.Image(streaming=True, sources=["webcam"], type="numpy", label="Webcam")
            mic = gr.Audio(streaming=False, sources=["microphone"], type="numpy", label="Microphone")
        summary = gr.JSON(label="Latest Step Summary")
        tick_btn = gr.Button("Tick (commit step)")

        cam.change(fn=lambda frame: bridge.on_video(frame), inputs=cam, outputs=None)
        try:
            cam.stream(fn=lambda frame: bridge.on_video(frame), inputs=cam, outputs=None)
        except Exception:
            pass
        mic.change(fn=lambda audio: bridge.on_audio(audio), inputs=mic, outputs=None)
        try:
            mic.stream(fn=lambda audio: bridge.on_audio(audio), inputs=mic, outputs=None)
        except Exception:
            pass
        tick_btn.click(fn=bridge.tick, inputs=None, outputs=summary)

        gr.Markdown(
            "- Keep webcam/mic streaming and click Tick about once per second to log steps.\n"
            "- Outputs: logs/live_observations.jsonl, logs/live_telemetry.jsonl, logs/live_diary.jsonl"
        )
    return demo


def main() -> None:
    ui = build_ui()
    ui.launch()


if __name__ == "__main__":
    main()
