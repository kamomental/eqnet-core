# Phase 1 Preparation – Sensor ➔ 9-Axis ➔ Qualia

This checklist captures the concrete steps for wiring physiological signals into the qualia loop without breaking the existing observation pipeline.

## 1. Raw Frame Daemon
- [ ] Create `tools/mock_sensor_daemon.py` that produces `raw_frame` dicts (heart_rate, breath, pose, activity surrogates).
- [x] Allow switching between deterministic (seeded) and noisy waveforms for reproducibility.
- [ ] Emit JSONL to `logs/streaming_sensor_raw.jsonl` and optionally POST to a local Hub endpoint (future hook).
- [x] Provide CLI flags for interval ms, duration, and baseline HR.

## 2. Hub Hook (planned after daemon)
- [x] Add `RuntimeSensors.tick(raw_frame)` helper (or similar) that invokes `StreamingSensorState.from_raw` and stores the latest snapshot.
- [x] Thread the snapshot through `MomentLogEntry metrics` even when no user input arrives.
- [x] Ensure telemetry includes `activity_level`, `heart_rate_motion`, `heart_rate_emotion`, and `body_state_flag`.

## 3. Sensor ➔ 9-axis Blend
- [x] Implement `sensor_to_emotion_axes(snapshot)` returning a 9-D vector aligned with `terrain.emotion.AXES`.
- [x] Blend text-derived and sensor-derived axes via weights that depend on activity/privacy/fog.
- [x] Update `build_qualia_vec_v0` (or v1) to concatenate the blended axes before culture/text features.

## 4. Verification Hooks
- [x] CLI smoke test: run daemon + `eqnet/tests/test_streaming_sensor.py` to ensure metrics stay finite.
- [ ] Add a telemetry viewer pane (“Sensor Trace”) that plots HR baseline/delta vs. resulting axes weights.
- [x] Confirm MomentLog rows contain `heart_rate`, `activity_level`, `emotion_axes_sensor`, and `qualia_vec` simultaneously.

Status legend: `[ ]` pending, `[~]` in-progress, `[x]` done.

## Running the mock sensor daemon
```
python tools/mock_sensor_daemon.py --interval-ms 400 --duration 30 --emit-metrics
```
- Emits `raw_frame` payloads into `logs/streaming_sensor_raw.jsonl`.
- `--emit-metrics` routes each frame through `StreamingSensorState` so the JSONL includes both the raw sensor packet and the derived metrics/fused vector.
- Use `--seed` for deterministic traces when comparing telemetry runs.
- The daemon is intentionally stand-alone: later phases will add a `RuntimeSensors.tick(raw_frame)` hook so Hub/Loop instances can subscribe to the same stream without modifying existing observation loops.
- Swap the generator by passing `--source-class some.pkg:RealSensorSource --source-kwargs '{"port": "/dev/ttyUSB0"}'`. Any class implementing `next_raw_frame(t)` can replace the mock without code changes.
- Run `pytest tests/test_streaming_sensor.py tests/test_qualia_model.py` after wiring sensors to ensure the capture + qualia blend stay finite.
