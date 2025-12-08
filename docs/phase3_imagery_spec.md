# Phase3 Spec – Imagery Feed-forward (Positive Futures)

## Context
- Project: EQNet emotional runtime
- Phase: Phase3 (simulate_future imagery mode)
- Relevant code:
  - `eqnet/qualia_model.py` (`simulate_future` with `ReplayMode.IMAGERY`)
  - `emot_terrain_lab/hub/runtime.py`, `control/mcp.py` (TalkMode/Tone controls)
  - `eqnet/logs/moment_log.py`, `logs/*.jsonl` for Self-Report/Narrative/Monument integration
  - `tools/eqnet_telemetry_viewer.py` for visualization

## Goals
1. Add a helper that runs `simulate_future(..., mode=ReplayMode.IMAGERY)` with the current intention vector and returns `future_hopefulness` (or similar) indicating how positive the imagined trajectory is (e.g., average valence/love increase). Log it per episode (e.g., `logs/future_imagery.jsonl`).
2. If `future_hopefulness > THRESHOLD`, gently upshift TalkMode (e.g., force `AFFIRM`/`TALK` or lower tone_soften slightly). Keep the first version minimal—only TalkMode adjustment and a log entry; no fastpath/tone changes yet.
3. Thread the imagery metadata into Self-Report/Narrative/Monument logic so that episodes tagged as imagery-positive can boost salience/Monument promotion (fast+slow cues). Ensure these fields are optional/backward-compatible.
4. In the Streamlit viewer, add a “Future Imagery” pane that plots `future_hopefulness` alongside the resulting affect metrics (e.g., love, valence) so we can see whether positive imagery correlates with warmer behavior.

## Verification
- Tests: run existing suites (`pytest tests/test_streaming_sensor.py tests/test_qualia_model.py`) plus any new ones you add for imagery helpers/logging.
- Manual: run the viewer and confirm the new Future Imagery pane displays the logged values, and TalkMode shifts only when hopefulness exceeds the threshold.

## Constraints
- Keep existing APIs backward compatible (e.g., MomentLog entries without imagery fields should still parse).
- Logging must follow current JSONL conventions (episode_id/tag when available).
- Imagery adjustment should be gentle: no changes to fastpath/tone in v1.
