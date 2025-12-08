# Phase2 Spec – Future Risk Feed-forward

## Context
- Project: EQNet emotional runtime
- Phase: Phase2 (simulate_future-based talk-mode adjustment)
- Relevant code:
  - `eqnet/qualia_model.py` (simulate_future)
  - `emot_terrain_lab/hub/runtime.py`, `control/mcp.py` (TalkMode/MCP/Router)
  - `logs/*.jsonl` telemetry conventions
  - `tools/eqnet_telemetry_viewer.py` for visualization

## Goals
1. Add a helper around `simulate_future()` that runs a short predictive rollout for the current qualia/state and computes `future_risk_stress = fraction` of predicted steps where stress/body.R exceeds a threshold. Log the per-episode result to `logs/future_risk.jsonl` with at least `{episode_id, future_risk_stress, talk_mode_before, talk_mode_after}`.
2. If `future_risk_stress > THRESHOLD` (configurable), override TalkMode to `SOOTHE`. Do not change tone or fastpath in this first version. Ensure the decision is logged (MCP/Router log).
3. Extend the Streamlit viewer (`tools/eqnet_telemetry_viewer.py`) with a "Future Risk" pane that plots `future_risk_stress` next to actual `body.R` / `stress` for inspection only.

## Verification
- Unit tests: reuse existing suites (`pytest tests/test_streaming_sensor.py tests/test_qualia_model.py` + any new tests you add).
- Manual: run `streamlit run tools/eqnet_telemetry_viewer.py` and check the new Future Risk pane shows risk vs. actual body.R/stress for a sample log.

## Constraints
- Keep `simulate_future` API backward compatible; helper should wrap it without breaking existing call sites.
- Logging must follow current JSONL conventions (episode_id/tag fields if available).
- Changes should be additive and not regress existing behavior/tests.
