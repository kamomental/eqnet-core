# EQNet Observation Loop Notes

## 1. What's New

- **SelfReporter** (`devlife/mind/self_model.py`) logs per-episode reflections with `mood`, `social_tone`, `confidence`, `stress_level`, and metrics (`love`, `intent_trust`, `fastpath_override`, `body.R`, `tension`).
- **Narrative rollup** (`tools/narrative_rollup.py`) batches self-reports into coarse stories that summarize tag composition, average affect, climate shifts, and a short description.
- **Telemetry Viewer updates** now load `self_report.jsonl` and `narrative_log.jsonl` alongside KPI/MCP logs so a single episode page shows:
  - body/affect/value/fastpath plots
  - the latest self-report card
  - any narrative chunk covering the episode
- **Green impulse tooling** (`emot_terrain_lab/core/green_kernel.py` + `tools/analyze_green_impulse.py`) records Φ/Ψ impulse responses, estimates decay τ and dominant frequencies, and can emit `logs/green_modes.jsonl` for KPI tracking.
- **Monument builder** (`tools/build_monuments.py`) fuses Self-Report (fast signals) and Narrative (slow meaning) into `logs/monuments.jsonl`, so “important memories” are based on EQNet’s own feelings + later interpretation.

## 2. Daily Loop Recipe

1. **Tag experiments** – run `scripts/run_quick_loop.py` with different `--tag` and affect thresholds to gather KPI/self-report data.
2. **Narrative rollup** – `python tools/narrative_rollup.py --window <N>` after a batch to produce `logs/narrative_log.jsonl`.
3. **Monument detection** – `python tools/build_monuments.py --out logs/monuments.jsonl` to capture moments where fast feelings + slow meaning align.
4. **Viewer pass** – `streamlit run tools/eqnet_telemetry_viewer.py` and filter by tag/episode to inspect the full “heart monitor + self-report + narrative” panel.
5. **Green-mode check (optional)** – `python emot_terrain_lab/core/green_kernel.py ...` followed by `python tools/analyze_green_impulse.py ... --emit logs/green_modes.jsonl` when the Lenia/GRN body is re-tuned.

## 3. Fast vs Slow Memory Signals

- **Fast triggers** (Self-Report): high override bursts, “warm despite low love,” guarded mood paired with high love, supportive tone under tension, or quiet overrides when everything else looks calm.
- **Slow triggers** (Narrative): theme transitions, emotional trends like `warm`/`transition`/`breakthrough`, or descriptions marking a period as special/meaningful.
- Only episodes that satisfy both are promoted to Monument status, matching the “瞬間のトキメキ + 後からの意味づけ” philosophy.

## 4. Next Observational Goals

- Build a `tools/tag_trait_matrix.py` helper to print per-tag averages (`body.R`, `love`, `intent_trust`, mood modes) for quick temperament maps.
- Add tag filters + Monument overlays directly inside the Streamlit viewer.
- Gradually upgrade SelfReporter summaries (rule → LLM) so the textual reflections feel richer while still staying deterministic when needed.
- Keep emitting `logs/green_modes.jsonl` whenever the Lenia/NCA body is tweaked so KPI dashboards can relate body τ to climate/mood stability.
