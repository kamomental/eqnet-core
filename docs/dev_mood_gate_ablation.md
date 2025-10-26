Mood Gate Wiring & A/B (Notes)

- Hub metrics now accept optional `mood_*` fields merged from environment:
  - `EQNET_MOOD_METRICS`: JSON string, e.g. `{ "mood_v": 0.2, "mood_a": 0.3, "mood_effort": 0.1, "mood_uncertainty": 0.2 }`
  - `EQNET_MOOD_METRICS_FILE`: Path to a JSON file with the same keys
- When present, `PolicyHead` applies a small-gain, serial mood gate to controls.
- A/B scaffold:
  - `python ops/eval_mood_ablation.py --trials 30 --out reports/mood_ablation.jsonl`
  - Produces per-trial controls + metrics for m-OFF/m-ON in JSONL under `reports/`.

