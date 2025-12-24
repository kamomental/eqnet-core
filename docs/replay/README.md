# Ops Log RPG Replay

This folder contains a minimal replay viewer that renders trace_v1 logs
as a lightweight RPG-style animation with subtitles.

## Files

- `make_replay.py` - builds `replay.json` from trace_v1 jsonl files.
- `replay.html` - HTML viewer that plays `replay.json` in a browser.

## Generate replay.json

```
python docs/replay/make_replay.py --trace_dir trace_runs/<run_id>/YYYY-MM-DD --out docs/replay/replay.json
```

## Run the viewer

```
cd docs/replay
python -m http.server 8000
```

Open: `http://localhost:8000/replay.html`

## One-click (Windows)

```
docs\replay\run_replay.bat
```

This script finds the latest run/day, builds `replay.json`, and starts the server.

## Notes

- Do not open `replay.html` via `file://` (fetch will fail). Always use `http://localhost`.
- decision mapping: `execute -> PASS`, `cancel -> VETO`, `world_transition -> HOLD`.
- input source: trace_v1 jsonl (activation_traces.jsonl is not used).
