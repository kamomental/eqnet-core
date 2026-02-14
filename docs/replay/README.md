# Ops Log RPG Replay

This folder contains a minimal replay viewer that renders trace_v1 logs
as a lightweight RPG-style animation with subtitles.

## Files

- `make_replay.py` - builds `replay.json` from trace_v1 jsonl files.
- `replay_rules.json` - derivation rules for `growth_state` / `expression_diff` / `emotion_view`.
- `replay.html` - HTML viewer that plays `replay.json` in a browser.
- `locales/ja.json` - replay UI文言・字幕テンプレート・ラベル辞書（固定文言をここに集約）。

## Generate replay.json

```
python docs/replay/make_replay.py --trace_dir trace_runs/<run_id>/YYYY-MM-DD --out docs/replay/replay.json
```

Optional: pass custom rules.

```
python docs/replay/make_replay.py --trace_dir trace_runs/<run_id>/YYYY-MM-DD --out docs/replay/replay.json --rules docs/replay/replay_rules.json
```

## Run the viewer

```
python -m http.server 8000
```

Open: `http://localhost:8000/docs/replay/replay.html`

## One-click (Windows)

```
docs\replay\run_replay.bat
```

This script finds the latest run/day, builds `replay.json`, and starts the server.

## Notes

- Do not open `replay.html` via `file://` (fetch will fail). Always use `http://localhost`.
- decision mapping: `execute -> PASS`, `cancel -> VETO`, `world_transition -> HOLD`.
- input source: trace_v1 jsonl (activation_traces.jsonl is not used).
- replay payload now includes:
  - `growth_state` (bond/stability/curiosity with value + delta)
  - `expression_diff` (face/pose/voice IDs + intensity)
  - `emotion_view` (primary/secondary/stability)
  - `reaction_line.tokens`
- text rendering:
  - `subtitle_templates` と `subtitle_decision` を用いて字幕生成
  - `axis_labels` / `face_labels` / `pose_labels` / `reaction_tokens` で表示名を差し替え可能
  - `motion` で 2D の `lipsync` / `bounce` / `hair` パラメータを調整可能
  - サイドUIは `RAW` 表示領域をバストアップ表示に置換し、`RAW JSON` は折りたたみで確認可能
  - `assets/replay/pngtuber_manifest.json` に PNG レイヤーを登録すると Motion PNGTuber 方式で描画
  - 口形は字幕文字列から推定した `a/i/u/e/o/n` で進行（完全音素一致ではないが時間同期を優先）

## One-Image PNGTuber Setup

1) Build layer pack from one source image (auto split).

```
python tools/replay/build_pngtuber_pack.py --source path/to/character.png --out-dir assets/replay/character/default --manifest assets/replay/pngtuber_manifest.json
```

`--erase-mouth true` が既定で有効です（headレイヤーの口を埋め、口差分オーバーレイに適した状態にします）。

2) (Optional) Improve quality with custom masks.

```
python tools/replay/build_pngtuber_pack.py --source path/to/character.png --out-dir assets/replay/character/default --manifest assets/replay/pngtuber_manifest.json --head-mask masks/head.png --body-mask masks/body.png --hair-front-mask masks/hair_front.png --mouth-a-mask masks/mouth_a.png --mouth-i-mask masks/mouth_i.png --mouth-u-mask masks/mouth_u.png --mouth-e-mask masks/mouth_e.png --mouth-o-mask masks/mouth_o.png --mouth-n-mask masks/mouth_n.png --auto-split false
```

3) Start viewer. If assets are missing, viewer falls back to built-in pixel portrait.
