---
name: frontend
description: EQNet / Inner OS の表出、UI、会話 surface、dashboard、runtime compaction を改善するときに使う。自然な日本語、短い応答、視認性、運用画面の読みやすさをまとめて扱う。
---

# Frontend

この skill は、自然な表出と UI の両方を崩さずに改善するときに使います。

## 使う場面

- `runtime.py` の compaction や route surface を触るとき
- `locales/ja.json` の日本語文面を直すとき
- dashboard / viewer の表示を変えるとき
- 一般層に伝わる自然さを上げたいとき

## この repo で先に見るもの

1. `AGENTS.md`
2. `locales/ja.json`
3. `emot_terrain_lab/hub/runtime.py`
4. `emot_terrain_lab/ops/dashboard.py`
5. `docs/core/evaluation_criteria.md`
6. `docs/core/evaluation_targets.md`

## 原則

- 日本語 UI 文面は `locales/ja.json` に集約する
- 表出は説明を増やすのでなく、短く自然にする
- 保護は文の長さや圧で出し、相談窓口文体にしない
- route hack ではなく、inner_os の readout を活かす

## 変更時に必ず確認すること

1. 一文目で何を受け取ったか分かるか
2. `大丈夫です` などの反復が増えていないか
3. `もちろんです。どこから続けますか？` のような generic fallback に戻っていないか
4. short final が冷たく切れていないか
5. 近傍の runtime テストが通るか

## 最低限回すテスト

- `tests/test_runtime_short_content_sequence.py`
- `tests/test_runtime_deep_reflection_compaction.py`
- `tests/test_runtime_route_prompts.py`
- `tests/test_multi_turn_deep_talk_surface.py`

変更前後の観点整理には `checklists/ui-review.md` を使う。

## 出力する内容

変更後は次を短く報告すること。

1. 何を自然にしたか
2. どの文面や selector を変えたか
3. どのテストで確認したか
4. 残る違和感は何か
