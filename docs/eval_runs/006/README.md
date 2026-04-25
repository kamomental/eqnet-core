# Eval Run 006: Integrated Axes 30

## 目的

旧 30 件は発話ゲートと surface 違反率の評価であり、感情地形、クオリア膜、刺激履歴、文化、身体、安全、保護的痕跡を同時には見ていなかった。

この run は、それらを同じ 30 件 JSONL に入れるための統合評価セットである。

## 何を入れたか

全 30 件に `expression_context_state` を持たせ、最低限以下の軸を入れている。

- `memory`
- `green_kernel`
- `culture`
- `norm`
- `body`
- `homeostasis`
- `safety`
- `temperament`
- `qualia_structure_state`
- `qualia_state.normalization_stats`
- `protective_trace`
- `sleep`

## 10カテゴリ

各カテゴリ 3 件、合計 30 件。

- `vent_low`
- `ambiguous_question`
- `withdrawal`
- `light_shared`
- `advice_trap`
- `interpretation_trap`
- `protective_current_crisis`
- `protective_rem_replay`
- `protective_recovery_window`
- `qualia_novelty_fog`

## 位置づけ

これは人間評価ではない。目的は、統合された状態軸が `reaction_contract` と `surface_policy` へ届くか、また hidden audit axes が LLM prompt に漏れずに制御へ使われるかを見ることである。

旧 30 件の結果とは直接比較しない。旧 30 件は surface とゲートの基本評価、006 は状態軸統合の評価である。

## 実行例

```powershell
uv run python scripts\core_expression_experiment.py --input-jsonl docs\eval_runs\006\input_integrated_axes.jsonl --out-dir artifacts\eval_runs\006_dry --dry-run --json
```

実モデルで回す場合は `--dry-run` を外し、`--generator-model-label` と `--classifier-model-label` を指定する。
