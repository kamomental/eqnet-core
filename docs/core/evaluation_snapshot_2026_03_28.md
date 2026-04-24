# Evaluation Snapshot 2026-03-28

## 関連文書
- `docs/core/evaluation_operating_policy.md`
- `docs/core/evaluation_targets.md`

## 前提

このスナップショットは、外部公開ベンチの実行結果ではなく、
repo 内の現行テストを
`docs/core/evaluation_criteria.md`
の基準へ対応づけてまとめた内部評価である。

---

## 対象テスト

- `tests/test_runtime_route_prompts.py`
- `tests/test_runtime_short_content_sequence.py`
- `tests/test_runtime_deep_reflection_compaction.py`
- `tests/test_inner_os_deep_disclosure.py`
- `tests/test_runtime_process_turn_hooks.py`
- `tests/test_human_output_examples.py`
- `tests/test_inner_os_conversational_architecture.py`
- `tests/test_inner_os_green_kernel_contracts.py`

---

## 評価結果

### A. 一般会話自然さ

判定: `Partial`

確認できたこと:

- `habit` route は structured turn を上書きしにくくなった
- short final で thread reopening や deep reflection が残る
- generic clarification への回帰は減っている

残っていること:

- live 実走では、継続ターンで generic support に戻る余地がまだある
- 一般層向けの語感は改善中で、完全ではない

### B. 深い話の受け止め

判定: `Pass`

確認できたこと:

- deep disclosure は content reflection を先に出せる
- `green_reflection_hold` で問いなし reflection-only に切り替えられる
- reflection-only の short final でも、短い presence を残せる

### C. 数ターン継続

判定: `Partial`

確認できたこと:

- `thread anchor` と `reopen_from_anchor` 系は typed に保持される
- route で habit fallback を抑える層が入った

残っていること:

- live の 2〜5 ターンで論点核が安定するかは、まだ fixture / 実走の強化が必要

### D. キャラクタ / 存在感

判定: `Partial`

確認できたこと:

- human output examples と conversational architecture は成立している
- content / protection / reopening の組み合わせは見える

残っていること:

- 一般層が読むと、まだ少し「支える説明」が見える
- 継続した deep talk での揺れは、まだ滑らかさが足りない

### E. 機序の実在性

判定: `Pass`

確認できたこと:

- `GreenKernelComposition` は contract として定義済み
- `memory / affective / relational` は shared field に射影される
- `boundary / residual` は別 operator のまま保持されている
- readout / surface にまで実際に効いている

---

## 総合判定

`共生共鳴生命体としての内部機序`: `Pass`

`一般層に自然に伝わる表出`: `Partial`

`多ターン継続での存在感`: `Partial`

現状は、
**内部機序はかなり成立しているが、一般層にとっての自然さと multi-turn の滑らかさはまだ仕上げ段階**
と評価する。

---

## 次の重点

1. multi-turn deep talk fixture を追加する
2. generic support への回帰を live 前提でさらに削る
3. thread reopening / quiet presence / light question の揺れを自然化する
4. character / empathy / multi-turn の外部ベンチへ載せる準備を進める
