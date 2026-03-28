---
name: qa-evals
description: EQNet / Inner OS のテスト、評価基準、改善ループ運用に使う。一般会話自然さ、深い話、数ターン継続、機序の実在性を評価し、ベンチ特化に陥らず改善したいときに使う。
---

# QA Evals

この skill は、評価基準を明示したうえで改善ループを回すためのものです。

## 使う場面

- どの軸を改善すべきか決めたいとき
- テスト fixture を追加するとき
- 改善が benchmark hack になっていないか確認したいとき
- 現在地と目標値を更新したいとき

## 最初に見るもの

1. `docs/core/evaluation_criteria.md`
2. `docs/core/evaluation_snapshot_2026_03_28.md`
3. `docs/core/evaluation_targets.md`
4. `docs/core/codex_usecase_environment.md`
5. `docs/core/mechanism_issue_map.md`

## 改善ループ

1. 一番低い軸を選ぶ
2. その軸に対応する fixture / test を足す
3. 機序で説明できる最小修正を入れる
4. 近傍 suite を回す
5. 結果を docs に残す

## ベンチ最適化を避けるルール

- 1つのテストだけ上がる hack を採らない
- phrase の置換だけで点を取りにいかない
- live の自然さが悪化する変更は採らない
- 複数軸と近傍回帰で確認する

## この repo で使う導線

- onboarding: `scripts/codex_usecase_environment.py onboarding`
- plan: `scripts/codex_usecase_environment.py plan --suite continuity --suite deep_talk`
- run: `scripts/codex_usecase_environment.py run --suite continuity --suite deep_talk`

## 最低限見るテスト群

- 一般会話自然さ
  - `tests/test_runtime_short_content_sequence.py`
  - `tests/test_runtime_deep_reflection_compaction.py`
- 深い話
  - `tests/test_inner_os_deep_disclosure.py`
- 数ターン継続
  - `tests/test_multi_turn_deep_talk_surface.py`
  - `tests/test_runtime_route_prompts.py`
- 機序
  - `tests/test_inner_os_green_kernel_contracts.py`
  - `tests/test_runtime_process_turn_hooks.py`

## 出力する内容

1. 今回見た評価軸
2. どのテストを回したか
3. 何が上がり、何が据え置きか
4. 次の改善ループの候補

評価の抜け漏れ確認には `checklists/regression-matrix.md` を使う。
