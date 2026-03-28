---
name: api
description: runtime hook、planner、integration hook、script CLI など API や契約面の変更を行うときに使う。typed interface、責務分離、回帰テスト、運用導線を揃えたいときに有効。
---

# API

この skill は、repo 内の API / contract / hook の変更を安全に進めるためのものです。

## 使う場面

- `inner_os` の typed interface を追加・変更するとき
- `integration_hooks.py` や `response_planner.py` を触るとき
- runtime と summary / dashboard の橋渡しを変えるとき
- script CLI を追加するとき

## この repo で先に見るもの

1. `AGENTS.md`
2. `inner_os/README.md`
3. `docs/core/mechanism_issue_map.md`
4. `docs/core/evaluation_criteria.md`
5. 変更先 module の contract test

## 原則

- dataclass / typed interface を優先する
- silent fallback を避ける
- state update と expression bridge を混ぜない
- 新しい state は observability と test を一緒に追加する
- raw observation を直接 LLM bridge に流さない

## 変更時の確認項目

1. interface 境界が明示されているか
2. runtime 側の key 名と summary 側の key 名がずれていないか
3. docs の責務説明が古くなっていないか
4. `py_compile` と近傍 pytest が通るか

## 最低限回すもの

- 近傍 contract test
- `tests/test_inner_os_bootstrap.py`
- `tests/test_runtime_process_turn_hooks.py`

## 出力する内容

1. 追加・変更した contract
2. どの層の責務を変えたか
3. 互換性に関する注意
4. 実行した確認

契約変更時の抜け漏れ確認には `checklists/integration-upgrade.md` を使う。
