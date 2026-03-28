---
name: codebase-understanding
description: EQNet / Inner OS リポジトリの大規模コード理解に使う。アプリ全体の入口、責務分割、データフロー、変更候補ファイル、評価導線を短時間で整理したいときに使う。
---

# Codebase Understanding

この skill は、変更前に repo 全体の責務と入口を短く把握するためのものです。

## 使う場面

- 新しい作業を始める前に、どこから読むべきか決めたいとき
- ある機能がどの層で実装されているかを特定したいとき
- 変更影響範囲を絞りたいとき
- API / frontend / evaluation のどこまで触るべきか見積もりたいとき

## 最初に読むもの

1. `AGENTS.md`
2. `README.md`
3. `docs/core/README.md`
4. `docs/core/codex_usecase_environment.md`
5. `docs/core/evaluation_criteria.md`
6. `docs/core/evaluation_targets.md`
7. `inner_os/README.md`

## 重点的に見るディレクトリ

- `inner_os/`
  - 本体 state / evaluation / expression の責務を見る
- `emot_terrain_lab/`
  - runtime / hub / dashboard など表出と運用の入口を見る
- `locales/`
  - 日本語表出の実文面を見る
- `tests/`
  - 回帰の入口と fixture を確認する
- `docs/core/`
  - 現在の評価軸と設計意図を確認する

## 出力する内容

必ず次を短くまとめること。

1. 入口ファイル
2. 主な責務の分割
3. 変更候補ファイル
4. 影響を受けるテスト
5. 先に読むべき docs

必要なら `templates/architecture-summary.md` を埋めて、同じ形式で比較できるようにする。

## この repo での読み方

- LLM は本体ではなく表出層とみなす
- `inner_os` core と runtime surface を混ぜて説明しない
- まず `contact -> event化 -> shared field/readout -> boundary -> surface` のどこを触るかを切る

## 使える補助

- onboarding は `scripts/codex_usecase_environment.py onboarding`
- suite の確認は `scripts/codex_usecase_environment.py plan --suite continuity --suite deep_talk`

## 避けること

- 全ファイルを列挙して終わること
- 責務を曖昧なまま「この辺」と言うこと
- docs と実装で責務がずれているのに未指摘で終えること
