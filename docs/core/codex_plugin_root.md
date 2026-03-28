# Codex Plugin Root

repo 最上位に、再利用可能な `plugin + skills` の骨組みを置くための整理です。

## 目的

- repo 固有の常識は `AGENTS.md` に残す
- 繰り返し使う作業手順は `skills/` に切り出す
- 他 repo にも移植しやすいように `.codex-plugin/plugin.json` で束ねる

## 現在の構成

- `AGENTS.md`
  - repo 全体の作業規約
- `.codex-plugin/plugin.json`
  - plugin manifest
- `skills/codebase-understanding/`
  - 大規模コード理解
- `skills/frontend/`
  - 表出 / UI / dashboard
- `skills/api/`
  - typed contract / hook / runtime 変更
- `skills/qa-evals/`
  - 評価基準と改善ループ

## bundled resources

- `skills/codebase-understanding/templates/architecture-summary.md`
  - コード理解の出力雛形
- `skills/frontend/checklists/ui-review.md`
  - 自然な表出と UI 確認用 checklist
- `skills/api/checklists/integration-upgrade.md`
  - contract / hook 変更用 checklist
- `skills/qa-evals/checklists/regression-matrix.md`
  - 改善ループの回帰観点一覧

## 役割分担

### AGENTS.md

- repo 固有ルール
- アーキテクチャ制約
- 日本語や apply_patch のような実務ルール

### skills/

- 繰り返し使うワークフロー
- いつ使うか
- 最初に何を読むか
- 何を出力するか

### plugin manifest

- skill 群の配布単位
- 他 repo へ持ち出すときの入口

## この repo に合わせた使い方

- コード理解から始めるとき
  - `codebase-understanding`
- 表出や UI を詰めるとき
  - `frontend`
- hook や contract を触るとき
  - `api`
- 評価ループを回すとき
  - `qa-evals`

## 補足

- `.mcp.json` や `.codex/config.toml` は、必要になった時点で追加する
- まずは最小の plugin root と skills を育て、再利用可能だと固まってから広げる
