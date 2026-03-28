# Codex Use-Case Environment

## 目的

OpenAI の Codex use case で示されている

- codebase onboarding
- iterate on difficult problems

を、この repo でそのまま回せるようにする。

ここでの狙いは、ベンチ最適化ではなく

- どこを読むべきか
- どの評価スイートを回すべきか
- どの改善ループを回すべきか

を最小の道具で明示すること。

## 構成

### 1. typed evaluation environment

- `inner_os/evaluation/codex_environment.py`

役割:

- onboarding の読み順
- architecture summary の雛形出力
- 評価スイート定義
- suite ごとの pytest command 生成
- onboarding / summary / plan の Markdown 出力

### 2. CLI

- `scripts/codex_usecase_environment.py`

モード:

- `onboarding`
  - 読み順を出す
- `summary`
  - codebase-understanding 用の architecture summary 雛形を出す
- `plan`
  - 評価スイートと command plan を出す
- `run`
  - suite を実行し Markdown report を出す

## 使い方

### codebase onboarding

```powershell
& 'C:\Users\kouic\AppData\Local\Programs\Python\Python311\python.exe' scripts\codex_usecase_environment.py onboarding
```

### architecture summary

```powershell
& 'C:\Users\kouic\AppData\Local\Programs\Python\Python311\python.exe' scripts\codex_usecase_environment.py summary
```

必要なら skill template に書き出す:

```powershell
& 'C:\Users\kouic\AppData\Local\Programs\Python\Python311\python.exe' scripts\codex_usecase_environment.py summary --write reports\codex_eval\architecture_summary.md
```

### evaluation plan

```powershell
& 'C:\Users\kouic\AppData\Local\Programs\Python\Python311\python.exe' scripts\codex_usecase_environment.py plan --suite continuity --suite deep_talk
```

### evaluation run

```powershell
& 'C:\Users\kouic\AppData\Local\Programs\Python\Python311\python.exe' scripts\codex_usecase_environment.py run --suite continuity --suite deep_talk
```

既定では `reports/codex_eval/latest.md` に report を書く。

## スイート構成

- `general_surface`
  - 一般会話自然さの確認
- `deep_talk`
  - 深い話での内容反映と問いの圧の確認
- `continuity`
  - 数ターン継続、thread anchor、generic continuation 抑制
- `mechanisms`
  - Green / contact / residual / boundary の typed 機序

## skill との関係

- `skills/codebase-understanding/`
  - `summary` モードを入口にして architecture summary を起こす
- `skills/qa-evals/`
  - `plan` と `run` を入口にして改善ループを回す

## 改善ループ

- 最初に onboarding で repo を掴む
- 次に summary で責務と変更候補を整理する
- score key が最も低い suite を plan で選ぶ
- run で report を取り、mechanism-first の修正を入れる

## 位置づけ

これは最終的な外部ベンチではない。

- repo を素早く理解する
- 改善ループを一貫した形で回す
- 評価文書を更新する

ための内部運用環境である。
