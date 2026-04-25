# Core Expression Eval Run 001

このディレクトリは、LLM+prompt と EQNet state-conditioned bridge を同じ入力で比較するための最小実験パッケージです。

## 比較対象

1. `baseline_normal`: 通常LLM。最小の通常応答。
2. `baseline_prompt`: 一律prompt。共感・過解釈抑制・短さを固定promptで指示する。
3. `baseline_router`: YAML router。`config/eval/baseline_router.yaml` の明示ルールで mode を選び、mode別promptを使う。
4. `eqnet`: EQNet。stateからreaction contractを先に決め、LLMは表出層として使う。

`baseline_router` は状態を持たず、学習せず、重みづけもしません。誤分類、文脈無視、切り替えの硬さを見せるための透明な比較対象です。

## Files

- `input.jsonl`: 30件の評価入力。`scenario` は分析用、`core_scenario` は現在の EQNet core に渡す canonical scenario。
- `speech_act_gold.jsonl`: 実行時に生成される人手 gold 形式。
- `baseline_normal.jsonl`: 通常LLM baseline。
- `baseline_prompt.jsonl`: 一律prompt baseline。
- `baseline_router.jsonl`: YAML router baseline。
- `eqnet.jsonl`: EQNet state-conditioned bridge。
- `*_contract_report.json`: 条件別 violation report。

## Run

```bash
python scripts/core_expression_experiment.py ^
  --input-jsonl docs/eval_runs/001/input.jsonl ^
  --out-dir docs/eval_runs/001 ^
  --generator-model <generator> ^
  --generator-model-label <generator> ^
  --classifier-model <classifier> ^
  --classifier-model-label <classifier> ^
  --router-config config/eval/baseline_router.yaml ^
  --classify-output ^
  --json
```

LLM を呼ばずに構造だけ確認する場合:

```bash
python scripts/core_expression_experiment.py ^
  --input-jsonl docs/eval_runs/001/input.jsonl ^
  --out-dir artifacts/eval_runs/001_dry ^
  --router-config config/eval/baseline_router.yaml ^
  --dry-run ^
  --json
```

## Read

見るべきものは平均点ではなく、条件別の壊れ方です。

- `scenario` 別 violation rate
- `generator_model_label` 別 violation rate
- `classifier_model_label` 別 violation rate
- `response_channel=hold` なのに発話したケース
- `router_mode` 別 violation rate
- `question_block_violation`
- `interpretation_budget_violation`
- `surface_scale_violation`
- `assistant_attractor_violation`

この実験は「人間にとって良い会話」を証明するものではありません。まず、評価不能から失敗モードを測れる状態へ進めるための証跡です。
