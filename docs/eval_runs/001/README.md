# Core Expression Eval Run 001

このディレクトリは、LLM+prompt と EQNet state-conditioned bridge を同じ入力で比較するための最小実験パッケージです。

## Files

- `input.jsonl`: 30件の評価入力。`scenario` は分析用、`core_scenario` は現在の EQNet core に渡す canonical scenario。
- `speech_act_gold.jsonl`: `scripts/core_expression_experiment.py` 実行時に生成される人手 gold 形式。
- `baseline_normal.jsonl`: 通常 LLM baseline。
- `baseline_prompt.jsonl`: empathy prompt baseline。
- `eqnet.jsonl`: EQNet state-conditioned bridge。
- `*_contract_report.json`: 条件別 violation report。

## Run

```bash
python scripts/core_expression_experiment.py ^
  --input-jsonl docs/eval_runs/001/input.jsonl ^
  --out-dir docs/eval_runs/001 ^
  --generator-model-label <generator> ^
  --classifier-model-label <classifier> ^
  --classify-output ^
  --json
```

LLM を呼ばずに構造だけ確認する場合:

```bash
python scripts/core_expression_experiment.py ^
  --input-jsonl docs/eval_runs/001/input.jsonl ^
  --out-dir artifacts/eval_runs/001_dry ^
  --dry-run ^
  --json
```

## Read

見るべきものは平均点ではなく、条件別の壊れ方です。

- `scenario` 別 violation rate
- `generator_model_label` 別 violation rate
- `classifier_model_label` 別 violation rate
- `response_channel=hold` なのに発話したケース
- `question_block_violation`
- `interpretation_budget_violation`
- `surface_scale_violation`
- `assistant_attractor_violation`

この実験は「人間にとって良い会話」を証明するものではありません。まず、評価不能から失敗モードを測れる状態へ進めるための最初の証跡です。
