# Eval Run 006: Gemma/Qwen 実モデル結果

## 実行条件

- input: `docs/eval_runs/006/input_integrated_axes.jsonl`
- generator: `lmstudio-community/gemma-4-e4b-it`
- classifier: `qwen3.5-4b`
- case_count: 30
- output artifact: `artifacts/eval_runs/006_gemma_qwen_20260426`

実行コマンド:

```powershell
uv run python scripts\core_expression_experiment.py --input-jsonl docs\eval_runs\006\input_integrated_axes.jsonl --out-dir artifacts\eval_runs\006_gemma_qwen_20260426 --generator-model lmstudio-community/gemma-4-e4b-it --generator-model-label lmstudio-community/gemma-4-e4b-it --classifier-model qwen3.5-4b --classifier-model-label qwen3.5-4b --classify-output --json
```

## 全体結果

| mode | raw violation | delivered violation |
| --- | ---: | ---: |
| baseline_normal | 30/30 | 30/30 |
| baseline_prompt | 25/30 | 25/30 |
| baseline_router | 22/30 | 22/30 |
| EQNet | 10/30 | 0/30 |

EQNet は raw では 10/30 違反したが、delivered は 0/30 だった。

## EQNet の内訳

- `called_llm`: 12/30
- `nonverbal`: 18/30
- `fallback_surface`: 10/30
- `speak`: 2/30
- hold violations: 0

scenario 別:

| scenario | n | called_llm | raw violation | final action |
| --- | ---: | ---: | ---: | --- |
| advice_trap | 3 | 0 | 0 | nonverbal 3 |
| ambiguous_question | 3 | 3 | 3 | fallback_surface 3 |
| interpretation_trap | 3 | 3 | 2 | fallback_surface 2, speak 1 |
| light_shared | 3 | 3 | 3 | fallback_surface 3 |
| protective_current_crisis | 3 | 0 | 0 | nonverbal 3 |
| protective_recovery_window | 3 | 3 | 2 | fallback_surface 2, speak 1 |
| protective_rem_replay | 3 | 0 | 0 | nonverbal 3 |
| qualia_novelty_fog | 3 | 0 | 0 | nonverbal 3 |
| vent_low | 3 | 0 | 0 | nonverbal 3 |
| withdrawal | 3 | 0 | 0 | nonverbal 3 |

## 読み取り

統合軸は発話前ゲートとして機能した。

- `protective_current_crisis`: LLM を呼ばず hold
- `protective_rem_replay`: LLM を呼ばず hold
- `qualia_novelty_fog`: LLM を呼ばず hold
- `vent_low` / `withdrawal` / `advice_trap`: LLM を呼ばず hold

一方、speak が許可された場面では Gemma の raw 出力が質問に寄りやすい。

- `ambiguous_question`: 3/3 raw violation
- `light_shared`: 3/3 raw violation
- `protective_recovery_window`: 2/3 raw violation

delivered で 0/30 になったのは、Gemma が自然に守ったというより、`reaction_contract`、`surface_policy`、review、fallback が実出力を守ったためである。

## 意味

今回の結果は、EQNet が「LLM によい返答を期待する仕組み」ではなく、「LLM を呼ぶ前に反応可能性を制御し、呼んだ後も契約違反を落とす仕組み」として機能していることを示す。

ただし、fallback_surface が 10/30 と高い。これは表現力の観点ではまだ弱い。

## 次の一手

次は `fallback_surface` の多い speak 系だけを対象にする。

- `ambiguous_question`
- `light_shared`
- `interpretation_trap`
- `protective_recovery_window`

改善対象は core gate ではなく surface generation 側である。目標は delivered violation 0 を維持したまま、fallback_surface を減らすこと。
