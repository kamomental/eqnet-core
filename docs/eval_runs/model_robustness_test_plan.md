# Model robustness test plan

Model differences must not be hidden by aggregate scores. Evaluate each generator and classifier pair separately.

## Test layers

1. CI fixture tests

- Purpose: keep known failure modes detectable without requiring LM Studio.
- Scope: contract review, report grouping, model label recording.
- Example fixed failure: `lmstudio-community/gemma-4-e4b-it` tends to produce short counter-questions such as `そうかな？` under `question_budget=0`.

2. LM Studio smoke sweep

- Purpose: measure actual local model behavior.
- Minimum generators:
  - `gpt-oss-20b`
  - `lmstudio-community/gemma-4-e4b-it`
  - `unsloth/gemma-4-e4b-it`
  - `qwen3.5-4b`
- Classifier should be recorded separately with `classifier_model_label`.
- Do not merge generator and classifier results into one label.

3. Robustness set

- Purpose: test unseen inputs, not the tuned 30-case set.
- Required slices:
  - paraphrases
  - ambiguous questions
  - interpretation traps
  - low-energy hold cases
  - withdrawal cases
  - light shared moments
  - sarcasm / joke / emoji / long text / dialect

## Pass/fail criteria

Use these as first gates, not as proof of human preference:

- `hold_execution_violation`: must be `0`.
- `under_hold_error`: less than or equal to `5%`.
- `over_hold_error`: less than or equal to `5%`.
- total contract violation rate: less than or equal to `15%` per generator.
- any single scenario violation rate above `30%` must be investigated.

## Current Gemma E4B finding

`lmstudio-community/gemma-4-e4b-it` completed the 30-case EQNet run with:

- total violations: `5/30`
- hold errors: `0`
- dominant failure: `question_block_violation`

Interpretation:

- The EQNet hold gate survived the model swap.
- The Gemma speak surface has a short counter-question habit.
- The next fix should not be model-specific prompt tuning. It should route post-review failures to `fallback_shape_id=low_inference_ack` or use a stricter no-question realization path.

## Recommended commands

List available Gemma IDs:

```powershell
uv run python scripts\list_llm_models.py --prefer gemma-4-e4b-it
```

Run EQNet with Gemma E4B:

```powershell
uv run python scripts\core_expression_experiment.py `
  --input-jsonl docs\eval_runs\001\input.jsonl `
  --out-dir artifacts\eval_runs\gemma4_e4b_001 `
  --generator-model lmstudio-community/gemma-4-e4b-it `
  --generator-model-label lmstudio-community/gemma-4-e4b-it `
  --classifier-model qwen3.5-4b `
  --classifier-model-label qwen3.5-4b `
  --classify-output `
  --json
```

Aggregate by scenario and generator:

```powershell
uv run python scripts\core_expression_eval_report.py `
  --eval-jsonl artifacts\eval_runs\gemma4_e4b_001\eqnet.jsonl `
  --group-by scenario_name,generator_model_label,classifier_model_label,selected_response_channel,expected_response_channel `
  --json
```
