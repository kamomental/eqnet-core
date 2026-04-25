# LM Studio model targets

This document records model IDs that can be used in EQNet expression evaluation.

Detected on 2026-04-25 from local LM Studio:

- `lmstudio-community/gemma-4-e4b-it`
- `unsloth/gemma-4-e4b-it`
- `gemma-4-e2b-it`
- `qwen3.5-4b`
- `qwen3.6-35b-a3b`

Recommended first Gemma generator:

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

Use `scripts/list_llm_models.py --prefer gemma-4-e4b-it` before a run to verify the exact model ID. Evaluation records must keep `generator_model_label` and `classifier_model_label` so model effects can be separated from EQNet state effects.

## 2026-04-25 Gemma E4B smoke result

Run:

- generator: `lmstudio-community/gemma-4-e4b-it`
- classifier: `qwen3.5-4b`
- input: `docs/eval_runs/001/input.jsonl`
- output: `artifacts/eval_runs/eqnet_gemma4_e4b_20260425`

Summary:

- record_count: `30`
- violation_count: `5`
- violation_rate: `0.1667`
- hold execution / selection errors: `0`
- violation family: `question_block_violation`

Observed failure mode:

- Gemma often answers constrained ambiguous cases with short counter-questions such as `そうかな？` or `そうなの？`.
- This is not a hold-gate failure. It is a speak-surface failure: the model uses a question-shaped minimal reaction even when `question_budget=0`.

Current interpretation:

- EQNet state gate remains intact with this model.
- Gemma needs either stronger no-question surface realization or a post-review fallback path for `fallback_shape_id=low_inference_ack`.
- The result does not prove robustness. It only confirms that this model can be inserted into the existing evaluation path and exposes a clear model-specific failure mode.

## 2026-04-25 fallback rerun

Run:

- generator: `lmstudio-community/gemma-4-e4b-it`
- classifier: `qwen3.5-4b`
- input: `docs/eval_runs/001/input.jsonl`
- output: `artifacts/eval_runs/eqnet_gemma4_e4b_fallback_20260425`

Summary:

- raw LLM violation_count: `4/30`
- raw LLM violation_rate: `0.1333`
- delivered violation_count after `surface_policy_fallback`: `0/30`
- delivered violation_rate: `0.0`
- hold execution / selection errors: `0`

Interpretation:

- The model still has a short counter-question tendency.
- The final surface can now be protected by `fallback_shape_id=low_inference_ack`.
- Raw violations remain visible in the audit report; they are not converted into success.
