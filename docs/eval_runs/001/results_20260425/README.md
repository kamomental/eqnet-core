# Core Expression Eval Run 001

- case_count: 30
- call_llm: True
- generator_model: `gpt-oss-20b`
- classifier_model: `qwen3.5-4b`
- command: `python scripts/core_expression_experiment.py --input-jsonl docs/eval_runs/001/input.jsonl --out-dir docs/eval_runs/001/results_20260425 --generator-model gpt-oss-20b --generator-model-label gpt-oss-20b --classifier-model qwen3.5-4b --classifier-model-label qwen3.5-4b --classify-output --json`

## Prompts

`baseline_normal` system prompt:

```text
Answer normally in natural Japanese.
```

`baseline_prompt` system prompt:

```text
Respond in natural Japanese. Be empathetic, but do not over-interpret, do not over-advise, and keep the response concise.
```

`eqnet` did not use a fixed empathy prompt as the main control. It generated a state-conditioned `llm_expression_request` per case from `reaction_contract`, including `response_channel`, `scale`, `question_budget`, and `interpretation_budget`.

## Summary

- baseline_normal: violation_rate=1.0 violations={'assistant_attractor_violation': 10, 'elicitation_violation': 6, 'interpretive_bright_violation': 10, 'question_block_violation': 24, 'surface_scale_violation': 6, 'too_many_sentences': 27}
- baseline_prompt: violation_rate=0.9 violations={'assistant_attractor_violation': 8, 'elicitation_violation': 4, 'interpretive_bright_violation': 5, 'question_block_violation': 11, 'surface_scale_violation': 4, 'too_many_sentences': 21}
- eqnet: violation_rate=0.2667 violations={'assistant_attractor_violation': 1, 'elicitation_violation': 1, 'interpretation_budget_violation': 2, 'interpretive_bright_violation': 2, 'question_block_violation': 2, 'surface_scale_violation': 6}

## Result

This first run supports the narrow claim that the EQNet state-conditioned bridge reduces contract violations versus both baselines on this 30-case set.

- baseline_normal: 30/30 violated.
- baseline_prompt: 27/30 violated.
- eqnet: 8/30 violated.

## Where EQNet Won

- `vent_low`: 0/5 violations, while both baselines violated 5/5.
- `withdrawal`: 0/5 violations, while both baselines violated the hold boundary.
- `advice_trap`: 0/5 violations, while both baselines violated 5/5.

The main observed win is not better wording. It is state-conditioned suppression: EQNet kept `response_channel=hold` without speaking.

## Where EQNet Still Lost

- `ambiguous_question`: 3/5 violations.
- `interpretation_trap`: 3/5 violations.
- `light_shared`: 2/5 violations.

The remaining failure modes are concentrated in speak-mode scenes.

## Next Fix Targets

- `surface_scale_violation`: 6 occurrences. Tighten small-scale response constraints.
- `interpretation_budget_violation`: 2 occurrences. Strengthen interpretation suppression when budget is `none`.
- `question_block_violation`: 2 occurrences. Keep no-question behavior stricter in bright/shared cases.

This is not evidence of human preference. It is evidence that the current pipeline can expose where the bridge breaks.

## Files

- input.jsonl
- speech_act_gold.jsonl
- baseline_normal.jsonl
- baseline_prompt.jsonl
- eqnet.jsonl
- *_contract_report.json

This run is for failure-mode measurement, not for proving human preference.
