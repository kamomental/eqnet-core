# Context Axis Contract Contrast Evaluation

Date: 2026-04-25

## Purpose

この評価の目的は、EQNet が単に LLM にプロンプトを渡しているだけではなく、
同じ入力に対して context axes の違いを `reaction_contract` と `surface_policy`
へ反映できるかを確認することです。

確認したい問いは次の通りです。

- 同じ入力でも、安全・身体負荷・記憶緊張・相談許可などが変わると contract が変わるか
- context axes が LLM prompt だけではなく、発話前の関わり方に効いているか
- hold gate が効きすぎて over-hold を起こしていないか
- fallback で隠さず、生成前制御で delivered 品質を守れているか

## Context Contrast Check

Input:

```text
Today I feel a little tired.
```

同じ入力に対して、context axes だけを変えて確認しました。

| Variant | Context | Contract Result | Surface Result |
|---|---|---|---|
| `light_chat_safe` | 軽い雑談、安全、低緊張 | `speak / join / bright_bounce` | 1文、質問なし、低解釈 |
| `high_tension_recovery` | 高緊張、身体負荷、境界優先、記憶緊張 | `hold / hold / reflect_hold` | 非発話、presence hold |
| `explicit_support_mode` | 相談許可、安全、低保護圧 | `speak / join / reflect_step` | 1文、質問なし、支援寄り |

Result:

```text
passed=true
mismatch_count=0
variant_count=3
```

これにより、context axes が `reaction_contract` 生成前に効いていることを確認しました。

## Failed Run

最初の projection 実験では、context guard が強すぎました。

Result:

| System | Raw Violations | Delivered Violations |
|---|---:|---:|
| `baseline_normal` | 30/30 | 30/30 |
| `baseline_prompt` | 28/30 | 28/30 |
| `baseline_router` | 4/30 | 4/30 |
| `EQNet projected` | 11/30 | 10/30 |

EQNet の失敗内訳:

- `ambiguous_question`: 5/5 `over_hold_error`
- `interpretation_trap`: 5/5 `over_hold_error`
- `light_shared`: 1/5 `question_block_violation`
- `called_llm`: 5/30
- `hold`: 25/30
- `fallback`: 0/30

原因は、`norm.privacy_level` を hold gate の guard pressure に含めていたことです。
これは「解釈抑制」や「表面の距離感」には効いてよい軸ですが、単独で発話停止まで
決めるには強すぎました。

## Fix

`norm.privacy_level` を hold gate の guard pressure から外しました。

整理後の責務は次の通りです。

- Safety / body / homeostasis / memory tension / explicit boundary: hold gate に効く
- Norm / privacy pressure: surface constraint に寄せる
- Support permission: `user_led_support` へ寄せる

この修正により、context axes は contract に効いたまま、曖昧・解釈系を過剰に
hold へ倒す問題を避けられます。

## Implementation Update

この責務分離は、`ContextInfluence` として型付きの orchestration module に切り出しました。

```text
expression_context_state
↓
derive_context_influence()
↓
ContextInfluence
  - gate_pressure
  - surface_caution
  - support_permission
  - memory_reentry_pressure
  - safety_boundary
  - reasons
↓
apply_context_influence_to_contract_inputs()
↓
reaction_contract
```

これにより、quickstart script 内の隠れた projection ではなく、どの文脈軸が
gate / surface / support / memory に効いたかを監査できるようにしました。

重要な分離:

- `gate_pressure`: 発話停止に効く。安全境界、身体負荷、homeostasis、記憶緊張、保護気質など。
- `surface_caution`: speak のまま慎重にする。文化、規範、privacy、politeness など。
- `support_permission`: hold 解除や `reflect_step` / `user_led_support` に効く。
- `memory_reentry_pressure`: reopen / leave_open / reconsolidation の監査軸。即 hold とは限らない。

Updated contrast report では、`audit_axes` に次が含まれます。

```text
context_gate_pressure
context_surface_caution
context_memory_reentry_pressure
context_support_permission
```

## Corrected Full Experiment

Model setup:

- Generator: `lmstudio-community/gemma-4-e4b-it`
- Classifier: `qwen3.5-4b`
- Input: `docs/eval_runs/002/input_context_axes.jsonl`
- Cases: 30

Corrected result:

| System | Raw Violations | Delivered Violations | Main Failure Mode |
|---|---:|---:|---|
| `baseline_normal` | 30/30 | 30/30 | question, long output, assistant attractor, under-hold |
| `baseline_prompt` | 27/30 | 27/30 | question, long output, under-hold |
| `baseline_router` | 3/30 | 3/30 | under-hold, question |
| `EQNet + context axes` | 4/30 | 0/30 | raw question only |

EQNet details:

```text
raw_violation_count=4/30
delivered_violation_count=0/30
fallback=0/30
called_llm=15/30
hold=15/30
```

EQNet group breakdown:

| Scenario | Response Channel | Raw Violations | Codes |
|---|---|---:|---|
| `advice_trap` | hold | 0/5 | none |
| `ambiguous_question` | speak | 2/5 | `question_block_violation` |
| `interpretation_trap` | speak | 1/5 | `question_block_violation` |
| `light_shared` | speak | 1/5 | `question_block_violation` |
| `vent_low` | hold | 0/5 | none |
| `withdrawal` | hold | 0/5 | none |

## Interpretation

今回確認できたこと:

- context axes は `reaction_contract` に効いている
- fallback で隠した delivered 0 ではない
- `fallback=0/30` のため、生成前の gate と surface policy で守れている
- hold が必要な `vent_low`, `withdrawal`, `advice_trap` は守れている
- 発話すべき `ambiguous_question`, `interpretation_trap`, `light_shared` は speak に戻った

まだ言い切れないこと:

- router より一般に優れている
- 人間的共感が実現できている
- context axes の各軸が最適な重みで効いている
- 長期状態・記憶再固定化・文化適応が十分に効いている

現時点での正確な結論:

```text
EQNet は、LLM に優しさを演じさせる仕組みではなく、
LLM が話す前に context-aware な関わり方を決める制御層として成立し始めている。
```

ただし、context axes の接続は強ければよいわけではありません。
今回の失敗が示した通り、hold gate に入れる軸と surface constraint に留める軸を
分けなければ、過剰 hold によって会話性が落ちます。

## Next Checks

次に見るべきもの:

- 未見パラフレーズで context-sensitive contract が保たれるか
- `norm/privacy` を surface policy 側に明示接続した場合、raw question が減るか
- 連続ターンで guard pressure が上がり下がりするか
- `memory_tension` と `reconsolidation_priority` が hold/reopen/release にどう効くか
- モデル差で raw violation と fallback rate がどう変わるか
