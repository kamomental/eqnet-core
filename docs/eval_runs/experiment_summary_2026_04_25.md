# 2026-04-25 expression evaluation summary

## 目的

EQNet の評価対象を「LLMが一度で良い文を出したか」だけにしない。

次の4つを分けて見る。

- raw violation: 生成モデルの癖・失敗
- delivered violation: 実際にユーザーへ出す文の契約違反
- hold error: hold / speak の選択・実行ミス
- fallback rate: fallback に依存した割合

この分離により、モデル差を隠さず、同時に実出力の品質を守れるかを見る。

## 結果

| Run | Generator | Raw violation | Delivered violation | Hold error | Fallback rate | 主な失敗 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| old EQNet core | `gpt-oss-20b` | `8/30` (`26.67%`) | 未測定 | `0` | 未測定 | scale / interpretation / question |
| router v3 | `gpt-oss-20b` | `3/30` (`10.00%`) | 未測定 | `under_hold_error=2` | 未測定 | hold選択ミス |
| EQNet + SurfacePolicy guard | `gpt-oss-20b` | `2/30` (`6.67%`) | 未測定 | `0` | 未測定 | interpretation |
| EQNet + SurfacePolicy text | `gpt-oss-20b` | `1/30` (`3.33%`) | 未測定 | `0` | `0/30` | question |
| EQNet + SurfacePolicy | `lmstudio-community/gemma-4-e4b-it` | `5/30` (`16.67%`) | 未測定 | `0` | 未測定 | short counter-question |
| EQNet + SurfacePolicy + fallback | `lmstudio-community/gemma-4-e4b-it` | `4/30` (`13.33%`) | `0/30` (`0.00%`) | `0` | `4/30` (`13.33%`) | short counter-question |

## 読み取り

### 1. EQNet core の強みは hold gate

old EQNet core は raw violation が `8/30` と高かったが、hold系の破綻は `0` だった。

router v3 は raw violation が `3/30` まで下がったが、`under_hold_error=2` が残った。

この差は重要。

router v3 は表面ルールとして強いが、hold選択の境界で失敗する。
EQNet は発話surfaceが弱くても、hold gate は壊れにくい。

### 2. SurfacePolicy は speak surface の弱さを潰した

EQNet + SurfacePolicy text では、`gpt-oss-20b` の raw violation が `1/30` まで下がった。

これは新しい判断器を足したからではない。
`reaction_contract` の制約を発話面へ明示的に投影し、LLMが守るべき出力空間を狭めた結果。

### 3. Gemma はモデル固有の反問癖を持つ

`lmstudio-community/gemma-4-e4b-it` は、`question_budget=0` の場面でも `そうかな？` / `そうなの？` 型の短い反問に逃げる傾向があった。

これは hold gate の失敗ではない。
selected channel は正しく、問題は speak surface にある。

### 4. fallback により delivered品質は守れた

fallback後のGemma再実測では、

- raw violation: `4/30`
- delivered violation: `0/30`
- fallback rate: `4/30`
- hold error: `0`

となった。

raw violation は監査ログに残る。
delivered violation は `0` まで落ちた。

したがって、失敗を隠して成功扱いしているわけではない。
モデルの癖は観測しつつ、最終出力だけを安全化している。

## KPIの分離

今後は次のように読む。

| 指標 | 意味 |
| --- | --- |
| raw violation rate | モデルがどれだけ契約から逸脱したか |
| delivered violation rate | ユーザーに出す最終文がどれだけ契約違反したか |
| fallback rate | 表現力をどれだけfallbackに依存したか |
| hold error rate | 話す / 待つ の判断がどれだけ壊れたか |

品質保証の最低条件は `delivered violation=0`。

ただし、それだけでは十分ではない。
fallback rate が高いモデルは、安全だが表現力が弱い。

## 現時点の結論

1. EQNet の hold gate は、router v3 より構造的に意味がある可能性が高い。
2. SurfacePolicy は、EQNet の判断を壊さず speak surface を改善した。
3. Gemma E4B は使えるが、短い反問癖がある。
4. fallback により実出力は守れるが、fallback rate は表現力KPIとして追跡すべき。
5. まだ robustness は証明されていない。30件セットは観測済みであり、次は未見パラフレーズで見る必要がある。

## 次の実験

次は tuned 30件ではなく、未見文で見る。

- paraphrase set: 各scenario 10件、合計60件
- generators: `gpt-oss-20b`, `lmstudio-community/gemma-4-e4b-it`, `unsloth/gemma-4-e4b-it`, `qwen3.5-4b`
- classifier: `qwen3.5-4b` から開始し、後で別classifierも比較
- group_by: `scenario_name`, `generator_model_label`, `classifier_model_label`, `selected_response_channel`, `expected_response_channel`

評価の合格線:

- delivered violation rate: `0%`
- hold_execution_violation: `0`
- under_hold_error: `<= 5%`
- over_hold_error: `<= 5%`
- raw violation rate: `<= 15%`
- fallback rate: 低いほどよい。上限は未定だが、まず `<= 15%` を暫定目標にする。
