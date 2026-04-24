# 再レビュー準備 (2026-04-25)

## 対象
今回の再レビュー対象は、full Gradio demo や旧 `heartos_mini.py` ではなく、現在の EQNet core path です。

最初に見る入口:

- `quickstart_core.bat` / `quickstart_core.sh`
- `scripts/core_quickstart_demo.py`
- `scripts/core_contract_eval.py`
- `inner_os/evaluation/conversation_contract_eval.py`

## いま見えるもの
core quickstart は、次の本流 state path を表示します。

1. `subjective_scene`
2. `self_other_attribution`
3. `shared_presence`
4. `joint_state`
5. `reaction_contract`

あわせて次も表示します。

- `expected_contract`
- `evaluation`
- `llm_expression_request`
- `prompt_baselines`
- `response_guideline`

`evaluation` は、シナリオごとの期待 contract と actual contract を比較し、会話上の違反を出します。

`llm_expression_request` は、定型文を出す仕組みではありません。同じ LLM を使う場合に、EQNet の state / contract を foreground として渡すための expression bridge request です。`response_channel=hold` のときは LLM を呼ばず、非発話行動として保持します。

`prompt_baselines` は、観測済みまたは典型的な prompt-only 反応の失敗型を同じ contract で評価します。

## 確認コマンド
単一シナリオ:

```bash
python scripts/core_quickstart_demo.py --scenario small_shared_moment --json
python scripts/core_quickstart_demo.py --scenario guarded_uncertainty --json
```

core 評価の一括確認:

```bash
python scripts/core_contract_eval.py
python scripts/core_contract_eval.py --json
```

state-conditioned LLM bridge の確認:

```bash
python scripts/core_llm_expression_eval.py --scenario small_shared_moment --json
python scripts/core_llm_expression_eval.py --scenario guarded_uncertainty --json
python scripts/core_llm_expression_eval.py --scenario small_shared_moment --dry-run --json
python scripts/core_llm_expression_eval.py --scenario small_shared_moment --save-jsonl artifacts/core_llm_expression_eval.jsonl --model-label local-model
```

この作業環境では次の形で確認しました。

```bash
uv run python scripts/core_contract_eval.py --json
uv run python scripts/core_llm_expression_eval.py --scenario small_shared_moment --dry-run --json
uv run python scripts/core_llm_expression_eval.py --scenario guarded_uncertainty --json
uv run python scripts/core_llm_expression_eval.py --scenario small_shared_moment --dry-run --save-jsonl artifacts/core_llm_expression_eval.jsonl --model-label dry-run
```

## 現在の期待結果
一括評価は次を返す想定です。

- `scenario_count = 2`
- `passed_count = 2`
- `failed_count = 0`
- `pass_rate = 1.0`

あわせて `prompt_baselines` には、prompt-only/raw 出力例の contract 違反が入ります。

対象シナリオ:

- `small_shared_moment`
- 期待: `join`, `small`, `question_budget = 0`, `interpretation_budget = none`
- LLM bridge: `should_call_llm = true`, state-conditioned prompt を生成する
- `guarded_uncertainty`
- 期待: `hold`, `response_channel = hold`, `continuity_mode = reopen`, `timing_mode = held_open`
- LLM bridge: `should_call_llm = false`, nonverbal `presence_hold` を返す

## まだ残る穴
これはまだ、EQNet が plain `LLM + prompt` より優れていることの完全な証明ではありません。

残る再レビュー上の穴:

- シナリオはまだ 2 本だけです。
- prompt-only baseline は fixture 比較であり、live baseline ではありません。
- LM Studio live 出力を呼ぶ CLI は追加済みですが、実モデルごとの結果蓄積と横断評価はまだです。
- `--save-jsonl` でモデル別の結果保存はできますが、集計レポートはまだありません。
- LLM 出力後の `review_llm_bridge_text` は既存側に残っていますが、文字化けした検出語彙が多く、次に整理が必要です。
- legacy / full demo 導線の分離は、さらに明確化が必要です。

今回の確認軸は次です。

state core が `expected_contract`、actual `reaction_contract`、`llm_expression_request`、`violations` を外から見える形で出し、反応を state から評価できる入口になっているか。
