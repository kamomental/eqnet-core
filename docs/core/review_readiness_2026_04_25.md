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
- `response_guideline`

`evaluation` は、シナリオごとの期待 contract と actual contract を比較し、会話上の違反を出します。

## 確認コマンド
単一シナリオ:

```bash
python scripts/core_quickstart_demo.py --scenario small_shared_moment --json
python scripts/core_quickstart_demo.py --scenario guarded_uncertainty --json
```

現在の core 評価を一括確認:

```bash
python scripts/core_contract_eval.py
python scripts/core_contract_eval.py --json
```

この作業環境では、次の形で確認しました。

```bash
uv run python scripts/core_contract_eval.py --json
```

## 現在の期待結果
一括評価は次を返す想定です。

- `scenario_count = 2`
- `passed_count = 2`
- `failed_count = 0`
- `pass_rate = 1.0`

対象シナリオ:

- `small_shared_moment`
  - 期待: `join`, `small`, `question_budget = 0`, `interpretation_budget = none`
- `guarded_uncertainty`
  - 期待: `hold`, `response_channel = hold`, `continuity_mode = reopen`, `timing_mode = held_open`

## まだ残る穴
これはまだ、EQNet が plain `LLM + prompt` より優れていることの完全な証明ではありません。

残る再レビュー上の穴:

- シナリオはまだ 2 本だけです。
- prompt-only baseline との比較はまだ入っていません。
- LM Studio live 出力は同じ evaluator にまだ接続していません。
- legacy / full demo 導線の分離は、さらに明確化が必要です。

今回の確認点は次です。

state core が `expected_contract`、actual `reaction_contract`、`violations` を外から見える形で出し、反応を state から評価できる入口になっているか。
