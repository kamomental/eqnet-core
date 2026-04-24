# Core Quickstart (2026-04-25)

## 目的
この quickstart は、EQNet の本流を最短で確認するための入口です。  
重い UI や LLM 比較を挟まず、次の core loop をそのまま見せます。

1. `subjective_scene`
2. `self_other_attribution`
3. `shared_presence`
4. `joint_state`
5. `reaction_contract`

## 実行
Windows:

```bat
quickstart_core.bat
```

macOS / Linux:

```bash
./quickstart_core.sh
```

JSON で確認する場合:

```bash
python scripts/core_quickstart_demo.py --json
```

全シナリオをまとめて評価する場合:

```bash
python scripts/core_contract_eval.py
python scripts/core_contract_eval.py --json
```

## 既定シナリオ

- `small_shared_moment`
  - 小さい共有モーメントを小さく一緒に受ける
- `guarded_uncertainty`
  - 踏み込みを控えて hold / defer に寄る

## 何が出るか

- `subjective_scene`
- `self_other_attribution`
- `shared_presence`
- `joint_state`
- `expected_contract`
- `reaction_contract`
- `evaluation`
- `response_guideline`

`evaluation` では、シナリオごとに定義した期待 contract と actual contract を比較し、
`stance / scale / question_budget / interpretation_budget / response_channel / continuity / timing / distance`
の違反を機械的に確認できます。

`core_contract_eval.py` は既定シナリオを一括で回し、
`passed / score / violations` を一覧化します。

## 位置づけ

- `quickstart_core.*`
  - 正規の入口
- `quickstart_llm.bat`
  - 比較・研究用の入口
- `quickstart_gradio.*`
  - full demo / 実験導線
- `heartos_mini.py`
  - 旧実験系。core quickstart の代表ではない
