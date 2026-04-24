# Core Quickstart (2026-04-25)

## 目的

この quickstart は、EQNet の本流を最短で確認するための入口です。  
豪華な UI や LLM 比較ではなく、次の core loop をそのまま見せます。

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

JSON で確認したい場合:

```bash
python scripts/core_quickstart_demo.py --json
```

## 既定シナリオ

- `small_shared_moment`
  - 小さい共有モーメントを小さく一緒に受ける。
- `guarded_uncertainty`
  - 踏み込みを抑えて hold / defer に寄る。

## 位置づけ

- `quickstart_core.*`
  - 正規の入口
- `quickstart_llm.bat`
  - 比較・研究用の入口
- `quickstart_gradio.*`
  - full demo / 実験的な可視化入口
- `heartos_mini.py`
  - 旧簡易実験。core quickstart ではない
