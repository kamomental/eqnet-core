# LM Studio Pipeline Probe

LM Studio の生出力と、EQNet/heartOS を通った最終出力を横並びで確認するための検証メモです。

目的:
- LM Studio 自体が返している文面を確認する
- EQNet 後段で何が抑制・変形しているかを見る
- `qualia_gate` が narrative を止めたのか、surface fallback が置き換えたのかを切り分ける

関連ファイル:
- [emot_terrain_lab/hub/lmstudio_pipeline_probe.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/hub/lmstudio_pipeline_probe.py)
- [scripts/check_lmstudio_pipeline.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/scripts/check_lmstudio_pipeline.py)

主な表示項目:
- `LM Raw Output`
  - LM Studio bridge が最初に返した文面
- `最終応答`
  - qualia gate / surface shaping 後に実際に返る文面
- `Qualia Gate`
  - `allow`
  - `suppress_narrative`
  - `reason`
  - `u_t / m_t / load_t / p_t / p_ema / theta`
- `Interaction Constraints / Turn Delta / Content Sequence`
  - どの骨格で返そうとしていたか

補足:
- runtime の guarded surface は `ja-JP` locale を明示して `content_sequence` を組み直す
- そのため、LM raw が日本語でも gate 後だけ英語になる問題は、`content_policy` と surface shaping の locale 化で追う

使い方:

```bash
python scripts/check_lmstudio_pipeline.py --prompt "今は無理に整理せず、引っかかっているところだけ一緒に見たいです。"
```

JSON 出力:

```bash
python scripts/check_lmstudio_pipeline.py --prompt "少ししんどいです。" --json
```

LM bridge を強制確認:

```bash
python scripts/check_lmstudio_pipeline.py --prompt "何が引っかかっているかだけ見たいです。" --force-llm-bridge
```

読み方:
- `llm_bridge_called=True` かつ `LM Raw Output` がある
  - LM Studio までは到達している
- `llm_raw_differs_from_final=True`
  - LM の生文面と最終応答が途中で分岐している
- `qualia_gate.allow=False`
  - narrative foreground が gate で抑えられている可能性が高い
- `qualia_gate.allow=True` なのに最終応答が大きく違う
  - surface shaping / fallback 側の確認が必要
## Reaction Contract

probe では `reaction_contract` も見る。

- `stance`
- `scale`
- `initiative`
- `question_budget`
- `interpretation_budget`
- `response_channel`
- `timing_mode`
- `continuity_mode`
- `distance_mode`
- `closure_mode`

見るポイントは文面そのものではなく、最終反応がこの contract と整合しているかどうか。
特に `brief_shared_smile` のような小さい共有モーメントでは、

- `scale=small`
- `question_budget=0`
- `interpretation_budget=none`
- `continuity_mode=continue`

が揃っているかを優先して確認する。
