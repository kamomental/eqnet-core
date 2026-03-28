# Evaluation Loop: Anchor Progress (2026-03-28)

## 目的

`evaluation_targets.md` で最優先に置いた
`数ターン継続 64 -> 76`
に対して、thread anchor を表面文ではなく論点核に寄せる。

## 今回の変更

- `inner_os/anchor_normalization.py` を追加
  - quote 抽出
  - clause 切り出し
  - anchor tail の圧縮
  - 複数候補からの anchor 選択
- `recent_dialogue_state.py`
  - `recent_anchor` を共通正規化に通す
- `discussion_thread_state.py`
  - `topic_anchor` を共通正規化に通す
- `discussion_thread_registry.py`
  - registry へ保存する anchor を正規化
- `turn_delta.py`
  - `first non-empty` ではなく `select_anchor_hint(...)` で anchor を選ぶ
- `content_policy.py`
  - surface 用の `anchor_hint` も同じ正規化を通す
- `content_policy.py`
  - `green_reflection_hold` でも、継続 thread が十分立った turn では
    `light question` を戻せるようにした
  - 初手の guarded turn は reflection-only を保ちつつ、
    turn2 以降で毎回同じ止め方にならないようにした
- `runtime.py`
  - reflection-only compact phrase が直近の surface 履歴と重なるときは
    代替の companion line を選ぶようにした
  - deep reflection の短い止め方が毎回同型になりにくくなった
  - thread reopening の compact companion line も
    直近履歴と重なるときは別候補へ揺らぐようにした

## 追加した評価

- `tests/test_inner_os_anchor_normalization.py`
  - quote 付き anchor の抽出
  - surface sentence からの論点核圧縮
  - `turn_delta` が長文より concise anchor を選ぶこと
- `tests/test_inner_os_deep_disclosure.py`
  - `green_reflection_hold` でも continuing thread では
    `gentle_question` を戻せること
- `tests/test_runtime_deep_reflection_compaction.py`
  - 直近履歴に同じ compact presence があるとき、
    別の companion line へ揺らぐこと
- `tests/test_runtime_short_content_sequence.py`
  - thread reopening の short compaction でも
    同じ closing line を繰り返しにくいこと

## 近傍回帰

次を回して全通過:

- `tests/test_inner_os_anchor_normalization.py`
- `tests/test_inner_os_interaction_constraints.py`
- `tests/test_inner_os_discussion_thread_registry.py`
- `tests/test_runtime_route_prompts.py`
- `tests/test_multi_turn_deep_talk_surface.py`

結果:

- `31 passed, 1 warning`
- warning は既存の `python_multipart`

追加の deep-talk 近傍回帰:

- `15 passed, 1 warning`
- warning は既存の `python_multipart`

## 評価上の位置づけ

今回の変更は phrase hack ではなく、

- 履歴
- discussion registry
- turn delta
- surface

の全段で **同じ論点核を参照しやすくする** 機序改善です。

## 暫定見立て

- 数ターン継続: `64 -> 69`
- 一般会話自然さ: `68 -> 71`

まだ live 実走の multi-turn deep talk で確認していないため、
この更新は **暫定値** とする。

## 次のループ

次は live 実走で、

- turn2〜turn4 で generic support に戻らないか
- `reflect_only / quiet_presence / light question` が自然に揺れるか
- anchor reopening が表面文ではなく論点核で保たれるか

を確認し、必要なら route / compaction を追加調整する。
