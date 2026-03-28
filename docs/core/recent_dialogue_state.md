# Recent Dialogue State

`recent_dialogue_state` は、長期 arc や overnight carry とは別に、直近 2〜5 ターン程度の短い対話継続を typed state として扱うための層です。

状態は次の 3 つです。

- `fresh_opening`
  - 新しく話し始めている
- `continuing_thread`
  - 直近の話題線をそのまま持っている
- `reopening_thread`
  - いったん間を置いた話を戻そうとしている

この層は、次を直接決めません。

- 長期人格そのもの
- relation arc そのもの
- emergency posture の最終判断

代わりに、same-turn continuity を `turn_delta` と `content_sequence` に渡し、`keep_shared_thread_visible` や `leave_return_point` を prompt 依存ではなく state 依存で選べるようにします。

主な観測値は次です。

- `overlap_score`
- `reopen_pressure`
- `thread_carry`
- `recent_anchor`
- `dominant_inputs`

ここで見る履歴は user 発話だけに固定しません。assistant 側が残した reopening / return-point の文面も thread history に合流させ、前回どこで止めていたかを reopening の anchor として再利用できます。

この state は次へ渡されます。

- `response_planner`
- `integration_hooks.response_gate`
- `continuity_summary`

用途は、same-turn observability と short-turn continuity の両方です。
