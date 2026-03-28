# Discussion Thread State

`discussion_thread_state` は、短い対話の中で
「いま何の論点が未解決で、どこまで戻りやすいか」を持つ sidecar です。

`recent_dialogue_state` が主に

- 続きか
- 再開か
- 新規か

を見ているのに対して、こちらは

- `fresh_issue`
- `active_issue`
- `revisit_issue`
- `settling_issue`

のように、論点そのものの状態を扱います。

最小指標:

- `topic_anchor`
- `unresolved_pressure`
- `revisit_readiness`
- `thread_visibility`
- `dominant_inputs`

この state は現時点では、

- `turn_delta`
- `content_sequence`
- `continuity_summary`

に通り、same-turn で
「その場の支え方」だけでなく
「どの論点線を保つか / いったん閉じるか」
を決める補助として使われます。
