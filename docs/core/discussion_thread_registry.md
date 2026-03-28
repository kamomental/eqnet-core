# Discussion Thread Registry

`discussion_thread_registry_snapshot` は、`recent_dialogue_state` / `discussion_thread_state` /
`issue_state` を 1 ターンで捨てず、短い論点の続きとして保持する short-horizon registry です。

## 役割

- 直近の会話が `fresh_opening` なのか `continuing_thread` なのかを残す
- `revisit_issue / unresolved / pausing / resolving` の位相を prompt ではなく typed state で持つ
- 同じ話の「続き」や「戻り先」を runtime carry として扱う

## Summary フィールド

- `dominant_thread_id`
- `dominant_anchor`
- `dominant_issue_state`
- `total_threads`
- `top_thread_ids`
- `thread_scores`
- `uncertainty`

## Thread node の主な値

- `last_recent_dialogue_state`
- `last_discussion_state`
- `last_issue_state`
- `thread_carry`
- `reopen_pressure`
- `unresolved_pressure`
- `revisit_readiness`
- `thread_visibility`
- `question_pressure`
- `pause_readiness`
- `resolution_readiness`
- `count`
- `confidence`

## Carry の流れ

- same-turn では `current_state["discussion_thread_registry_snapshot"]` に載る
- expression bridge では reopen / pause のときに `dominant_anchor` を短い reopening line の候補に使える
- runtime 後段では `continuity_summary` と `persona_meta["inner_os"]` で読める
- nightly では `discussion_thread_trace` memory record から再構成され、
  `inner_os_discussion_thread_registry_summary` として出る
- sleep bridge では nightly summary から `current_state` に戻る
- transfer package では carry summary と runtime seed の両方に載り、
  warm start 後も `dominant_anchor / dominant_issue_state` を引き継ぐ

## 位置づけ

これは長期の arc registry ではありません。`identity_arc` や `relation_arc` に対して、
「直近数ターンで何の論点をどこで止めていたか」を保持する短期 registry です。
