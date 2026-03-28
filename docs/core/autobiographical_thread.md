# Autobiographical Thread

`autobiographical_thread` は、`recent_dialogue_state`、`discussion_thread_state`、`issue_state`、
`residual_reflection` から出る「まだ少し残っている話」を、いきなり長期 identity arc に固定せずに
保持するための弱い summary layer です。

目的は 2 つです。

1. その場で消えたくない未完了の話題を、数ターンから数日ほど薄く残すこと
2. ただちに自己物語や関係アークへ過剰昇格させず、再開圧だけを弱く持ち越すこと

## 何を束ねるか

- `recent_dialogue_state`
  - 直近の reopen / continue の圧
- `discussion_thread_state`
  - unresolved / revisit / settle の圧
- `issue_state`
  - explore / pause / resolve の位相
- `residual_reflection`
  - 言えなかったこと、弱められたことの残り

これらをそのまま長期人格へ入れず、次の summary にまとめます。

- `mode`
- `anchor`
- `focus`
- `strength`
- `reason_tokens`

## 位置づけ

この層は、`discussion_thread_registry` より長く、`identity_arc` より弱い中間層です。

- prompt の偶然に依存した reopening を減らす
- turn をまたいでも「前に少し引っかかっていた話」を残す
- ただし long-term self-story へ直結させない

という役割を持ちます。

## 現在の接続

1. `derive_autobiographical_thread_summary(current_state)`
2. `WorkingMemoryCore.snapshot(...)` で summary を作る
3. `working_memory_trace` に記録する
4. `inner_os_working_memory_bridge` で日中 snapshot に戻す
5. `continuity_summary` と `turn_delta` に流し、reopening anchor の fallback に使う
6. `sleep_consolidation_core` / `ops.nightly` / `transfer_package` に carry する

これにより、same-turn だけでなく nightly / transfer 後も、
`autobiographical_thread` が弱い reopening / return-point bias として残ります。

特に `discussion_thread` や `recent_dialogue` の anchor が薄い場合でも、
`autobiographical_thread_anchor` が十分に残っていれば
`turn_delta` 側で `reopen_from_anchor` / `leave_return_point_from_anchor` の fallback に使われます。

## まだしていないこと

- identity arc を直接更新しない
- relation arc を直接更新しない
- 本格的な自伝記憶にしない

ここではあくまで、
「まだ少し残っている話」と
「長い自己物語」
の間にある弱い持続層として扱います。

## 設計意図

- 本体は短期の thread / issue / residual に置く
- `autobiographical_thread` は summary だけにする
- reopening の anchor fallback には使うが、常に前面化しない

この layer の役割は、
「ただの短期記憶」と「長い自己物語」の間に、
弱い時間的持続を差し込むことです。
