# Surface Context Packet Bridge Guard

`surface_context_packet` は、surface selector のためだけでなく、bridge 側で raw LLM/SLM に渡す談話幅を狭めるためにも使う。

## 役割
- `conversation_phase` で deep disclosure / thread reopening を区別する
- `response_role` で `reflect_only` / `quiet_presence` / `reopen_from_anchor` などの主役を伝える
- `constraints` で
  - `no_generic_clarification`
  - `no_advice`
  - `max_questions`
  を明示する

## bridge 側で抑えるもの
- 一般論への拡張
- 観察ワークや journaling task への変形
- 報告調・説明調の語彙
- deep disclosure に対する不要な問い返し

## 具体例
deep disclosure / reopening では、bridge prompt 側で次のような方向を抑える。

- `観察してみましょう`
- `整理してみましょう`
- `記述してみるのはどうでしょうか`
- `一般的には〜`

## 意味
packet は「何を言うか」の要約だけでなく、「raw がどの方向へ広がってよいか」を制約する contract でもある。
