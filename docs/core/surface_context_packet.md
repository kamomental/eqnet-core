# Surface Context Packet

`surface_context_packet` は、内部状態をそのまま説明文へ潰さずに、表出用の中間表現として下流へ渡すための packet です。

## 目的

- `anchor + act + style` だけでは足りない状況情報をまとめる
- `前の内容を教えてください` のような generic clarification を避ける
- deep talk / reopening で、何が共有済みで何が未表出かを下流へ渡す

## 含めるもの

- `conversation_phase`
  - `reopening_thread`
  - `discussion_revisit`
  - `issue_pause`
  - など
- `shared_core`
  - `anchor`
  - `already_shared`
  - `not_yet_shared`
- `response_role`
  - `primary`
  - `secondary`
- `constraints`
  - `no_generic_clarification`
  - `no_advice`
  - `max_questions`
  - `keep_thread_visible`
- `surface_profile`
  - `response_length`
  - `cultural_register`
  - `group_register`
  - `sentence_temperature`
  - `brightness`
  - `playfulness`
  - `tempo`
- `source_state`
  - `recent_dialogue_state`
  - `discussion_thread_state`
  - `issue_state`
  - `green_*`
  - `live_engagement_*`
  - `lightness_*`

## 方針

- 短期継続は retrieval ではなく state / registry から組む
- 長期想起だけ memory retrieval を補助的に使う
- 下流へは散文説明ではなく packet を渡す

## 現在の位置づけ

現在は `ResponsePlan.llm_payload["surface_context_packet"]` に入り、さらに runtime の `inner_os_llm_guidance` と `LLMHub` の `[inner_os_policy]` prompt を通って、LLM/SLM に渡すための正式な contract になっています。
今後は runtime observability と live eval にも露出して、deep talk / reopening の改善ループの主軸にします。
