# llm_hub Response Cause

## 目的
- raw LLM が assistant / counselor / analyst 側へ戻る引力を下げる
- `surface_context_packet.source_state` に散っていた理由断片を、raw prompt で読める最小の cause contract に束ねる

## 位置づけ
- `llm_hub.py` の `system_prompt` だけではなく、`[inner_os_policy]` 本文にも `response_cause` を入れる
- `response_cause` は thought chain ではなく、発話直前の理由 foreground である

## response_cause の中身
- `immediate`
  - 今回の小さい出来事
  - shared shift
  - relation / world update
- `memory_link`
  - 既知 thread との結びつき
  - anchor
  - memory frame
  - activation confidence
- `joint_position`
  - common ground
  - mutual room
  - shared delight / tension
- `stance`
  - organism posture
  - external field
  - terrain flow
  - protective tension / play window
- `reply_rule`
  - offer
  - preserve
  - question policy
  - tone hint

## 設計意図
- raw LLM に「何を言うか」ではなく「なぜここでそう返すか」を短く固定する
- generic helpfulness より先に
  - 共有済みの小出来事
  - 既知の記憶 thread
  - self-other coupling
  - current organism posture
  を読ませる

## 非目標
- 長い内面独白を LLM に書かせること
- chain-of-thought を表に出すこと
- `response_cause` を final の固定テンプレにすること
