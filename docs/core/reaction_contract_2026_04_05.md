# Reaction Contract (2026-04-05)

## 目的

モデルごとの差を消すことではなく、違うモデルでも自然な会話に入るための不変条件を
`reaction_contract` として明示する。

## contract 軸

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
- `reason_tags`

## 導出元

- `interaction_policy`
- `action_posture`
- `actuation_plan`
- `discourse_shape`
- `surface_context_packet`
- `turn_delta`

## 現在の扱い

- planner
  - `ResponsePlan.reaction_contract`
  - `llm_payload["reaction_contract"]`
- llm_hub
  - `[inner_os_policy]` に contract を明示
- runtime
  - `inner_os_llm_guidance["reaction_contract"]`
  - `controls_used["inner_os_reaction_contract"]`
- probe
  - `LMStudioPipelineProbe.reaction_contract`
  - raw review の `review_llm_bridge_text(...)` に渡す

## 今回の意味

`brief_shared_smile` のような low-entropy 場面で、モデルごとの癖を直接抑え込むのでなく、
先に

- 今は一緒に受けるのか
- 反応の大きさはどの程度か
- 質問してよいか
- 解釈を足してよいか
- どの timing で入るか

を固定する。

これにより、`gpt-oss-20b` の「聞きに行く assistant」寄り、`qwen3.5-4b` の
「眺めて解釈する narrator」寄りという差があっても、自然な会話の範囲を保ちやすくする。

## 次の焦点

- `reaction_contract` を live evaluation の主要指標に入れる
- wording review ではなく conversation contract review を主評価にする
- low-entropy scene では deterministic shaping をさらに強める
