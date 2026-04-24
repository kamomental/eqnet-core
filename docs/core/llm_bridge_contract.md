# LLM Bridge Contract

`llm_hub` の prompt-only 制御では削りきれない assistant attractor を、
runtime 直前の contract で止めるための層です。

## 目的

- `question_policy=none` なのに raw が follow-up question を出す
- `brief_shared_smile / keep_it_small` なのに raw が支援文・解釈文へ広がる

このような bridge 契約違反を、`final` へ入る前に検出して落とします。

## 現在の対象

- `question_block_violation`
  - `? / ？ / ですか / でしょうか / 聞かせて / 教えて` など
- `assistant_attractor_violation`
  - `お疲れ様 / 体調管理 / いつでも聞きます` など
- `interpretive_bright_violation`
  - `きっかけ / 証拠 / ただの出来事というより` など
- `too_many_sentences`
  - `keep_it_small` なのに 3 文以上へ広がる

現在は phrase の一致だけでなく、**文単位の構造判定**も併用しています。

- assistant lead
  - `おっしゃる通りですね / 昨日の続きですね / その後はどう...`
- interpretive prose
  - `...かもしれません / ...のでしょう / 証拠 / サイン`
- elicitation / counselor move
  - `見守って / 整理して / 受け止めてください`

このため、live の raw が wording を少しずらしても、
`small shared moment` に対して質問・支援文・解釈文へ戻れば違反として拾えます。

## 動作

1. raw LLM text を review する
2. 違反がなければそのまま使う
3. 違反があれば、current `content_sequence / discourse_shape / surface_context_packet`
   から作った fallback を優先する
4. `original / effective / violations` を controls と probe に残す

加えて `probe` では、runtime 途中の packet が弱かった場合でも、
**current final + discourse shape** から post-review をかけ直して
`effective raw` を再構成します。

## 位置づけ

これは言語品質判定器ではなく、`Inner OS` の canonical state に対する
**bridge 契約チェック** です。

- 本体: `organism_state + joint_state + utterance_reason + memory_dynamics`
- raw LLM: 表出候補
- contract: 候補が state 契約を破ったら落とす

## 次段

- `joint_state` と `memory_dynamics` をさらに増やしたとき、
  contract も state 由来の rule へ寄せる
- prompt-only では削れない raw attractor が残るなら、
  その後で小さな conversational SFT / DPO を検討する
