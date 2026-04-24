# Discourse Shape Reasoning

`discourse_shape` は、単なる `bright` / `deep` の mode から決めるのでなく、
`utterance_reason_packet` が示す「なぜこの返しをするか」からも決まる。

現在の最小 contract は次の chain。

`shared_moment`
-> `appraisal_state`
-> `meaning_update_state`
-> `utterance_reason_packet`
-> `discourse_shape`
-> `surface_realization`

## ねらい

- `bright_continuity` だから明るく返す、で終わらせない
- 小さい出来事の型に応じて shape を変える
- `question_budget` も reason から抑える

## 現在の rule

### `brief_shared_smile`

- `appraisal_event = laugh_break`
- `meaning_update_relation = shared_smile_window`
- `utterance_reason_offer = brief_shared_smile`

がそろうと、`turn_delta` が弱くても `bright_bounce` を優先する。

### `keep_it_small`

`utterance_reason_preserve = keep_it_small` かつ
`utterance_reason_question_policy = none`
のときは、`bright_bounce` でも follow-up 質問を立てない。

## 意味

これで shape の選択が

- mode 駆動
- lightness 駆動

だけでなく、

- 何が起きたか
- 何が少し変わったか
- だから何を差し出すか

の順で決まるようになる。
