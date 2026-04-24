# Memory Dynamics Reasoning Bridge

`memory_dynamics` はこれまで

- summary / carry
- runtime metrics
- terrain dynamics の入力

には出ていたが、`なぜ今この返しになるか` の chain にはまだ薄かった。

今回の接続では、`memory_dynamics` を

- `appraisal_state`
- `meaning_update_state`
- `utterance_reason_packet`
- `surface_context_packet`
- `LLMHub` の `[inner_os_policy]`

まで通した。

## 何が変わったか

bright / live の小さい共有出来事について、

- ただ今ちょっと笑えた
- ただ今ちょっとほっとした

だけでなく、

- それが既知の thread の上で起きたのか
- その記憶がいま点火しているのか
- monument 的に残っている話題なのか

を `reason chain` が参照できるようになった。

## 追加された見方

`appraisal_state`
- `memory_mode`
- `recall_anchor`
- `memory_resonance`

`meaning_update_state`
- `memory_update`
- `recall_anchor`
- `memory_resonance`

`utterance_reason_packet`
- `memory_frame`
- `memory_anchor`

## 意味

これで `memory_dynamics` は「記憶の計測値」だけではなく、

- この場で既知 thread をどう近くに保つか
- 小さい出来事をただの単発として扱わないか
- どこまで言わずに共有だけ残すか

に効く canonical state になり始めた。

## 次

次の本命は 2 本。

1. `memory_dynamics` を `route / response_strategy / discourse_shape` の upstream にも効かせる
2. `joint_state` を立てて、memory の動きが self だけでなく common ground 側にも効くようにする
