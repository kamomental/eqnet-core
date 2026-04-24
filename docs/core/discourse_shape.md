# Discourse Shape

`discourse_shape` は、`act` と `surface realization` のあいだに置く談話骨格である。

ここで決めるのは、具体文ではなく次のような骨格だけに留める。

- 何文で返すか
- 問いを置くか
- anchor を明示するか
- どの closing family を使うか
- bright / deep / reopen のどのエネルギー帯か

## 役割

`turn_delta` だけでは、

- 何をしたいか

までは分かるが、

- 1文で止めるのか
- 2文で返すのか
- return point を残すのか

までは安定しない。

`discourse_shape` はこの談話骨格を小さな typed contract として切り出す。

## 現在の shape

- `anchor_reopen`
- `reflect_hold`
- `bright_bounce`
- `reflect_step`

## 位置づけ

- `act`
  - 何をするか
- `discourse_shape`
  - どういう骨格で返すか
- `surface realization`
  - 実際の日本語をどう言うか

この分離によって、deep / bright / live を別システム化せず、
同じ表出系の中で重み違いとして扱いやすくする。
