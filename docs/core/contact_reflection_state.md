# Contact Reflection State

Date: 2026-03-28

## Why This Layer Exists

`contact field -> contact dynamics -> access projection` までは既にありましたが、
そこからすぐ `event` や `content sequence` へ飛ぶと、

- 何が内部へ通ったか
- 何がその場で反射として返るか
- 何が吸収されて残るか
- 何が遮断されるか

が曖昧なままでした。

`ContactReflectionState` は、この接触境界の結果を小さく明示する層です。

## Stack Position

```text
contact field
    -> contact dynamics
    -> access projection
    -> contact reflection state
    -> disclosure / thread / issue event 化
    -> content policy
```

## Current Meaning

`ContactReflectionState` は、入力をそのまま event 化しません。
先に接触境界の結果として次をまとめます。

- `transmit_share`
  - 内部へ通る比率
- `reflect_share`
  - その場で返答へ反射しやすい比率
- `absorb_share`
  - いったん飲み込み、残差側へ回りやすい比率
- `block_share`
  - 今は通さず止める比率

加えて、今の接触が

- `open_reflection`
- `guarded_reflection`
- `absorbing_contact`
- `blocked_contact`

のどれに近いかを `state` として持ちます。

## Why It Helps Expression

deep disclosure で毎回同じように問い返すと、
「支えてはいるが機械的」になりやすいです。

この層があると、

- 開いているときは `reflect_then_question`
- guarded なときは `reflect_only`
- 強く止めるべきときは `boundary_only`

のように、接触状態に応じて返答の形を変えられます。

## Current Scope

いまは `v0` の heuristic 層です。

提供するもの:

- 接触境界の結果を typed に読む
- deep disclosure で `問い返す / 受け止めだけにする` を分ける
- mechanism issue map の欠けていた `接触 / 反射 / 吸収 / 遮断` を実装側に接続する

まだやっていないもの:

- full な contact scattering 方程式
- nightly carry への昇格
- boundary / residual との統一 contract
