# Memory Dynamics Canonicalization (2026-04-01)

## 今回やったこと

`memory_dynamics` を、単なる

- `dominant_mode`
- `palace_topology`
- `monument_salience`

の薄い要約状態から、

- `palace_mode`
- `monument_mode`
- `ignition_mode`
- `reconsolidation_mode`
- `activation_confidence`
- `recall_anchor`
- `trace`

を持つ canonical state へ拡張した。

## 何を統合したか

今回の `memory_dynamics` は次の断片を 1 本に束ねる。

- palace:
  - associative topology
  - memory palace の active density
- monument:
  - monument salience / monument kind
- ignition:
  - activation trace
  - confidence curve
  - recall anchor
- reconsolidation:
  - replay priority
  - reconsolidation priority
  - autobiographical pull
  - forgetting pressure

## 位置づけ

`memory_dynamics` は記憶そのものではない。

- 記憶内容
- 記憶候補
- scene replay

の上にある、記憶の動き方を表す state である。

つまり、

- どの link が前景化しやすいか
- monument がどれだけ残っているか
- recall がどれだけ発火しやすいか
- nightly に何が再固定化されやすいか

を 1 つの contract に圧縮している。

## 追加した観点

- `palace_mode`
  - `ambient / diffuse / anchored / clustered / sparse`
- `monument_mode`
  - `ambient / tagged / rising / engraved`
- `ignition_mode`
  - `idle / arming / primed / active`
- `reconsolidation_mode`
  - `settle / replaying / reconsolidating / defragmenting`

## trace の意味

`trace` は毎ターンの詳細ログではなく、

- palace / monument / ignition / reconsolidation

の姿勢がどう移ったかを見るための短い履歴である。

これにより、`memory_dynamics` は same-turn の score ではなく、
slow-state として carry できる。

## 次にやること

次の本命は 2 つ。

1. `memory_dynamics` を `utterance_reason` と `joint_state` に接続して、
   「この場でなぜこの記憶側へ引かれるのか」を表出理由へ通す。
2. `activation_trace` と `memory_palace_state` を runtime 側でも明示 carry して、
   recall の起点が live probe から見えるようにする。
