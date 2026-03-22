# Inner OS Migration Note

## 目的
既存の flat `inner_os/*.py` core を壊さずに、新しい 4層圧縮 skeleton を共存させる。

## 方針
- 既存 core:
  - runtime 実装に近い reusable primitives
- 新規 skeleton:
  - contract / layering / bootstrap 用の研究 OS 面

## 共存ルール
- 既存 import path は変えない
- 新規 `inner_os/<package>/` は設計面と新規 module contract を受け持つ
- flat core を即座に置換しない
- integration は contract 経由で段階的に行う

## 当面の使い分け
- 実運用 hook / runtime 連携:
  - 既存 flat core を継続利用
- 研究用 state architecture / migration:
  - 新規 skeleton を使う
