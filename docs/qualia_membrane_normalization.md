# Qualia Membrane Normalization

## 目的

クオリア膜は、入力刺激をそのまま強度として扱わず、観測場の読み取り信頼度を含めて正規化する。
これは「刺激が強いから即座に反応する」のではなく、局所地形が安定して読めるか、外れ値や霧で見通しが落ちているかを分けるための層である。

## 現在の実装

`inner_os/orchestration/field_normalization.py` が場の正規化を担う。

- `local_range`: 現在の場から推定した局所 range
- `global_range`: 平均的な参照 range
- `coefficient_variation`: 隣接勾配のばらつき
- `range_trust`: 局所 range を信じる度合い
- `effective_range`: local/global を補間した実効 range
- `fog_density`: 外部場の見通しの悪さ
- `gradient_confidence`: 勾配を読める度合い

`BasicQualiaProjector` はこの正規化を `observability`、`precision`、`body_coupling`、`value_grad`、`habituation` に適用する。
結果は `QualiaState.normalization_stats` に残るため、発話前の監査や評価 JSONL に接続できる。

## 意味

単一の外れ値がある場合、従来の最大値割りでは他の軸が潰れて見える。
新しい正規化では、隣接勾配のばらつきが大きい場合に local range の信頼を下げ、global range 側へ補間する。

霧が濃い場合は、同じ入力値でも `range_trust` と `gradient_confidence` が下がる。
これは「反応場が存在しない」のではなく、「今はその場を十分に読めていない」という状態として扱う。

## まだ未実装の範囲

- 長期履歴から `global_range` を学習・更新する registry
- 環境ノイズや文化場から `fog_density` を自動注入する接続
- 初体験スパイクと慣れの曲率変化を、評価レポートに直接出す指標

現段階では、正規化の型とクオリア投影への接続、監査出力までを実装した。
