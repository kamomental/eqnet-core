# Qualia Structure State

`qualia_structure_state` は、語彙から逆算した静的な感情距離や、
Plutchik 型の外部配置だけでは表せない、
`Inner OS` 内部の時間的なクオリア構造を保持するための state 契約です。

## 目的

次の 3 つを同時に持つことを目的にします。

- そのターンで立ち上がっている内部中心
- 直前状態からどちらへ流れているかという変化量
- 記憶・再入・継続圧から来る時間的な構造

これにより、UI や runtime は
「言語から推定した感情ラベルの距離」
ではなく、
「内部から立ち上がっている状態の相」
としてクオリアを扱えます。

## 入力

導出では主に次を使います。

- `qualia_state`
  - `qualia`
  - `gate`
  - `habituation`
  - `body_coupling`
- `qualia_membrane_temporal`
  - `timeline_coherence`
  - `reentry_pull`
  - `continuity_pressure`
  - `relation_reentry_pull`
  - `supersession_pressure`
- `qualia_planner_view`
  - `trust`
  - `felt_energy`
  - `body_load`
  - `protection_bias`
  - `degraded`

## 主な出力

- `center`
  - 現在の内部中心
- `momentum`
  - 直前中心との差分
- `phase`
  - `fragmenting / rising / echoing / settling / shifting / holding`
- `emergence`
  - 立ち上がりの強さ
- `stability`
  - まとまり
- `memory_resonance`
  - 記憶再入の共鳴
- `temporal_coherence`
  - 時系列の連続性
- `drift`
  - 状態移動の大きさ
- `trace`
  - 短い履歴

## 位置づけ

`qualia_structure_state` は、
巨大な知識層ではなく小さい知能核に属する state です。

- `qualia_graph`
  - 静的な位置関係の近似
- `qualia_structure_state`
  - 内部から立ち上がる時間構造

という役割分担で使います。

## 現在の配線

現在は次に通しています。

- `integration_hooks`
- `transfer_package`
- `runtime.persona_meta["inner_os"]`
- `continuity_summary`

したがって downstream 側は、
静的な離散パラメータだけでなく、
この state を参照して UI / probe / planner を組めます。
