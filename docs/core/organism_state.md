# organism_state

`organism_state` は、既存の projection を置き換えるものではなく、

- `growth_state`
- `epistemic_state`
- `qualia_structure_state`
- `heartbeat_structure_state`
- `relation_competition_state`
- `social_topology_state`

を束ねる **canonical latent** です。

## 目的

この state の役割は、個別 module の値をそのまま prose に流すことではなく、
runtime / carry / continuity が参照する「いまの個体姿勢」を 1 つの contract にすることです。

## 主な軸

- `attunement`
- `coherence`
- `grounding`
- `protective_tension`
- `expressive_readiness`
- `play_window`
- `relation_pull`
- `social_exposure`

これらは感情ラベルそのものではなく、既存 state の射影を束ねた low-dimensional view です。

## posture

`dominant_posture` は現在の organism posture を表します。

- `steady`
- `attune`
- `open`
- `play`
- `protect`
- `recover`
- `verify`

これは persona の固定口調ではなく、状態遷移の結果です。

## relation / social

- `relation_focus`: 現在もっとも引かれている相手
- `social_mode`: `one_to_one / threaded_group / public_visible / hierarchical / ambient`

## trace

`trace` には短い時系列フレームを残します。  
これにより `organism_state` も単発値ではなく、carry 可能な temporal contract として扱えます。

## 現在の接続先

- `inner_os.integration_hooks`
- `inner_os.transfer_package`
- `inner_os.continuity_summary`
- `emot_terrain_lab.hub.runtime`

## 注意

`organism_state` は本体 state の唯一の真実ではありません。  
本体はあくまで各 core にあり、`organism_state` はそれらを runtime / carry で束ねるための canonical latent です。
