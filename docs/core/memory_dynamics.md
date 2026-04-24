# Memory Dynamics

## 目的
`memory_dynamics` は、repo に散っている

- palace
- monument
- ignition
- reconsolidation
- forgetting

の断片を、1 つの typed slow-state として束ねるための canonical contract です。

ここでの役割は「記憶そのものを保存すること」ではなく、
**どの連想が立ち上がりやすいか、どの記念碑が残りやすいか、どの再想起が発火しやすいか**
を低次元で持つことにあります。

## 入力断片
- [memory_palace.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/terrain/memory_palace.py)
- [memory_orchestration_core.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/memory_orchestration_core.py)
- [association_graph.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/association_graph.py)
- [recall_engine.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/hub/recall_engine.py)
- [forgetting_core.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/forgetting_core.py)
- [sleep_consolidation_core.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/sleep_consolidation_core.py)

## 主な state
- `palace_topology`
  - 連想リンクと conscious mosaic の密度をまとめた軸
- `dominant_link_key`
  - 現在もっとも前景化されている associative link
- `dominant_link_inputs`
  - その link を押し上げた入力群
- `monument_salience`
  - 記念碑的な記憶の残りやすさ
- `monument_kind`
  - どの種類の monument か
- `ignition_readiness`
  - recall / replay が発火しやすいか
- `consolidation_pull`
  - 再固定化や nightly replay へ向かう引力
- `forgetting_pressure`
  - 忘却圧
- `memory_tension`
  - 干渉・圧縮・前景化競合の強さ
- `prospective_pull`
  - まだ来ていない想起や次回参照への pull
- `dominant_mode`
  - `ignite / reconsolidate / prospect / protect / stabilize`

## packet axes
- `topology`
- `salience`
- `ignition`
- `consolidation`
- `tension`

これらは記憶の本体ではなく、runtime / transfer / summary / evaluation へ渡す観測インターフェースです。

## 接続先
- `integration_hooks`
  - `pre_turn_update / memory_recall / post_turn_update`
- `transfer_package`
  - `portable_state.carry`
  - `runtime_seed`
  - `normalize`
- `continuity_summary`
  - same-turn / overnight summary
- `runtime`
  - `persona_meta["inner_os"]`
  - metrics

## 設計意図
`memory_dynamics` は、palace / monument / ignition を新しく再発明するための層ではありません。
既存断片を `inner_os` 側から使えるようにし、今後の

- `organism_state`
- `memory_dynamics`
- `projection_observables`

の統合へ向かうための中間 contract です。
