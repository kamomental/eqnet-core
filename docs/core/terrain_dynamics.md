# Terrain Dynamics

## 役割

`terrain_dynamics` は、

- `organism_state`
- `external_field_state`
- `memory_dynamics`
- `qualia_structure_state`
- `heartbeat_structure_state`

の上に乗る、時系列の力学 contract です。

ここでは感情を固定ラベルではなく、

- basin
- flow
- energy
- entropy
- ignition
- barrier
- recovery

として扱います。

## 含めるもの

- `dominant_basin`
- `dominant_flow`
- `terrain_energy`
- `entropy`
- `ignition_pressure`
- `barrier_height`
- `recovery_gradient`
- `basin_pull`
- `trace`

## 位置づけ

- `qualia_projector` は観測断面
- `qualia_structure_state` は temporal observable
- `terrain_dynamics` は canonical dynamics

という分担です。

## 現在の接続先

- `integration_hooks`
- `transfer_package`
- `continuity_summary`
- `runtime persona_meta / metrics`
- `surface_context_packet`

## 次の本命

- `appraisal -> meaning_update -> utterance_reason` を dynamics 起点へ寄せる
- explorer を `distance` 中心から `trajectory / basin / flow` 中心へ移す
