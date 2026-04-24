# Joint State

`joint_state` は、自己側の posture だけでは捉えきれない
`shared_tension / shared_delight / repair_readiness / common_ground`
を canonical latent として束ねる層です。

## 役割

- `shared_moment_state`
- `listener_action_state`
- `live_engagement_state`
- `meaning_update_state`
- `organism_state`
- `external_field_state`
- `terrain_dynamics_state`
- `memory_dynamics_state`

を 1 本の self-other / common-ground state に圧縮する。

## 主要軸

- `shared_tension`
- `shared_delight`
- `repair_readiness`
- `common_ground`
- `joint_attention`
- `mutual_room`
- `coupling_strength`
- `dominant_mode`

## dominant_mode

- `delighted_jointness`
- `repair_attunement`
- `strained_jointness`
- `shared_attention`
- `ambient`

## 位置づけ

`joint_state` は prose を作る層ではない。
「なぜこの場でそう返すか」の手前にある canonical latent であり、
`surface_context_packet` と `llm_hub` はこの state を観測して表出を選ぶ。

## 現在の配線

- `policy_packet`
- `surface_context_packet`
- `response_planner`
- `runtime`
- `llm_hub`
- `integration_hooks`
- `transfer_package`
- `continuity_summary`

## 今後

`joint_state` は次に `memory_dynamics` と結び直して、
`shared delight / repair / common ground` を単発の label ではなく、
記憶と場の再構成に支えられた力学として扱う。
