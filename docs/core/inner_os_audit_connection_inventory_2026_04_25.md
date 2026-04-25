# Inner OS audit connection inventory 2026-04-25

## Position

このメモは、Inner OS の部品が「存在するか」ではなく、どこまで監査経路に接続されているかを整理する。

結論は次の通り。

- 部品は広く存在する。
- `inner_os/expression/response_planner.py` の本流では、かなりの状態が `interaction_audit_bundle`、`surface_context_packet`、`llm_payload` に接続されている。
- ただし、直近の 30 件 `core_llm_expression_eval` は簡略 quickstart 経路なので、全 Inner OS 監査経路を通っているとは言えない。
- したがって、現在の課題は「部品がない」ではなく、「実験経路と本流監査経路を一致させること」。

## Main audit-connected route

`inner_os/expression/response_planner.py` の本流は、概ね次の順で状態を束ねる。

```text
foreground / memory / scene
  -> interaction_judgement_summary
  -> interaction_condition_report
  -> interaction_audit_bundle
  -> interaction_policy
  -> green_kernel_composition
  -> action_posture
  -> actuation_plan
  -> surface_context_packet
  -> reaction_contract
  -> llm_payload / expression bridge
```

この経路では、`interaction_audit_bundle` と `surface_context_packet` が監査用の主な出口になる。

## Audit bundle

`inner_os/interaction_audit_bundle.py` は、会話上の判断を人間が読める監査行に圧縮する。

主な出力は次の通り。

- `observed_lines`
- `inferred_lines`
- `selected_object_lines`
- `deferred_object_lines`
- `operation_lines`
- `intended_effect_lines`
- `scene_lines`
- `relation_lines`
- `memory_lines`
- `integration_lines`
- `inspection_lines`
- `key_metrics`

`key_metrics` には、少なくとも次が入る。

- `pressure_balance`
- `question_budget`
- `question_pressure`
- `defer_dominance`
- `effect_count`
- `resonance_score`
- `recommended_family_id`
- `detail_room_level`
- `acknowledgement_need_level`
- `pressure_sensitivity_level`
- `next_step_room_level`

これは「何を見たか」「何を推定したか」「何を選び、何を保留したか」「どの圧力が効いたか」を監査するための層。

## Surface context packet

`inner_os/expression/surface_context_packet.py` は、発話表面へ落とす直前の監査可能な状態投影である。

ここには、以下の系統がすでに入る。

### Dialogue and thread

- `recent_dialogue_state`
- `discussion_thread_state`
- `issue_state`
- `turn_delta`
- `boundary_transform`
- `residual_reflection`

### Joint and shared state

- `shared_moment_state`
- `listener_action_state`
- `joint_state`
- `appraisal_state`
- `meaning_update_state`
- `utterance_reason_packet`

具体的には次が投影される。

- `joint_shared_delight`
- `joint_shared_tension`
- `joint_repair_readiness`
- `joint_common_ground`
- `joint_attention`
- `joint_mutual_room`
- `joint_coupling_strength`

### Action and actuation

- `action_posture`
- `actuation_plan`

具体的には次が投影される。

- `action_posture_mode`
- `actuation_execution_mode`
- `actuation_primary_action`
- `actuation_response_channel`

### Organism

- `organism_state`

具体的には次が投影される。

- `organism_posture`
- `organism_relation_focus`
- `organism_social_mode`
- `organism_attunement`
- `organism_coherence`
- `organism_grounding`
- `organism_protective_tension`
- `organism_expressive_readiness`
- `organism_play_window`
- `organism_relation_pull`
- `organism_social_exposure`

### External field

- `external_field_state`

具体的には次が投影される。

- `external_field_dominant`
- `external_field_social_mode`
- `external_field_thread_mode`
- `external_field_environmental_load`
- `external_field_social_pressure`
- `external_field_continuity_pull`
- `external_field_ambiguity_load`
- `external_field_safety_envelope`
- `external_field_novelty`

### Terrain

- `terrain_dynamics_state`

具体的には次が投影される。

- `terrain_dominant_basin`
- `terrain_dominant_flow`
- `terrain_energy`
- `terrain_entropy`
- `terrain_ignition_pressure`
- `terrain_barrier_height`
- `terrain_recovery_gradient`
- `terrain_basin_pull`

### Memory

- `memory_dynamics_state`

具体的には次が投影される。

- `memory_dynamics_mode`
- `memory_dominant_relation_type`
- `memory_relation_generation_mode`
- `memory_dominant_causal_type`
- `memory_causal_generation_mode`
- `memory_palace_mode`
- `memory_monument_mode`
- `memory_ignition_mode`
- `memory_reconsolidation_mode`
- `memory_recall_anchor`
- `memory_monument_salience`
- `memory_activation_confidence`
- `memory_tension`

### Heartbeat

- `heartbeat_structure_state`

具体的には次が投影される。

- `heartbeat_pulse_band`
- `heartbeat_phase_window`
- `heartbeat_reaction`
- `heartbeat_activation_drive`
- `heartbeat_containment_bias`
- `heartbeat_bounce_room`
- `heartbeat_response_tempo`

## Safety, environment, and emergency

安全系は少なくとも次に分かれている。

- `inner_os/situation_risk_state.py`
- `inner_os/emergency_posture.py`
- `inner_os/expression/content_policy.py`
- `inner_os/expression/interaction_constraints.py`
- `emot_terrain_lab/hub/safety_orchestrator.py`

環境系は少なくとも次に分かれている。

- `inner_os/environment_pressure_core.py`
- `inner_os/external_field_state.py`

`external_field_state` は、環境負荷、社会圧、継続性、曖昧さ、安全包絡、 novelty を持つ。

```text
environment_pressure
  -> external_field_state
  -> joint_state / terrain_dynamics / action_posture / surface_context_packet
```

ただし、すべての簡略評価経路でこの流れが通るわけではない。

## Culture and norm

文化・規範系は少なくとも次に分かれている。

- `eqnet/culture_model.py`
- `inner_os/cultural_conversation_state.py`
- `inner_os/scene_state.py`
- `inner_os/constraint_field.py`
- `inner_os/expression/response_planner.py`

文化系は、主に次の値へ落ちる。

- `politeness_pressure`
- `directness_ceiling`
- `joke_ratio_ceiling`
- `group_attunement`
- `norm_pressure`
- `cultural_register`

注意点として、`eqnet/culture_model.py` には文字化けした検出パターンが残っているため、文化記念碑検出を正本化する前に修正または隔離が必要。

## Temperament, development, and growth

遺伝そのものではないが、実装上の「気質・体質・長期傾向」に相当するものはある。

- `inner_os/temperament_estimate.py`
- `inner_os/development_core.py`
- `inner_os/growth_state.py`
- `inner_os/sleep_consolidation_core.py`
- `inner_os/daily_carry_summary.py`

`temperament_estimate` は次を持つ。

- `risk_tolerance`
- `ambiguity_tolerance`
- `curiosity_drive`
- `bond_drive`
- `recovery_discipline`
- `protect_floor`
- `initiative_persistence`
- `leader_tendency`
- `hero_tendency`
- `forward_trace`
- `guard_trace`
- `bond_trace`
- `recovery_trace`

`daily_carry_summary.py` では、same-turn と sleep carry の両方で次が監査対象に入る。

- `growth_*`
- `memory_dynamics_*`
- `temperament_*`
- `homeostasis_budget_*`
- `body_homeostasis_*`
- `boundary_*`
- `residual_reflection_*`
- `autobiographical_thread_*`
- `temporal_membrane_*`

つまり、成長・老い・持ち越し・再固定化の監査出口はすでに存在する。

## What is connected vs not yet proven

接続済みと言えるもの。

- `response_planner` 本流では、`interaction_audit_bundle` に判断要約が出る。
- `surface_context_packet` には、organism / external field / terrain / memory / heartbeat / joint / action / actuation が投影される。
- `daily_carry_summary` には、same-turn と sleep carry の監査項目がある。
- `reaction_contract` には `action_posture` と `actuation_plan` が渡る。

まだ未証明なもの。

- 直近の 30 件 `core_llm_expression_eval` が、上記の全経路を通っていること。
- 文化・環境・気質・記憶発火が、実際に `reaction_contract` の選択差を生んでいること。
- `surface_policy` と fallback の改善が、これらの内面状態に由来する改善であること。
- 文化記念碑検出の文字化けパターンが安全に処理されていること。

## Immediate correction

前回までの言い方は少し粗かった。

正確には、「監査に接続されていない」のではない。

正しくは次。

```text
本流 response_planner では監査接続されている。
しかし直近の比較実験経路は簡略版なので、
その監査接続を十分に使っているとはまだ言えない。
```

したがって次に必要なのは、新規概念追加ではなく、実験経路を本流監査経路へ寄せること。

## Next step

次の実装候補は、`core_llm_expression_eval` の結果 JSONL に次を追加すること。

- `surface_context_packet.source_state`
- `surface_context_packet.surface_profile`
- `interaction_audit_bundle.key_metrics`
- `action_posture`
- `actuation_plan`
- `organism_state`
- `external_field_state`
- `terrain_dynamics_state`
- `memory_dynamics_state`
- `heartbeat_structure_state`

これにより、評価が「発話違反率」だけでなく、どの内部状態からその反応になったかの監査に戻る。
