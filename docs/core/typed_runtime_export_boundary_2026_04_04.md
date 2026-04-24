# Typed Runtime Export Boundary (2026-04-04)

## 目的

`runtime.py` の内部では contract を保持しつつ、
外向きの `guidance` と `response_meta` だけを plain dict にそろえる。

## 今回の変更

- [runtime.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/hub/runtime.py)
  に module-level の export helper を追加
  - `_export_runtime_value(...)`
  - `_export_runtime_mapping(...)`
- `_serialize_response_meta(...)` の
  - `safety`
  - `perception_summary`
  - `retrieval_summary`
  を shallow な `dict(...)` ではなく export helper 経由にした
- `_build_inner_os_llm_guidance(...)` の返り値で
  - `conversation_contract`
  - `conversational_objects`
  - `object_operations`
  - `interaction_effects`
  - `interaction_judgement_view`
  - `interaction_judgement_summary`
  - `interaction_condition_report`
  - `interaction_inspection_report`
  - `interaction_audit_bundle`
  - `interaction_audit_casebook`
  - `interaction_audit_report`
  - `interaction_audit_reference_case_meta`
  を export helper 経由へ寄せた

## 意味

これで runtime でも、

- 内部: contract / Mapping
- export: plain dict

の境界を少し明確にできた。

`dict(...)` の defensive copy をゼロにはしていないが、
少なくとも guidance / response_meta の export では
「とりあえず shallow copy」から抜け始めている。

## 回帰

- `pytest tests\test_runtime_process_turn_hooks.py tests\test_inner_os_integration_hooks.py tests\test_lmstudio_pipeline_probe.py -q`
  - `143 passed, 1 warning`
- `pytest tests\test_runtime_process_turn_hooks.py tests\test_inner_os_integration_hooks.py tests\test_inner_os_bootstrap.py tests\test_lmstudio_pipeline_probe.py -q`
  - `155 passed, 1 warning`

## 追加メモ

- `controls_used` 同期でも
  - `inner_os_planned_content_sequence`
  - `inner_os_discourse_shape`
  - `inner_os_surface_profile`
  - `inner_os_surface_context_packet`
  - `inner_os_boundary_transform`
  - `inner_os_residual_reflection`
  を export helper 経由へ寄せた
- `persona_meta["inner_os"]` の
  - `boundary_transform`
  - `residual_reflection`
  も shallow `dict(...)` ではなく export helper 経由にした
- `current_state.setdefault(...)` の
  - `epistemic_state`
  - `memory_dynamics_state`
  - `qualia_structure_state`
  - `heartbeat_structure_state`
  - `organism_state`
  - `joint_state`
  - `external_field_state`
  - `terrain_dynamics_state`
  と各 axes も export helper 経由にそろえた
