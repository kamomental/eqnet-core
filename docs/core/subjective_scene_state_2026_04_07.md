# Subjective Scene State (2026-04-07)

## 目的

`self-view` を VQA 的な「何が見えたか」ではなく、主観空間の online 構築として扱う。
この段階では VLM 直結ではなく、`camera_observation` を自己基準の state に写像する薄い state core として実装する。

## 追加した state

### 1. `SubjectiveSceneState`

場所:
- `inner_os/world_model/subjective_scene_state.py`

役割:
- 観測空間を自己基準の lived space に変換する
- 幾何層、帰属の前段、情動トーンをまとめる

主な軸:
- `egocentric_closeness`
- `workspace_proximity`
- `frontal_alignment`
- `motion_salience`
- `self_related_salience`
- `shared_scene_potential`
- `familiarity`
- `comfort`
- `curiosity`
- `tension`
- `uncertainty`

### 2. `SelfOtherAttributionState`

場所:
- `inner_os/self_model/self_other_attribution_state.py`

役割:
- self / other / shared / unknown を二値分類ではなく連続量として推定する
- self-body image の投射と回収を、appearance / contingency / perspective / sensorimotor consistency の整合として扱う

主な軸:
- `self_likelihood`
- `other_likelihood`
- `shared_likelihood`
- `unknown_likelihood`
- `appearance_match`
- `contingency_match`
- `perspective_match`
- `sensorimotor_consistency`
- `attribution_confidence`
- `ambiguity`

### 3. `SharedPresenceState`

場所:
- `inner_os/shared_presence_state.py`

役割:
- self-view を「私が見た」から「私たちの場がどう立ち上がっているか」へ上げる
- AITuber 的な object report ではなく、companion 的な co-presence を state として保持する

主な軸:
- `co_presence`
- `shared_attention`
- `shared_scene_salience`
- `self_projection_strength`
- `other_projection_receptivity`
- `boundary_stability`

## 既存 core への接続

今回の接続は最小限に留めている。

- `derive_joint_state(...)` に
  - `subjective_scene_state`
  - `self_other_attribution_state`
  - `shared_presence_state`
  を追加
- `common_ground`
- `joint_attention`
- `mutual_room`
- `coupling_strength`

へ、共在感と shared attribution の寄与を薄く加えた。

重要なのは、`joint_state` の mode 判定を置き換えていないこと。
既存の conversational path を壊さず、主観空間の state が joint 更新に入る入口だけを作った。

## 位置づけ

これは最終形ではない。
現段階では、

- `camera_observation`
- `subjective_scene`
- `self_other_attribution`
- `shared_presence`
- `joint_state / terrain_dynamics`

の細い導線を引いた段階である。

次に進めるなら、

- `terrain_dynamics` への接続
- `reaction_contract` への接続
- `response_selection` での `inhabitation` 利用

を追加して、self-view を実際の反応オーケストレーションに効かせる。
