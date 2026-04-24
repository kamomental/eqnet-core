# Typed Policy Packet Contract (2026-04-03)

## 目的

`policy_packet` は `inner_os` の中心にあるが、返り値が巨大な `dict[str, object]` だったため、

- planner 内で後から追記される
- test でキー存在確認が増える
- state-first でなく packet-first に戻る

という圧力が強かった。

今回の目的は、`policy_packet` 全面改修ではなく、
**返り値境界だけを mutable contract に包む** ことだった。

## 変更内容

### 1. mutable mapping contract を追加

`inner_os/policy_packet.py` に以下を追加した。

- `InteractionPolicyPacketContract`
- `coerce_interaction_policy_packet(...)`

この contract は `MutableMapping[str, object]` として振る舞うため、

- `packet["..."]`
- `packet.get("...")`
- `dict(packet)`
- `packet["new_key"] = value`

をそのまま維持できる。

### 2. `derive_interaction_policy_packet(...)` の返り値を contract 化

トップレベルの return を `InteractionPolicyPacketContract({...})` に変更した。

helper 群の大量の `return {...}` には触れていない。
つまり今回は、
**policy packet の出口だけ** を typed 化した段階である。

### 3. planner 側の型注釈を追従

- `inner_os/expression/models.py`
  - `ResponsePlan.interaction_policy` を `MutableMapping[str, object]`
- `inner_os/expression/response_planner.py`
  - `_apply_interaction_policy_surface_bias(...)` を `Mapping[str, object]`

に更新した。

## 設計意図

`interaction_policy` は planner の途中で

- `contact_reflection_state`
- `conversation_contract`
- `recent_dialogue_state`
- `discussion_thread_state`
- `issue_state`
- `green_kernel_composition`

などを後から書き足している。

そのため今回は immutable dataclass にせず、
**mutable contract を正本にし、巨大 dict 直返しだけをやめる**
方針を取った。

これは最終形ではなく、

1. 返り値境界を contract 化する
2. 次に core field を抽出する

という2段階のうちの第1段である。

## まだ残っている課題

- `policy_packet` の中身自体は依然として巨大
- semantic な core field はまだ列挙していない
- bootstrap test はなお key-heavy
- runtime / integration_hooks の God object 問題は未解消

つまり今回は、
**packet-first の再回収を少し止めるための出口圧縮** に留めている。

## 確認

- `py_compile`
  - `inner_os/policy_packet.py`
  - `inner_os/__init__.py`
  - `inner_os/expression/models.py`
  - `inner_os/expression/response_planner.py`
  - `tests/test_inner_os_policy_packet_opening_request.py`
- `pytest tests/test_inner_os_policy_packet_opening_request.py tests/test_inner_os_conversational_architecture.py tests/test_inner_os_bootstrap.py tests/test_inner_os_live_engagement_state.py tests/test_inner_os_integration_hooks.py -q`
  - `126 passed, 1 warning`

warning は既存の `python_multipart` のみ。
