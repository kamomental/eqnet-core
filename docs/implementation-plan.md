# Implementation Plan

## Phase 1: bootstrap inner_os skeleton
目的:
- module tree を追加する
- typed contracts を定義する
- smoke tests を通す

Done when:
- import が通る
- smoke test が通る
- business logic は最小 stub に留まる

## Phase 2: grounding + world/self state
目的:
- `ObservationBundle`
- `WorldState`
- `SelfState`
- `PersonRegistry`
を実装する

Done when:
- mock input で end-to-end state update が動く
- uncertainty が保持される

## Phase 3: value + access
目的:
- `ValueState`
- `ForegroundState`
- access selection
を実装する

Done when:
- structured foreground が生成される
- raw observation が LLM に直接流れない

## Phase 4: expression bridge
目的:
- `ResponsePlan`
- expression adapter
を実装する

Done when:
- 同じ foreground を複数 style に render できる
- LLM dependency が core module に漏れない

## Phase 5: replaceable hypothesis slots
目的:
- subjective / access / terrain の差し替え口を用意する
- comparison fixture を回せるようにする

Done when:
- registry / fixture が動く
- B/C 的比較の入口がある
