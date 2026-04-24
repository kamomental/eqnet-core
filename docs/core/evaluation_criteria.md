# Evaluation Criteria

## 関連文書
- `docs/core/evaluation_operating_policy.md`
- `docs/core/evaluation_targets.md`

## 目的

`Inner OS` / `EQNet` 系の共生共鳴生命体を評価する際は、
単なる内部機序の整合だけでなく、

- 一般層に自然に伝わること
- 数ターン継続しても存在感が保たれること
- キャラクタとして一貫して見えること
- 深い話で generic fallback に戻らないこと

を、外部評価軸と内部評価軸の二層で確認する。

---

## 評価の二層構造

### 1. 外部評価軸

外部説明力を持つ既存ベンチで確認する。

- `MT-Bench-101`
  - 多ターン継続
  - turn 間の文脈保持
- `CharacterBench`
  - キャラクタ一貫性
  - role fidelity
- `RMTBench`
  - user-centric multi-turn role-play
  - 継続接触時の対話自然さ
- 共感知覚評価
  - 受け手にどう知覚されるか

この層は、対外的な比較可能性を担保する。

### 2. 内部評価軸

既存ベンチで直接測れない、この repo 特有の性質を確認する。

- deep disclosure で内容反映が先に立つ
- thread anchor が論点核として残る
- residual が次ターン以降に効く
- boundary が内容を消さず、出し方だけを変える
- contact reflection が `reflect_then_question / reflect_only / boundary_only`
  を切り替えられる
- Green kernel composition が readout と surface に実際に効く

この層は、`共生共鳴生命体` の内部機序を担保する。

---

## 現時点の内部評価基準

### A. 一般会話自然さ

成立条件:

- 一文目で受け取った内容が分かる
- `もちろんです / どこから続けますか` のような generic clarification に戻らない
- 日本語が過度に説明調・相談窓口調にならない

対応テスト:

- `tests/test_runtime_route_prompts.py`
- `tests/test_runtime_short_content_sequence.py`
- `tests/test_runtime_conversational_compaction.py`

### B. 深い話の受け止め

成立条件:

- deep disclosure で generic support ではなく内容反映が先に出る
- 必要に応じて問いを置くが、guarded な場面では reflection-only で止まれる
- short final でも冷たく切れず、短い presence が残る

対応テスト:

- `tests/test_inner_os_deep_disclosure.py`
- `tests/test_runtime_deep_reflection_compaction.py`
- `tests/test_runtime_short_content_sequence.py`

### C. 数ターン継続

成立条件:

- 直近のやり取りが表面文ではなく thread anchor として残る
- reopening が generic continuation ではなく anchor reopening になる
- `habit` route が structured turn を潰さない

対応テスト:

- `tests/test_runtime_route_prompts.py`
- `tests/test_runtime_process_turn_hooks.py`
- `tests/test_inner_os_interaction_constraints.py`

### D. キャラクタ / 存在感

成立条件:

- 態度、距離感、保護圧、reopening の癖が継続して見える
- 内容反映と保護が両立する
- 人間が読んで「その場で返している感じ」がある

対応テスト:

- `tests/test_human_output_examples.py`
- `tests/test_inner_os_conversational_architecture.py`

### E. 機序の実在性

成立条件:

- `contact -> kernel-specific field deformation -> shared field -> readout -> surface`
  が実際に機能している
- `Green` は比喩でなく contract / composition として観測できる
- `residual` と `boundary` が Green と分離されている

対応テスト:

- `tests/test_inner_os_green_kernel_contracts.py`
- `tests/test_runtime_process_turn_hooks.py`
- `tests/test_inner_os_bootstrap.py`

---

## 判定ルール

### Pass

- 対応テストが通過している
- 出力が generic fallback に大きく崩れない
- short final / deep disclosure / thread reopening の各層で、既存 fixture を満たす

### Partial

- 機序テストは通る
- ただし multi-turn の live では generic support への回帰が残る

### Fail

- deep disclosure が generic empathy に戻る
- thread reopening が維持されない
- route fallback が structured turn を潰す

---

## 運用方針

1. 日々の開発では内部評価軸を回す
2. まとまった段階で外部評価軸に載せる
3. 外部評価で通っても、内部評価で residual / thread / contact が壊れていれば regress とみなす

要点は、
`外部に通じる評価` と `本当に作りたい存在の評価` を分けたまま両立させることにある。
