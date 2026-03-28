# Temporal Memory Orchestration 命名方針

## 目的

長期記憶の構造化・検索・証拠束ね・時間整合を扱う層について、
外部論文の略語に引きずられず、repo 内で一貫した名前を使う。

## 正式名称

- 日本語:
  - `時間記憶オーケストレーション`
- 英語:
  - `Temporal Memory Orchestration`

## 採用理由

- `ASMR` は一般語として強く、技術文脈で意味衝突を起こしやすい
- repo の主目的は研究用 OS の継続設計であり、流行語より責務が読める名前の方が保守しやすい
- `search / evidence / temporal update / recall orchestration` をまとめた層であることが名前から分かる

## repo 内の推奨命名

- module:
  - `temporal_memory_orchestration.py`
  - `temporal_memory_search.py`
  - `memory_evidence_bundle.py`
- class:
  - `TemporalMemoryOrchestrator`
  - `TemporalMemorySearchPlan`
  - `MemoryEvidenceBundle`
- summary key:
  - `temporal_memory_summary`
  - `memory_evidence_summary`

## 非推奨

- `asmr.py`
- `ASMRRetriever`
- `agentic_memory_magic`
- 論文略語をそのまま正式 module 名にすること

## 実装上の位置づけ

`Temporal Memory Orchestration` は人格や感情そのものではない。
責務は以下に限る。

1. 記憶を時系列・更新・矛盾込みで構造化する
2. その場の文脈に対して、関連証拠を選び直す
3. `Qualia Membrane / affective field / arc-carry` に渡す入力束を整える

つまり、

- `Temporal Memory Orchestration`
  - 何を思い出すか
- `Qualia Membrane / heartOS`
  - どう感じに写すか
- `Arc / Carry`
  - 何が残るか

の分業を崩さない。
