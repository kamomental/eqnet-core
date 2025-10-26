# Skill Registry Guide

EQNetのハブは「コア（EQNet）」と「外部エージェント（LLM/TTS など）」を組み合わせ、`tools/` 配下でスキルを管理します。このガイドではスキルの追加手順と、ハブとの連携ポイントを簡潔にまとめます。

---

## 1. スキル管理ディレクトリ
- `tools/registry.py`: スキルの読み込み・検索・実行を担うレジストリ。
- `config/tools.yaml`: スキル定義を記述する YAML。以下が基本構造です。

```yaml
skills:
  - id: "chitchat_basic"
    name: "Basic Chitchat"
    intent: "chitchat"       # hub.generate(intent=...) と連動
    description: "短い世間話を行う"
    llm: "llm-fast"          # config/hub.yaml に登録されたモデル名
    context_sources: []       # 例: ["rag"], ["code"], [] など
    priority: 1               # 同じ intent 内での優先度
```

`intent` に対応した LLM は `hub/llm_hub.py` から選択されます。標準 intent は `chitchat` と `qa` で、必要に応じて拡張可能です。

---

## 2. 新しいスキルを追加する手順
1. `config/tools.yaml` の `skills` 配列に定義を追加します。  
   - `intent`: `hub.generate(intent=...)` で指定する呼び出しキー。  
   - `llm`: `config/hub.yaml` の `llms[].name` と一致させます。  
   - `context_sources`: RAG や SQL など、追加コンテキストが必要な場合に指定。  
   - `priority`: 同じ `intent` 内で最初に選ばれるスキルを制御。  
2. 必要なら `SkillRegistry.reload()`（再起動でも可）を呼び出して定義を再読み込み。
3. `hub.generate(intent="...", ...)` を呼ぶと、優先度が高いスキルの `llm` と `context_sources` が自動的に適用されます。

**例: コード支援スキル**
```yaml
  - id: "code_helper"
    name: "Code Helper"
    intent: "code"
    description: "軽いコード修正やデバッグ相談を受け付ける"
    llm: "llm-code"
    context_sources: ["code"]
    priority: 3
```

---

## 3. ハブとの連携ポイント
- LLM Hub (`hub/llm_hub.py`) は intent を受け取り、`SkillRegistry.find_by_intent(intent)` で最適スキルを取得します。
- スキルに紐づく `llm` は `terrain.llm` へ接続され、EQNet の Policy が意図ごとに異なる制御を行います。
- `context_sources` はスキル実行前に準備する追加コンテキストを指し、RAG や SQL クエリなどを柔軟に差し込めます。

---

## 4. 運用のヒント
- 依頼頻度が高いタスク（例: RAG、SQL、コードレビュー）は専用 intent を用意し、個別に優先度を設定しましょう。
- スキル追加後はテレメトリやユーザーフィードバックを確認し、`description` や `priority` を定期的に調整します。
- 将来 GUI や外部ツールと連携する場合は、intent 名とトリガー条件を整理したシートを維持すると管理が楽になります。

---

スキル定義は軽量で柔軟です。EQNet の文化層や sensibility 設定（`config/sensibility.yaml`）と併用し、状況に応じて最適な対話スタイルと機能を切り替えてください。
