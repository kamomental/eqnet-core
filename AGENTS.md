# AGENTS.md

## Goal
EQNet 系の共生共鳴生命体に向けた `Inner OS` を構築する。
このリポジトリでは、LLM を本体状態モデルにせず、状態更新系の下流にある
表出層として扱う。

## Context
- このリポジトリは最終理論の固定版ではなく、研究用 OS である
- 既存 `inner_os` は flat な core 群として残っている
- 新規の module skeleton は既存 core を壊さず、共存可能な形で追加する

重要原則:
- LLM を主要な内部状態モデルにしない
- LLM は reportable foreground を受け取る expression bridge に留める

## Constraints
- 日本語のコメント / docstring / docs
- typed Python interfaces
- 可能な限りハードコードを避ける
- uncertainty は明示的に表現する
- continuity / person identity は stateful registry として扱う
- subjective / access / terrain の仮説差し替えを前提に疎結合で保つ
- raw observation を直接 LLM に流さない

## Architectural rules
### 4-layer compression
1. Grounding
2. State Core
3. Value & Access
4. Expression Bridge

### Core packages
- `inner_os/grounding`
- `inner_os/world_model`
- `inner_os/self_model`
- `inner_os/value_system`
- `inner_os/access`
- `inner_os/expression`
- `inner_os/memory`
- `inner_os/evaluation`

### Separation rules
以下は 1 つのファイルに潰さない:
- perception / grounding
- state update
- value computation
- access selection
- llm expression

## Coding rules
- 小さいファイルを優先
- `dataclass` / typed model を優先
- silent fallback を避ける
- ambiguity は ambiguity のまま保持する
- 既存 `inner_os` core と新規 skeleton の責務を混ぜない

## Testing rules
新規追加時は以下を含める:
- contract unit test
- state update smoke test
- replaceable model comparison fixture の入口

## Done when
1. module boundary が明示されている
2. interface が typed である
3. 新規 module の test がある
4. docs が更新されている
5. model behavior が変わる時は evaluation hook も追加されている
