# Inner OS Bootstrap Task

## Goal
この repo に 4層圧縮の Inner OS skeleton を追加する。

## Context
- repo は research OS であり monolithic final model ではない
- LLM は expression layer に留める
- 既存 `inner_os` flat core と共存できる構成を優先する

## Constraints
- 日本語 comments / docs
- typed Python interfaces
- explicit uncertainty
- continuity / person identity は registry として持つ
- raw observation を直接 LLM へ流さない

## Done when
1. `inner_os/` 配下に bootstrap package tree ができる
2. typed model が定義される
3. stub update functions がある
4. smoke test が通る
5. migration note が追加される
