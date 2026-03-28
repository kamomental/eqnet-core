# Integration Upgrade Checklist

## 契約

- typed interface が追加・更新されている
- key 名が runtime / summary / docs で揃っている
- silent fallback を増やしていない

## 責務

- state update と expression bridge を混ぜていない
- summary / dashboard へ出す責務が明確
- raw observation を直接 LLM へ流していない

## 回帰

- py_compile
- contract test
- bootstrap
- runtime process hooks

## docs

- README / core docs の入口更新
- behavior change に対応する docs 更新
- evaluation hook の追加有無確認
