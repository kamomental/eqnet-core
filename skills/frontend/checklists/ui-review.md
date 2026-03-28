# UI Review Checklist

## 日本語と会話感

- 一文目で受け取った内容が分かる
- 相談窓口文体に寄りすぎていない
- `大丈夫です` などの反復が増えていない
- generic fallback に戻っていない

## 表出設計

- 内容反映が保護文に潰されていない
- short final が冷たく切れていない
- quiet presence が過剰説明になっていない
- route hack でなく inner_os readout が効いている

## UI / dashboard

- 情報のまとまりが読みやすい
- 文言が locale に集約されている
- mobile / narrow width でも崩れにくい
- 運用面で見たい state が snapshot に出ている

## 確認

- runtime 近傍テスト
- locale 変更の影響
- live 実走の違和感
