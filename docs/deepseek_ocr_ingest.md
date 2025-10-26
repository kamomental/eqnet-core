DeepSeek‑OCR Ingest (Vision→Markdown→RAG)

概要
- DeepSeek‑OCR 等の「視覚→Markdown」圧縮出力を EQNet の RAG にそのまま取り込む最小経路を提供します。
- HTTP/CLI のどちらでも接続可能。環境変数で URL/コマンドを設定できます。

環境変数
- HTTP: `DEEPSEEK_OCR_ENDPOINT` に推論エンドポイント（例: `http://localhost:8080/infer`）
- CLI:  `DEEPSEEK_OCR_BIN` に CLI 実行ファイル名（例: `deepseek-ocr`、フルパス可）

使い方
1) HTTP サーバ接続
   - 環境変数を設定（PowerShell）
     `$env:DEEPSEEK_OCR_ENDPOINT = "http://localhost:8080/infer"`
   - 取り込み
     `python scripts/ingest_pdf_to_rag.py --pdf_dir ./docs --backend deepseek-http`

2) CLI 接続
   - 環境変数を設定（任意）
     `$env:DEEPSEEK_OCR_BIN = "deepseek-ocr"`
   - 取り込み
     `python scripts/ingest_pdf_to_rag.py --pdf_dir ./docs --backend deepseek-cli`

3) 引数で直接指定（環境変数不要）
   - HTTP: `--backend deepseek-http --endpoint http://localhost:8080/infer`
   - CLI:  `--backend deepseek-cli  --bin deepseek-ocr`

実装箇所
- バックエンド: `emot_terrain_lab/ingest/deepseek_backend.py:1`
- 入口: `emot_terrain_lab/ingest/vision_to_md.py:1`
- スクリプト: `scripts/ingest_pdf_to_rag.py:1`

メモ
- このリポではネットワーク・GPU に依存しないように疑似エンベディングを同梱しています。品質検証時には実エンベディングへ差し替えてください。
- Markdown を章単位（`##` 区切り）で分割し、RAG へ upsert します。JSON/表/数式はそのままテキスト化されます。

