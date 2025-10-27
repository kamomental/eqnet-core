## FastMCP / Agent-to-Agent Integration Guide

### 1. サーバ起動
```bash
EQNET_NIGHTLY_PATH=reports/nightly.json \
EQNET_A2A_LOG_DIR=logs/a2a \
uvicorn scripts.run_mcp_agent:app --host 0.0.0.0 --port 8055 --reload
```

オプション:
- `EQNET_MCP_READ_ONLY=1` を指定すると書き込み系ツール（telemetry / a2a-turn / score / close）を無効化できます。

公開される主なエンドポイント
- `GET /mcp/capabilities` … 利用可能なリソース／ツール一覧
- `GET /mcp/resources/resonance/summary` … Nightly 由来の共鳴サマリ
- `GET /mcp/resources/vision/snapshot` … Vision 集計（counts / pose など）
- `GET /mcp/resources/culture/feedback` … politeness / intimacy 操作ログ
- `POST /mcp/tools/telemetry/vision.push` … Vision イベントを telemetry に送信
- `POST /mcp/tools/a2a/contract.open` … セッション契約（scopes / guardrails / expires_at）
- `POST /mcp/tools/a2a/turn.post` … ターン投げ込み
- `POST /mcp/tools/a2a/score.report` … 候補スコア共有
- `GET /mcp/a2a/session/{session_id}` … 監査スナップショット取得

### 2. Vision からの呼び出し例
```bash
curl -X POST http://localhost:8055/mcp/tools/telemetry/vision.push \
  -H "Content-Type: application/json" \
  -d '{"ts_ms": 1730000000, "counts_by_kind": {"bow": 3}}'
```

### 3. Agent-to-Agent セッション例
```bash
curl -X POST http://localhost:8055/mcp/tools/a2a/contract.open \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sess-example",
    "version": "2025.10",
    "intent": "delegate.route_plan",
    "from": "planner-A",
    "to": "eqnet",
    "scopes": ["read:resonance", "read:vision"],
    "guardrails": {"max_steps": 6, "tool_timeout_s": 8, "no_recursive": true},
    "expires_at": "2025-10-29T00:00:00Z"
  }'
```
1) `turn.post` でターンを進め、2) `score.report` で合意候補を共有、3) 必要に応じて `session.close` で終了。
監査ログは `logs/a2a/<session_id>.jsonl` に追記され、Nightly からも参照できます。
