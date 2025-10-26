# Terrain Log Compaction & Retention Gameplan

## 0. 目的
- 60GB級 `terrain.json` を安全に保持しつつ、日々の可観測性・検索を確保する
- Sunyata / Candor / Senseレシートで増え続けるログを HOT/WARM/COLD に整理

---

## 1. 原本の可逆退避（zstd + NDJSON）
```bash
# 可逆圧縮
zstd -19 --rm terrain.json            # => terrain.json.zst

# 巨大配列なら NDJSON へ展開
zstd -dc terrain.json.zst | jq -c '.[]' | zstd -19 > terrain.jsonl.zst

# 1GBローテーション（安全ネット）
zstd -dc terrain.jsonl.zst | split -d -a 4 -b 1024m - terrain.jsonl.part.
```

ポイント: 以降の全パイプラインは NDJSON (行単位) を前提にする。

---

## 2. Parquet(ZSTD) + 日付パーティション化
### DuckDB 一発変換
```bash
python -m ops.terrain_compact \
  --input 'logs/terrain/terrain.jsonl.part.*' \
  --output dataset/terrain_parquet \
  --timestamp-column ts \
  --threads 8
```

内部では
```sql
COPY (
  SELECT *,
         CAST(ts AS TIMESTAMP)                       AS ts_ts,
         CAST(date_trunc('day', CAST(ts AS TIMESTAMP)) AS DATE) AS date
  FROM read_ndjson_auto('logs/terrain/terrain.jsonl.part.*')
) TO 'dataset/terrain_parquet'
  (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (date));
```

### 読み込み例（DuckDB / Polars / PyArrow 共通）
```sql
SELECT count(*)
FROM read_parquet('dataset/terrain_parquet/*/*.parquet')
WHERE date BETWEEN DATE '2025-10-01' AND DATE '2025-10-25';
```
→ Hive 風 `date=YYYY-MM-DD/` ディレクトリがフィルタで自動プルーニング。

---

## 3. HOT / WARM / COLD 運用の目安
|層|期間|保持内容|出力例|
|---|---|---|---|
|HOT|0-14日|Sunyata.do_topk / self.roles / s_index / candor / gaze.referent|`dataset/terrain_parquet/`|
|WARM|15-90日|日次ロールアップ (p50/p95, misref率, candor分布)|`dataset/terrain_rollup/`|
|COLD|90日超|統計スケッチ (t-digest, HLL, Count-Min) + 小サンプル|`dataset/terrain_cold/`|

Nightly自動化（例）
```bash
python ops/nightly.py --terrain-compact \
  --hot-days 14 --warm-days 90 \
  --parquet-root dataset/terrain_parquet \
  --rollup-out dataset/terrain_rollup
```
※ nightly.py のオプション化は TODO。現状は ops/terrain_compact.py を直接実行。

---

## 4. よくある落とし穴
- **日付が文字列のまま** → DATE型にキャストしてから `PARTITION_BY (date)`
- **パーティションなしで書き出し** → すべてフルスキャンに。必ず partition を指定
- **圧縮混在** → ZSTD で統一。古い Spark を併用する場合のみ SNAPPY を検討
- **フィールド持ち過ぎ** → HOT 期間を過ぎたら engine_trace / protected_spans など重い列を落とす

---

## 5. クエリテンプレ
```sql
-- Candor 必要度の推移
SELECT date, candor.level, COUNT(*) AS n
FROM read_parquet('dataset/terrain_parquet/*/*.parquet')
GROUP BY 1,2 ORDER BY 1,2;

-- Clinging 警報頻度
SELECT date, SUM((sunyata.clinging_triggered)::INT) AS triggered
FROM read_parquet('dataset/terrain_parquet/*/*.parquet')
GROUP BY 1 ORDER BY 1;

-- 共同注視の誤参照率
SELECT date, AVG((gaze.misrefer)::INT) AS misref_rate
FROM read_parquet('dataset/terrain_parquet/*/*.parquet')
WHERE kind='receipt'
GROUP BY 1 ORDER BY 1;
```

---

## 6. まとめ
- zstd 可逆圧縮 → NDJSON 行形式 → Parquet(ZSTD) + Hiveパーティション
- ops/terrain_compact.py で一括変換、DuckDB/Polars/PyArrow から秒単位クエリ
- Nightly で HOT/WARM/COLD をGCし、Sunyata/S-index の可観測性を保つ
