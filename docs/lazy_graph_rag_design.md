LazyGraphRAG 最小設計（EQNet向け）

目的
- 事前の大規模 index 化を避け、必要な時だけ検索する
- 「関係性（Graph）」を一次情報として扱い、RAGは補助にする

結論
- index は必須ではない
- 小規模・検証段階は Lazy で十分
- 将来のスケールでのみ index を導入すれば良い

全体像（3レイヤ）
1) Graph 層（主）
- 人・出来事・関係性をノードとエッジで保持
- 例: 家族 / 友人 / 仕事 / 重要イベント / 感情タグ

2) Lazy Retrieval 層（補助）
- クエリが来た時だけ、必要範囲を探索
- 例: Graph で候補を絞り、JSONL/Parquet を走査

3) 生成層（LLM）
- 取り出した文脈だけを与えて応答
- RAGは「必要な時だけ」差し込む

最小構成（実装の最短）
- ストレージ: JSONL / Parquet
- Graph: DO-Graph or 単純な adjacency JSON
- 検索:
  - 1段目: Graph で候補 ID を絞る
  - 2段目: 候補 ID に一致する記録だけ抽出

動作フロー
1) 受信: ユーザ発話 / 意図
2) Graph 参照: 関係ノードを絞る（例: family, work, past_event）
3) Lazy 検索: 対象 ID の記録だけ抽出
4) 応答: 抽出文脈 + EQNet の判断で返す

LazyGraphRAG に向くケース
- データが小さい / 中規模
- 「関係性」が重要（親密度・文化・共鳴）
- 反応の正確さより「意味の一貫性」を優先したい

index を導入すべきタイミング
- データが数十万以上に増えた
- p95 レイテンシが許容を超えた
- 同時アクセスが増えた

移行の仕方（段階的）
Step 1: Lazy 構成で運用開始
Step 2: 利用頻度の高い部分だけ index 化（warm cache）
Step 3: 全体 index へ段階移行

EQNet との整合
- boundary / decision / resonance の分離運用に干渉しない
- Graph が文脈の意味を決める
- RAGは「必要な時だけ」補助線として使う

要点の一言
「Graph が意味を決め、RAG は必要な時だけ補う」。

最終運用コード（回帰テスト用）
python - <<'PY'
from pathlib import Path
from collections import Counter
import re, json
from emot_terrain_lab.rag.lazy_rag import LazyRAG, LazyRAGConfig

GRAPH = Path("data/state/memory_graph.json")
JSONL = Path("data/logs.jsonl")

# ===== ここだけ編集 =====
ANCHORS = [
  {"anchor_id":"turn-001", "expect":"実装メモ（在庫差分検知）"},
  {"anchor_id":"turn-002", "expect":"Decision相当：Graph主導・RAG補助の方針決定"},
  {"anchor_id":"turn-003", "expect":"Lesson相当：ID正規化不一致の原因と対処"},
]
# =======================

TOPK_KW = 12
N_QUERIES = 5
MIN_TERMS, MAX_TERMS = 2, 4

WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30FF\u3400-\u9FFF]+")

def toks(s): return [t for t in WORD_RE.findall((s or "").strip()) if len(t) >= 2]
def pick_kw(text):
    return [w for w,_ in Counter(toks(text)).most_common(TOPK_KW)]

def make_queries(kw):
    qs=[]
    for k in range(MIN_TERMS, MAX_TERMS+1):
        for i in range(0, len(kw)-k+1):
            qs.append(" ".join(kw[i:i+k]))
            if len(qs) >= N_QUERIES: return qs
    return qs

def row_id(r):
    return str(r.get("turn_id") or r.get("node_id") or r.get("id") or "")

def row_text(r):
    if not r:
        return ""
    for k in ("text","message","content","summary","note","body"):
        v = r.get(k)
        if isinstance(v, str) and v.strip(): return v.strip()
    d = r.get("data")
    if isinstance(d, dict):
        for k in ("text","message","content","summary","note","body"):
            v = d.get(k)
            if isinstance(v, str) and v.strip(): return v.strip()
    return ""

def graph_title(node_id):
    if not GRAPH.exists(): return ""
    g = json.loads(GRAPH.read_text(encoding="utf-8-sig"))
    for n in (g.get("nodes") or []):
        nid = str(n.get("id") or n.get("node_id") or "")
        if nid == node_id:
            return str(n.get("title") or n.get("text") or n.get("summary") or "")
    return ""

def jsonl_find(node_id):
    if not JSONL.exists(): return None
    for line in JSONL.read_text(encoding="utf-8-sig").splitlines():
        line=line.strip()
        if not line: continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if row_id(r) == node_id: return r
    return None

cfg = LazyRAGConfig(graph_path=GRAPH, memory_jsonl_path=JSONL)
rag = LazyRAG(cfg)

overall = True
print("=== LazyGraphRAG Regression (short) ===")

for a in ANCHORS:
    aid = a["anchor_id"]
    base = (graph_title(aid) + "\n" + row_text(jsonl_find(aid))).strip()
    kw = pick_kw(base)
    qs = make_queries(kw)

    ok = False
    for q in qs:
        cand = rag.candidate_ids(q)
        matched = rag.fetch_rows_by_ids(cand)
        if any(row_id(r) == aid for r in matched):
            ok = True
            break

    print(f"- {aid}  {'PASS' if ok else 'FAIL'}  expect={a.get('expect','')}")
    overall = overall and ok

print("OVERALL:", "PASS" if overall else "FAIL")
PY
