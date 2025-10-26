#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nightly Report Generator (P1)

Aggregates and renders a nightly snapshot:
- RAG quality: NDCG@10
- Diversity: H4 (Hill number proxy)
- Latency: p95 (ms)
- ETL: redundancy rate, reuse rate

Compares against a baseline (latest prior report by default), and emits
Markdown + SVG figures under reports/YYYY-MM-DD/ .
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # header lines like {"schema": ...} are tolerated
                if not (i == 0 and "schema" in line):
                    continue
    return rows


def _percent(v: float) -> str:
    return f"{v*100:.1f}%" if np.isfinite(v) else "NaN%"


def load_metrics(paths: Dict[str, str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    # RAG NDCG@10
    p = paths.get("rag")
    if p and Path(p).exists():
        rows = _read_jsonl(Path(p))
        out["ndcg10"] = np.array([r.get("ndcg@10", np.nan) for r in rows if isinstance(r, dict)])
    else:
        out["ndcg10"] = np.array([np.nan])
    # Diversity H4
    p = paths.get("div")
    if p and Path(p).exists():
        rows = _read_jsonl(Path(p))
        out["h4"] = np.array([r.get("h4", np.nan) for r in rows if isinstance(r, dict)])
    else:
        out["h4"] = np.array([np.nan])
    # Latency p95
    p = paths.get("lat")
    if p and Path(p).exists():
        rows = _read_jsonl(Path(p))
        out["p95"] = np.array([r.get("p95_ms", np.nan) for r in rows if isinstance(r, dict)])
    else:
        out["p95"] = np.array([np.nan])
    # ETL redundancy / reuse
    p = paths.get("etl")
    if p and Path(p).exists():
        rows = _read_jsonl(Path(p))
        out["redund_rate"] = np.array([r.get("redundancy_rate", np.nan) for r in rows if isinstance(r, dict)])
        out["reuse_rate"] = np.array([r.get("reuse_rate", np.nan) for r in rows if isinstance(r, dict)])
    else:
        out["redund_rate"] = np.array([np.nan])
        out["reuse_rate"] = np.array([np.nan])
    return out


def summarize(m: Dict[str, np.ndarray]) -> Dict[str, float]:
    def mean_ok(a: np.ndarray) -> float:
        if a is None:
            return float("nan")
        a = a[~np.isnan(a)]
        return float(np.mean(a)) if a.size else float("nan")

    return {
        "ndcg10": mean_ok(m.get("ndcg10")),
        "h4": mean_ok(m.get("h4")),
        "p95": mean_ok(m.get("p95")),
        "redund": mean_ok(m.get("redund_rate")),
        "reuse": mean_ok(m.get("reuse_rate")),
    }


def rel_delta(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    denom = max(abs(b), 1e-9)
    return (a - b) / denom


def find_latest_report_dir(base: Path) -> Path | None:
    cand = sorted([p for p in base.glob("*") if p.is_dir()], reverse=True)
    return cand[0] if cand else None


def ensure_dirs(report_dir: Path) -> None:
    (report_dir / "assets").mkdir(parents=True, exist_ok=True)


def plot_figures(current: Dict[str, np.ndarray], baseline: Dict[str, np.ndarray] | None, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        # If matplotlib is not available, skip plots gracefully
        return
    # NDCG hist
    c = current.get("ndcg10", np.array([]))
    b = baseline.get("ndcg10", np.array([])) if baseline else None
    plt.figure(figsize=(6, 4))
    if c.size:
        plt.hist(c[~np.isnan(c)], bins=20, alpha=0.8, label="current")
    if b is not None and b.size:
        plt.hist(b[~np.isnan(b)], bins=20, alpha=0.4, label="baseline")
    plt.xlabel("NDCG@10"); plt.ylabel("count"); plt.title("NDCG@10 distribution"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "assets" / "ndcg_hist.svg"); plt.close()
    # H4 bar
    s_curr = summarize(current); s_base = summarize(baseline) if baseline else {"h4": float("nan")}
    plt.figure(figsize=(5, 4))
    xs = ["current"]; ys = [s_curr.get("h4", float("nan"))]
    if np.isfinite(s_base.get("h4", float("nan"))):
        xs = ["baseline", "current"]; ys = [s_base["h4"], s_curr["h4"]]
    plt.bar(xs, ys); plt.ylabel("H4 diversity"); plt.title("Diversity (H4)"); plt.tight_layout()
    plt.savefig(out_dir / "assets" / "h4_bar.svg"); plt.close()
    # p95 latency box
    plt.figure(figsize=(6, 4))
    data = [current.get("p95", np.array([]))[~np.isnan(current.get("p95", np.array([])))]]; labels = ["current"]
    if baseline and baseline.get("p95") is not None:
        bdat = baseline["p95"][~np.isnan(baseline["p95"])]
        if bdat.size:
            data = [bdat, data[0]]; labels = ["baseline", "current"]
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("p95 latency (ms)"); plt.title("Latency (p95)"); plt.tight_layout()
    plt.savefig(out_dir / "assets" / "latency_box.svg"); plt.close()


def adopt_decision(s_curr: Dict[str, float], s_base: Dict[str, float]) -> tuple[str, str]:
    ndcg_up = rel_delta(s_curr.get("ndcg10", np.nan), s_base.get("ndcg10", np.nan))
    h4_up = rel_delta(s_curr.get("h4", np.nan), s_base.get("h4", np.nan))
    lat_up = rel_delta(s_curr.get("p95", np.nan), s_base.get("p95", np.nan))  # 上がる=悪化
    if (np.isfinite(ndcg_up) and ndcg_up >= 0.05) and (np.isfinite(h4_up) and h4_up >= 0.10) and (np.isfinite(lat_up) and lat_up <= 0.0):
        return "ADOPT", f"NDCG +{_percent(ndcg_up)}, H4 +{_percent(h4_up)}, p95 ≤ baseline"
    if (np.isfinite(ndcg_up) and ndcg_up <= -0.02) or (np.isfinite(h4_up) and h4_up <= -0.10) or (np.isfinite(lat_up) and lat_up >= 0.05):
        return "ROLLBACK", f"NDCG {_percent(ndcg_up)}, H4 {_percent(h4_up)}, p95 {_percent(lat_up)}"
    return "KEEP", f"NDCG {_percent(ndcg_up)}, H4 {_percent(h4_up)}, p95 {_percent(lat_up)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current_dir", default="eval/current", help="metrics dir for today")
    ap.add_argument("--baseline_dir", default="", help="baseline metrics dir (optional)")
    ap.add_argument("--report_dir", default="reports", help="reports output root")
    args = ap.parse_args()

    today = dt.datetime.now().strftime("%Y-%m-%d")
    report_root = Path(args.report_dir) / today
    ensure_dirs(report_root)

    def pack(base: Path) -> Dict[str, str]:
        return {
            "rag": str(base / "rag_results.jsonl"),
            "div": str(base / "diversity.jsonl"),
            "lat": str(base / "latency.jsonl"),
            "etl": str(base / "etl_stats.jsonl"),
        }

    curr_paths = pack(Path(args.current_dir))
    base_dir = Path(args.baseline_dir) if args.baseline_dir else find_latest_report_dir(Path(args.report_dir))
    base_paths = pack(base_dir / "artifacts") if base_dir else {}

    current = load_metrics(curr_paths)
    baseline = load_metrics(base_paths) if base_dir else None
    s_curr = summarize(current)
    s_base = summarize(baseline) if baseline else {"ndcg10": float("nan"), "h4": float("nan"), "p95": float("nan"), "redund": float("nan"), "reuse": float("nan")}

    plot_figures(current, baseline, report_root)
    decision, reason = adopt_decision(s_curr, s_base)

    header_path = Path("reports/templates/header.md")
    header = header_path.read_text(encoding="utf-8") if header_path.exists() else "# Nightly Report\n"
    md = []
    md.append(header.strip())
    md.append(f"\n**Date:** {today}\n")
    md.append("## Summary\n")
    md.append(f"- Decision: **{decision}**  \n- Reason: {reason}\n")
    md.append("## Metrics\n")
    md.append(f"- NDCG@10: current={s_curr['ndcg10']:.3f} / baseline={s_base['ndcg10']:.3f}\n")
    md.append(f"- H4 diversity: current={s_curr['h4']:.3f} / baseline={s_base['h4']:.3f}\n")
    md.append(f"- p95 latency (ms): current={s_curr['p95']:.1f} / baseline={s_base['p95']:.1f}\n")
    md.append(f"- ETL redundancy: current={_percent(s_curr['redund'])} / baseline={_percent(s_base['redund'])}\n")
    md.append(f"- ETL reuse: current={_percent(s_curr['reuse'])} / baseline={_percent(s_base['reuse'])}\n")
    md.append("\n## Figures\n")
    md.append("![NDCG@10](assets/ndcg_hist.svg)\n")
    md.append("![H4](assets/h4_bar.svg)\n")
    md.append("![Latency p95](assets/latency_box.svg)\n")

    out_md = report_root / f"{today}.md"
    (report_root / "artifacts").mkdir(exist_ok=True, parents=True)
    # copy current artifacts into report directory for baseline reference
    for key, p in curr_paths.items():
        try:
            src = Path(p)
            if src.exists():
                dst = report_root / "artifacts" / src.name
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"✓ Report written: {out_md}")
    print(f"✓ Assets: {(report_root / 'assets').as_posix()}")


if __name__ == "__main__":
    main()

