# -*- coding: utf-8 -*-
"""
Run simple Granger causality tests on the exported time-series.

Usage:
    python scripts/granger_analysis.py --csv exports/timeseries.csv --out exports/granger_results.json --maxlag 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def load_timeseries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if not {"entropy", "enthalpy_mean", "rest_flag"}.issubset(df.columns):
        raise ValueError("CSV must contain entropy, enthalpy_mean, rest_flag columns.")
    df.sort_values("timestamp", inplace=True)
    df["date"] = df["timestamp"].dt.date
    daily = (
        df.groupby("date")
        .agg(
            entropy=("entropy", "mean"),
            enthalpy_mean=("enthalpy_mean", "mean"),
            rest_flag=("rest_flag", "max"),
        )
        .reset_index()
    )
    daily["rest_flag"] = daily["rest_flag"].astype(int)
    return daily


def run_granger(daily: pd.DataFrame, maxlag: int) -> dict:
    results = {}
    pairs = {
        "entropy_to_rest": ("entropy", "rest_flag"),
        "enthalpy_to_rest": ("enthalpy_mean", "rest_flag"),
        "rest_to_entropy": ("rest_flag", "entropy"),
        "rest_to_enthalpy": ("rest_flag", "enthalpy_mean"),
    }
    for name, (cause, effect) in pairs.items():
        data = daily[[effect, cause]].dropna().to_numpy(dtype=float)
        if data.shape[0] <= maxlag + 2:
            results[name] = {"error": "insufficient data"}
            continue
        try:
            test = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            results[name] = {
                f"lag_{lag}": {
                    "ssr_ftest_pvalue": float(test[lag][0]["ssr_ftest"][1]),
                    "params_ftest_pvalue": float(test[lag][0]["params_ftest"][1]),
                }
                for lag in range(1, maxlag + 1)
            }
        except Exception as exc:
            results[name] = {"error": str(exc)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to timeseries CSV")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    parser.add_argument("--maxlag", type=int, default=3, help="Maximum lag for Granger tests")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    daily = load_timeseries(csv_path)
    results = run_granger(daily, maxlag=args.maxlag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Granger results to {out_path}")


if __name__ == "__main__":
    main()
