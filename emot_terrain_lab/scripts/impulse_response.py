# -*- coding: utf-8 -*-
"""
Compute impulse response functions (IRFs) using a VAR model on the exported time series.

Usage:
    python scripts/impulse_response.py --csv exports/timeseries.csv --horizon 7 --lag 1 --out exports/irf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from statsmodels.tsa.api import VAR

try:  # pragma: no cover - allow running as script
    from scripts.granger_analysis import load_timeseries
except ImportError:
    from granger_analysis import load_timeseries


def compute_irf(daily: pd.DataFrame, lag: int, horizon: int) -> dict:
    model = VAR(daily[["entropy", "enthalpy_mean", "rest_flag"]])
    fitted = model.fit(lag)
    irf = fitted.irf(horizon)

    response = {}
    for shock_idx, shock_name in enumerate(["entropy", "enthalpy_mean", "rest_flag"]):
        response[shock_name] = {}
        for response_idx, response_name in enumerate(["entropy", "enthalpy_mean", "rest_flag"]):
            series = irf.irfs[:, response_idx, shock_idx].tolist()
            response[shock_name][response_name] = series
    return {
        "lag": lag,
        "horizon": horizon,
        "irf": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to timeseries CSV")
    parser.add_argument("--lag", type=int, default=1, help="VAR lag order")
    parser.add_argument("--horizon", type=int, default=7, help="Impulse response horizon")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    daily = load_timeseries(Path(args.csv))
    irf_payload = compute_irf(daily, lag=args.lag, horizon=args.horizon)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(irf_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved impulse response results to {out_path}")


if __name__ == "__main__":
    main()
