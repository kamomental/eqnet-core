"""Pairwise resonance analytics for EQNet telemetry logs."""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


WINDOW_FUNCTIONS = {"none", "hann"}


@dataclass
class SeriesData:
    label: str
    time: np.ndarray  # seconds or step index
    values: np.ndarray


def _load_rho_series(label: str, log_path: Path) -> SeriesData:
    """Extract time/rho series from a telemetry JSONL log."""
    if not log_path.exists():
        raise FileNotFoundError(f"telemetry log not found: {log_path}")
    times: List[float] = []
    values: List[float] = []
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "field.metrics":
            continue
        data = row.get("data") or {}
        ts = row.get("ts")
        if ts is None:
            ts = data.get("step")
        if ts is None:
            continue
        try:
            rho = float(data.get("rho"))
        except (TypeError, ValueError):
            continue
        try:
            times.append(float(ts))
        except (TypeError, ValueError):
            continue
        values.append(rho)
    time_arr = np.asarray(times, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    if time_arr.size == 0:
        return SeriesData(label, np.array([], dtype=float), np.array([], dtype=float))
    order = np.argsort(time_arr)
    return SeriesData(label, time_arr[order], value_arr[order])


def _detrend(series: np.ndarray) -> np.ndarray:
    if series.size < 3:
        return series
    x = np.arange(series.size, dtype=float)
    slope, intercept = np.polyfit(x, series, 1)
    return series - (slope * x + intercept)


def _apply_window(series: np.ndarray, window: str) -> np.ndarray:
    if window == "hann" and series.size >= 2:
        return series * np.hanning(series.size)
    return series


def _zscore(series: np.ndarray) -> np.ndarray:
    mean = float(np.mean(series))
    std = float(np.std(series))
    if std <= 1e-12:
        return series - mean
    return (series - mean) / std


def _first_lag_autocorr(series: np.ndarray) -> float:
    if series.size < 2:
        return 0.0
    mean = float(np.mean(series))
    numerator = float(np.sum((series[:-1] - mean) * (series[1:] - mean)))
    denominator = float(np.sum((series - mean) ** 2))
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a_sel = a[mask]
    b_sel = b[mask]
    if a_sel.size < 2 or b_sel.size < 2:
        return float("nan")
    try:
        return float(np.corrcoef(a_sel, b_sel)[0, 1])
    except Exception:
        return float("nan")


def _residual(series: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if controls.size == 0:
        return series - np.mean(series)
    X = np.column_stack([np.ones(series.size), controls])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, series, rcond=None)
        pred = X @ beta
    except np.linalg.LinAlgError:
        pred = np.zeros_like(series)
    return series - pred


def _partial_corr(series_a: np.ndarray, series_b: np.ndarray, controls: Optional[np.ndarray]) -> float:
    if controls is None or controls.size == 0:
        return _safe_corr(series_a, series_b)
    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    if controls.shape[0] != series_a.size:
        return _safe_corr(series_a, series_b)
    residual_a = _residual(series_a, controls)
    residual_b = _residual(series_b, controls)
    return _safe_corr(residual_a, residual_b)


def _lag_matrix(values: np.ndarray, lag: int) -> np.ndarray:
    n = values.size
    if lag < 1 or n <= lag:
        return np.empty((0, lag), dtype=float)
    rows = n - lag
    mat = np.empty((rows, lag), dtype=float)
    for j in range(lag):
        mat[:, j] = values[lag - j - 1 : n - j - 1]
    return mat


def _ols_sse(y: np.ndarray, X: np.ndarray) -> Optional[float]:
    if X.size == 0 or y.size == 0 or X.shape[0] != y.size:
        return None
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    resid = y - X @ beta
    return float(np.dot(resid, resid))


def _granger_f(predictor: np.ndarray, target: np.ndarray, lag: int) -> float:
    if lag < 1 or predictor.size <= lag or target.size <= lag:
        return float("nan")
    Y = target[lag:]
    X_y = _lag_matrix(target, lag)
    X_x = _lag_matrix(predictor, lag)
    if X_y.size == 0 or X_x.size == 0 or X_y.shape[0] != Y.size:
        return float("nan")
    X_restricted = np.column_stack([np.ones(Y.size), X_y])
    sse_r = _ols_sse(Y, X_restricted)
    if sse_r is None:
        return float("nan")
    X_full = np.column_stack([X_restricted, X_x])
    sse_f = _ols_sse(Y, X_full)
    if sse_f is None:
        return float("nan")
    if sse_r <= sse_f:
        return 0.0
    d = X_full.shape[1] - X_restricted.shape[1]
    n_obs = Y.size
    if d <= 0 or n_obs <= X_full.shape[1]:
        return float("nan")
    denom = sse_f / max(n_obs - X_full.shape[1], 1)
    if denom <= 1e-12:
        return float("nan")
    num = (sse_r - sse_f) / d
    return float(max(num / denom, 0.0))


def _max_cross_correlation(a: np.ndarray, b: np.ndarray, dt: float) -> Tuple[float, int, float]:
    """Return (peak_correlation, lag_samples, lag_refined_time)."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan"), 0, float("nan")
    series_a = a[mask]
    series_b = b[mask]
    if series_a.size < 2 or series_b.size < 2:
        return float("nan"), 0, float("nan")
    a_center = series_a - series_a.mean()
    b_center = series_b - series_b.mean()
    denom = np.sqrt(np.sum(a_center ** 2) * np.sum(b_center ** 2))
    if denom <= 1e-12:
        return float("nan"), 0, float("nan")
    corr_full = np.correlate(a_center, b_center, mode="full") / denom
    idx = int(np.argmax(corr_full))
    lag_samples = idx - (len(b_center) - 1)
    refined_samples = float(lag_samples)
    if 0 < idx < len(corr_full) - 1:
        y1, y2, y3 = corr_full[idx - 1], corr_full[idx], corr_full[idx + 1]
        denom_parab = (y1 - 2 * y2 + y3)
        if abs(denom_parab) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom_parab
            refined_samples = lag_samples + float(np.clip(delta, -1.0, 1.0))
    return float(corr_full[idx]), lag_samples, float(refined_samples * dt)


def _fisher_ci(r: float, n_eff: float) -> Tuple[float, float]:
    if not math.isfinite(r) or n_eff <= 3.0 or abs(r) >= 0.999999:
        return float("nan"), float("nan")
    try:
        z = 0.5 * math.log((1 + r) / (1 - r))
    except ZeroDivisionError:
        return float("nan"), float("nan")
    se = 1.0 / math.sqrt(max(n_eff - 3.0, 1e-6))
    delta = 1.96 * se
    lower = math.tanh(z - delta)
    upper = math.tanh(z + delta)
    return float(lower), float(upper)


def _resample_series(data: SeriesData, start: float, stop: float, step: float) -> np.ndarray:
    grid = np.arange(start, stop + step * 0.5, step, dtype=float)
    if grid.size == 0:
        return grid, np.array([], dtype=float)
    values = np.interp(grid, data.time, data.values)
    return grid, values


def _prepare_pair_series(
    a: SeriesData,
    b: SeriesData,
    *,
    resample_ms: Optional[float],
    zscore: bool,
    detrend: bool,
    window: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if a.time.size < 2 or b.time.size < 2:
        return np.array([]), np.array([]), np.array([]), 1.0
    if resample_ms:
        dt = resample_ms / 1000.0
    else:
        dt_a = np.median(np.diff(a.time)) if a.time.size > 1 else 1.0
        dt_b = np.median(np.diff(b.time)) if b.time.size > 1 else 1.0
        dt = float(min(max(dt_a, 1e-6), max(dt_b, 1e-6)))
    start = max(a.time[0], b.time[0])
    stop = min(a.time[-1], b.time[-1])
    if stop - start < 2 * dt:
        return np.array([]), np.array([]), np.array([]), dt
    grid, series_a = _resample_series(a, start, stop, dt)
    _, series_b = _resample_series(b, start, stop, dt)
    if grid.size < 3:
        return np.array([]), np.array([]), np.array([]), dt
    if detrend:
        series_a = _detrend(series_a)
        series_b = _detrend(series_b)
    if zscore:
        series_a = _zscore(series_a)
        series_b = _zscore(series_b)
    if window and window != "none":
        series_a = _apply_window(series_a, window)
        series_b = _apply_window(series_b, window)
    mask = np.isfinite(series_a) & np.isfinite(series_b)
    if mask.sum() < 3:
        return np.array([]), np.array([]), np.array([]), dt
    return grid[mask], series_a[mask], series_b[mask], dt


def compute_resonance_metrics(
    logs: Sequence[Tuple[str, Path]],
    *,
    resample_ms: Optional[float] = None,
    zscore: bool = False,
    detrend: bool = False,
    window: str = "none",
    alpha: float = 0.0,
    beta: float = 0.0,
    granger_lag: int = 1,
    return_series: bool = False,
) -> Dict[str, object]:
    """Compute pairwise resonance metrics with advanced options."""
    if window not in WINDOW_FUNCTIONS:
        raise ValueError(f"window must be one of {WINDOW_FUNCTIONS}")
    granger_lag = max(int(granger_lag), 0)
    series_map: Dict[str, SeriesData] = {}
    for label, path in logs:
        series_map[label] = _load_rho_series(label, path)

    pairs: List[Dict[str, object]] = []
    detail: Dict[str, Dict[str, List[float]]] = {}
    labels = list(series_map.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_label = labels[i]
            b_label = labels[j]
            data_a = series_map[a_label]
            data_b = series_map[b_label]
            grid, series_a, series_b, dt = _prepare_pair_series(
                data_a,
                data_b,
                resample_ms=resample_ms,
                zscore=zscore,
                detrend=detrend,
                window=window,
            )
            if grid.size < 3:
                pairs.append(
                    {
                        "agents": [a_label, b_label],
                        "rho_corr": float("nan"),
                        "rho_corr_ci95": [float("nan"), float("nan")],
                        "rho_cross_corr_peak": float("nan"),
                        "rho_cross_corr_lag": float("nan"),
                        "rho_cross_corr_lag_refined": float("nan"),
                        "energy": float("nan"),
                        "n_eff": 0.0,
                        "objective": float("nan"),
                        "lengths": [int(data_a.values.size), int(data_b.values.size)],
                        "note": "insufficient_overlap",
                    }
                )
                continue

            corr_rho = _safe_corr(series_a, series_b)
            ac1_a = _first_lag_autocorr(series_a)
            ac1_b = _first_lag_autocorr(series_b)
            n_eff = grid.size * (1 - ac1_a * ac1_b) / max(1 + ac1_a * ac1_b, 1e-6)
            n_eff = float(max(n_eff, 2.0))
            ci_low, ci_high = _fisher_ci(corr_rho, n_eff)
            peak, lag_samples, lag_refined = _max_cross_correlation(series_a, series_b, dt)
            energy = float(np.mean((series_a - series_b) ** 2))
            objective = (
                corr_rho
                - float(alpha) * abs(lag_refined if math.isfinite(lag_refined) else 0.0)
                - float(beta) * energy
            )
            controls_matrix = None
            if len(labels) > 2:
                control_series = []
                start = grid[0]
                stop = grid[-1]
                for other_label in labels:
                    if other_label in (a_label, b_label):
                        continue
                    other_data = series_map[other_label]
                    _, series_c = _resample_series(other_data, start, stop, dt)
                    if series_c.size != grid.size:
                        continue
                    if detrend:
                        series_c = _detrend(series_c)
                    if zscore:
                        series_c = _zscore(series_c)
                    if window and window != "none":
                        series_c = _apply_window(series_c, window)
                    control_series.append(series_c)
                if control_series:
                    controls_matrix = np.column_stack([c[mask] for c in control_series])
            partial_corr = _partial_corr(series_a, series_b, controls_matrix)
            granger_ab = _granger_f(series_a, series_b, granger_lag)
            granger_ba = _granger_f(series_b, series_a, granger_lag)
            pair_entry = {
                "agents": [a_label, b_label],
                "rho_corr": corr_rho,
                "rho_corr_ci95": [ci_low, ci_high],
                "rho_cross_corr_peak": peak,
                "rho_cross_corr_lag": lag_samples,
                "rho_cross_corr_lag_refined": lag_refined,
                "energy": energy,
                "n_eff": n_eff,
                "objective": objective,
                "partial_corr": partial_corr,
                "granger": {
                    "lag": int(granger_lag),
                    "a_to_b_f": granger_ab,
                    "b_to_a_f": granger_ba,
                },
                "lengths": [int(data_a.values.size), int(data_b.values.size)],
            }
            if return_series:
                pair_entry["time"] = grid.tolist()
                pair_entry["rho_a"] = series_a.tolist()
                pair_entry["rho_b"] = series_b.tolist()
            pairs.append(pair_entry)

    result: Dict[str, object] = {"schema": "resonance.v1", "pairs": pairs}
    if return_series:
        result["series"] = {
            label: {"time": data.time.tolist(), "rho": data.values.tolist()}
            for label, data in series_map.items()
        }
    return result


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute resonance metrics from telemetry logs.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="List of log paths (label=path or raw path, glob allowed).",
    )
    parser.add_argument("--out", type=str, default="reports/resonance.json", help="Output JSON path.")
    parser.add_argument("--plots-dir", type=str, default="", help="Optional directory for plots.")
    parser.add_argument("--resample-ms", type=float, default=None, help="Resample interval in milliseconds.")
    parser.add_argument("--zscore", action="store_true", help="Apply z-score normalisation.")
    parser.add_argument("--detrend", action="store_true", help="Remove linear trend before analysis.")
    parser.add_argument("--window", choices=sorted(WINDOW_FUNCTIONS), default="none", help="Window function.")
    parser.add_argument("--alpha", type=float, default=0.0, help="Objective coefficient for |lag|.")
    parser.add_argument("--beta", type=float, default=0.0, help="Objective coefficient for energy term.")
    parser.add_argument("--granger-lag", type=int, default=1, help="Lag (in samples) for Granger causality F-stat.")
    parser.add_argument("--matrix", action="store_true", help="Emit correlation matrix for all agents.")
    return parser.parse_args(argv)


def _normalize_log_specs(values: Sequence[str]) -> List[Tuple[str, Path]]:
    results: List[Tuple[str, Path]] = []
    for entry in values:
        expanded = glob.glob(entry)
        paths = expanded or [entry]
        for item in paths:
            if "=" in item:
                label, path_str = item.split("=", 1)
                label = label.strip()
                path = Path(path_str.strip())
            else:
                path = Path(item)
                label = path.stem
            results.append((label or path.stem, path))
    return results


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    logs = _normalize_log_specs(args.logs)
    metrics = compute_resonance_metrics(
        logs,
        resample_ms=args.resample_ms,
        zscore=args.zscore,
        detrend=args.detrend,
        window=args.window,
        alpha=args.alpha,
        beta=args.beta,
        granger_lag=args.granger_lag,
        return_series=bool(args.plots_dir),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pair_count = len(metrics.get("pairs", []))
    print(f"[resonance] wrote {pair_count} pair metrics -> {out_path}")

    if args.matrix:
        labels = [label for label, _ in logs]
        corr_matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
        for pair in metrics.get("pairs", []):
            i = labels.index(pair["agents"][0])
            j = labels.index(pair["agents"][1])
            corr_matrix[i, j] = corr_matrix[j, i] = pair.get("rho_corr", float("nan"))
        matrix_path = out_path.with_suffix(".matrix.json")
        matrix_payload = {"labels": labels, "corr": corr_matrix.tolist()}
        matrix_path.write_text(json.dumps(matrix_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[resonance] correlation matrix -> {matrix_path}")

    plots_dir = Path(args.plots_dir) if args.plots_dir else None
    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            from telemetry import plot_resonance

            created = plot_resonance.render_resonance_plots(metrics, plots_dir)
            if created:
                print(f"[resonance] plots -> {plots_dir}")
        except Exception as exc:  # pragma: no cover - plotting optional
            print(f"[resonance] warning: failed to render plots ({exc})")


if __name__ == "__main__":
    main()
