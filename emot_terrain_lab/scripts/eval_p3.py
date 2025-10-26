"""Quick evaluation for Wave P3 multi-agent sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import yaml

from ops.p3_metrics import kuramoto_R, propagation_rate


def load_rows(path: Path) -> List[dict]:
    data = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data.append(json.loads(line))
    return data


def filter_rows(rows: Iterable[dict], **criteria) -> List[dict]:
    output = []
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            output.append(row)
    return output


def series(rows: Iterable[dict], agent: str, key: str) -> np.ndarray:
    return np.array([r.get(key, 0.0) for r in rows if r.get("agent") == agent], dtype=float)


def events(rows: Iterable[dict], agent: str, threshold: float) -> np.ndarray:
    return (series(rows, agent, "bud_score") > threshold).astype(int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wave P3 session metrics.")
    parser.add_argument("--path", type=Path, default=Path("data/logs/session.jsonl"))
    parser.add_argument("--tau", type=float, default=5.0, help="Propagation window size (seconds).")
    parser.add_argument("--delta-min", type=float, default=0.12)
    parser.add_argument("--r-max", type=float, default=0.78)
    parser.add_argument("--bud-threshold", type=float, default=0.42)
    parser.add_argument("--template", type=Path, help="Optional YAML thresholds (uses key 'p3').")
    parser.add_argument("--out", type=Path, help="Write summary JSON here.")
    parser.add_argument("--field-log", type=Path, help="Optional FIELD response log for τ(ρ).")
    parser.add_argument("--gofr-log", type=Path, help="Optional Phase log for g(R).")
    return parser.parse_args()


def load_template(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("p3", {}) if isinstance(data, dict) else {}


@dataclass
class FieldStats:
    rho: float
    tau: float


def evaluate_field(path: Path, *, beta: float) -> dict:
    rows = load_rows(path)
    if not rows:
        return {}
    samples: List[FieldStats] = []
    for row in rows:
        rho = row.get("rho")
        tau = row.get("tau")
        if rho is None or tau is None:
            continue
        samples.append(FieldStats(float(rho), float(tau)))
    if not samples:
        return {}
    rho_vals = np.array([s.rho for s in samples], dtype=float)
    tau_vals = np.array([s.tau for s in samples], dtype=float)
    corr = float(np.corrcoef(rho_vals, tau_vals)[0, 1])
    return {
        "samples": len(samples),
        "corr_rho_tau": corr,
        "rho_mean": float(rho_vals.mean()),
        "tau_mean": float(tau_vals.mean()),
        "beta": beta,
    }


def evaluate_phase(path: Path, band: tuple[float, float]) -> dict:
    rows = load_rows(path)
    if not rows:
        return {}
    values = [(row.get("R"), row.get("dPsi_dt")) for row in rows if "R" in row and "dPsi_dt" in row]
    if not values:
        return {}
    R_vals = np.array([float(r) for r, _ in values], dtype=float)
    g_vals = np.array([float(g) for _, g in values], dtype=float)
    peak_idx = int(np.argmax(g_vals))
    peak_R = float(R_vals[peak_idx])
    peak_g = float(g_vals[peak_idx])
    in_band = band[0] <= peak_R <= band[1]
    return {
        "samples": len(values),
        "peak_R": peak_R,
        "peak_g": peak_g,
        "peak_in_band": in_band,
        "band": {"low": band[0], "high": band[1]},
    }


def main(args: argparse.Namespace) -> int:
    rows = load_rows(args.path)
    if not rows:
        print("No rows loaded.")
        return 1

    thresholds = {
        "delta_min": args.delta_min,
        "R_max": args.r_max,
        "bud_threshold": args.bud_threshold,
    }
    if args.template and args.template.exists():
        preset = load_template(args.template)
        thresholds.update(
            {
                "delta_min": float(preset.get("delta_p_min", thresholds["delta_min"])),
                "R_max": float(preset.get("R_max", thresholds["R_max"])),
                "bud_threshold": float(preset.get("bud_threshold", thresholds["bud_threshold"])),
            }
        )

    phases = {
        "independent": filter_rows(rows, phase="independent"),
        "strong": filter_rows(rows, phase="strong"),
    }
    if not phases["independent"] or not phases["strong"]:
        print("Missing phases; ensure independent and strong data exist.")
        return 1

    eA_ind = events(phases["independent"], "A", thresholds["bud_threshold"])
    eB_ind = events(phases["independent"], "B", thresholds["bud_threshold"])
    eA_str = events(phases["strong"], "A", thresholds["bud_threshold"])
    eB_str = events(phases["strong"], "B", thresholds["bud_threshold"])

    P_ind = propagation_rate(eA_ind, eB_ind, args.tau)
    P_str = propagation_rate(eA_str, eB_str, args.tau)
    delta_p = P_str - P_ind
    delta_pass = delta_p >= thresholds["delta_min"]
    print(f"ΔP(A→B)={delta_p:.3f}  (ind={P_ind:.3f} → strong={P_str:.3f})  PASS? {delta_pass}")

    R_values = [row.get("R", 0.0) for row in rows if "R" in row]
    if R_values:
        Rmax = max(R_values)
    else:
        groups = {}
        for row in rows:
            ts = row.get("ts")
            groups.setdefault(ts, []).append(row.get("theta", 0.0))
        Rmax = 0.0
        for values in groups.values():
            Rmax = max(Rmax, kuramoto_R(np.array(values, dtype=float)))
    R_pass = Rmax <= thresholds["R_max"]
    print(f"max(R)={Rmax:.2f}  SAFE? {R_pass}")

    summary = {
        "delta_p": delta_p,
        "delta_pass": delta_pass,
        "R_max": Rmax,
        "R_pass": R_pass,
        "thresholds": thresholds,
    }

    if args.field_log and args.field_log.exists():
        tau_summary = evaluate_field(args.field_log, beta=thresholds.get("field_tau_beta", 0.1))
        summary["field"] = tau_summary
        print(f"Field relaxation: corr(rho, tau)={tau_summary.get('corr_rho_tau'):.3f}")

    if args.gofr_log and args.gofr_log.exists():
        band = (thresholds.get("phase_band_low", 0.3), thresholds.get("phase_band_high", 0.6))
        g_summary = evaluate_phase(args.gofr_log, band)
        summary["phase"] = g_summary
        print(f"Phase g(R): peak_in_band={g_summary.get('peak_in_band')}  peak_R={g_summary.get('peak_R'):.3f}")

    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
