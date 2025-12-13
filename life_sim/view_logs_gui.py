from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from matplotlib.widgets import Slider
from run_life_sim import generate_logs

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Noto Sans CJK JP", "sans-serif"]
HAZARD_THRESHOLD = 0.6


def read_log(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_row(label: str, row: Dict[str, Any]) -> str:
    per = row.get("perception") or {}
    internal = row.get("internal") or {}
    action = row.get("action") or {}
    comment = row.get("comment", "")

    offer = per.get("offer_type", "-" if per.get("offer_seen") else "なし")
    ctx = per.get("context", "-")
    loc = per.get("location", "-")
    hazard = per.get("hazard_hint", "-")
    intent = per.get("social_intent", "-")

    layer = internal.get("self_layer", internal.get("mode", "-"))
    margin = internal.get("winner_margin")
    boundary = internal.get("boundary_score")
    interp = internal.get("interpretation", "-")

    v_scale = action.get("v_scale")
    d_target = action.get("d_target")
    pause_s = action.get("pause_s")

    return (
        f"{label}\n"
        f"  感覚: offer={offer} ctx={ctx} loc={loc} hazard={hazard} intent={intent}\n"
        f"  内面: layer={layer} margin={margin} boundary={boundary} interp={interp}\n"
        f"  行動: v={v_scale} d_target={d_target} pause={pause_s}s\n"
        f"  {comment}"
    )


def event_label(row: Dict[str, Any]) -> str:
    internal = row.get("internal") or {}
    interp = internal.get("interpretation")
    if internal.get("reset"):
        return "⏸ 考え直し"
    if interp == "playful_temptation":
        return "🍬 甘い誘惑（許容）"
    if interp == "contextually_safe_tool":
        return "🔧 道具：文脈一致"
    if interp == "ambiguous_hazard_offer":
        return "⚠ 文脈不明→境界"
    return "🧠 状況観察中"


def diff_label(off_row: Dict[str, Any], on_row: Dict[str, Any]) -> str:
    if on_row.get("internal", {}).get("reset") and not off_row.get("internal", {}).get("reset"):
        return "差分: ONは考え直し、OFFは即応"
    off_d = _float(off_row.get("action", {}).get("d_target"))
    on_d = _float(on_row.get("action", {}).get("d_target"))
    if on_d - off_d > 0.05:
        return "差分: ONは距離を保つ"
    off_v = _float(off_row.get("action", {}).get("v_scale"), 1.0)
    on_v = _float(on_row.get("action", {}).get("v_scale"), 1.0)
    if on_v < off_v - 0.1:
        return "差分: ONは減速・様子見"
    return "差分: 小"


def build_motion(rows: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    xs = [0.0]
    ys = [0.0]
    angles = [0.0]
    radii = [0.7 + _float(rows[0].get("action", {}).get("d_target"), 0.0)]

    for prev, cur in zip(rows, rows[1:]):
        dt = max(_float(cur.get("t"), 0.0) - _float(prev.get("t"), 0.0), 0.1)
        pause = _float(cur.get("action", {}).get("pause_s"), 0.0)
        effective_dt = max(dt - pause, 0.0)
        v = _float(cur.get("action", {}).get("v_scale"), 1.0)
        xs.append(xs[-1] + v * effective_dt * 0.2)
        ys.append(0.0)
        jitter = _float(cur.get("action", {}).get("jitter"), 0.0)
        angles.append(jitter * 4.0)
        radii.append(0.7 + _float(cur.get("action", {}).get("d_target"), 0.0))

    return {"xs": xs, "ys": ys, "angles": angles, "radii": radii}


def init_motion_ax(ax, label: str, color: str):
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.2, 4.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(label)
    path_line, = ax.plot([], [], color=color, lw=2)
    body_dot, = ax.plot([], [], marker="o", color=color, markersize=10)
    head_line, = ax.plot([], [], color=color, lw=2)
    ring = patches.Circle((0, 0), radius=0.7, fill=False, linestyle="--", color=color, alpha=0.6)
    ax.add_patch(ring)
    return {"path": path_line, "body": body_dot, "head": head_line, "ring": ring}


def update_motion(art, motion, idx):
    x = motion["xs"][idx]
    y = motion["ys"][idx]
    art["path"].set_data(motion["xs"][: idx + 1], motion["ys"][: idx + 1])
    art["body"].set_data([x], [y])

    L = 0.3
    hx = x + L * (1.0)
    hy = y + L * motion["angles"][idx]
    art["head"].set_data([x, hx], [y, hy])

    art["ring"].center = (x, y)
    art["ring"].radius = motion["radii"][idx]


def set_hazard_bg(ax, is_hazard: bool):
    ax.set_facecolor((1.0, 0.92, 0.92) if is_hazard else (1.0, 1.0, 1.0))


def build_scatter(rows: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    times = []
    colors = []
    for row in rows:
        per = row.get("perception") or {}
        if per.get("offer_seen"):
            t = row.get("t", 0.0)
            offer = per.get("offer_type", "?")
            colors.append("#f39c12" if offer == "sweet" else "#c0392b")
            times.append(t)
    return {"times": times, "colors": colors}


def spans_from_threshold(times: List[float], values: List[float], threshold: float) -> List[tuple[float, float]]:
    spans = []
    active = False
    start = times[0] if times else 0.0
    for t, val in zip(times, values):
        if val >= threshold and not active:
            active = True
            start = t
        elif val < threshold and active:
            active = False
            spans.append((start, t))
    if active and times:
        spans.append((start, times[-1]))
    return spans


def scene_markers(rows: List[Dict[str, Any]]) -> List[tuple[float, str]]:
    markers = []
    prev = None
    for row in rows:
        ctx = row.get("perception", {}).get("context")
        if ctx and ctx != prev:
            markers.append((row.get("t", 0.0), ctx))
            prev = ctx
    return markers


def ensure_logs(log_dir: Path, stimulus_path: Path) -> None:
    off = log_dir / "no_eqnet.jsonl"
    on = log_dir / "with_eqnet.jsonl"
    if off.exists() and on.exists():
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    generate_logs(stimulus_path, log_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize EQNet logs with animation")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--stimulus", default="life_sim/stimulus.jsonl")
    parser.add_argument("--interval", type=int, default=800)
    args = parser.parse_args()

    log_dir = Path(args.logs)
    ensure_logs(log_dir, Path(args.stimulus))
    off_rows = read_log(log_dir / "no_eqnet.jsonl")
    on_rows = read_log(log_dir / "with_eqnet.jsonl")
    n = min(len(off_rows), len(on_rows))
    off_rows = off_rows[:n]
    on_rows = on_rows[:n]

    times = [row["t"] for row in off_rows]
    boundary_off = [_float(row.get("internal", {}).get("boundary_score")) for row in off_rows]
    boundary_on = [_float(row.get("internal", {}).get("boundary_score")) for row in on_rows]

    scatter_off = build_scatter(off_rows)
    scatter_on = build_scatter(on_rows)
    markers = scene_markers(on_rows)

    motion_off = build_motion(off_rows)
    motion_on = build_motion(on_rows)

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1])

    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.set_title("Boundary timeline (OFF vs ON)")
    ax_timeline.set_xlim(times[0], times[-1])
    ax_timeline.set_ylim(0, 1)
    ax_timeline.plot(times, boundary_off, label="OFF boundary", color="#1f77b4")
    ax_timeline.plot(times, boundary_on, label="ON boundary", color="#c44172")
    ax_timeline.scatter(scatter_off["times"], [0.9] * len(scatter_off["times"]), c=scatter_off["colors"], marker="v", alpha=0.5)
    ax_timeline.scatter(scatter_on["times"], [0.1] * len(scatter_on["times"]), c=scatter_on["colors"], marker="^", alpha=0.5)
    ax_timeline.legend(loc="upper right")
    for start, end in spans_from_threshold(times, boundary_off, HAZARD_THRESHOLD):
        ax_timeline.axvspan(start, end, ymin=0.55, ymax=1.0, alpha=0.12, color="#f8c0c0")
    for start, end in spans_from_threshold(times, boundary_on, HAZARD_THRESHOLD):
        ax_timeline.axvspan(start, end, ymin=0.0, ymax=0.45, alpha=0.12, color="#f8c0c0")
    for t, label in markers:
        ax_timeline.axvline(t, color="#bbbbbb", linestyle=":", alpha=0.6)
        ax_timeline.text(t, 0.5, label, rotation=90, va="center", ha="center", fontsize=9, alpha=0.7)

    cursor = ax_timeline.axvline(times[0], color="#444", linestyle="--")

    ax_anim_off = fig.add_subplot(gs[1, 0])
    ax_anim_on = fig.add_subplot(gs[1, 1])
    anim_off = init_motion_ax(ax_anim_off, "OFF", "#1f77b4")
    anim_on = init_motion_ax(ax_anim_on, "ON", "#c44172")

    text_off = fig.add_subplot(gs[2, 0])
    text_off.axis("off")
    text_on = fig.add_subplot(gs[2, 1])
    text_on.axis("off")

    text_off_title = text_off.text(0.0, 1.0, "", transform=text_off.transAxes, va="top", ha="left", fontfamily="Yu Gothic")
    text_on_title = text_on.text(0.0, 1.0, "", transform=text_on.transAxes, va="top", ha="left", fontfamily="Yu Gothic")

    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    slider = Slider(slider_ax, "t", times[0], times[-1], valinit=times[0])
    fig.text(0.5, 0.015, "p:一時停止  1/2/3:速度  q/Esc:終了", ha="center", fontsize=11)
    event_text_off = fig.text(0.25, 0.095, "", ha="center", fontsize=12, color="#1f77b4")
    event_text_on = fig.text(0.75, 0.095, "", ha="center", fontsize=12, color="#c44172")
    diff_text = fig.text(0.5, 0.08, "", ha="center", fontsize=11, color="#333333")

    state = {"frame": 0, "paused": False, "speed": 1, "slider_internal": False}

    def update(frame: int):
        idx = state["frame"]
        if not state["paused"]:
            idx = (idx + max(state["speed"], 1)) % n
            state["frame"] = idx

        cursor.set_xdata([times[idx], times[idx]])
        update_motion(anim_off, motion_off, idx)
        update_motion(anim_on, motion_on, idx)
        set_hazard_bg(ax_anim_off, boundary_off[idx] > HAZARD_THRESHOLD)
        set_hazard_bg(ax_anim_on, boundary_on[idx] > HAZARD_THRESHOLD)

        npc_off = off_rows[idx].get("npc") or {}
        npc_on = on_rows[idx].get("npc") or {}
        utter_off = npc_off.get("utterance") or "…"
        utter_on = npc_on.get("utterance") or "…"
        event_text_off.set_text(f"{event_label(off_rows[idx])} / 目的:{npc_off.get('goal','-')} / 状態:{npc_off.get('state','-')} /『{utter_off}』")
        event_text_on.set_text(f"{event_label(on_rows[idx])} / 目的:{npc_on.get('goal','-')} / 状態:{npc_on.get('state','-')} /『{utter_on}』")
        diff_text.set_text(diff_label(off_rows[idx], on_rows[idx]))

        state["slider_internal"] = True
        slider.set_val(times[idx])
        state["slider_internal"] = False

        text_off_title.set_text(summarize_row("OFF", off_rows[idx]))
        text_on_title.set_text(summarize_row("ON", on_rows[idx]))
        return cursor, anim_off["body"], anim_on["body"], text_off_title, text_on_title

    def on_key(event):
        if event.key == "p":
            state["paused"] = not state["paused"]
        elif event.key == "1":
            state["speed"] = 1
        elif event.key == "2":
            state["speed"] = 2
        elif event.key == "3":
            state["speed"] = 3
        elif event.key in ("q", "escape"):
            plt.close(fig)

    def on_slide(val):
        if state.get("slider_internal"):
            return
        idx = int(np.argmin(np.abs(np.asarray(times) - float(val))))
        state["frame"] = idx
        state["paused"] = True
        fig.canvas.draw_idle()

    slider.on_changed(on_slide)

    fig.canvas.mpl_connect("key_press_event", on_key)
    ani = FuncAnimation(fig, update, interval=args.interval, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
