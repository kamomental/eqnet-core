import math
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class Plan:
    target_xy: np.ndarray
    v_planner: float
    base_distance: float
    heading: float


@dataclass
class EqnetScalars:
    v_scale: float
    jitter: float
    d_target: float
    phase_tag: str = "normal"


@dataclass
class Knobs:
    tau_recover: float = 1.5
    jitter_amp_cap: float = 0.08
    d_target_cap: float = 0.25


@dataclass
class SimResult:
    dt: float
    target: np.ndarray
    xs: list
    ys: list
    head_dirs: list
    tags: list
    rows: list
    distance_goals: list
    reset_progress: list
    ci_result: str


def add_upper_body_micro_noise(head_angle: float, jitter_amp: float, rng: np.random.Generator):
    return head_angle + jitter_amp * (0.6 * rng.normal() + 0.4 * math.sin(rng.uniform(0, 2 * math.pi)))


def apply_eqnet_modulation(plan_in: Plan, s: EqnetScalars, knobs: Knobs, rng: np.random.Generator):
    assert 0.0 <= s.v_scale <= 1.2
    assert abs(s.d_target) <= knobs.d_target_cap

    jitter_amp = min(abs(s.jitter), knobs.jitter_amp_cap)

    v_out = plan_in.v_planner * s.v_scale
    d_out = plan_in.base_distance + s.d_target
    target_xy = plan_in.target_xy.copy()
    head_angle = add_upper_body_micro_noise(plan_in.heading, jitter_amp, rng)
    return v_out, d_out, head_angle, target_xy


def scenario_scalars(t: float, *, eqnet_on: bool) -> EqnetScalars:
    if not eqnet_on:
        return EqnetScalars(v_scale=1.0, jitter=0.0, d_target=0.0, phase_tag="normal")

    v_scale = 1.0
    jitter = 0.0
    d_target = 0.0
    tag = "normal"

    if 3.0 <= t < 6.0:
        v_scale = 0.85
        jitter = 0.06
    elif 6.0 <= t < 8.0:
        v_scale = 0.65
        jitter = 0.05
        d_target = 0.18
    elif 8.0 <= t < 10.0:
        v_scale = 0.25
        jitter = 0.08
        d_target = 0.22
        tag = "reset_ritual"
    elif 10.0 <= t < 14.0:
        v_scale = 0.25 + (1.0 - 0.25) * (1.0 - math.exp(-(t - 10.0) / 1.5))
        jitter = 0.03 * math.exp(-(t - 10.0) / 1.0)
        d_target = 0.22 * math.exp(-(t - 10.0) / 1.2)

    return EqnetScalars(
        v_scale=float(v_scale),
        jitter=float(jitter),
        d_target=float(np.clip(d_target, -0.25, 0.25)),
        phase_tag=tag,
    )


def compute_energy(prev_head: float, head: float, prev_pos: np.ndarray, pos: np.ndarray, dt: float):
    omega_head = (head - prev_head) / max(dt, 1e-6)
    v_base = np.linalg.norm((pos - prev_pos) / max(dt, 1e-6))
    return omega_head**2, v_base**2


def ci_check_jitter_leak(rows, ratio_thr=0.15, ratio_thr_reset=0.20, EU_MIN=0.02, consec_N=3, consec_N_reset=5):
    def worst_consec(values, thr, N):
        c = 0
        worst = 0.0
        for x in values:
            if x > thr:
                c += 1
                worst = max(worst, x)
                if c >= N:
                    return True, worst
            else:
                c = 0
        return False, worst

    normal = [r["lower_ratio"] for r in rows if r["phase_tag"] != "reset_ritual" and r["E_U"] >= EU_MIN]
    reset = [r["lower_ratio"] for r in rows if r["phase_tag"] == "reset_ritual" and r["E_U"] >= EU_MIN]

    bad_n, worst_n = worst_consec(normal, ratio_thr, consec_N)
    bad_r, worst_r = worst_consec(reset, ratio_thr_reset, consec_N_reset)

    if bad_n:
        raise SystemExit(f"CI FAIL: lower_ratio leak normal (worst={worst_n:.3f} thr={ratio_thr})")
    if bad_r:
        raise SystemExit(f"CI FAIL: lower_ratio leak reset (worst={worst_r:.3f} thr={ratio_thr_reset})")


def simulate(eqnet_on: bool, seed=0) -> SimResult:
    rng = np.random.default_rng(seed)
    dt = 0.05
    T = 14.0
    steps = int(T / dt)

    pos = np.array([0.0, 0.0], dtype=float)
    heading = 0.0
    head_angle = heading
    target = np.array([2.0, 0.0], dtype=float)
    knobs = Knobs()

    rows = []
    xs: list[float] = []
    ys: list[float] = []
    head_dirs: list[float] = []
    tags: list[str] = []
    distance_goals: list[float] = []

    prev_pos = pos.copy()
    prev_head = head_angle

    for k in range(steps):
        t = k * dt
        vec = target - pos
        dist = np.linalg.norm(vec) + 1e-9
        dir_xy = vec / dist

        heading = math.atan2(dir_xy[1], dir_xy[0])
        plan = Plan(target_xy=target, v_planner=0.25, base_distance=0.7, heading=heading)

        scalars = scenario_scalars(t, eqnet_on=eqnet_on)
        v_out, d_out, head_angle, target_out = apply_eqnet_modulation(plan, scalars, knobs, rng)
        assert np.allclose(target_out, target), "Target must not drift"

        dist_err = dist - d_out
        v_eff = v_out * float(np.clip(dist_err / 0.6, 0.0, 1.0))
        pos = pos + dir_xy * v_eff * dt

        E_U, E_L = compute_energy(prev_head, head_angle, prev_pos, pos, dt)
        lower_ratio = float(E_L / (E_U + 1e-6))

        rows.append(
            {
                "t": float(t),
                "phase_tag": scalars.phase_tag,
                "v_scale": float(scalars.v_scale),
                "jitter": float(scalars.jitter),
                "d_target": float(scalars.d_target),
                "E_U": float(E_U),
                "E_L": float(E_L),
                "lower_ratio": float(lower_ratio),
            }
        )

        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        head_dirs.append(float(head_angle))
        tags.append(scalars.phase_tag)
        distance_goals.append(float(d_out))

        prev_pos = pos.copy()
        prev_head = head_angle

    reset_progress = [0.0] * len(rows)
    start = None
    for idx, tag in enumerate(tags + ["_END_SENTINEL_"]):
        if tag == "reset_ritual":
            if start is None:
                start = idx
        else:
            if start is not None:
                end = idx
                length = end - start
                for j in range(start, end):
                    denom = max(length - 1, 1)
                    reset_progress[j] = (j - start) / denom
                start = None

    try:
        ci_check_jitter_leak(rows)
        ci_result = "PASS"
    except SystemExit as exc:
        ci_result = f"FAIL: {exc}"

    return SimResult(
        dt=dt,
        target=target,
        xs=xs,
        ys=ys,
        head_dirs=head_dirs,
        tags=tags,
        rows=rows,
        distance_goals=distance_goals,
        reset_progress=reset_progress,
        ci_result=ci_result,
    )


def _init_panel(ax, sim: SimResult, label: str, color: str):
    legend = ax.text(0.02, 0.1, "◯=距離の安心帯  ￣￣=呼吸  帯=儀式", transform=ax.transAxes, fontsize=8, color="#444", ha="left", va="bottom")
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.2, 2.4)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(label)
    ax.text(0.02, 0.86, "◯=距離の安心帯", transform=ax.transAxes, fontsize=7, ha="left", va="top", color="#555")
    ax.text(0.02, 0.80, "帯=考え直し", transform=ax.transAxes, fontsize=7, ha="left", va="top", color="#555")
    ax.text(0.02, 0.74, "呼吸バー=儀式", transform=ax.transAxes, fontsize=7, ha="left", va="top", color="#555")
    ax.plot([sim.target[0]], [sim.target[1]], marker="*", markersize=10, color="#6c6c6c")

    path_line, = ax.plot([], [], lw=2, color=color)
    body_dot, = ax.plot([], [], marker="o", markersize=8, color=color)
    head_line, = ax.plot([], [], lw=2, color=color)

    ring = patches.Circle((sim.xs[0], sim.ys[0]), radius=sim.distance_goals[0], fill=False, linestyle="--", color=color, alpha=0.7)
    ax.add_patch(ring)

    breath = patches.Rectangle((0.15, -0.08), 0.0, 0.05, transform=ax.transAxes, color=color, alpha=0.25, clip_on=False)
    ax.add_patch(breath)

    phase_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, ha="left", va="top", fontsize=9)

    return {
        "ax": ax,
        "path": path_line,
        "body": body_dot,
        "head": head_line,
        "ring": ring,
        "breath": breath,
        "phase": phase_text,
        "color": color,
    }


def _update_panel(art, sim: SimResult, frame: int):
    x = sim.xs[frame]
    y = sim.ys[frame]
    art["path"].set_data(sim.xs[:frame], sim.ys[:frame])
    art["body"].set_data([x], [y])

    L = 0.25
    hx = x + L * math.cos(sim.head_dirs[frame])
    hy = y + L * math.sin(sim.head_dirs[frame])
    art["head"].set_data([x, hx], [y, hy])

    art["ring"].center = (x, y)
    art["ring"].radius = sim.distance_goals[frame]

    phase = sim.tags[frame]
    art["ax"].set_facecolor("#f6f1ff" if phase == "reset_ritual" else "white")
    art["phase"].set_text(f"phase={phase}")

    progress = sim.reset_progress[frame]
    if progress > 0:
        art["breath"].set_width(0.7 * progress)
        art["breath"].set_visible(True)
    else:
        art["breath"].set_width(0.0)
        art["breath"].set_visible(False)


def run_gui(out_dir="sim_out_gui", seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_a = simulate(eqnet_on=False, seed=seed)
    sim_b = simulate(eqnet_on=True, seed=seed)

    assert len(sim_a.xs) == len(sim_b.xs), "Sim lengths must match"
    total_frames = len(sim_a.xs)

    state = {
        "frame": 0,
        "speed": 1.0,
        "paused": False,
        "taps": {"A": [], "B": []},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.subplots_adjust(bottom=0.2, top=0.88, wspace=0.15)

    panel_a = _init_panel(axes[0], sim_a, "A  (EQNet OFF)", color="#1f77b4")
    panel_b = _init_panel(axes[1], sim_b, "B  (EQNet ON)", color="#c44172")

    info = fig.text(0.5, 0.94, "", ha="center", va="center", fontsize=11)
    controls = fig.text(
        0.5,
        0.05,
        "p=Pause  1/2/3=Speed  a=Tap A  b=Tap B  s=Save taps  q/Esc=Exit",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    def save_taps():
        ts = int(time.time())
        for label in ("A", "B"):
            taps = state["taps"][label]
            if not taps:
                continue
            path = out_dir / f"taps_{label}_{ts}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for t in taps:
                    f.write(json.dumps({"panel": label, "t": t}) + "\n")
            print(f"[saved] {len(taps)} taps -> {path}")

    def init():
        _update_panel(panel_a, sim_a, 0)
        _update_panel(panel_b, sim_b, 0)
        info.set_text("Press a/b when共鳴した瞬間をタップ")
        return []

    def update(_):
        frame = state["frame"]
        if not state["paused"]:
            advance = max(int(state["speed"]), 1)
            frame = (frame + advance) % total_frames
            state["frame"] = frame

        _update_panel(panel_a, sim_a, frame)
        _update_panel(panel_b, sim_b, frame)

        t = sim_a.rows[frame]["t"]
        info.set_text(
            f"t={t:5.2f}s   speed={state['speed']}x   tapsA={len(state['taps']['A'])}   tapsB={len(state['taps']['B'])}"
        )
        return []

    def on_key(event):
        key = event.key
        if key == "p":
            state["paused"] = not state["paused"]
        elif key == "1":
            state["speed"] = 0.5
        elif key == "2":
            state["speed"] = 1.0
        elif key == "3":
            state["speed"] = 2.0
        elif key == "a":
            t = sim_a.rows[state["frame"]]["t"]
            state["taps"]["A"].append(float(t))
            print(f"Tap A #{len(state['taps']['A'])} @ {t:.2f}s")
        elif key == "b":
            t = sim_b.rows[state["frame"]]["t"]
            state["taps"]["B"].append(float(t))
            print(f"Tap B #{len(state['taps']['B'])} @ {t:.2f}s")
        elif key == "s":
            save_taps()
        elif key in ("q", "escape"):
            save_taps()
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    ani = FuncAnimation(fig, update, init_func=init, interval=60, blit=False, cache_frame_data=False)
    plt.show()


def simulate_and_render(out_dir: Path, *, eqnet_on: bool, seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = simulate(eqnet_on=eqnet_on, seed=seed)
    (out_dir / "motion_log.jsonl").write_text("\n".join(json.dumps(r) for r in result.rows), encoding="utf-8")

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.2, 2.4)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(f"EQNet {'ON' if eqnet_on else 'OFF'} | CI={result.ci_result}")

    ax.plot([result.target[0]], [result.target[1]], marker="*", markersize=12)
    (path_line,) = ax.plot([], [], lw=2)
    (body_dot,) = ax.plot([], [], marker="o", markersize=8)
    (head_line,) = ax.plot([], [], lw=2)

    def init():
        path_line.set_data([], [])
        body_dot.set_data([], [])
        head_line.set_data([], [])
        return path_line, body_dot, head_line

    def update(i):
        path_line.set_data(result.xs[:i], result.ys[:i])
        body_dot.set_data([result.xs[i]], [result.ys[i]])
        L = 0.25
        hx = result.xs[i] + L * math.cos(result.head_dirs[i])
        hy = result.ys[i] + L * math.sin(result.head_dirs[i])
        head_line.set_data([result.xs[i], hx], [result.ys[i], hy])
        return path_line, body_dot, head_line

    ani = FuncAnimation(fig, update, frames=len(result.xs), init_func=init, interval=50, blit=True)
    ani.save(out_dir / "sim.gif", writer=PillowWriter(fps=int(1 / result.dt)))
    plt.close(fig)


def main():
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "--gui":
        run_gui()
        return

    base = Path("sim_out")
    simulate_and_render(base / "A_eqnet_off", eqnet_on=False, seed=0)
    simulate_and_render(base / "B_eqnet_on", eqnet_on=True, seed=0)
    print("Wrote GIFs + logs under:", base.resolve())
    print("Run GUI: python sim_eqnet_motion_ab.py --gui")


if __name__ == "__main__":
    main()
