# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pillow",
#     "scipy",
# ]
# ///
"""
Baseline docking trajectory generation + mid_x optimization + animation.

Phases:
  1. In-place rotation: face the midpoint (mid_x, 0)
  2. Straight advance:  trapezoidal velocity to midpoint
  3. In-place rotation: align to theta=0 (rear toward dock)
  4. Reverse drive:     trapezoidal decel/cruise to dock (-charger_x, 0)

Subcommands:
  uv run baseline.py              -- trajectory generation + GIF
  uv run baseline.py optimize     -- mid_x optimization plot
"""

import math
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import timeopt_common
from timeopt_common import (
    CHARGER_X, START, DOCK, MID_POINT, V_MAX, W_MAX, A_MAX, ALPHA_MAX, CHARGE_V,
    DT, baseline_gif, baseline_csv, midx_png,
    normalize_angle, save_csv, animate_trajectory,
)


def trapezoid_rotation(angle: float, w_max: float, alpha_max: float, dt: float):
    """Return (t, omega) arrays for in-place rotation with trapezoidal profile."""
    sign = 1.0 if angle >= 0 else -1.0
    dist = abs(angle)
    if dist < 1e-9:
        return [], []

    t_ramp = w_max / alpha_max
    dist_ramp = alpha_max * t_ramp**2
    if dist <= dist_ramp:
        t_ramp = math.sqrt(dist / alpha_max)
        t_total = 2 * t_ramp
        t_cruise = 0.0
        w_peak = alpha_max * t_ramp
    else:
        t_cruise = (dist - dist_ramp) / w_max
        t_total = 2 * t_ramp + t_cruise
        w_peak = w_max

    ts, ws = [], []
    t = 0.0
    while t < t_total + dt * 0.5:
        t_clamped = min(t, t_total)
        if t_clamped < t_ramp:
            w = alpha_max * t_clamped
        elif t_clamped < t_ramp + t_cruise:
            w = w_peak
        else:
            w = w_peak - alpha_max * (t_clamped - t_ramp - t_cruise)
        w = max(w, 0.0)
        ts.append(t_clamped)
        ws.append(sign * w)
        t += dt

    return ts, ws


def trapezoid_linear(distance: float, v_max: float, a_max: float, dt: float, backward: bool = False):
    """Return (t, v) arrays for straight-line motion with trapezoidal profile."""
    dist = abs(distance)
    if dist < 1e-9:
        return [], []

    t_ramp = v_max / a_max
    dist_ramp = a_max * t_ramp**2
    if dist <= dist_ramp:
        t_ramp = math.sqrt(dist / a_max)
        t_cruise = 0.0
        v_peak = a_max * t_ramp
    else:
        t_cruise = (dist - dist_ramp) / v_max
        v_peak = v_max

    t_total = 2 * t_ramp + t_cruise

    ts, vs = [], []
    t = 0.0
    while t < t_total + dt * 0.5:
        t_clamped = min(t, t_total)
        if t_clamped < t_ramp:
            v = a_max * t_clamped
        elif t_clamped < t_ramp + t_cruise:
            v = v_peak
        else:
            v = v_peak - a_max * (t_clamped - t_ramp - t_cruise)
        v = max(v, 0.0)
        if backward:
            v = -v
        ts.append(t_clamped)
        vs.append(v)
        t += dt

    return ts, vs


def _trapezoid_time(distance: float, v_max: float, a_max: float) -> float:
    """Analytical time for trapezoidal linear motion."""
    if distance < 1e-12:
        return 0.0
    t_ramp = v_max / a_max
    dist_ramp = a_max * t_ramp**2
    if distance <= dist_ramp:
        return 2 * math.sqrt(distance / a_max)
    return 2 * t_ramp + (distance - dist_ramp) / v_max


def _rotation_time(angle: float, w_max: float, alpha_max: float) -> float:
    """Analytical time for trapezoidal rotation."""
    dist = abs(angle)
    if dist < 1e-12:
        return 0.0
    t_ramp = w_max / alpha_max
    dist_ramp = alpha_max * t_ramp**2
    if dist <= dist_ramp:
        return 2 * math.sqrt(dist / alpha_max)
    return 2 * t_ramp + (dist - dist_ramp) / w_max


def _total_time_for_midx(mid_x: float) -> float:
    """Total time of the baseline method for a given mid_x."""
    sx, sy, sth = START
    dock_x = DOCK[0]

    target_angle = math.atan2(0.0 - sy, mid_x - sx)
    delta1 = abs(normalize_angle(target_angle - sth))
    t1 = _rotation_time(delta1, W_MAX, ALPHA_MAX)

    dist = math.hypot(mid_x - sx, 0.0 - sy)
    t2 = _trapezoid_time(dist, V_MAX, A_MAX)

    delta3 = abs(normalize_angle(0.0 - target_angle))
    t3 = _rotation_time(delta3, W_MAX, ALPHA_MAX)

    accel_time = CHARGE_V / A_MAX
    accel_dist = 0.5 * A_MAX * accel_time**2
    cruise_dist = max(0, abs(dock_x - mid_x) - accel_dist)
    t4 = accel_time + cruise_dist / CHARGE_V

    return t1 + t2 + t3 + t4


def generate_baseline_trajectory():
    """Generate the full baseline trajectory."""
    sx, sy, sth = START
    mx, my = MID_POINT
    dx, dy, dth = DOCK

    trajectory = []
    x, y, th = sx, sy, sth
    t_global = 0.0

    # Phase 1: in-place rotation to face midpoint
    target_angle = math.atan2(my - sy, mx - sx)
    delta_angle = normalize_angle(target_angle - th)
    ts_rot, ws_rot = trapezoid_rotation(delta_angle, W_MAX, ALPHA_MAX, DT)
    for i, (t_loc, w) in enumerate(zip(ts_rot, ws_rot)):
        trajectory.append((t_global, x, y, th, 0.0, w))
        if i < len(ts_rot) - 1:
            dt_step = ts_rot[i + 1] - t_loc
            th += w * dt_step
            t_global += dt_step
    th = normalize_angle(th)

    # Phase 2: straight advance to midpoint
    dist_to_mid = math.hypot(mx - x, my - y)
    ts_lin, vs_lin = trapezoid_linear(dist_to_mid, V_MAX, A_MAX, DT)
    for i, (t_loc, v) in enumerate(zip(ts_lin, vs_lin)):
        trajectory.append((t_global, x, y, th, v, 0.0))
        if i < len(ts_lin) - 1:
            dt_step = ts_lin[i + 1] - t_loc
            x += v * math.cos(th) * dt_step
            y += v * math.sin(th) * dt_step
            t_global += dt_step

    # Phase 3: in-place rotation to theta=0
    delta_angle2 = normalize_angle(0.0 - th)
    ts_rot2, ws_rot2 = trapezoid_rotation(delta_angle2, W_MAX, ALPHA_MAX, DT)
    for i, (t_loc, w) in enumerate(zip(ts_rot2, ws_rot2)):
        trajectory.append((t_global, x, y, th, 0.0, w))
        if i < len(ts_rot2) - 1:
            dt_step = ts_rot2[i + 1] - t_loc
            th += w * dt_step
            t_global += dt_step
    th = normalize_angle(th)

    # Phase 4: accelerate + constant-speed reverse to dock
    accel_time = CHARGE_V / A_MAX
    t_loc = 0.0
    while t_loc < accel_time - DT * 0.5:
        v = -(A_MAX * t_loc)
        trajectory.append((t_global, x, y, th, v, 0.0))
        dt_step = min(DT, accel_time - t_loc)
        x += v * math.cos(th) * dt_step
        y += v * math.sin(th) * dt_step
        t_global += dt_step
        t_loc += dt_step

    dist_remaining = abs(dx - x)
    cruise_time = dist_remaining / CHARGE_V
    t_loc = 0.0
    while t_loc < cruise_time - DT * 0.5:
        trajectory.append((t_global, x, y, th, -CHARGE_V, 0.0))
        dt_step = min(DT, cruise_time - t_loc)
        x += -CHARGE_V * math.cos(th) * dt_step
        y += -CHARGE_V * math.sin(th) * dt_step
        t_global += dt_step
        t_loc += dt_step

    trajectory.append((t_global, dx, dy, th, 0.0, 0.0))
    return trajectory


def optimize_midx():
    """Find the optimal mid_x and save the plot."""
    lower, upper = 0.0, max(1.0, abs(START[0]) + 0.5)
    mid_xs = np.linspace(lower, upper, 1000)
    times = [_total_time_for_midx(mx) for mx in mid_xs]

    result = minimize_scalar(_total_time_for_midx, bounds=(lower, upper), method="bounded")
    opt_mid_x = result.x
    opt_time = result.fun

    print(f"Optimal mid_x = {opt_mid_x:.4f}")
    print(f"Minimum total time = {opt_time:.4f} s")
    print()

    sx, sy, sth = START
    dock_x = DOCK[0]
    target_angle = math.atan2(0.0 - sy, opt_mid_x - sx)
    delta1 = abs(normalize_angle(target_angle - sth))
    dist = math.hypot(opt_mid_x - sx, 0.0 - sy)
    delta3 = abs(normalize_angle(0.0 - target_angle))
    accel_time = CHARGE_V / A_MAX
    accel_dist = 0.5 * A_MAX * accel_time**2
    cruise_dist = max(0, abs(dock_x - opt_mid_x) - accel_dist)

    print(f"Phase 1 (rotate):  {_rotation_time(delta1, W_MAX, ALPHA_MAX):.3f} s  (\u0394\u03b8={math.degrees(delta1):.1f}\u00b0)")
    print(f"Phase 2 (straight): {_trapezoid_time(dist, V_MAX, A_MAX):.3f} s  (d={dist:.3f} m)")
    print(f"Phase 3 (rotate):  {_rotation_time(delta3, W_MAX, ALPHA_MAX):.3f} s  (\u0394\u03b8={math.degrees(delta3):.1f}\u00b0)")
    print(f"Phase 4 (back):    {accel_time + cruise_dist / CHARGE_V:.3f} s  (d={abs(dock_x - opt_mid_x):.3f} m)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mid_xs, times, "b-", linewidth=1.5)
    ax.axvline(opt_mid_x, color="r", linestyle="--", alpha=0.7)
    ax.plot(opt_mid_x, opt_time, "ro", markersize=8)
    ax.text(opt_mid_x + 0.01, opt_time + 0.2,
            f"mid_x = {opt_mid_x:.3f}\nT = {opt_time:.3f} s",
            fontsize=10, color="red")
    ax.set_xlabel("mid_x [m]")
    ax.set_ylabel("Total time [s]")
    ax.set_title("Baseline total time vs mid_x")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = midx_png()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")
    plt.close()


def main():
    trajectory = generate_baseline_trajectory()
    total_time = trajectory[-1][0]
    print(f"Total time: {total_time:.3f} s")
    save_csv(trajectory, baseline_csv())
    animate_trajectory(
        trajectory, baseline_gif(),
        "Baseline: Rotate \u2192 Straight \u2192 Rotate \u2192 Back",
        markers=[(MID_POINT[0], MID_POINT[1], "s", 8, "Mid point")],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "THETA"))
    parser.add_argument("command", nargs="?", default="run", choices=["run", "optimize"])
    args = parser.parse_args()

    if args.start:
        timeopt_common.set_start(*args.start)
        START = timeopt_common.START

    if args.command == "optimize":
        optimize_midx()
    else:
        main()
