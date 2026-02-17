"""Shared constants, utilities, CSV I/O, animation helpers, and comparison visualization."""

import csv
import math
import os

import numpy as np

CHARGER_X = 0.2
START = (0.5, 0.2, -math.pi)  # (x, y, theta)
DOCK = (-CHARGER_X, 0.0, 0.0)
MID_POINT = (0.0, 0.0)

V_MAX = 0.3        # [m/s]
W_MAX = 1.5        # [rad/s]
A_MAX = 0.5        # [m/s^2]
ALPHA_MAX = 2.5    # [rad/s^2]
CHARGE_V = 0.05    # dock approach speed [m/s]
TREAD = 0.2        # [m] wheel-to-wheel distance
V_WHEEL_MAX = 0.4  # [m/s] max individual wheel speed (None to disable)
A_WHEEL_MAX = 0.7  # [m/s^2] max individual wheel acceleration (None to disable)

ROBOT_WIDTH = 0.240    # [m]
ROBOT_LENGTH = 0.387   # [m]
FPS = 30
DT = 0.01  # simulation time step [s]

def start_suffix():
    """Generate a filename suffix from the current START position."""
    return f"_x{START[0]}_y{START[1]}_t{START[2]:.2f}"


def baseline_gif():  return f"output/baseline_trajectory{start_suffix()}.gif"
def baseline_csv():  return f"output/baseline_trajectory{start_suffix()}.csv"
def optimized_gif(): return f"output/optimized_trajectory{start_suffix()}.gif"
def optimized_csv(): return f"output/optimized_trajectory{start_suffix()}.csv"
def comparison_gif():return f"output/comparison_trajectory{start_suffix()}.gif"
def comparison_png():return f"output/comparison_velocity{start_suffix()}.png"
def midx_png():      return f"output/midx_optimization{start_suffix()}.png"


def set_start(x: float, y: float, theta: float):
    """Override the global START."""
    global START
    START = (x, y, theta)


def wheel_speeds(v, omega):
    """Compute (v_left, v_right) for differential drive.

    Works with scalars, numpy arrays, or CasADi symbolics.
    """
    half_tread = TREAD / 2.0
    return v - omega * half_tread, v + omega * half_tread


def normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def save_csv(trajectory, path: str, precision: int = 6):
    """Save trajectory to CSV. trajectory is a list of (t, x, y, theta, v, omega)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fmt = f"{{:.{precision}f}}"
    with open(path, "w") as f:
        f.write("t,x,y,theta,v,omega\n")
        for row in trajectory:
            f.write(",".join(fmt.format(v) for v in row) + "\n")
    print(f"Wrote {path} ({len(trajectory)} rows)")


def load_csv(path: str) -> dict:
    """Load trajectory CSV and return a dict of numpy arrays."""
    ts, xs, ys, thetas, vs, omegas = [], [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row["t"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            thetas.append(float(row["theta"]))
            vs.append(float(row["v"]))
            omegas.append(float(row["omega"]))
    return dict(t=np.array(ts), x=np.array(xs), y=np.array(ys),
                theta=np.array(thetas), v=np.array(vs), omega=np.array(omegas))


def compute_frame_indices(ts, total_time, fps=FPS):
    """Compute frame indices from timestamps for animation."""
    n_frames = int(total_time * fps) + 1
    frame_times = np.linspace(0, total_time, n_frames)
    frame_indices = np.searchsorted(ts, frame_times, side="right") - 1
    frame_indices = np.clip(frame_indices, 0, len(ts) - 1)
    return n_frames, frame_times, frame_indices


def setup_ax(ax, xs, ys, title, margin=0.3):
    """Common axis setup (limits, aspect ratio, grid)."""
    ax.set_xlim(min(xs.min(), DOCK[0]) - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def draw_dock(ax):
    """Draw the dock (gray rectangle + target marker)."""
    import matplotlib.pyplot as plt
    dock_depth = 0.10
    dock_width = ROBOT_WIDTH + 0.04
    dock_center_x = DOCK[0] - ROBOT_LENGTH / 2
    dock_rect = plt.Rectangle(
        (dock_center_x - dock_depth / 2, -dock_width / 2), dock_depth, dock_width,
        fill=True, facecolor="lightgray", edgecolor="gray", linewidth=1, alpha=0.6,
    )
    ax.add_patch(dock_rect)
    ax.plot(DOCK[0], DOCK[1], "+", color="gray", markersize=8, markeredgewidth=1.5)


def create_robot_artist(ax, color="black"):
    """Create robot rectangle and direction arrow artists."""
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (-ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2), ROBOT_LENGTH, ROBOT_WIDTH,
        fill=False, edgecolor=color, linewidth=2,
    )
    ax.add_patch(rect)
    arrow_color = "red" if color == "black" else color
    arrow = ax.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2),
    )
    return rect, arrow


def update_robot_artist(rect, arrow, ax, x, y, theta):
    """Update robot rectangle and arrow positions for a frame."""
    from matplotlib.transforms import Affine2D
    arrow_len = ROBOT_LENGTH / 2
    rect.set_transform(Affine2D().rotate(theta).translate(x, y) + ax.transData)
    arrow.xy = (x + arrow_len * math.cos(theta),
                y + arrow_len * math.sin(theta))
    arrow.set_position((x, y))


def animate_trajectory(trajectory, path, title, markers=None):
    """Animate a single trajectory and save as GIF.

    markers: list of (x, y, style, size, label) for additional plot markers.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    os.makedirs(os.path.dirname(path), exist_ok=True)

    ts = np.array([r[0] for r in trajectory])
    xs = np.array([r[1] for r in trajectory])
    ys = np.array([r[2] for r in trajectory])
    thetas = np.array([r[3] for r in trajectory])
    vs = np.array([r[4] for r in trajectory])

    n_frames, frame_times, frame_indices = compute_frame_indices(ts, ts[-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, xs, ys, title)
    draw_dock(ax)

    ax.plot(START[0], START[1], "go", markersize=10, label="Start")
    if markers:
        for mx, my, style, size, label in markers:
            ax.plot(mx, my, style, markersize=size, label=label)

    (trail_line,) = ax.plot([], [], "b-", linewidth=1.5, alpha=0.6)
    robot_rect, arrow = create_robot_artist(ax)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11)
    ax.legend(loc="upper right")

    def update(frame):
        idx = frame_indices[frame]
        x_r, y_r, theta = xs[idx], ys[idx], thetas[idx]
        trail_line.set_data(xs[:idx + 1], ys[:idx + 1])
        update_robot_artist(robot_rect, arrow, ax, x_r, y_r, theta)
        time_text.set_text(f"t = {ts[idx]:.2f} s  v = {vs[idx]:.2f} m/s")
        return trail_line, robot_rect, arrow, time_text

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False)
    anim.save(path, writer=PillowWriter(fps=FPS))
    print(f"Saved {path} ({n_frames} frames, {ts[-1]:.2f} s)")
    plt.close()


def animate_comparison(bl, opt, path, vertical=False):
    """Generate a comparison animation GIF with baseline and optimized in two panels."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    os.makedirs(os.path.dirname(path), exist_ok=True)

    total_time = max(bl["t"][-1], opt["t"][-1])
    n_frames = int(total_time * FPS) + 1
    frame_times = np.linspace(0, total_time, n_frames)

    bl_indices = np.searchsorted(bl["t"], frame_times, side="right") - 1
    bl_indices = np.clip(bl_indices, 0, len(bl["t"]) - 1)
    opt_indices = np.searchsorted(opt["t"], frame_times, side="right") - 1
    opt_indices = np.clip(opt_indices, 0, len(opt["t"]) - 1)

    all_x = np.concatenate([bl["x"], opt["x"]])
    all_y = np.concatenate([bl["y"], opt["y"]])
    margin = 0.3
    xlim = (min(all_x.min(), DOCK[0]) - margin, all_x.max() + margin)
    ylim = (all_y.min() - margin, all_y.max() + margin)

    if vertical:
        fig, (ax_bl, ax_opt) = plt.subplots(2, 1, figsize=(10, 10))
    else:
        fig, (ax_bl, ax_opt) = plt.subplots(1, 2, figsize=(14, 6))

    panels = []
    for ax, label, color, data, total_s in [
        (ax_bl, "Baseline", "blue", bl, bl["t"][-1]),
        (ax_opt, "Optimized", "red", opt, opt["t"][-1]),
    ]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{label} ({total_s:.2f} s)")
        ax.grid(True, alpha=0.3)

        draw_dock(ax)
        ax.plot(data["x"][0], data["y"][0], "go", markersize=8)
        ax.plot(0, 0, "s", color="gray", markersize=6)

        (trail,) = ax.plot([], [], "-", color=color, linewidth=1.5, alpha=0.5)
        rect, arrow = create_robot_artist(ax, color=color)
        time_txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=11)
        panels.append(dict(trail=trail, rect=rect, arrow=arrow, time_txt=time_txt))

    fig.tight_layout()

    def update(frame):
        t_now = frame_times[frame]
        for p, indices, data in [
            (panels[0], bl_indices, bl),
            (panels[1], opt_indices, opt),
        ]:
            idx = indices[frame]
            xr, yr, th = data["x"][idx], data["y"][idx], data["theta"][idx]
            p["trail"].set_data(data["x"][:idx + 1], data["y"][:idx + 1])
            parent_ax = p["rect"].axes
            update_robot_artist(p["rect"], p["arrow"], parent_ax, xr, yr, th)
            p["time_txt"].set_text(f"t = {t_now:.2f} s  v = {data['v'][idx]:.2f} m/s")
        return [v for p in panels for v in (p["trail"], p["rect"], p["arrow"], p["time_txt"])]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False)
    anim.save(path, writer=PillowWriter(fps=FPS))
    print(f"Saved {path} ({n_frames} frames, {total_time:.2f} s)")
    plt.close()


def plot_velocity_comparison(bl, opt, path):
    """Plot velocity and angular velocity comparison between baseline and optimized."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_rows = 3 if V_WHEEL_MAX is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=True)

    ax = axes[0]
    ax.plot(bl["t"], bl["v"], "b-", linewidth=1.5, alpha=0.8, label="Baseline")
    ax.plot(opt["t"], opt["v"], "r-", linewidth=1.5, alpha=0.8, label="Optimized")
    ax.axhline(V_MAX, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-V_MAX, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("v [m/s]")
    ax.set_title("Velocity comparison: Baseline vs Optimized")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(bl["t"], bl["omega"], "b-", linewidth=1.5, alpha=0.8, label="Baseline")
    ax.plot(opt["t"], opt["omega"], "r-", linewidth=1.5, alpha=0.8, label="Optimized")
    ax.axhline(W_MAX, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-W_MAX, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("\u03c9 [rad/s]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if V_WHEEL_MAX is not None:
        bl_vl, bl_vr = wheel_speeds(bl["v"], bl["omega"])
        opt_vl, opt_vr = wheel_speeds(opt["v"], opt["omega"])

        ax = axes[2]
        ax.plot(bl["t"], bl_vl, "b-", linewidth=1.2, alpha=0.6, label="Baseline L")
        ax.plot(bl["t"], bl_vr, "b--", linewidth=1.2, alpha=0.6, label="Baseline R")
        ax.plot(opt["t"], opt_vl, "r-", linewidth=1.2, alpha=0.6, label="Optimized L")
        ax.plot(opt["t"], opt_vr, "r--", linewidth=1.2, alpha=0.6, label="Optimized R")
        ax.axhline(V_WHEEL_MAX, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(-V_WHEEL_MAX, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("v_wheel [m/s]")
        ax.legend(ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


def compare(vertical=False):
    """Generate comparison visualization of baseline and optimized."""
    bl = load_csv(baseline_csv())
    opt = load_csv(optimized_csv())
    animate_comparison(bl, opt, comparison_gif(), vertical=vertical)
    plot_velocity_comparison(bl, opt, comparison_png())
