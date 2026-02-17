# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "casadi",
#     "numpy",
#     "matplotlib",
#     "pillow",
# ]
# ///
"""
Time-optimal trajectory planning via CasADi + IPOPT.

Optimization segment: Start (sx, sy, sth) -> waypoint (0, 0, 0)
Terminal conditions:   v = -cv, omega = 0
Reverse segment:       waypoint -> dock at constant cv (not optimized)

Decision variables (trapezoidal collocation, N+1 nodes):
  x[k], y[k], theta[k], v[k], omega[k]  (k=0..N)
  T  (total time)

Subcommands:
  uv run optimize.py              -- optimize + GIF
  uv run optimize.py compare      -- comparison visualization with baseline
"""

import time

import casadi as ca
import numpy as np

import timeopt_common
from timeopt_common import (
    CHARGER_X, START, DOCK, V_MAX, W_MAX, A_MAX, ALPHA_MAX, CHARGE_V,
    TREAD, V_WHEEL_MAX, A_WHEEL_MAX, DT,
    baseline_csv, optimized_gif, optimized_csv,
    save_csv, load_csv, animate_trajectory, compare, wheel_speeds,
)

N = 60  # number of discretization intervals


def build_and_solve():
    """Build the NLP and solve with IPOPT. Returns optimal T and state arrays."""
    sx, sy, sth = START
    opti = ca.Opti()

    X = opti.variable(N + 1)
    Y = opti.variable(N + 1)
    TH = opti.variable(N + 1)
    V = opti.variable(N + 1)
    W = opti.variable(N + 1)
    T = opti.variable()

    dt = T / N

    opti.minimize(T)

    for k in range(N):
        opti.subject_to(X[k + 1] == X[k] + (dt / 2) * (V[k] * ca.cos(TH[k]) + V[k + 1] * ca.cos(TH[k + 1])))
        opti.subject_to(Y[k + 1] == Y[k] + (dt / 2) * (V[k] * ca.sin(TH[k]) + V[k + 1] * ca.sin(TH[k + 1])))
        opti.subject_to(TH[k + 1] == TH[k] + (dt / 2) * (W[k] + W[k + 1]))

    opti.subject_to(X[0] == sx)
    opti.subject_to(Y[0] == sy)
    opti.subject_to(TH[0] == sth)
    opti.subject_to(V[0] == 0.0)
    opti.subject_to(W[0] == 0.0)

    opti.subject_to(X[N] == 0.0)
    opti.subject_to(Y[N] == 0.0)
    opti.subject_to(TH[N] == 0.0)
    opti.subject_to(V[N] == -CHARGE_V)
    opti.subject_to(W[N] == 0.0)

    for k in range(N + 1):
        opti.subject_to(opti.bounded(-V_MAX, V[k], V_MAX))
        opti.subject_to(opti.bounded(-W_MAX, W[k], W_MAX))

    for k in range(N):
        dv = (V[k + 1] - V[k]) / dt
        dw = (W[k + 1] - W[k]) / dt
        opti.subject_to(opti.bounded(-A_MAX, dv, A_MAX))
        opti.subject_to(opti.bounded(-ALPHA_MAX, dw, ALPHA_MAX))

    if V_WHEEL_MAX is not None:
        for k in range(N + 1):
            v_l, v_r = wheel_speeds(V[k], W[k])
            opti.subject_to(opti.bounded(-V_WHEEL_MAX, v_l, V_WHEEL_MAX))
            opti.subject_to(opti.bounded(-V_WHEEL_MAX, v_r, V_WHEEL_MAX))

    if A_WHEEL_MAX is not None:
        half_tread = TREAD / 2.0
        for k in range(N):
            dv = (V[k + 1] - V[k]) / dt
            dw = (W[k + 1] - W[k]) / dt
            a_l = dv - dw * half_tread
            a_r = dv + dw * half_tread
            opti.subject_to(opti.bounded(-A_WHEEL_MAX, a_l, A_WHEEL_MAX))
            opti.subject_to(opti.bounded(-A_WHEEL_MAX, a_r, A_WHEEL_MAX))

    opti.subject_to(T >= 0.1)

    z0_x = np.zeros(N + 1)
    z0_y = np.zeros(N + 1)
    z0_th = np.zeros(N + 1)
    z0_v = np.zeros(N + 1)
    z0_w = np.zeros(N + 1)
    T_guess = 5.0

    try:
        bl = load_csv(baseline_csv())
        best_idx = int(np.argmin(np.abs(bl["x"]) + np.abs(bl["y"]) + np.abs(bl["theta"])))

        T_guess = bl["t"][best_idx]
        target_ts = np.linspace(0, T_guess, N + 1)
        baseline_ts = bl["t"][:best_idx + 1]

        for field, arr in [("x", z0_x), ("y", z0_y), ("theta", z0_th),
                           ("v", z0_v), ("omega", z0_w)]:
            arr[:] = np.interp(target_ts, baseline_ts, bl[field][:best_idx + 1])

        z0_v[0] = 0.0
        z0_w[0] = 0.0
        z0_x[-1] = 0.0
        z0_y[-1] = 0.0
        z0_th[-1] = 0.0
        z0_v[-1] = -CHARGE_V
        z0_w[-1] = 0.0
        print(f"Loaded baseline initial guess (T_guess={T_guess:.2f} s)")
    except FileNotFoundError:
        for k in range(N + 1):
            a = k / N
            z0_x[k] = sx * (1 - a)
            z0_y[k] = sy * (1 - a)
            z0_th[k] = sth + a * (0.0 - sth)
            z0_v[k] = -CHARGE_V * a
        T_guess = 5.0
        print("No baseline CSV found, using linear interpolation")

    opti.set_initial(X, z0_x)
    opti.set_initial(Y, z0_y)
    opti.set_initial(TH, z0_th)
    opti.set_initial(V, z0_v)
    opti.set_initial(W, z0_w)
    opti.set_initial(T, T_guess)

    opti.solver("ipopt", {}, {
        "max_iter": 3000,
        "tol": 1e-6,
        "print_level": 5,
        "hessian_approximation": "limited-memory",
    })

    print(f"Variables: {5 * (N + 1) + 1}, N={N}")
    print("Solving with CasADi + IPOPT...")

    t_start = time.perf_counter()
    sol = opti.solve()
    solve_time = time.perf_counter() - t_start

    T_opt = float(sol.value(T))
    print(f"\nOptimal T = {T_opt:.4f} s")
    print(f"Solve wall time: {solve_time:.3f} s")
    print(f"Total (with dock approach): {T_opt + CHARGER_X / CHARGE_V:.4f} s")

    x_sol = np.array(sol.value(X)).flatten()
    y_sol = np.array(sol.value(Y)).flatten()
    th_sol = np.array(sol.value(TH)).flatten()
    v_sol = np.array(sol.value(V)).flatten()
    w_sol = np.array(sol.value(W)).flatten()

    print("\n=== Constraint Check ===")
    dt_val = T_opt / N

    print(f"Start: x={x_sol[0]:.8f} y={y_sol[0]:.8f} th={th_sol[0]:.8f} v={v_sol[0]:.8f} w={w_sol[0]:.8f}")
    print(f"End:   x={x_sol[-1]:.8f} y={y_sol[-1]:.8f} th={th_sol[-1]:.8f} v={v_sol[-1]:.8f} w={w_sol[-1]:.8f}")

    v_viol = max(abs(v_sol)) - V_MAX
    w_viol = max(abs(w_sol)) - W_MAX
    print(f"|v| max: {max(abs(v_sol)):.8f}  (limit {V_MAX})  viol={v_viol:.2e}  {'OK' if v_viol <= 1e-6 else 'VIOLATED'}")
    print(f"|w| max: {max(abs(w_sol)):.8f}  (limit {W_MAX})  viol={w_viol:.2e}  {'OK' if w_viol <= 1e-6 else 'VIOLATED'}")

    accels = np.diff(v_sol) / dt_val
    alphas = np.diff(w_sol) / dt_val
    a_viol = max(abs(accels)) - A_MAX
    al_viol = max(abs(alphas)) - ALPHA_MAX
    print(f"|dv/dt| max: {max(abs(accels)):.8f}  (limit {A_MAX})  viol={a_viol:.2e}  {'OK' if a_viol <= 1e-4 else 'VIOLATED'}")
    print(f"|dw/dt| max: {max(abs(alphas)):.8f}  (limit {ALPHA_MAX})  viol={al_viol:.2e}  {'OK' if al_viol <= 1e-4 else 'VIOLATED'}")

    if V_WHEEL_MAX is not None:
        vl_sol, vr_sol = wheel_speeds(v_sol, w_sol)
        vl_max = max(abs(vl_sol))
        vr_max = max(abs(vr_sol))
        vl_viol = vl_max - V_WHEEL_MAX
        vr_viol = vr_max - V_WHEEL_MAX
        print(f"|v_L| max: {vl_max:.8f}  (limit {V_WHEEL_MAX})  viol={vl_viol:.2e}  {'OK' if vl_viol <= 1e-6 else 'VIOLATED'}")
        print(f"|v_R| max: {vr_max:.8f}  (limit {V_WHEEL_MAX})  viol={vr_viol:.2e}  {'OK' if vr_viol <= 1e-6 else 'VIOLATED'}")

    if A_WHEEL_MAX is not None:
        half_tread = TREAD / 2.0
        al_sol = accels - alphas * half_tread
        ar_sol = accels + alphas * half_tread
        al_max = max(abs(al_sol))
        ar_max = max(abs(ar_sol))
        al_viol = al_max - A_WHEEL_MAX
        ar_viol = ar_max - A_WHEEL_MAX
        print(f"|a_L| max: {al_max:.8f}  (limit {A_WHEEL_MAX})  viol={al_viol:.2e}  {'OK' if al_viol <= 1e-4 else 'VIOLATED'}")
        print(f"|a_R| max: {ar_max:.8f}  (limit {A_WHEEL_MAX})  viol={ar_viol:.2e}  {'OK' if ar_viol <= 1e-4 else 'VIOLATED'}")

    dyn_max = 0.0
    for k in range(N):
        ex = abs(x_sol[k+1] - x_sol[k] - (dt_val/2)*(v_sol[k]*np.cos(th_sol[k]) + v_sol[k+1]*np.cos(th_sol[k+1])))
        ey = abs(y_sol[k+1] - y_sol[k] - (dt_val/2)*(v_sol[k]*np.sin(th_sol[k]) + v_sol[k+1]*np.sin(th_sol[k+1])))
        eth = abs(th_sol[k+1] - th_sol[k] - (dt_val/2)*(w_sol[k] + w_sol[k+1]))
        dyn_max = max(dyn_max, ex, ey, eth)
    print(f"Dynamics max residual: {dyn_max:.2e}  {'OK' if dyn_max < 1e-6 else 'CHECK'}")

    return T_opt, x_sol, y_sol, th_sol, v_sol, w_sol


def extract_trajectory(T_opt, x_sol, y_sol, th_sol, v_sol, w_sol):
    """Combine optimization result with constant-speed reverse segment."""
    dt = T_opt / N
    rows = []

    for k in range(N + 1):
        t = k * dt
        rows.append((t, x_sol[k], y_sol[k], th_sol[k], v_sol[k], w_sol[k]))

    dock_time = CHARGER_X / CHARGE_V
    t_back = 0.0
    x = 0.0
    while t_back < dock_time - DT * 0.5:
        t_back += DT
        x += -CHARGE_V * DT
        rows.append((T_opt + t_back, x, 0.0, 0.0, -CHARGE_V, 0.0))

    rows.append((T_opt + dock_time, DOCK[0], 0.0, 0.0, 0.0, 0.0))
    return rows


def main():
    T_opt, x_sol, y_sol, th_sol, v_sol, w_sol = build_and_solve()

    trajectory = extract_trajectory(T_opt, x_sol, y_sol, th_sol, v_sol, w_sol)
    total_time = trajectory[-1][0]
    print(f"\nTotal time (incl. dock approach): {total_time:.3f} s")

    save_csv(trajectory, optimized_csv(), precision=10)
    animate_trajectory(
        trajectory, optimized_gif(),
        "CasADi+IPOPT Optimized: Time-Optimal Trajectory",
        markers=[(0, 0, "s", 8, "Waypoint")],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "THETA"))
    parser.add_argument("--vertical", action="store_true", help="Use vertical (2x1) layout for comparison GIF")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "compare"])
    args = parser.parse_args()

    if args.start:
        timeopt_common.set_start(*args.start)
        START = timeopt_common.START

    if args.command == "compare":
        compare(vertical=args.vertical)
    else:
        main()
