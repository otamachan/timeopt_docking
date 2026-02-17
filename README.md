# Time-Optimal Trajectory Planning for Differential-Drive Robot Docking

> **Experimental** — This is a simplified study for exploring time-optimal trajectory planning, not production-ready code.

A small experiment comparing a naive analytical baseline with NLP-based time-optimal trajectory planning for a differential-drive robot (like [Kachaka](https://kachaka.life/)) docking at a charging station, using CasADi + IPOPT.

## Motivation

A typical docking trajectory follows a rotate-translate-rotate-reverse pattern. How much faster can we dock if we solve for the time-optimal trajectory instead? This toy problem explores that question under kinematic and actuator constraints (dynamics such as inertia and friction are not modeled).

![Image](https://github.com/user-attachments/assets/929c20bb-0f96-490a-9632-babd99eac864)

## Problem Definition

### Robot Model

Differential-drive robot with state $(x, y, \theta)$ and control inputs $(v, \omega)$.

**Kinematics:**

$$\dot{x} = v \cos\theta, \quad \dot{y} = v \sin\theta, \quad \dot{\theta} = \omega$$

**Wheel speeds** (differential-drive coupling):

$$v_L = v - \omega \cdot \frac{d}{2}, \quad v_R = v + \omega \cdot \frac{d}{2}$$

where $d$ is the wheel tread (track width).

### Docking Scenario

```
              y
              ^
              |     Start (x, y, θ)
              |       *
              |
Dock          | Waypoint
(-cx, 0, 0)  | (0,0)
  < < < < < < +-------------------> x
  reverse at
  constant speed
```

- **Start**: configurable $(x, y, \theta)$, default $(0.5, 0.2, -\pi)$
- **Waypoint**: $(0, 0, 0)$ — the robot must pass through this pose with $v = -c_v$, $\omega = 0$
- **Dock**: $(-c_x, 0, 0)$ — reached by constant-speed reverse from the waypoint (not optimized)
- **Optimization objective**: minimize travel time $T$ from Start to Waypoint

### Constraints

| Constraint | Symbol | Value | Description |
|---|---|---|---|
| Body velocity | $\|v\| \le V_{max}$ | 0.3 m/s | Translational speed limit |
| Body angular velocity | $\|\omega\| \le \omega_{max}$ | 1.5 rad/s | Rotational speed limit |
| Body acceleration | $\|\dot{v}\| \le a_{max}$ | 0.5 m/s² | Translational acceleration limit |
| Body angular acceleration | $\|\dot{\omega}\| \le \alpha_{max}$ | 2.5 rad/s² | Rotational acceleration limit |
| Wheel velocity | $\|v_{L,R}\| \le V_{wheel}$ | 0.4 m/s | Per-wheel speed limit |
| Wheel acceleration | $\|\dot{v}_{L,R}\| \le a_{wheel}$ | 0.7 m/s² | Per-wheel acceleration limit |
| Tread width | $d$ | 0.2 m | Wheel-to-wheel distance |

## Baseline

The baseline generates a four-phase trajectory:

1. **Rotate** in place to face the waypoint
2. **Translate** straight toward the waypoint (trapezoidal velocity profile)
3. **Rotate** in place to align heading to 0
4. **Reverse** at constant $c_v = 0.05$ m/s to the dock

Each phase uses trapezoidal acceleration/deceleration profiles. The intermediate x-coordinate (`mid_x`) is optimized via scipy to minimize total time.

## Optimization Method

### NLP Formulation

**Decision variables** ($5(N+1) + 1 = 306$ variables with $N = 60$):

$$x_k, \; y_k, \; \theta_k, \; v_k, \; \omega_k \quad (k = 0, \dots, N), \quad T$$

**Objective:**

$$\min \; T$$

**Dynamics** (trapezoidal collocation):

$$x_{k+1} = x_k + \frac{\Delta t}{2}\bigl(v_k \cos\theta_k + v_{k+1} \cos\theta_{k+1}\bigr)$$

$$y_{k+1} = y_k + \frac{\Delta t}{2}\bigl(v_k \sin\theta_k + v_{k+1} \sin\theta_{k+1}\bigr)$$

$$\theta_{k+1} = \theta_k + \frac{\Delta t}{2}\bigl(\omega_k + \omega_{k+1}\bigr)$$

where $\Delta t = T / N$.

**Boundary conditions:**

$$\mathbf{z}_0 = (s_x,\; s_y,\; s_\theta,\; 0,\; 0), \quad \mathbf{z}_N = (0,\; 0,\; 0,\; -c_v,\; 0)$$

**Box constraints:**

$$|v_k| \le V_{max}, \quad |\omega_k| \le \omega_{max}$$

$$\left|\frac{v_{k+1} - v_k}{\Delta t}\right| \le a_{max}, \quad \left|\frac{\omega_{k+1} - \omega_k}{\Delta t}\right| \le \alpha_{max}$$

**Wheel coupling constraints:**

$$|v_k \pm \omega_k \cdot d/2| \le V_{wheel}$$

$$\left|\frac{v_{k+1} - v_k}{\Delta t} \pm \frac{\omega_{k+1} - \omega_k}{\Delta t} \cdot \frac{d}{2}\right| \le a_{wheel}$$

### Solver Configuration

- **Solver**: [CasADi](https://web.casadi.org/) + [IPOPT](https://github.com/coin-or/Ipopt)
- **Hessian**: L-BFGS approximation (`limited-memory`)
- **Tolerance**: $10^{-6}$
- **Initial guess**: interpolated from the baseline trajectory

The optimization covers only the Start-to-Waypoint segment. The reverse segment (Waypoint-to-Dock) is appended as a fixed constant-speed phase.

## Results

Reverse segment (Waypoint to Dock) is fixed at 4.0 s. Times in parentheses exclude this segment.

| Start $(x, y, \theta)$ | Baseline | Optimized | Improvement | Solve time |
|---|---|---|---|---|
| $(0.6, 0.0, -\pi)$ | 9.35 s (5.35) | 7.49 s (3.49) | ~20% (~35%) | ~0.5 s |
| $(0.5, 0.3, -\pi/2)$ | 10.21 s (6.21) | 7.65 s (3.65) | ~25% (~41%) | ~0.4 s |
| $(0.5, 0.2, -\pi)$ | 9.67 s (5.67) | 7.29 s (3.29) | ~25% (~42%) | ~0.5 s |

### Start $(0.6, 0.0, -\pi)$

![Comparison](https://github.com/user-attachments/assets/2b20136a-41b4-45b8-b7da-7d18944e3198)

<img width="750" alt="Velocity" src="https://github.com/user-attachments/assets/84d5e951-f330-41b5-ba0a-c257da463d10" />

### Start $(0.5, 0.3, -\pi/2)$

![Comparison](https://github.com/user-attachments/assets/de64e8dd-e50e-41f3-93ab-23e896cffa63)

<img width="750" alt="Velocity" src="https://github.com/user-attachments/assets/5ccd05b1-7932-4cad-a945-54cec6d3fb66" />

### Start $(0.5, 0.2, -\pi)$

![Comparison](https://github.com/user-attachments/assets/929c20bb-0f96-490a-9632-babd99eac864)

<img width="750" alt="Velocity" src="https://github.com/user-attachments/assets/79e0cfa7-9965-44f7-b2ed-42ac11144c01" />

## Discussion

The baseline decomposes the trajectory into sequential rotate-translate-rotate phases, so angular velocity stays near zero during translation and vice versa. The optimizer overlaps rotation and translation: at full translational speed ($v = 0.3$ m/s), the wheel velocity coupling ($|v \pm \omega \cdot d/2| \le 0.4$ m/s) still permits $\omega$ up to 1.0 rad/s — 67% of the body limit. By turning while moving, the optimized trajectory shortens or removes the dedicated rotation phases, which is the main source of time savings.

## Usage

All scripts use [uv](https://docs.astral.sh/uv/) with inline script metadata for dependency management. No virtual environment setup needed.

```bash
# Generate baseline trajectory + GIF
uv run baseline.py

# Optimize mid_x for baseline
uv run baseline.py optimize

# Run NLP optimization + GIF
uv run optimize.py

# Generate comparison GIF + velocity plot
uv run optimize.py compare

# Run all of the above at once
uv run run_all.py

# Custom start position (x y theta)
uv run run_all.py --start 0.5 0.0 3.14159
```

Outputs are saved to `output/` with a suffix indicating the start position (e.g., `_x0.5_y0.2_t-3.14` for the default start):
- `baseline_trajectory_x0.5_y0.2_t-3.14.gif` / `.csv`
- `optimized_trajectory_x0.5_y0.2_t-3.14.gif` / `.csv`
- `comparison_trajectory_x0.5_y0.2_t-3.14.gif`
- `comparison_velocity_x0.5_y0.2_t-3.14.png`
- `midx_optimization_x0.5_y0.2_t-3.14.png`

## Project Structure

```
.
├── timeopt_common.py   # Shared constants, utilities, CSV I/O, visualization
├── baseline.py         # Analytical baseline trajectory generation
├── optimize.py         # CasADi + IPOPT time-optimal NLP solver
├── run_all.py          # Orchestrator script to run all commands
└── output/             # Generated GIFs, PNGs, and CSVs
```

## License

Apache License 2.0
