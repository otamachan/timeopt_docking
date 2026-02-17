# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "casadi",
#     "matplotlib",
#     "numpy",
#     "pillow",
#     "scipy",
# ]
# ///
"""Run all trajectory computations and visualizations.

Usage:
  uv run run_all.py                          # default start
  uv run run_all.py --start 0.3 0.5 -1.0    # custom start
"""

import subprocess
import sys


def main():
    start_args = []
    vertical_args = []
    argv = sys.argv[1:]
    if "--start" in argv:
        idx = argv.index("--start")
        start_args = argv[idx : idx + 4]
    if "--vertical" in argv:
        vertical_args = ["--vertical"]

    commands = [
        ["uv", "run", "baseline.py"] + start_args,
        ["uv", "run", "baseline.py", "optimize"] + start_args,
        ["uv", "run", "optimize.py"] + start_args,
        ["uv", "run", "optimize.py", "compare"] + vertical_args + start_args,
    ]

    for cmd in commands:
        print(f"\n{'=' * 60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'=' * 60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {' '.join(cmd)}")
            sys.exit(result.returncode)

    print(f"\n{'=' * 60}")
    print("All done. Check output/ for results.")


if __name__ == "__main__":
    main()
