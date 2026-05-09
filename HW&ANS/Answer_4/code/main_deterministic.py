"""Entry point for deterministic SIR experiments."""

from __future__ import annotations

import argparse
import time

import config as cfg
from experiments_deterministic import run_all_deterministic_experiments


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run deterministic SIR periodic-outbreak experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="full", help="Use quick for testing or full for final outputs.")
    parser.add_argument("--method", choices=["rk4", "solve_ivp", "ivp", "scipy"], default="rk4", help="ODE solver for non-vectorized deterministic experiments.")
    return parser.parse_args()


def main() -> None:
    """Run all deterministic experiments."""
    args = parse_args()
    start = time.perf_counter()
    run_all_deterministic_experiments(method=args.method, mode=args.mode)
    elapsed = time.perf_counter() - start
    print("Deterministic SIR experiments completed.")
    print(f"Mode: {args.mode}")
    print(f"Solver method: {args.method}")
    print(f"Figures saved to: {cfg.FIG_DIR}")
    print(f"CSV results saved to: {cfg.RESULT_DIR}")
    print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
