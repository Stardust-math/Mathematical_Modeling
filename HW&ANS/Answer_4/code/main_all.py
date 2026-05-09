"""One-click entry point for all deterministic and stochastic experiments."""

from __future__ import annotations

import argparse
import time

import config as cfg
from experiments_deterministic import run_all_deterministic_experiments
from experiments_stochastic import run_all_stochastic_experiments


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run all SIR periodic-outbreak experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="full", help="quick is for testing; full generates formal outputs.")
    parser.add_argument("--method", choices=["rk4", "solve_ivp", "ivp", "scipy"], default="rk4", help="ODE solver for non-vectorized deterministic experiments.")
    return parser.parse_args()


def main() -> None:
    """Run deterministic and stochastic experiments."""
    args = parse_args()
    start = time.perf_counter()
    print(f"Running all SIR experiments in {args.mode!r} mode...")
    run_all_deterministic_experiments(method=args.method, mode=args.mode)
    run_all_stochastic_experiments(mode=args.mode)
    elapsed = time.perf_counter() - start
    print("All SIR experiments completed.")
    print(f"Mode: {args.mode}")
    print(f"Figures saved to: {cfg.FIG_DIR}")
    print(f"CSV results saved to: {cfg.RESULT_DIR}")
    print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
