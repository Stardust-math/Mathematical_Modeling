"""Entry point for stochastic Gillespie SIR experiments."""

from __future__ import annotations

import argparse
import time

import config as cfg
from experiments_stochastic import run_all_stochastic_experiments


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run stochastic Gillespie SIR experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="full", help="Use quick for testing or full for final outputs.")
    return parser.parse_args()


def main() -> None:
    """Run stochastic experiments."""
    args = parse_args()
    start = time.perf_counter()
    run_all_stochastic_experiments(mode=args.mode)
    elapsed = time.perf_counter() - start
    print("Stochastic Gillespie SIR experiments completed.")
    print(f"Mode: {args.mode}")
    print(f"Figures saved to: {cfg.FIG_DIR}")
    print(f"CSV results saved to: {cfg.RESULT_DIR}")
    print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
