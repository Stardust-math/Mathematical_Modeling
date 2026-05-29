from __future__ import annotations
from .config import ensure_directories
from .experiments_synthetic import run_synthetic_experiments


def main():
    ensure_directories()
    return run_synthetic_experiments(generate_figures=False)


if __name__ == '__main__':
    main()
